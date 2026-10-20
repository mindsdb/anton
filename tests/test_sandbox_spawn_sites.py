"""Every spawn of agent code under a sandbox policy goes through the launcher.

Each capture test reads the policy descriptor the spawn was handed, so it
checks what the launcher would apply: the target program, the allowlisted
environment and the rules. Refusal tests check that a launcher refusal is
named and never costs the person their venv or saved cell state.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import anton.core.backends.local as local
from anton.core.artifacts import backend_launcher
from anton.core.backends import sandbox
from anton.core.backends.local import _BOOT_SCRIPT_PATH, LocalScratchpadRuntime, snapshot_file
from anton.core.backends.manager import ScratchpadManager
from anton.core.backends.sandbox import BACKENDS_OFF_MESSAGE, CellSandbox, SandboxUnavailable
from anton.core.tools import tool_handlers
from anton.core.tools.generate_artifact import orchestrator, verifiers
from anton.core.tools.registry import ToolOutcome
from anton.core.utils.scratchpad import prepare_scratchpad_exec

# The launcher, its descriptors and these POSIX paths need a POSIX host.
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the sandbox launcher is POSIX-only")

_LAUNCHER_ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"}
_SECRETS = {
    "COWORK_AUTH_TOKEN": "tok-secret",
    "DATABASE_URI": "postgresql://cowork:db-secret@postgres/cowork",
    "MINDS_API_KEY": "minds-secret",
    "ANTON_MINDS_API_KEY": "anton-minds-secret",
}
# A mirror's password and token pass the allowlist but no spawn under a policy
# gets them: installs and the health check run the venv's Python, which a
# cell can rewrite.
_PIP_SERVER_ENV = {
    "PIP_INDEX_URL": "https://mirror-user:pip-secret@mirror.internal:8443/pypi/simple",
    "PIP_EXTRA_INDEX_URL": "https://extra.internal/simple?token=pip-token-secret",
}
_PIP_AGENT_ENV = {
    "PIP_INDEX_URL": "https://mirror.internal:8443/pypi/simple",
    "PIP_EXTRA_INDEX_URL": "https://extra.internal/simple",
}


@pytest.fixture
def server_env(monkeypatch) -> None:
    for key in list(os.environ):
        monkeypatch.delenv(key)
    for key, value in {
        **_SECRETS,
        **_PIP_SERVER_ENV,
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/anton",
        "LD_PRELOAD": "/opt/preload.so",
        "ANTON_SCRATCHPAD_PERSIST_SESSION": "true",
        "ANTON_LANGFUSE_HEADERS": "1",
        "DS_SERVER__PASSWORD": "server-ds",
    }.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    folder = tmp_path / "project"
    folder.mkdir()
    return folder


@pytest.fixture
def policy(tmp_path: Path, project: Path) -> CellSandbox:
    for name in ("home", "tmp"):
        (tmp_path / name).mkdir()
    return CellSandbox(project_dir=project, home_dir=tmp_path / "home", tmp_dir=tmp_path / "tmp", user_name="bob")


def _payload(kwargs: dict) -> dict:
    fd, _refusal_write = kwargs["pass_fds"]
    os.lseek(fd, 0, os.SEEK_SET)
    chunks = []
    while chunk := os.read(fd, 1 << 16):
        chunks.append(chunk)
    return json.loads(b"".join(chunks))


def _launcher_argv(args: tuple, kwargs: dict) -> list[str]:
    """The part of a launch argv after the launcher and its two descriptors: the target."""
    fd, refusal_write = kwargs["pass_fds"]
    head = [
        sys.executable, "-I", "-S", str(sandbox.launcher_path()),
        "--policy-fd", str(fd), "--refusal-fd", str(refusal_write), "--",
    ]
    assert list(args[: len(head)]) == head
    return list(args[len(head):])


def _assert_no_server_secret(payload: dict, args: tuple, policy: CellSandbox) -> None:
    env = payload["env"]
    for key in (*_SECRETS, "LD_PRELOAD", "ANTON_UV_PATH"):
        assert key not in env, key
    text = " ".join(args) + json.dumps(env)
    for secret in ("tok-secret", "db-secret", "minds-secret", "pip-secret", "pip-token-secret"):
        assert secret not in text
    assert env["HOME"] == str(policy.home_dir)
    assert env["TMPDIR"] == str(policy.tmp_dir)
    assert env["USER"] == "bob"


class _Proc:
    def __init__(self, *, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0) -> None:
        self._out = (stdout, stderr)
        self.returncode = returncode
        self.pid = 4242
        self.stdin = SimpleNamespace(write=lambda _b: None, close=lambda: None, is_closing=lambda: False)

    async def communicate(self):
        return self._out

    async def wait(self):
        return self.returncode

    def kill(self):
        pass


def _venv(project: Path, name: str = "main") -> Path:
    venv = project / ".anton" / "scratchpad-venvs" / name
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text("")
    (venv / ".python_version").write_text(f"{sys.version_info.major}.{sys.version_info.minor}\n")
    return venv


def _pad(project: Path, policy: CellSandbox | None, **kwargs) -> LocalScratchpadRuntime:
    venv = project / ".anton" / "scratchpad-venvs" / "main"
    if not venv.exists():
        _venv(project)
    pad = LocalScratchpadRuntime(
        "main",
        coding_provider="openai",
        coding_model="gpt-test",
        coding_api_key="coding-key",
        coding_base_url="https://model-route/v1",
        workspace_path=project,
        cell_sandbox=policy,
        **kwargs,
    )
    pad._venv_dir = str(venv)
    pad._venv_python = str(venv / "bin" / "python")
    return pad


# ── The sites ────────────────────────────────────────────────────────────────


async def test_cell_launch_preparation_leaves_the_event_loop(project, policy, monkeypatch, server_env):
    pad = _pad(project, policy)
    monkeypatch.setattr(pad, "_verify_venv_python", lambda: True)
    loop_thread = threading.get_ident()
    original = CellSandbox.launch
    preparations = []
    launches = []

    def prepare(self, *args, **kwargs):
        preparations.append(threading.get_ident())
        assert threading.get_ident() != loop_thread, "launch preparation blocks the event loop"
        launch = original(self, *args, **kwargs)
        launches.append(launch)
        return launch

    async def capture(*args, **kwargs):
        assert threading.get_ident() == loop_thread, "subprocess creation moved off its event loop"
        return _Proc()

    monkeypatch.setattr(CellSandbox, "launch", prepare)
    monkeypatch.setattr(local.asyncio, "create_subprocess_exec", capture)
    try:
        await pad.start()
        assert len(preparations) == 1
    finally:
        await pad.close()
    for fd in (*launches[0].pass_fds, launches[0].refusal_fd):
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.parametrize("failure", [OSError, asyncio.CancelledError])
async def test_cell_spawn_failure_closes_prepared_descriptors(project, policy, monkeypatch, failure):
    pad = _pad(project, policy)
    monkeypatch.setattr(pad, "_verify_venv_python", lambda: True)
    original = CellSandbox.launch
    launches = []
    saved = _snapshot(project)

    def prepare(self, *args, **kwargs):
        launch = original(self, *args, **kwargs)
        launches.append(launch)
        return launch

    async def fail(*args, **kwargs):
        assert _payload(kwargs)["rules"]
        raise failure("spawn stopped")

    monkeypatch.setattr(CellSandbox, "launch", prepare)
    monkeypatch.setattr(local.asyncio, "create_subprocess_exec", fail)
    with pytest.raises(SandboxUnavailable if failure is OSError else failure, match="spawn stopped"):
        await pad.start()
    assert saved.read_bytes() == b"saved namespace"
    assert Path(pad._venv_python).exists()
    assert pad._proc is None
    assert len(launches) == 1
    for fd in (*launches[0].pass_fds, launches[0].refusal_fd):
        with pytest.raises(OSError):
            os.fstat(fd)


async def test_cell_launch_preparation_refusal_preserves_venv_and_snapshot(project, policy, monkeypatch):
    pad = _pad(project, policy)
    monkeypatch.setattr(pad, "_verify_venv_python", lambda: True)
    saved = _snapshot(project)

    def refuse(self, *args, **kwargs):
        raise SandboxUnavailable("launch preparation refused")

    async def spawn(*args, **kwargs):
        pytest.fail("a refused preparation started a process")

    monkeypatch.setattr(CellSandbox, "launch", refuse)
    monkeypatch.setattr(local.asyncio, "create_subprocess_exec", spawn)
    with pytest.raises(SandboxUnavailable, match="launch preparation refused"):
        await pad.start()
    assert saved.read_bytes() == b"saved namespace"
    assert Path(pad._venv_python).exists()
    assert pad._proc is None


async def test_close_waits_for_the_process_prepared_by_a_running_start(project, policy, monkeypatch):
    pad = _pad(project, policy)
    monkeypatch.setattr(pad, "_verify_venv_python", lambda: True)
    ready, release = threading.Event(), threading.Event()
    spawning, spawned = asyncio.Event(), asyncio.Event()
    original = CellSandbox.launch
    launches = []

    def prepare(self, *args, **kwargs):
        launch = original(self, *args, **kwargs)
        launches.append(launch)
        ready.set()
        assert release.wait(5)
        return launch

    async def capture(*args, **kwargs):
        spawning.set()
        await spawned.wait()
        return _Proc()

    monkeypatch.setattr(CellSandbox, "launch", prepare)
    monkeypatch.setattr(local.asyncio, "create_subprocess_exec", capture)
    starting = asyncio.create_task(pad.start())
    closing = None
    try:
        assert await asyncio.to_thread(ready.wait, 1)
        closing = asyncio.create_task(pad.close())
        await asyncio.sleep(0)
        release.set()
        await asyncio.wait_for(spawning.wait(), 5)
        await asyncio.sleep(0.02)
        assert not closing.done(), "close returned before the starting process became owned"
        spawned.set()
        await asyncio.wait_for(asyncio.gather(starting, closing), 5)
        assert pad._proc is None
        assert pad._cell_launch is None
        for fd in (*launches[0].pass_fds, launches[0].refusal_fd):
            with pytest.raises(OSError):
                os.fstat(fd)
    finally:
        release.set()
        spawned.set()
        await asyncio.gather(*(task for task in (starting, closing) if task is not None), return_exceptions=True)
        await pad.close()


@pytest.mark.parametrize("teardown", ["close", "cleanup"])
async def test_teardown_during_venv_verification_waits_for_the_following_launch(
    project, policy, monkeypatch, teardown,
):
    pad = _pad(project, policy)
    verifying, verified = threading.Event(), threading.Event()
    preparing, prepared = threading.Event(), threading.Event()
    original = CellSandbox.launch
    launches = []

    def verify():
        verifying.set()
        assert verified.wait(5)
        return True

    def prepare(self, *args, **kwargs):
        launch = original(self, *args, **kwargs)
        launches.append(launch)
        preparing.set()
        assert prepared.wait(5)
        return launch

    async def capture(*args, **kwargs):
        return _Proc()

    monkeypatch.setattr(pad, "_verify_venv_python", verify)
    monkeypatch.setattr(CellSandbox, "launch", prepare)
    monkeypatch.setattr(local.asyncio, "create_subprocess_exec", capture)
    starting = asyncio.create_task(pad.start())
    closing = None
    try:
        assert await asyncio.to_thread(verifying.wait, 1)
        closing = asyncio.create_task(getattr(pad, teardown)())
        await asyncio.sleep(0)
        verified.set()
        assert await asyncio.to_thread(preparing.wait, 1)
        await asyncio.sleep(0.02)
        assert not closing.done(), "teardown returned while launch preparation was pending"
        prepared.set()
        await asyncio.wait_for(asyncio.gather(starting, closing), 5)
        assert pad._proc is None
        assert pad._cell_launch is None
        for fd in (*launches[0].pass_fds, launches[0].refusal_fd):
            with pytest.raises(OSError):
                os.fstat(fd)
    finally:
        verified.set()
        prepared.set()
        await asyncio.gather(*(task for task in (starting, closing) if task is not None), return_exceptions=True)
        await pad.close()


@pytest.mark.parametrize("teardown", ["close", "cleanup"])
@pytest.mark.parametrize("cancel_teardown", [False, True])
async def test_cancelled_launch_preparation_drains_before_teardown(
    project, policy, monkeypatch, server_env, teardown, cancel_teardown,
):
    pad = _pad(project, policy)
    monkeypatch.setattr(pad, "_verify_venv_python", lambda: True)
    loop_thread = threading.get_ident()
    ready, release = threading.Event(), threading.Event()
    original = CellSandbox.launch
    launches = []
    spawned = []

    def prepare(self, *args, **kwargs):
        assert threading.get_ident() != loop_thread, "launch preparation blocks the event loop"
        launch = original(self, *args, **kwargs)
        launches.append(launch)
        ready.set()
        assert release.wait(5), "the test did not release launch preparation"
        # Concurrent teardown must not clear runtime paths while work uses them.
        assert pad._venv_python is not None
        return launch

    async def capture(*args, **kwargs):
        spawned.append(args)
        return _Proc()

    monkeypatch.setattr(CellSandbox, "launch", prepare)
    monkeypatch.setattr(local.asyncio, "create_subprocess_exec", capture)
    starting = asyncio.create_task(pad.start())
    closing = None
    try:
        assert await asyncio.to_thread(ready.wait, 1), "launch preparation never started"
        starting.cancel()
        await asyncio.sleep(0)
        starting.cancel()
        closing = asyncio.create_task(getattr(pad, teardown)())
        await asyncio.sleep(0.02)
        if cancel_teardown:
            closing.cancel()
            await asyncio.sleep(0)
            closing.cancel()
        assert not starting.done(), "Stop abandoned launch preparation"
        assert not closing.done(), "teardown abandoned launch preparation"
        release.set()
        await asyncio.wait_for(asyncio.gather(starting, closing, return_exceptions=True), 5)
        assert starting.cancelled()
        assert closing.cancelled() is cancel_teardown
        assert not spawned, "a canceled launch started a process"
        assert pad._proc is None
        assert len(launches) == 1
        launch = launches[0]
        for fd in (*launch.pass_fds, launch.refusal_fd):
            with pytest.raises(OSError):
                os.fstat(fd)
    finally:
        release.set()
        await asyncio.gather(*(task for task in (starting, closing) if task is not None), return_exceptions=True)
        await pad.close()


@pytest.mark.parametrize("prepared_before_cancel", [False, True])
async def test_cancelled_preparation_task_closes_launch_descriptors(
    project, policy, monkeypatch, prepared_before_cancel,
):
    """A cancelled asyncio task does not mean its preparation thread has finished."""
    pad = _pad(project, policy)
    monkeypatch.setattr(pad, "_verify_venv_python", lambda: True)
    loop = asyncio.get_running_loop()
    ready, finished = asyncio.Event(), asyncio.Event()
    release = threading.Event()
    original_launch = CellSandbox.launch
    original_work = pad._run_venv_work
    original_close = os.close
    launches = []
    closes = {}

    def prepare(self, *args, **kwargs):
        if not prepared_before_cancel:
            loop.call_soon_threadsafe(ready.set)
            assert release.wait(5), "the test did not release launch preparation"
        launch = original_launch(self, *args, **kwargs)
        launches.append(launch)
        closes.update({fd: 0 for fd in (*launch.pass_fds, launch.refusal_fd)})
        return launch

    async def tracked_work(work):
        if work.__name__ != "prepare_launch":
            return await original_work(work)

        def finish_work():
            try:
                result = work()
                if prepared_before_cancel:
                    loop.call_soon_threadsafe(ready.set)
                    assert release.wait(5), "the test did not release the prepared launch"
                return result
            finally:
                loop.call_soon_threadsafe(finished.set)

        return await original_work(finish_work)

    def close(fd):
        if fd in closes:
            closes[fd] += 1
        original_close(fd)

    async def spawn(*args, **kwargs):
        pytest.fail("a cancelled preparation started a process")

    monkeypatch.setattr(CellSandbox, "launch", prepare)
    monkeypatch.setattr(pad, "_run_venv_work", tracked_work)
    monkeypatch.setattr(local.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(os, "close", close)
    starting = asyncio.create_task(pad.start())
    try:
        await asyncio.wait_for(ready.wait(), 5)
        assert len(pad._venv_workers) == 1
        worker = next(iter(pad._venv_workers))
        starting.cancel("start cancelled")
        worker.cancel("preparation task cancelled")
        with pytest.raises(asyncio.CancelledError, match="start cancelled"):
            await asyncio.wait_for(starting, 5)
        assert pad._proc is None
        assert not pad._pending_starts
        release.set()
        await asyncio.wait_for(finished.wait(), 5)
        assert len(launches) == 1
        assert list(closes.values()) == [1, 1, 1]
        for fd in closes:
            with pytest.raises(OSError):
                os.fstat(fd)
    finally:
        release.set()
        await asyncio.gather(starting, return_exceptions=True)
        await asyncio.wait_for(finished.wait(), 5)
        for launch in launches:
            launch.close()
            launch.refusal()
        await pad.close()


async def test_the_cell_runs_the_installed_boot_script_through_the_launcher(
    project, policy, monkeypatch, server_env
):
    # A uv the server could find must not reach the cell.
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: "/home/anton/.local/bin/uv"))
    pad = _pad(
        project,
        policy,
        scratchpad_ds_env={"DS_PG_MAIN__PASSWORD": "pad-ds"},
        workspace_env_overlay={"FOO": "from-dotenv", "PYTHONPATH": "/evil", "LD_PRELOAD": "/evil.so", "HOME": "/evil"},
    )
    monkeypatch.setattr(pad, "_verify_venv_python", lambda: True)
    seen: dict = {}

    async def capture(*args, **kwargs):
        seen.update(args=args, kwargs=kwargs, payload=_payload(kwargs))
        return _Proc()

    monkeypatch.setattr(local.asyncio, "create_subprocess_exec", capture)
    await pad.start()

    args, kwargs, payload = seen["args"], seen["kwargs"], seen["payload"]
    assert _launcher_argv(args, kwargs) == [pad._venv_python, "-P", str(_BOOT_SCRIPT_PATH)]
    assert pad._boot_path is None
    assert kwargs == {
        "stdin": asyncio.subprocess.PIPE,
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
        "env": _LAUNCHER_ENV,
        "cwd": str(project),
        "start_new_session": sys.platform != "win32",
        "pass_fds": kwargs["pass_fds"],
    }
    _assert_no_server_secret(payload, args, policy)
    env = payload["env"]
    assert env["FOO"] == "from-dotenv"
    assert env["DS_PG_MAIN__PASSWORD"] == "pad-ds" and "DS_SERVER__PASSWORD" not in env
    assert env["OPENAI_API_KEY"] == "coding-key"
    assert env["ANTON_SCRATCHPAD_PERSIST_SESSION"] == "true"
    assert env["ANTON_LANGFUSE_HEADERS"] == "1"
    assert "/evil" not in env["PYTHONPATH"]
    assert env["ANTON_SCRATCHPAD_WORKSPACE_PATH"] == str(project.resolve())
    assert {key: env[key] for key in _PIP_AGENT_ENV} == _PIP_AGENT_ENV
    with pytest.raises(OSError):
        os.fstat(kwargs["pass_fds"][0])


async def test_a_cell_with_no_connections_of_its_own_gets_no_inherited_ds_values(
    project, policy, monkeypatch, server_env
):
    pad = _pad(project, policy)
    monkeypatch.setattr(pad, "_verify_venv_python", lambda: True)
    seen: dict = {}

    async def capture(*args, **kwargs):
        seen.update(payload=_payload(kwargs))
        return _Proc()

    monkeypatch.setattr(local.asyncio, "create_subprocess_exec", capture)
    await pad.start()
    assert not any(key.startswith("DS_") for key in seen["payload"]["env"])


async def test_a_dotenv_key_the_server_holds_never_carries_the_servers_value_in(
    project, policy, monkeypatch, server_env
):
    monkeypatch.setenv("OPERATOR_ONLY_SECRET", "server-value-not-for-cells")
    pad = _pad(
        project,
        policy,
        workspace_env_overlay={"OPERATOR_ONLY_SECRET": "x", "FOO": "from-dotenv"},
    )
    monkeypatch.setattr(pad, "_verify_venv_python", lambda: True)
    seen: dict = {}

    async def capture(*args, **kwargs):
        seen.update(payload=_payload(kwargs))
        return _Proc()

    monkeypatch.setattr(local.asyncio, "create_subprocess_exec", capture)
    await pad.start()

    env = seen["payload"]["env"]
    assert "OPERATOR_ONLY_SECRET" not in env
    assert "server-value-not-for-cells" not in json.dumps(env)
    assert env["FOO"] == "from-dotenv"


def test_the_venv_health_check_runs_through_the_launcher(project, policy, monkeypatch, server_env):
    import subprocess

    pad = _pad(project, policy, scratchpad_ds_env={"DS_PG_MAIN__PASSWORD": "pad-ds"})
    seen: dict = {}

    def capture(*args, **kwargs):
        seen.update(args=args[0], kwargs=kwargs, payload=_payload(kwargs))
        return subprocess.CompletedProcess(args[0], 0, b"ok\n", b"")

    monkeypatch.setattr(subprocess, "run", capture)
    assert pad._verify_venv_python() is True
    assert _launcher_argv(tuple(seen["args"]), seen["kwargs"]) == [pad._venv_python, "-c", "print('ok')"]
    assert seen["kwargs"] == {
        "capture_output": True, "timeout": 5, "env": _LAUNCHER_ENV, "pass_fds": seen["kwargs"]["pass_fds"],
    }
    _assert_no_server_secret(seen["payload"], tuple(seen["args"]), policy)
    assert "DS_SERVER__PASSWORD" not in seen["payload"]["env"]
    assert seen["payload"]["env"]["DS_PG_MAIN__PASSWORD"] == "pad-ds"
    assert {key: seen["payload"]["env"][key] for key in _PIP_AGENT_ENV} == _PIP_AGENT_ENV


def test_a_health_check_whose_target_fakes_a_refusal_rebuilds_the_venv(project, policy, monkeypatch):
    """Exit 87 and the marker on stderr are the venv Python's own (a planted
    .pth can print them); only the launcher's refusal pipe is a refusal."""
    import subprocess

    pad = _pad(project, policy)

    def fake(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 87, b"", b"anton-sandbox: refused: faked by a .pth\n")

    monkeypatch.setattr(subprocess, "run", fake)
    assert pad._verify_venv_python() is False
    assert "faked by a .pth" in pad._last_verify_error


async def test_installs_run_pip_through_the_launcher_and_never_uv(project, policy, monkeypatch, server_env):
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: "/home/anton/.local/bin/uv"))
    pad = _pad(project, policy)
    monkeypatch.setattr(pad, "_verify_venv_python", lambda: True)
    seen: dict = {}

    async def capture(*args, **kwargs):
        seen.update(args=args, kwargs=kwargs, payload=_payload(kwargs))
        return _Proc(stdout=b"installed")

    monkeypatch.setattr(local.asyncio, "create_subprocess_exec", capture)
    assert await pad.install_packages(["pandas"]) == "installed"
    args, kwargs = seen["args"], seen["kwargs"]
    assert _launcher_argv(args, kwargs) == [pad._venv_python, "-m", "pip", "install", "--no-input", "pandas"]
    assert kwargs == {
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.STDOUT,
        "env": _LAUNCHER_ENV,
        "cwd": str(project),
        "pass_fds": kwargs["pass_fds"],
    }
    _assert_no_server_secret(seen["payload"], args, policy)
    assert seen["payload"]["env"]["PIP_CACHE_DIR"] == str(policy.home_dir / ".cache" / "pip")
    assert not any(key.startswith("DS_") for key in seen["payload"]["env"])
    # pip in the venv is the cell's to rewrite, so the install gets the cell's URLs.
    assert {key: seen["payload"]["env"][key] for key in _PIP_AGENT_ENV} == _PIP_AGENT_ENV


def test_venvs_are_built_with_the_standard_library_under_the_sandbox(project, policy, monkeypatch):
    import subprocess

    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: "/home/anton/.local/bin/uv"))
    pad = LocalScratchpadRuntime(
        "fresh", coding_provider="anthropic", coding_model="", coding_api_key="", coding_base_url="",
        workspace_path=project, cell_sandbox=policy,
    )
    ran: list = []
    created: list = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: ran.append(a))
    monkeypatch.setattr(local.venv, "create", lambda path, **k: created.append(path))
    pad._create_venv()
    assert ran == []
    assert created == [str(project / ".anton" / "scratchpad-venvs" / "fresh")]


class _Pool:
    def __init__(self, python: str, policy: CellSandbox | None) -> None:
        self._python = python
        self.cell_sandbox = policy
        self.calls: list[str] = []

    async def get_or_create(self, name):
        self.calls.append(f"get_or_create {name}")
        return SimpleNamespace()

    async def venv_python(self, name):
        self.calls.append(f"venv_python {name}")
        return self._python


async def test_both_verifier_checks_run_through_the_launcher(tmp_path, project, policy, monkeypatch, server_env):
    artifact = project / "artifacts" / "app"
    artifact.mkdir(parents=True)
    (artifact / "backend.py").write_text("import os\nos.environ['DS_PG_MAIN__PASSWORD']\n", encoding="utf-8")
    calls: list[tuple] = []
    results = iter([_Proc(), _Proc(stdout=json.dumps({"import_ok": True}).encode())])

    async def capture(*args, **kwargs):
        calls.append((args, kwargs, _payload(kwargs)))
        return next(results)

    monkeypatch.setattr(verifiers.asyncio, "create_subprocess_exec", capture)
    await verifiers.verify_backend(
        scratchpad_pool=_Pool("/venv/bin/python", policy),
        slug="app",
        artifact_path=artifact,
        ds_env={"DS_PG_MAIN__PASSWORD": "artifact-ds"},
    )

    (compile_args, compile_kwargs, compile_payload), (import_args, import_kwargs, import_payload) = calls
    assert _launcher_argv(compile_args, compile_kwargs) == [
        "/venv/bin/python", "-I", "-B", "-X", "utf8", "-c", verifiers._SYNTAX_CHECK_SCRIPT,
    ]
    assert _launcher_argv(import_args, import_kwargs) == ["/venv/bin/python", "-B", "-c", verifiers._INTROSPECT_SCRIPT]
    for args, kwargs, payload in calls:
        assert kwargs["cwd"] == str(artifact)
        assert kwargs["env"] == _LAUNCHER_ENV
        _assert_no_server_secret(payload, args, policy)
    assert import_payload["env"]["DS_PG_MAIN__PASSWORD"] == "artifact-ds"
    assert "DS_SERVER__PASSWORD" not in import_payload["env"]
    assert {key: import_payload["env"][key] for key in _PIP_AGENT_ENV} == _PIP_AGENT_ENV
    assert "anton_state_pp" not in import_payload["env"].get("PYTHONPATH", "")
    assert not any(k.startswith("DS_") for k in compile_payload["env"])


async def test_the_compile_check_has_a_timeout_under_the_sandbox(tmp_path, policy, monkeypatch):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "backend.py").write_text("app = None\n", encoding="utf-8")
    killed: list[bool] = []

    class _Hung(_Proc):
        async def communicate(self):
            await asyncio.sleep(60)

        def kill(self):
            killed.append(True)

    async def spawn(*args, **kwargs):
        return _Hung()

    monkeypatch.setattr(verifiers, "_COMPILE_TIMEOUT", 0.05)
    monkeypatch.setattr(verifiers.asyncio, "create_subprocess_exec", spawn)
    verdict, _ = await verifiers.verify_backend(
        scratchpad_pool=_Pool("/venv/bin/python", policy), slug="x", artifact_path=artifact
    )
    assert verdict.errors == ["backend.py compile timed out after 0s."]
    assert killed == [True]


async def test_without_the_sandbox_the_compile_check_waits_as_long_as_it_takes(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "backend.py").write_text("app = None\n", encoding="utf-8")

    class _SlowCompile(_Proc):
        async def communicate(self):
            await asyncio.sleep(0.3)
            return b"", b""

        def kill(self):
            raise AssertionError("the compile check was killed")

    procs = iter([_SlowCompile(), _Proc(stdout=json.dumps({"import_ok": True}).encode())])

    async def spawn(*args, **kwargs):
        return next(procs)

    monkeypatch.setattr(verifiers, "_COMPILE_TIMEOUT", 0.05)
    monkeypatch.setattr(verifiers.asyncio, "create_subprocess_exec", spawn)
    verdict, _ = await verifiers.verify_backend(
        scratchpad_pool=_Pool("/venv/bin/python", None), slug="x", artifact_path=artifact
    )
    assert not any("compile timed out" in error for error in verdict.errors)


async def test_the_generator_hands_the_import_check_the_artifacts_own_connections(tmp_path, policy, monkeypatch):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "backend.py").write_text("x = os.environ['DS_PG_MAIN__PASSWORD']\n", encoding="utf-8")

    class _Vault:
        def list_connections(self):
            return [{"engine": "pg", "name": "main"}, {"engine": "pg", "name": "other"}]

        def env_for(self, engine, name):
            return {f"DS_{engine.upper()}_{name.upper()}__PASSWORD": f"{name}-pw"}

    state = SimpleNamespace(
        session=SimpleNamespace(_data_vault=_Vault(), _scratchpads=SimpleNamespace(cell_sandbox=policy)),
        artifact_path=artifact,
        slug="app",
    )
    assert orchestrator._verify_datasource_env(state) == {"DS_PG_MAIN__PASSWORD": "main-pw"}


async def test_backends_are_refused_before_the_pool_is_touched(tmp_path, policy):
    folder = tmp_path / "artifacts" / "app"
    folder.mkdir(parents=True)
    (folder / "backend.py").write_text("", encoding="utf-8")
    pool = _Pool("/venv/bin/python", policy)
    result = await backend_launcher.launch_artifact_backend(
        slug="app", artifact_folder=folder, scratchpad_pool=pool, tracked_backends={}
    )
    assert result == BACKENDS_OFF_MESSAGE
    assert pool.calls == []
    assert not (folder / "backend.log").exists()


async def test_the_launch_backend_tool_relays_the_refusal(tmp_path, policy, monkeypatch):
    folder = tmp_path / "artifacts" / "app"
    folder.mkdir(parents=True)
    store = SimpleNamespace(
        open=lambda slug: SimpleNamespace(datasources=[]),
        folder_for=lambda slug: folder,
    )
    monkeypatch.setattr(tool_handlers, "resolve_artifact_store", lambda session: store)
    session = SimpleNamespace(_scratchpads=_Pool("/venv/bin/python", policy), _data_vault=None)
    outcome = await tool_handlers.handle_launch_backend(session, {"slug": "app"})
    assert BACKENDS_OFF_MESSAGE in outcome.content
    assert outcome.ok is False


@pytest.mark.parametrize("entry", ["spec", "generate"])
async def test_a_fullstack_artifact_is_refused_before_the_spec_or_generation(
    tmp_path, policy, monkeypatch, entry
):
    from unittest.mock import AsyncMock

    from anton.core.tools.generate_artifact.discovery import checkpoint as cp
    from anton.core.tools.generate_artifact.state import GenState

    spent: list[str] = []

    async def must_not_run(*args, **kwargs):
        spent.append("ran")

    for node in ("_data_phase", "_write_tech_spec", "_make_api_spec", "_gen_verify_backend",
                 "_gen_verify_frontend", "_run_and_verify_app"):
        monkeypatch.setattr(orchestrator, node, must_not_run)
    session = AsyncMock()
    session._scratchpads = SimpleNamespace(cell_sandbox=policy)
    state = GenState(
        session=session, artifact_type="fullstack-stateless-app", artifact_path=tmp_path,
        slug="app", brief="b", is_fullstack=True,
    )
    entry_point = {"spec": cp.ENTRY_SPEC, "generate": cp.ENTRY_GENERATE}[entry]
    assert await orchestrator.run(state, entry=entry_point) == BACKENDS_OFF_MESSAGE
    assert spent == []


async def test_a_fullstack_artifact_is_refused_before_any_question(tmp_path, policy, monkeypatch):
    from unittest.mock import AsyncMock

    from anton.core.tools.generate_artifact.discovery import checkpoint as cp
    from anton.core.tools.generate_artifact.state import GenState

    asked: list[str] = []

    async def must_not_run(*args, **kwargs):
        asked.append("discovery")

    monkeypatch.setattr(orchestrator, "run_discovery", must_not_run)
    session = AsyncMock()
    session._scratchpads = SimpleNamespace(cell_sandbox=policy)
    state = GenState(
        session=session, artifact_type="fullstack-stateless-app", artifact_path=tmp_path,
        slug="app", brief="b",
    )
    assert await orchestrator.run(state, entry=cp.ENTRY_FULL) == BACKENDS_OFF_MESSAGE
    assert asked == []


# ── Lints that start a browser or LibreOffice ────────────────────────────────


def test_each_browser_and_office_check_refuses_under_a_policy_before_it_spawns(tmp_path, policy, monkeypatch):
    import subprocess

    from anton.core.artifacts import html_lint, office_open_check, xlsx_office_check

    monkeypatch.setenv("ANTON_HTML_LINT_BROWSER", sys.executable)
    monkeypatch.setenv("ANTON_XLSX_LINT_OFFICE", sys.executable)
    spawned: list[tuple] = []

    # Recorded, not only raised: each check swallows its own errors as None.
    def must_not_spawn(*args, **kwargs):
        spawned.append(args)
        raise AssertionError(f"spawned {args!r}")

    monkeypatch.setattr(html_lint.subprocess, "run", must_not_spawn)
    monkeypatch.setattr(xlsx_office_check.subprocess, "Popen", must_not_spawn)
    monkeypatch.setattr(subprocess, "run", must_not_spawn)
    page = tmp_path / "page.html"
    page.write_text("<html></html>")
    book = tmp_path / "book.xlsx"
    book.write_bytes(b"not a zip")
    assert html_lint.lint_html(page, cell_sandbox=policy) is None
    assert html_lint.lint_url("http://127.0.0.1:1/", cell_sandbox=policy) is None
    assert xlsx_office_check.check_xlsx_via_office(book, cell_sandbox=policy) is None
    assert office_open_check.check_opens_via_office(book, cell_sandbox=policy) is None
    assert verifiers.verify_frontend_live(page, cell_sandbox=policy) is None
    assert verifiers.verify_app_live("http://127.0.0.1:1/", cell_sandbox=policy) is None
    assert spawned == []


class _Store:
    def __init__(self, root: Path) -> None:
        self.root = root


def _artifact(root: Path, slug: str, filename: str, content: bytes) -> dict[str, float]:
    folder = root / slug
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text("{}")
    (folder / filename).write_bytes(content)
    return {slug: 0.0}


def test_browser_and_office_lints_are_skipped_and_say_so(tmp_path, policy, monkeypatch):
    import subprocess

    from anton.core.artifacts import html_lint, xlsx_office_check

    monkeypatch.setenv("ANTON_HTML_LINT_BROWSER", sys.executable)
    monkeypatch.setenv("ANTON_XLSX_LINT_OFFICE", sys.executable)

    def must_not_spawn(*args, **kwargs):
        raise AssertionError(f"spawned {args!r}")

    monkeypatch.setattr(html_lint.subprocess, "run", must_not_spawn)
    monkeypatch.setattr(xlsx_office_check.subprocess, "Popen", must_not_spawn)
    store = _Store(tmp_path / "artifacts")
    before = _artifact(store.root, "page", "page.html", b"<html><body>hi</body></html>")
    before |= _artifact(store.root, "deck", "deck.pptx", b"not a zip")

    messages = tool_handlers.lint_changed_artifact_files(store, before, cell_sandbox=policy)
    assert f"page/page.html — {tool_handlers._SANDBOX_BROWSER_SKIP}" in messages
    # The in-process structural lint still runs and reports the broken deck.
    assert any(m.startswith("deck/deck.pptx — ") and "LibreOffice" not in m for m in messages)


def test_skipped_lints_stay_silent_where_the_tool_is_not_installed(tmp_path, policy, monkeypatch):
    monkeypatch.delenv("ANTON_HTML_LINT_BROWSER", raising=False)
    store = _Store(tmp_path / "artifacts")
    before = _artifact(store.root, "page", "page.html", b"<html></html>")
    assert tool_handlers.lint_changed_artifact_files(store, before, cell_sandbox=policy) == []


def test_the_generator_skips_its_browser_check_and_says_why():
    assert "sandbox" in verifiers.browser_check_skip_reason(sandboxed=True)


# ── Refusals keep the venv and the saved state ───────────────────────────────


def _refusing_policy(tmp_path: Path, project: Path) -> CellSandbox:
    """A policy the real launcher refuses on every host: its HOME is missing."""
    (tmp_path / "tmp").mkdir(exist_ok=True)
    return CellSandbox(
        project_dir=project, home_dir=tmp_path / "no-such-home", tmp_dir=tmp_path / "tmp", user_name="bob"
    )


def _snapshot(project: Path) -> Path:
    path = snapshot_file(project / ".anton" / "scratchpad-venvs", "conv-1", "main")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"saved namespace")
    return path


async def test_a_refused_health_check_is_named_and_keeps_the_venv_and_state(tmp_path, project, monkeypatch):
    venv = _venv(project)
    (venv / "installed-marker").write_text("pandas")
    saved = _snapshot(project)
    pad = LocalScratchpadRuntime(
        "main", coding_provider="anthropic", coding_model="", coding_api_key="", coding_base_url="",
        workspace_path=project, session_id="conv-1", cell_sandbox=_refusing_policy(tmp_path, project),
    )
    with pytest.raises(SandboxUnavailable, match="SandboxUnavailable: anton-sandbox: refused:"):
        await pad.start()
    assert (venv / "installed-marker").read_text() == "pandas"
    assert saved.read_bytes() == b"saved namespace"


async def test_a_cell_spawn_that_fails_under_the_sandbox_keeps_the_venv(project, policy, monkeypatch):
    pad = _pad(project, policy)
    monkeypatch.setattr(pad, "_verify_venv_python", lambda: True)

    async def refuse(*args, **kwargs):
        raise OSError("launcher missing")

    monkeypatch.setattr(local.asyncio, "create_subprocess_exec", refuse)
    with pytest.raises(SandboxUnavailable, match="launcher missing"):
        await pad.start()
    assert Path(pad._venv_python).exists()


async def test_a_cell_the_launcher_refuses_reports_the_refusal_by_name(tmp_path, project, monkeypatch):
    saved = _snapshot(project)
    pad = _pad(project, _refusing_policy(tmp_path, project), session_id="conv-1")
    monkeypatch.setattr(pad, "_verify_venv_python", lambda: True)
    await pad.start()
    try:
        cell = None
        async for item in pad.execute_streaming("print(1)"):
            cell = item
        assert "SandboxUnavailable: anton-sandbox: refused:" in cell.error
    finally:
        await pad.close()
    assert Path(pad._venvs_base / "main" / "bin" / "python").exists()
    assert saved.read_bytes() == b"saved namespace"


async def test_refusals_never_count_toward_the_reset_that_discards_state(project, policy, monkeypatch):
    saved = _snapshot(project)
    pad = _pad(project, policy, session_id="conv-1")
    pad._proc = SimpleNamespace(returncode=1)
    resets: list[bool] = []

    async def refused_resume():
        raise SandboxUnavailable("anton-sandbox: refused: test")

    async def reset():
        resets.append(True)

    monkeypatch.setattr(pad, "resume", refused_resume)
    monkeypatch.setattr(pad, "reset", reset)
    for _ in range(5):
        cells = [item async for item in pad.execute_streaming("print(1)")]
        assert "SandboxUnavailable: anton-sandbox: refused: test" in cells[-1].error
    assert resets == []
    assert saved.read_bytes() == b"saved namespace"


async def test_after_a_refused_resume_the_agent_is_not_sent_to_reset(project, policy, monkeypatch):
    pad = _pad(project, policy, session_id="conv-1")
    pad._proc = SimpleNamespace(returncode=1)

    async def refused_resume():
        raise SandboxUnavailable("anton-sandbox: refused: ENOMEM")

    monkeypatch.setattr(pad, "resume", refused_resume)
    cells = [item async for item in pad.execute_streaming("print(1)")]
    error = cells[-1].error
    assert "SandboxUnavailable: anton-sandbox: refused: ENOMEM" in error
    assert "Use reset" not in error
    assert "saved state are kept" in error


async def test_a_resume_that_fails_for_another_reason_still_offers_reset(project, policy, monkeypatch):
    pad = _pad(project, policy, session_id="conv-1")
    pad._proc = SimpleNamespace(returncode=1)

    async def broken_resume():
        raise RuntimeError("venv is broken")

    monkeypatch.setattr(pad, "resume", broken_resume)
    cells = [item async for item in pad.execute_streaming("print(1)")]
    assert "Use reset to restart with a clean state." in cells[-1].error


async def test_a_reset_the_sandbox_refuses_keeps_the_saved_state(tmp_path, project, monkeypatch):
    saved = _snapshot(project)
    pad = _pad(project, _refusing_policy(tmp_path, project), session_id="conv-1")
    with pytest.raises(SandboxUnavailable, match="anton-sandbox: refused:"):
        await pad.reset()
    assert saved.read_bytes() == b"saved namespace"
    assert Path(pad._venvs_base / "main" / "bin" / "python").exists()


# ── Pad names ────────────────────────────────────────────────────────────────


@pytest.fixture
def victim(tmp_path: Path) -> Path:
    folder = tmp_path / "alice"
    folder.mkdir()
    (folder / "private.txt").write_text("alice's")
    return folder


def _unchanged(victim: Path) -> bool:
    return sorted(p.name for p in victim.iterdir()) == ["private.txt"] and (
        (victim / "private.txt").read_text() == "alice's"
    )


@pytest.mark.parametrize("kind", ["absolute", "dotdot", "nul"])
async def test_exec_and_install_refuse_a_pad_name_outside_the_venvs(victim, kind):
    name = {"absolute": str(victim), "dotdot": "../../../alice", "nul": "a\0b"}[kind]
    session = SimpleNamespace()
    outcome = await prepare_scratchpad_exec(session, {"name": name, "code": "print(1)"})
    assert isinstance(outcome, ToolOutcome) and outcome.ok is False
    assert outcome.reason == "scratchpad_invalid_name"
    for action in ("exec", "install", "view", "dump", "remove", "reset"):
        result = await tool_handlers.handle_scratchpad(
            session, {"action": action, "name": name, "code": "print(1)", "packages": ["pandas"]}
        )
        assert isinstance(result, ToolOutcome) and result.reason == "scratchpad_invalid_name", action
    assert _unchanged(victim)


@pytest.mark.parametrize("name", ["absolute", "../../alice"])
def test_a_runtime_refuses_a_pad_name_outside_the_venvs(project, victim, name):
    with pytest.raises(ValueError):
        LocalScratchpadRuntime(
            str(victim) if name == "absolute" else name,
            coding_provider="anthropic", coding_model="", coding_api_key="", coding_base_url="",
            workspace_path=project,
        )
    assert _unchanged(victim)


async def test_the_verifiers_slug_pad_refuses_a_name_outside_the_venvs(project, victim):
    manager = ScratchpadManager(
        runtime_factory=local.local_scratchpad_runtime_factory,
        coding_provider="anthropic", coding_model="", coding_api_key="", coding_base_url="",
        workspace_path=project,
    )
    artifact = project / "artifacts" / "app"
    artifact.mkdir(parents=True)
    (artifact / "backend.py").write_text("app = None\n", encoding="utf-8")
    for slug in (str(victim), "../../../alice"):
        with pytest.raises(ValueError):
            await verifiers.verify_backend(scratchpad_pool=manager, slug=slug, artifact_path=artifact)
    assert _unchanged(victim)


def test_a_venv_folder_that_resolves_outside_the_venvs_is_never_deleted(project, victim):
    base = project / ".anton" / "scratchpad-venvs"
    base.mkdir(parents=True)
    (base / "main").symlink_to(victim, target_is_directory=True)
    pad = LocalScratchpadRuntime(
        "main", coding_provider="anthropic", coding_model="", coding_api_key="", coding_base_url="",
        workspace_path=project,
    )
    with pytest.raises(ValueError, match="outside"):
        pad._ensure_venv()
    assert _unchanged(victim)
