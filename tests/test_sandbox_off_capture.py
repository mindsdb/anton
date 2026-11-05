"""With no sandbox policy, every spawn of agent code is the call it always was.

Each test captures one real spawn site's argv, environment, cwd and keyword
arguments under a fixed parent environment, and compares them with literal
expectations written from the call as it stood before the sandbox existed.
None of these tests names the sandbox, so the file runs unchanged against the
code before it.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import anton.core.backends.local as local
from anton.core.artifacts import backend_launcher
from anton.core.backends.local import LocalScratchpadRuntime
from anton.core.settings import CoreSettings
from anton.core.tools.generate_artifact import verifiers

# The parent's environment for every test here: server secrets, a package
# mirror's password, an inherited DS_* and a loader variable, all of which
# reach every child today.
_PARENT_ENV = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/home/anton",
    "COWORK_AUTH_TOKEN": "tok-secret",
    "DATABASE_URI": "postgresql://cowork:pw-secret@postgres/cowork",
    "MINDS_API_KEY": "minds-secret",
    "ANTON_OPENAI_API_KEY": "anton-openai-key",
    "PIP_INDEX_URL": "https://mirror-user:pip-secret@mirror.internal/simple",
    "DS_OLD__PASSWORD": "server-ds-secret",
    "LD_PRELOAD": "/opt/preload.so",
}


@pytest.fixture
def parent_env(monkeypatch) -> dict[str, str]:
    for key in list(os.environ):
        monkeypatch.delenv(key)
    for key, value in _PARENT_ENV.items():
        monkeypatch.setenv(key, value)
    return dict(_PARENT_ENV)


def _environ() -> dict[str, str]:
    """The parent environment as the spawn saw it (pytest adds its own marker)."""
    return dict(os.environ)


class _Proc:
    """A finished child: what the spawn sites read back from a process."""

    def __init__(self, *, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0) -> None:
        self._out = (stdout, stderr)
        self.returncode = returncode
        self.pid = 4242
        self.stdin = SimpleNamespace(write=lambda _b: None, close=lambda: None, is_closing=lambda: False)
        self.stdout = None
        self.stderr = None

    async def communicate(self):
        return self._out

    async def wait(self):
        return self.returncode

    def kill(self):
        pass


def _pad(tmp_path: Path, **kwargs) -> LocalScratchpadRuntime:
    venv = tmp_path / "project" / ".anton" / "scratchpad-venvs" / "main"
    (venv / "bin").mkdir(parents=True)
    python = venv / "bin" / "python"
    python.write_text("")
    pad = LocalScratchpadRuntime(
        "main",
        coding_provider=kwargs.pop("coding_provider", "anthropic"),
        coding_model=kwargs.pop("coding_model", ""),
        coding_api_key=kwargs.pop("coding_api_key", ""),
        coding_base_url=kwargs.pop("coding_base_url", ""),
        workspace_path=tmp_path / "project",
        **kwargs,
    )
    pad._venv_dir = str(venv)
    pad._venv_python = str(python)
    return pad


async def test_the_cell_spawn_is_unchanged(tmp_path, monkeypatch, parent_env):
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: None))
    pad = _pad(
        tmp_path,
        coding_provider="openai",
        coding_model="gpt-test",
        coding_api_key="coding-key",
        coding_base_url="https://llm.example/v1",
        scratchpad_ds_env={"DS_PG_MAIN__PASSWORD": "pad-ds"},
        workspace_env_overlay={"FOO": "from-dotenv", "PATH": "/evil"},
    )
    monkeypatch.setattr(pad, "_verify_venv_python", lambda: True)
    calls: list[tuple[tuple, dict]] = []

    async def capture(*args, **kwargs):
        calls.append((args, kwargs))
        boot = Path(args[1])
        calls.append(((boot.read_text(encoding="utf-8"),), {}))
        return _Proc()

    monkeypatch.setattr(local.asyncio, "create_subprocess_exec", capture)
    await pad.start()

    (args, kwargs), ((boot_text,), _) = calls
    assert args[0] == pad._venv_python
    boot = Path(args[1])
    assert len(args) == 2
    assert boot.parent == Path(tempfile.gettempdir())
    assert boot.name.startswith("anton_scratchpad_") and boot.suffix == ".py"
    assert boot_text == local._read_boot_script()
    anton_root = str(Path(local.__file__).resolve().parent.parent.parent.parent)
    expected_env = {
        **{k: v for k, v in _environ().items() if k != "DS_OLD__PASSWORD"},
        "PYTHONUTF8": "1",
        "FOO": "from-dotenv",
        "DS_PG_MAIN__PASSWORD": "pad-ds",
        "ANTON_SCRATCHPAD_MODEL": "gpt-test",
        "ANTON_SCRATCHPAD_PROVIDER": "openai",
        "OPENAI_API_KEY": "coding-key",
        "OPENAI_BASE_URL": "https://llm.example/v1",
        "ANTON_OPENAI_BASE_URL": "https://llm.example/v1",
        "ANTON_CELL_INSTALL_TIMEOUT": str(CoreSettings().cell_install_timeout),
        "ANTON_SCRATCHPAD_WORKSPACE_PATH": str((tmp_path / "project").resolve()),
        "PYTHONPATH": anton_root,
    }
    assert kwargs == {
        "stdin": asyncio.subprocess.PIPE,
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
        "env": expected_env,
        "cwd": str(tmp_path / "project"),
        "start_new_session": sys.platform != "win32",
    }
    await pad._stop_process()
    assert not boot.exists()


def test_the_venv_health_check_is_unchanged(tmp_path, monkeypatch, parent_env):
    import subprocess

    pad = _pad(tmp_path)
    calls: list[tuple[tuple, dict]] = []

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args[0], 0, b"ok\n", b"")

    monkeypatch.setattr(subprocess, "run", capture)
    assert pad._verify_venv_python() is True
    assert calls == [(([pad._venv_python, "-c", "print('ok')"],), {"capture_output": True, "timeout": 5})]


@pytest.mark.parametrize(
    ("uv", "expected_argv"),
    [
        (None, lambda py: (py, "-m", "pip", "install", "--no-input", "pandas")),
        ("/fake/uv", lambda py: ("/fake/uv", "pip", "install", "--python", py, "pandas")),
    ],
)
async def test_the_package_install_is_unchanged(tmp_path, monkeypatch, parent_env, uv, expected_argv):
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: uv))
    pad = _pad(tmp_path)
    monkeypatch.setattr(pad, "_verify_venv_python", lambda: True)
    calls: list[tuple[tuple, dict]] = []

    async def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return _Proc(stdout=b"installed")

    monkeypatch.setattr(local.asyncio, "create_subprocess_exec", capture)
    assert await pad.install_packages(["pandas"]) == "installed"
    assert calls == [
        (
            expected_argv(pad._venv_python),
            {
                "stdout": asyncio.subprocess.PIPE,
                "stderr": asyncio.subprocess.STDOUT,
                "env": {**_environ(), "PYTHONUTF8": "1"},
            },
        )
    ]


class _VerifierPool:
    """A pool as the verifier sees one: no sandbox policy attribute."""

    def __init__(self, python: str) -> None:
        self._python = python

    async def get_or_create(self, name):
        return SimpleNamespace(install_packages=None)

    async def venv_python(self, name):
        return self._python


async def test_the_verifier_checks_are_unchanged(tmp_path, monkeypatch, parent_env):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "backend.py").write_text("app = None\n", encoding="utf-8")
    calls: list[tuple[tuple, dict]] = []
    results = iter([_Proc(), _Proc(stdout=json.dumps({"import_ok": True}).encode())])

    async def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return next(results)

    monkeypatch.setattr(verifiers.asyncio, "create_subprocess_exec", capture)
    await verifiers.verify_backend(
        scratchpad_pool=_VerifierPool("/venv/bin/python"), slug="x", artifact_path=artifact
    )

    pythonpath = str(Path(tempfile.gettempdir()) / "anton_state_pp")
    assert calls == [
        (
            ("/venv/bin/python", "-I", "-B", "-X", "utf8", "-c", verifiers._SYNTAX_CHECK_SCRIPT),
            {
                "cwd": str(artifact),
                "stdout": asyncio.subprocess.PIPE,
                "stderr": asyncio.subprocess.PIPE,
            },
        ),
        (
            ("/venv/bin/python", "-B", "-c", verifiers._INTROSPECT_SCRIPT),
            {
                "cwd": str(artifact),
                "stdout": asyncio.subprocess.PIPE,
                "stderr": asyncio.subprocess.PIPE,
                "env": {**_environ(), "PYTHONPATH": pythonpath, "PYTHONDONTWRITEBYTECODE": "1"},
            },
        ),
    ]


async def test_the_backend_launch_is_unchanged(tmp_path, monkeypatch, parent_env):
    folder = tmp_path / "artifacts" / "app"
    folder.mkdir(parents=True)
    (folder / "backend.py").write_text("print('hi')\n", encoding="utf-8")
    seen: dict = {}

    async def capture(*args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        seen["stdout_inode"] = os.fstat(kwargs["stdout"].fileno()).st_ino
        raise OSError("captured")

    monkeypatch.setattr(backend_launcher.asyncio, "create_subprocess_exec", capture)
    result = await backend_launcher.launch_artifact_backend(
        slug="app",
        artifact_folder=folder,
        scratchpad_pool=_VerifierPool("/venv/bin/python"),
        tracked_backends={},
        extra_args=["--flag"],
        extra_env={"FROM_DOTENV": "1"},
        ds_env={"DS_PG_MAIN__PASSWORD": "artifact-ds"},
    )
    assert result == "Error: failed to spawn backend: captured"

    args, kwargs = seen["args"], seen["kwargs"]
    assert args[:3] == ("/venv/bin/python", str((folder / "backend.py").resolve()), "--port")
    assert args[3].isdigit() and args[4:] == ("--flag",)
    env = {k: v for k, v in _environ().items() if not k.startswith("DS_")}
    env.update(
        FROM_DOTENV="1",
        DS_PG_MAIN__PASSWORD="artifact-ds",
        PYTHONPATH=str(Path(tempfile.gettempdir()) / "anton_state_pp"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    assert set(kwargs) == {"cwd", "stdout", "stderr", "stdin", "preexec_fn", "env"}
    assert kwargs["cwd"] == str(folder)
    assert kwargs["stdout"] is kwargs["stderr"]
    assert seen["stdout_inode"] == (folder / "backend.log").stat().st_ino
    assert kwargs["stdin"] == asyncio.subprocess.DEVNULL
    assert (kwargs["preexec_fn"] is None) == (not sys.platform.startswith("linux"))
    assert kwargs["env"] == env
