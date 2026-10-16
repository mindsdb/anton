"""The Landlock launcher, run for real.

The tests that lock a process need Linux with Landlock ABI 3 or later and skip
elsewhere with that reason. Each locked child also performs one allowed action
in the same run, so a child that failed to start cannot pass a denial check.

Imports only the standard library, pytest, pydantic and the two sandbox
modules, so the file runs in a bare python image:

    docker run --rm --cap-drop ALL --security-opt no-new-privileges:true --init \
      -v "$PWD":/src:ro -w /src python:3.12.15-slim sh -c \
      "pip install -q pytest pydantic && cd /tmp && PYTHONPATH=/src \
       python -m pytest -q -p no:cacheprovider --noconftest /src/tests/test_landlock_launcher.py"
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from anton.core.backends import landlock_launcher, sandbox
from anton.core.backends.sandbox import (
    SANDBOX_REFUSED_EXIT,
    SANDBOX_REFUSED_MARKER,
    CellSandbox,
    SandboxUnavailable,
    refusal_from,
)

_ABI = sandbox.landlock_abi()
needs_landlock = pytest.mark.skipif(
    _ABI < sandbox.MIN_LANDLOCK_ABI,
    reason=f"needs Linux with Landlock ABI >= {sandbox.MIN_LANDLOCK_ABI}; this host reports {_ABI}",
)


def test_the_launcher_repeats_the_sandbox_constants():
    assert landlock_launcher.LAUNCHER_API_VERSION == sandbox.LAUNCHER_API_VERSION
    assert landlock_launcher.MIN_LANDLOCK_ABI == sandbox.MIN_LANDLOCK_ABI
    assert landlock_launcher.SANDBOX_REFUSED_EXIT == sandbox.SANDBOX_REFUSED_EXIT
    assert landlock_launcher.SANDBOX_REFUSED_MARKER == sandbox.SANDBOX_REFUSED_MARKER


def test_the_kernel_structs_have_the_uapi_sizes():
    assert ctypes.sizeof(landlock_launcher._PathBeneathAttr) == 12
    assert ctypes.sizeof(landlock_launcher._RulesetAttr) == 24


def test_every_right_the_policy_names_is_one_the_launcher_knows(tmp_path):
    policy = _policy(tmp_path, shared=True, read_only=True)
    for rule in policy.rules():
        assert set(rule.access) <= set(landlock_launcher._RIGHTS), rule


@pytest.fixture
def dirs(tmp_path: Path) -> dict[str, Path]:
    out = {name: tmp_path / name for name in ("project", "home", "tmp", "outside")}
    for folder in out.values():
        folder.mkdir()
    (out["outside"] / "secret.txt").write_text("not yours\n", encoding="utf-8")
    return out


def _policy(tmp_path: Path, *, shared: bool = False, read_only: bool = False, **overrides) -> CellSandbox:
    fields: dict = {
        "project_dir": tmp_path / "project",
        "home_dir": tmp_path / "home",
        "tmp_dir": tmp_path / "tmp",
        "user_name": "bob",
    }
    if shared:
        fields["shared_dir"] = tmp_path / "shared"
    if read_only:
        fields["read_only_dirs"] = (tmp_path / "skills",)
        fields["read_only_files"] = (tmp_path / "ca.pem",)
    fields.update(overrides)
    return CellSandbox(**fields)


def _run_locked(
    policy: CellSandbox, argv: list[str], *, env: dict[str, str] | None = None, timeout: float = 60
) -> subprocess.CompletedProcess[str]:
    spawn = policy.launch(argv, env=policy.child_env(env or {}))
    try:
        return subprocess.run(
            spawn.argv,
            env=spawn.env,
            pass_fds=spawn.pass_fds,
            cwd=str(policy.project_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    finally:
        spawn.close()


def _python(policy: CellSandbox, code: str, *args: str) -> subprocess.CompletedProcess[str]:
    result = _run_locked(policy, [sys.executable, "-c", textwrap.dedent(code), *args])
    assert result.returncode == 0, result.stderr
    return result


@needs_landlock
def test_a_locked_child_writes_its_project_and_is_denied_outside(tmp_path, dirs):
    result = _python(
        _policy(tmp_path),
        """
        import errno, os, sys
        open(os.path.join(sys.argv[1], "made.txt"), "w").write("ok")
        print("write ok")
        for probe in (lambda: open(sys.argv[2]).read(), lambda: os.listdir(sys.argv[3])):
            try:
                probe()
                print("allowed")
            except OSError as exc:
                print("denied", errno.errorcode[exc.errno])
        """,
        str(dirs["project"]),
        str(dirs["outside"] / "secret.txt"),
        str(dirs["outside"]),
    )
    assert result.stdout.split("\n")[:3] == ["write ok", "denied EACCES", "denied EACCES"]
    assert (dirs["project"] / "made.txt").read_text() == "ok"


def _environ_once_set(pid: int, *, timeout: float = 5.0) -> bytes:
    """A just-started process's environ, read as the unlocked test process.

    Popen returns when the child's close-on-exec pipe closes, which is before
    the kernel fills in the new program's environ, so an early read is empty.
    """
    deadline = time.monotonic() + timeout
    while True:
        data = Path(f"/proc/{pid}/environ").read_bytes()
        if data or time.monotonic() > deadline:
            return data
        time.sleep(0.01)


@needs_landlock
def test_a_locked_child_cannot_read_another_process_environ(tmp_path, dirs):
    neighbour = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        env={"NEIGHBOUR_SECRET": "s3cr3t", "PATH": os.environ.get("PATH", "")},
    )
    try:
        # The unlocked test process can read it, so a denial below is the lock's.
        assert b"NEIGHBOUR_SECRET" in _environ_once_set(neighbour.pid)
        result = _python(
            _policy(tmp_path),
            """
            import errno, sys
            print("self", "ok" if open("/proc/self/environ", "rb").read() is not None else "")
            for pid in sys.argv[1:]:
                try:
                    open(f"/proc/{pid}/environ", "rb").read()
                    print(pid, "allowed")
                except OSError as exc:
                    print(pid, errno.errorcode[exc.errno])
            """,
            str(neighbour.pid),
            "1",
        )
    finally:
        neighbour.kill()
        neighbour.wait()
    assert result.stdout.split("\n")[:3] == ["self ok", f"{neighbour.pid} EACCES", "1 EACCES"]


@needs_landlock
def test_devices_names_and_the_shm_folder_work_under_the_lock(tmp_path, dirs):
    result = _python(
        _policy(tmp_path),
        """
        import errno, getpass, os, socket, subprocess, sys
        print("devnull", subprocess.run([sys.executable, "-c", "print(1)"], stdout=subprocess.DEVNULL).returncode)
        with open(os.devnull, "w") as f:
            f.write("x")
        print("urandom", len(open("/dev/urandom", "rb").read(4)))
        name = f"/dev/shm/anton-launcher-test-{os.getpid()}"
        with open(name, "w") as f:
            f.write("shm")
        print("shm", open(name).read())
        os.unlink(name)
        try:
            os.listdir("/dev/shm")
            print("shm-list allowed")
        except OSError as exc:
            print("shm-list", errno.errorcode[exc.errno])
        print("user", getpass.getuser())
        print("localhost", bool(socket.getaddrinfo("localhost", 80)))
        print("cpus", os.cpu_count() > 0)
        """,
    )
    assert result.stdout.split("\n")[:8] == [
        "devnull 0",
        "urandom 4",
        "shm shm",
        "shm-list EACCES",
        "user bob",
        "localhost True",
        "cpus True",
        "",
    ]


@needs_landlock
def test_multiprocessing_pool_and_manager_run_under_the_lock(tmp_path, dirs):
    result = _python(
        _policy(tmp_path),
        """
        import multiprocessing
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(2) as pool:
            print("pool", sum(pool.map(abs, [-1, -2, -3])))
        with ctx.Manager() as manager:
            shared = manager.dict(a=1)
            print("manager", shared["a"])
        """,
    )
    assert result.stdout.split("\n")[:2] == ["pool 6", "manager 1"]


@needs_landlock
def test_a_pth_planted_in_the_venv_runs_only_under_the_lock(tmp_path, dirs):
    venv = dirs["project"] / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    site_packages = next(venv.glob("lib/python*/site-packages"))
    marker = dirs["project"] / "pth-ran.txt"
    secret = dirs["outside"] / "secret.txt"
    (site_packages / "planted.pth").write_text(
        "import os; exec("
        + repr(
            "try:\n"
            f"    open({str(secret)!r}).read(); verdict = 'read'\n"
            "except OSError:\n"
            "    verdict = 'denied'\n"
            f"open({str(marker)!r}, 'w').write(verdict)\n"
        )
        + ")\n",
        encoding="utf-8",
    )
    result = _run_locked(_policy(tmp_path), [str(venv / "bin" / "python"), "-c", "print('body')"])
    assert result.returncode == 0, result.stderr
    assert result.stdout == "body\n"
    assert marker.read_text() == "denied"


@needs_landlock
def test_a_missing_required_folder_refuses_with_the_marker(tmp_path, dirs):
    policy = _policy(tmp_path, home_dir=tmp_path / "no-such-home")
    result = _run_locked(policy, [sys.executable, "-c", "print('ran')"])
    assert result.returncode == SANDBOX_REFUSED_EXIT
    assert result.stdout == ""
    assert result.stderr.splitlines()[0].startswith(SANDBOX_REFUSED_MARKER)
    assert "no-such-home" in result.stderr
    assert isinstance(refusal_from(result.returncode, result.stderr), SandboxUnavailable)


@needs_landlock
def test_a_missing_optional_grant_is_skipped(tmp_path, dirs):
    policy = _policy(tmp_path, shared=True, read_only=True)
    assert not policy.shared_dir.exists()
    result = _python(policy, "print('ran')")
    assert result.stdout == "ran\n"


@needs_landlock
def test_a_missing_program_is_not_a_refusal(tmp_path, dirs):
    result = _run_locked(_policy(tmp_path), [str(dirs["project"] / "no-such-python")])
    assert result.returncode == 127
    assert SANDBOX_REFUSED_MARKER not in result.stderr
    assert refusal_from(result.returncode, result.stderr) is None


@needs_landlock
def test_the_self_test_passes_and_cleans_up(tmp_path):
    sandbox.self_test(scratch_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(_ABI >= sandbox.MIN_LANDLOCK_ABI, reason="this host has Landlock")
def test_the_launcher_refuses_where_landlock_is_unavailable(tmp_path, dirs):
    result = _run_locked(_policy(tmp_path), [sys.executable, "-c", "print('ran')"])
    assert result.returncode == SANDBOX_REFUSED_EXIT
    assert result.stdout == ""
    assert result.stderr.startswith(SANDBOX_REFUSED_MARKER)
    with pytest.raises(SandboxUnavailable, match="SandboxUnavailable: anton-sandbox: refused:"):
        sandbox.self_test(scratch_dir=tmp_path)
