"""The Landlock launcher, run for real.

The tests that lock a process carry the `landlock` mark: they need Linux with
Landlock ABI 3 or later, and tests/conftest.py skips them elsewhere with that
reason, or fails them on Linux under GitHub Actions or with
LANDLOCK_REQUIRED=1. Each locked child also performs one allowed action in the
same run, so a child that failed to start cannot pass a denial check.

Imports only the standard library, pytest, pydantic and the two sandbox
modules, so the file runs in a bare python image (`--noconftest` runs every
`landlock` test, so use a host with Landlock):

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
from dataclasses import dataclass
from pathlib import Path

import pytest

from anton.core.backends import landlock_launcher, sandbox
from anton.core.backends.sandbox import (
    SANDBOX_REFUSED_EXIT,
    SANDBOX_REFUSED_MARKER,
    CellSandbox,
    SandboxedSpawn,
    SandboxUnavailable,
)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the launcher and its tests are POSIX-only")

_ABI = sandbox.landlock_abi()
needs_landlock = pytest.mark.landlock


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


@dataclass
class _Run:
    """A finished launch: the process's own results and the launcher's refusal, if any."""

    returncode: int
    stdout: str
    stderr: str
    refusal: SandboxUnavailable | None


def _run_spawn(
    spawn: SandboxedSpawn,
    *,
    cwd: Path,
    argv: list[str] | None = None,
    pass_fds: tuple[int, ...] = (),
    timeout: float = 60,
) -> _Run:
    """Run ``spawn`` (or ``argv`` in its place) to the end and read its refusal pipe."""
    try:
        result = subprocess.run(
            argv or spawn.argv,
            env=spawn.env,
            pass_fds=(*spawn.pass_fds, *pass_fds),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    finally:
        spawn.close()
        refusal = spawn.refusal()
    return _Run(result.returncode, result.stdout, result.stderr, refusal)


def _run_locked(
    policy: CellSandbox,
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
    pass_fds: tuple[int, ...] = (),
    timeout: float = 60,
) -> _Run:
    spawn = policy.launch(argv, env=policy.child_env(env or {}))
    return _run_spawn(spawn, cwd=policy.project_dir, pass_fds=pass_fds, timeout=timeout)


def _python(policy: CellSandbox, code: str, *args: str, pass_fds: tuple[int, ...] = ()) -> _Run:
    result = _run_locked(policy, [sys.executable, "-c", textwrap.dedent(code), *args], pass_fds=pass_fds)
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


def _environ_once_set(pid: int, *, needle: bytes = b"", timeout: float = 5.0) -> bytes:
    """A just-started process's environ, read as the unlocked test process.

    Popen returns when the child's close-on-exec pipe closes, which is before
    the kernel fills in the new program's environ, so an early read is empty.
    A launched target's environ shows the launcher's own until the launcher
    execs it, so ``needle`` waits for a value only the target holds.
    """
    deadline = time.monotonic() + timeout
    while True:
        data = Path(f"/proc/{pid}/environ").read_bytes()
        if (data and needle in data) or time.monotonic() > deadline:
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
def test_the_os_release_files_pip_reads_are_readable_under_the_lock(tmp_path, dirs):
    # pip's vendored distro opens these to build its user agent and crashes on
    # anything but a missing file, so a denied read fails every install.
    present = [p for p in ("/etc/debian_version", "/etc/os-release") if os.path.exists(p)]
    if not present:
        pytest.skip("this host has neither /etc/debian_version nor /etc/os-release")
    result = _python(
        _policy(tmp_path),
        f"""
        for path in {present!r}:
            with open(path) as f:
                print(path, bool(f.read()))
        """,
    )
    assert result.stdout.split("\n")[: len(present)] == [f"{p} True" for p in present]


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
    assert isinstance(result.refusal, SandboxUnavailable)
    assert "no-such-home" in str(result.refusal)


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
    assert result.refusal is None


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
    assert str(result.refusal).startswith(f"SandboxUnavailable: {SANDBOX_REFUSED_MARKER}")
    with pytest.raises(SandboxUnavailable, match="SandboxUnavailable: anton-sandbox: refused:"):
        sandbox.self_test(scratch_dir=tmp_path)


# ── HOME and TMPDIR: read, write and execute ─────────────────────────────────


@needs_landlock
def test_a_temporary_file_moves_into_the_project_and_scripts_run_from_home_and_tmp(tmp_path, dirs):
    result = _python(
        _policy(tmp_path),
        """
        import os, subprocess, tempfile
        fd, path = tempfile.mkstemp()
        os.write(fd, b"data")
        os.close(fd)
        os.replace(path, "moved.txt")
        print("replace", open("moved.txt").read())
        for var in ("TMPDIR", "HOME"):
            script = os.path.join(os.environ[var], "tool.sh")
            with open(script, "w") as f:
                f.write("#!/bin/sh\\necho ran\\n")
            os.chmod(script, 0o755)
            print(var, subprocess.run([script], capture_output=True, text=True).stdout.strip())
        """,
    )
    assert result.stdout.split("\n")[:3] == ["replace data", "TMPDIR ran", "HOME ran"]
    assert (dirs["project"] / "moved.txt").read_text() == "data"


# ── Grants through links ─────────────────────────────────────────────────────


@needs_landlock
@pytest.mark.parametrize("where", ["at the grant", "above the grant"])
def test_a_required_grant_reached_through_a_link_refuses_the_launch(tmp_path, dirs, where):
    if where == "at the grant":
        (tmp_path / "home-link").symlink_to(dirs["home"], target_is_directory=True)
        home = tmp_path / "home-link"
    else:
        (tmp_path / "via").symlink_to(tmp_path, target_is_directory=True)
        home = tmp_path / "via" / "home"
    result = _run_locked(_policy(tmp_path, home_dir=home), [sys.executable, "-c", "print('ran')"])
    assert result.returncode == SANDBOX_REFUSED_EXIT
    assert result.stdout == ""
    assert f"required path {home} is a link or sits below one" in str(result.refusal)


@needs_landlock
def test_an_optional_grant_that_is_a_link_is_skipped(tmp_path, dirs):
    link = tmp_path / "skills-link"
    link.symlink_to(dirs["outside"], target_is_directory=True)
    result = _python(
        _policy(tmp_path, read_only_dirs=(link,)),
        """
        import errno, sys
        open("made.txt", "w").write("ok")
        print("write ok")
        try:
            open(sys.argv[1]).read()
            print("allowed")
        except OSError as exc:
            print("denied", errno.errorcode[exc.errno])
        """,
        str(link / "secret.txt"),
    )
    assert result.stdout.split("\n")[:2] == ["write ok", "denied EACCES"]
    assert result.refusal is None


# ── Only the launcher reports a refusal ──────────────────────────────────────


@needs_landlock
def test_a_target_that_prints_the_marker_and_exits_87_is_not_a_refusal(tmp_path, dirs):
    result = _run_locked(
        _policy(tmp_path),
        [sys.executable, "-c", f"import sys; sys.stderr.write({SANDBOX_REFUSED_MARKER!r} + ' fake\\n'); sys.exit(87)"],
    )
    assert result.returncode == SANDBOX_REFUSED_EXIT
    assert result.stderr.startswith(SANDBOX_REFUSED_MARKER)
    assert result.refusal is None


@needs_landlock
def test_a_target_cannot_write_a_refusal_into_the_launchers_pipe(tmp_path, dirs):
    policy = _policy(tmp_path)
    code = textwrap.dedent(
        f"""
        import errno, os, sys
        open("made.txt", "w").write("ok")
        print("write ok")
        try:
            os.write(int(sys.argv[1]), {SANDBOX_REFUSED_MARKER.encode()!r} + b" faked by the target\\n")
            print("wrote")
        except OSError as exc:
            print("denied", errno.errorcode[exc.errno])
        """
    )
    spawn = policy.launch([sys.executable, "-c", code, "REFUSAL_FD"], env=policy.child_env({}))
    # The target learns the pipe's number only once launch() has opened it.
    argv = [str(spawn.pass_fds[1]) if arg == "REFUSAL_FD" else arg for arg in spawn.argv]
    result = _run_spawn(spawn, cwd=policy.project_dir, argv=argv)
    assert result.stdout.split("\n")[:2] == ["write ok", "denied EBADF"]
    assert result.refusal is None


@needs_landlock
def test_a_target_outside_the_grants_is_refused_by_the_sandbox(tmp_path, dirs):
    target = dirs["outside"] / "tool"
    target.write_text("#!/bin/sh\necho ran\n")
    target.chmod(0o755)
    result = _run_locked(_policy(tmp_path), [str(target)])
    assert result.returncode == SANDBOX_REFUSED_EXIT
    assert result.stdout == ""
    assert f"the sandbox denies running {target}" in str(result.refusal)


@needs_landlock
def test_a_target_its_own_permissions_forbid_is_a_broken_file_not_a_refusal(tmp_path, dirs):
    target = dirs["project"] / "python"
    target.write_text("#!/bin/sh\necho ran\n")
    target.chmod(0o644)
    result = _run_locked(_policy(tmp_path), [str(target)])
    assert result.returncode == 126
    assert result.refusal is None
    assert f"anton-sandbox: cannot run {target}" in result.stderr


@needs_landlock
def test_an_environment_execve_rejects_is_refused_with_the_marker(tmp_path, dirs):
    policy = _policy(tmp_path)
    spawn = policy.launch([sys.executable, "-c", "print('ran')"], env={"": "x"})
    result = _run_spawn(spawn, cwd=policy.project_dir)
    assert result.returncode == SANDBOX_REFUSED_EXIT
    assert result.stdout == ""
    assert "execve refused the target's environment" in str(result.refusal)


# ── Inherited descriptors, locked neighbours and the ABI floor ───────────────


@needs_landlock
def test_a_descriptor_inherited_from_outside_the_grants_is_closed(tmp_path, dirs):
    fd = os.open(dirs["outside"] / "secret.txt", os.O_RDONLY)
    try:
        result = _python(
            _policy(tmp_path),
            """
            import errno, os, sys
            open("made.txt", "w").write("ok")
            print("write ok")
            try:
                os.read(int(sys.argv[1]), 64)
                print("read")
            except OSError as exc:
                print("denied", errno.errorcode[exc.errno])
            """,
            str(fd),
            pass_fds=(fd,),
        )
    finally:
        os.close(fd)
    assert result.stdout.split("\n")[:2] == ["write ok", "denied EBADF"]


def _person(tmp_path: Path, name: str) -> CellSandbox:
    root = tmp_path / name
    for folder in ("project", "home", "tmp"):
        (root / folder).mkdir(parents=True)
    return CellSandbox(
        project_dir=root / "project", home_dir=root / "home", tmp_dir=root / "tmp", user_name=name
    )


def _locked_sleeper(policy: CellSandbox, secret: str) -> tuple[subprocess.Popen, SandboxedSpawn]:
    spawn = policy.launch(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        env=policy.child_env({"ANTON_SCRATCHPAD_SECRET": secret}),
    )
    try:
        proc = subprocess.Popen(spawn.argv, env=spawn.env, pass_fds=spawn.pass_fds, cwd=str(policy.project_dir))
    finally:
        spawn.close()
    return proc, spawn


@needs_landlock
def test_two_locked_pads_cannot_read_each_others_environ(tmp_path):
    alice, bob = _person(tmp_path, "alice"), _person(tmp_path, "bob")
    for owner, reader in ((alice, bob), (bob, alice)):
        secret = f"{owner.user_name}-secret"
        sleeper, spawn = _locked_sleeper(owner, secret)
        try:
            # The unlocked test process can read it, so a denial below is the lock's.
            assert secret.encode() in _environ_once_set(sleeper.pid, needle=secret.encode())
            result = _python(
                reader,
                """
                import errno, sys
                open("made.txt", "w").write("ok")
                print("self", bool(open("/proc/self/environ", "rb").read()))
                try:
                    open(f"/proc/{sys.argv[1]}/environ", "rb").read()
                    print("neighbour allowed")
                except OSError as exc:
                    print("neighbour", errno.errorcode[exc.errno])
                """,
                str(sleeper.pid),
            )
        finally:
            sleeper.kill()
            sleeper.wait()
            spawn.refusal()
        assert result.stdout.split("\n")[:2] == ["self True", "neighbour EACCES"], owner.user_name


_ABI_CAP_WRAPPER = """\
import sys
sys.path.insert(0, {launcher_dir!r})
import landlock_launcher
cap = int(sys.argv[1])
probe = landlock_launcher._probe_abi
landlock_launcher._probe_abi = lambda libc: min(probe(libc), cap)
landlock_launcher.main(sys.argv[2:])
"""

_ABI_PROBE = """
import errno, fcntl, os, termios
open("made.txt", "w").write("ok")
with open("/dev/null", "rb") as f:
    try:
        fcntl.ioctl(f.fileno(), termios.TCGETS, bytes(64))
        print("ioctl allowed")
    except OSError as exc:
        print("ioctl", errno.errorcode[exc.errno])
try:
    os.kill(os.getppid(), 0)
    print("signal allowed")
except OSError as exc:
    print("signal", errno.errorcode[exc.errno])
"""


def _run_with_abi_cap(tmp_path: Path, policy: CellSandbox, cap: int) -> _Run:
    wrapper = tmp_path / "abi_cap.py"
    wrapper.write_text(_ABI_CAP_WRAPPER.format(launcher_dir=str(sandbox.launcher_path().parent)))
    spawn = policy.launch([sys.executable, "-c", _ABI_PROBE], env=policy.child_env({}))
    # The launcher's own argv with the wrapper and the cap in place of its path.
    argv = [*spawn.argv[:3], str(wrapper), str(cap), *spawn.argv[4:]]
    return _run_spawn(spawn, cwd=policy.project_dir, argv=argv)


@needs_landlock
def test_an_abi_below_3_refuses_with_the_marker(tmp_path, dirs):
    result = _run_with_abi_cap(tmp_path, _policy(tmp_path), 2)
    assert result.returncode == SANDBOX_REFUSED_EXIT
    assert result.stdout == ""
    assert "the kernel's Landlock ABI is 2, below 3" in str(result.refusal)


@needs_landlock
@pytest.mark.parametrize(
    ("cap", "expected"),
    [
        # ABI 3 and 4: device ioctls are not handled yet, so the driver answers.
        (3, ["ioctl ENOTTY", "signal allowed"]),
        (4, ["ioctl ENOTTY", "signal allowed"]),
        # ABI 5 handles device ioctls, and no grant carries the right.
        (5, ["ioctl EACCES", "signal allowed"]),
        # ABI 6 scopes signals to the sandbox, so the unlocked parent is out of reach.
        (6, ["ioctl EACCES", "signal EPERM"]),
    ],
)
def test_each_abi_handles_the_rights_and_scopes_it_knows(tmp_path, dirs, cap, expected):
    if cap > _ABI:
        pytest.skip(f"this kernel's Landlock ABI is {_ABI}, below {cap}")
    result = _run_with_abi_cap(tmp_path, _policy(tmp_path), cap)
    assert result.returncode == 0, result.stderr
    assert result.refusal is None
    assert result.stdout.split("\n")[:2] == expected
    assert (dirs["project"] / "made.txt").read_text() == "ok"
