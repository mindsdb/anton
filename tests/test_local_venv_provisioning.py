"""LocalScratchpadRuntime venv provisioning.

`_find_uv()` only checked a couple of hardcoded directories with no live PATH
search beyond `shutil.which`, so a uv installed via Homebrew/MacPorts/Linuxbrew
was invisible whenever the parent process's PATH didn't happen to include it
(e.g. cowork's Electron app launching cowork-server with a minimal PATH on
macOS). These tests pin the widened candidate list.
"""
from __future__ import annotations

import os
import asyncio
import subprocess
import sys
import threading
import time
from unittest.mock import MagicMock

import pytest

import anton.core.backends.local as local
from anton.core.backends.local import LocalScratchpadRuntime

_DEFAULTS = dict(
    coding_provider="anthropic",
    coding_model="",
    coding_api_key="",
    coding_base_url="",
)


def make_pad(tmp_path, name="probe"):
    return LocalScratchpadRuntime(name=name, _venvs_base=tmp_path, **_DEFAULTS)


@pytest.mark.parametrize(
    "expected_path",
    [
        "/opt/homebrew/bin/uv",  # Homebrew, Apple Silicon
        "/usr/local/bin/uv",  # Homebrew, Intel Mac
        "/opt/local/bin/uv",  # MacPorts
        "/home/linuxbrew/.linuxbrew/bin/uv",  # Linuxbrew
    ],
)
def test_find_uv_checks_extra_unix_locations(monkeypatch, expected_path):
    monkeypatch.setattr(local.shutil, "which", lambda _: None)
    monkeypatch.setattr(local.os.path, "isfile", lambda p: p == expected_path)
    monkeypatch.setattr(local.os, "access", lambda p, mode: True)

    assert local.LocalScratchpadRuntime._find_uv() == expected_path


def test_find_uv_still_prefers_shutil_which(monkeypatch):
    # A uv resolvable via PATH wins over every hardcoded fallback — no
    # regression to the existing, most-common path.
    monkeypatch.setattr(local.shutil, "which", lambda _: "/usr/bin/uv")
    monkeypatch.setattr(local.os.path, "isfile", lambda p: True)
    monkeypatch.setattr(local.os, "access", lambda p, mode: True)

    assert local.LocalScratchpadRuntime._find_uv() == "/usr/bin/uv"


def test_find_uv_returns_none_when_nowhere_found(monkeypatch):
    monkeypatch.setattr(local.shutil, "which", lambda _: None)
    monkeypatch.setattr(local.os.path, "isfile", lambda p: False)

    assert local.LocalScratchpadRuntime._find_uv() is None


def test_stdlib_fallback_symlinks_the_interpreter_on_posix(tmp_path, monkeypatch):
    # venv.create()'s library default is symlinks=False on every platform (only
    # the `python -m venv` CLI defaults it per-OS); a copied macOS Python binary
    # loses its @rpath and crashes on launch.
    if sys.platform == "win32":
        pytest.skip("posix-only: symlink semantics differ on Windows")
    pad = make_pad(tmp_path)
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: None))

    pad._create_venv()

    assert os.path.islink(pad._venv_python)


def test_stdlib_fallback_does_not_force_symlinks_on_windows(tmp_path, monkeypatch):
    # Creating a symlink on Windows needs Developer Mode / elevation; forcing
    # it would trade this crash for that one on machines without it.
    pad = make_pad(tmp_path)
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: None))
    monkeypatch.setattr(local.sys, "platform", "win32")
    monkeypatch.setattr(pad, "_add_windows_firewall_rule", lambda: None)
    create = MagicMock()
    monkeypatch.setattr(local.venv, "create", create)

    pad._create_venv()

    assert create.call_args.kwargs["symlinks"] is False


def test_create_venv_surfaces_uvs_stderr_on_failure(tmp_path, monkeypatch):
    # CalledProcessError.__str__ omits the captured stderr by default, so a
    # real uv failure (bad --python, disk full) reached the user as just
    # "returned non-zero exit status N" — the same masking this whole fix
    # was about, just at venv CREATION instead of verification.
    import subprocess

    pad = make_pad(tmp_path)
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: "/fake/uv"))

    def fake_run(args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=2, cmd=args, stderr=b"error: no interpreter found for python-3.99\n",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(Exception, match="no interpreter found for python-3.99"):
        pad._create_venv()


def test_create_venv_does_not_seed_pip(tmp_path, monkeypatch):
    # Installs always go through `uv pip install --python <venv>`, so a seeded
    # pip was never used; on cloud it cost ~15s writing onto EFS per venv.
    import subprocess

    pad = make_pad(tmp_path)
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: "/fake/uv"))
    run = MagicMock()
    monkeypatch.setattr(subprocess, "run", run)

    pad._create_venv()

    args = run.call_args.args[0]
    assert args[:2] == ["/fake/uv", "venv"]
    assert "--system-site-packages" in args
    assert "--seed" not in args


def _write_fake_python(tmp_path, *, exit_code, stderr_text):
    """A fake venv "python" that fails a specific way when invoked as
    ``<path> -c "..."`` — stands in for a real dyld crash without needing one."""
    script = tmp_path / "fake_python"
    script.write_text(f'#!/bin/sh\necho "{stderr_text}" >&2\nexit {exit_code}\n')
    script.chmod(0o755)
    return str(script)


def test_verify_captures_exit_code_and_stderr_on_failure(tmp_path, monkeypatch):
    if sys.platform == "win32":
        pytest.skip("posix-only: shebang scripts don't run directly on Windows")
    pad = make_pad(tmp_path)
    pad._venv_python = _write_fake_python(tmp_path, exit_code=7, stderr_text="boom")

    assert pad._verify_venv_python() is False
    assert pad._last_verify_error == "exit 7: boom"


def test_verify_clears_the_error_on_success(tmp_path):
    pad = make_pad(tmp_path)
    pad._last_verify_error = "stale error from a previous attempt"
    pad._venv_python = sys.executable

    assert pad._verify_venv_python() is True
    assert pad._last_verify_error is None


def test_verify_clears_a_stale_error_when_venv_python_is_unset(tmp_path):
    # A retry that never got as far as setting _venv_python must not carry
    # a PREVIOUS attempt's error into this attempt's (unrelated) failure.
    pad = make_pad(tmp_path)
    pad._last_verify_error = "stale error from a previous attempt"
    pad._venv_python = None

    assert pad._verify_venv_python() is False
    assert pad._last_verify_error is None


def test_verify_clears_a_stale_error_when_the_interpreter_is_missing(tmp_path):
    pad = make_pad(tmp_path)
    pad._last_verify_error = "stale error from a previous attempt"
    pad._venv_python = str(tmp_path / "does-not-exist")

    assert pad._verify_venv_python() is False
    assert pad._last_verify_error is None


def test_verify_captures_an_exception_reason(tmp_path):
    if sys.platform == "win32":
        pytest.skip("posix-only: exec-permission semantics differ on Windows")
    pad = make_pad(tmp_path)
    not_executable = tmp_path / "not_a_python"
    not_executable.write_text("not a real interpreter")
    not_executable.chmod(0o644)
    pad._venv_python = str(not_executable)

    assert pad._verify_venv_python() is False
    assert pad._last_verify_error


def _in_daemon_thread(*, fn, timeout: float = 30.0):
    """Run ``fn`` on a daemon thread, then return its result or raise its error.

    A failed build deletes its venv while it holds that venv's lock, so the
    lock must be reentrant. On a daemon thread, a lock that deadlocks fails
    the test instead of hanging the run.
    """
    results: list = []
    errors: list[Exception] = []

    def run():
        try:
            results.append(fn())
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=timeout)
    assert not worker.is_alive(), f"still running after {timeout} s: the venv lock deadlocked"
    if errors:
        raise errors[0]
    return results[0]


def test_ensure_venv_failure_message_includes_the_verify_detail(tmp_path, monkeypatch):
    pad = make_pad(tmp_path)

    def fake_verify():
        pad._last_verify_error = "exit 1: dyld: Library not loaded"
        return False

    monkeypatch.setattr(pad, "_create_venv", lambda: None)
    monkeypatch.setattr(pad, "_verify_venv_python", fake_verify)

    with pytest.raises(RuntimeError, match="dyld: Library not loaded"):
        _in_daemon_thread(fn=pad._ensure_venv)


@pytest.mark.parametrize(
    "uv_path, expected_method",
    [
        ("/opt/homebrew/bin/uv", "uv venv (/opt/homebrew/bin/uv)"),
        (None, "stdlib venv (uv not found)"),
    ],
)
def test_ensure_venv_failure_message_states_facts_without_fix_hints(
    tmp_path, monkeypatch, uv_path, expected_method
):
    # The model relays this message to the user; a generic hint such as
    # "run python3 -c ..." was repeated as a diagnosis of the user's system
    # Python, which the scratchpad venv is not built from.
    pad = make_pad(tmp_path)
    monkeypatch.setattr(pad, "_create_venv", lambda: None)
    monkeypatch.setattr(pad, "_verify_venv_python", lambda: False)
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: uv_path))

    with pytest.raises(RuntimeError) as exc_info:
        _in_daemon_thread(fn=pad._ensure_venv)

    message = str(exc_info.value)
    assert str(tmp_path / "probe") in message
    assert sys.executable in message
    assert expected_method in message
    assert "python3 -c" not in message


def test_find_uv_checks_scoop_on_windows(monkeypatch):
    # scoop (~/scoop/shims/uv.exe) is the Windows analogue of Homebrew — a
    # package-manager install invisible to a GUI-launched parent's PATH.
    monkeypatch.setattr(local.sys, "platform", "win32")
    monkeypatch.setattr(local.shutil, "which", lambda _: None)
    scoop_path = os.path.expanduser("~/scoop/shims/uv.exe")
    monkeypatch.setattr(local.os.path, "isfile", lambda p: p == scoop_path)
    monkeypatch.setattr(local.os, "access", lambda p, mode: True)

    assert local.LocalScratchpadRuntime._find_uv() == scoop_path


def test_find_uv_checks_winget_links_on_windows(monkeypatch):
    monkeypatch.setattr(local.sys, "platform", "win32")
    monkeypatch.setattr(local.shutil, "which", lambda _: None)
    monkeypatch.setenv("LOCALAPPDATA", "C:\\Users\\u\\AppData\\Local")
    winget_path = os.path.join(
        "C:\\Users\\u\\AppData\\Local", "Microsoft", "WinGet", "Links", "uv.exe"
    )
    monkeypatch.setattr(local.os.path, "isfile", lambda p: p == winget_path)
    monkeypatch.setattr(local.os, "access", lambda p, mode: True)

    assert local.LocalScratchpadRuntime._find_uv() == winget_path


def _record_threads(*, monkeypatch, threads: list[int]) -> None:
    """Record the thread of every venv subprocess, venv.create and rmtree call."""

    def recording(*, real):
        def call(*args, **kwargs):
            threads.append(threading.get_ident())
            return real(*args, **kwargs)

        return call

    monkeypatch.setattr(subprocess, "run", recording(real=subprocess.run))
    monkeypatch.setattr(local.venv, "create", recording(real=local.venv.create))
    monkeypatch.setattr(local.shutil, "rmtree", recording(real=local.shutil.rmtree))


async def test_start_runs_the_venv_work_off_the_event_loop(tmp_path, monkeypatch):
    # Every start runs a `python -c print('ok')` check, and the first one also
    # builds the venv. On the event loop thread, each start stalled every other
    # turn and request in the host's process until it finished.
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: None))
    threads: list[int] = []
    _record_threads(monkeypatch=monkeypatch, threads=threads)
    pad = make_pad(tmp_path)

    try:
        await pad.start()
    finally:
        await pad.close()

    assert threads, "start() neither built nor checked the venv"
    assert threading.get_ident() not in threads


async def test_reset_and_cleanup_run_the_venv_work_off_the_event_loop(tmp_path, monkeypatch):
    # reset() checks the venv again before its restart, and cleanup() deletes
    # it: both reach the same blocking work as start().
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: None))
    pad = make_pad(tmp_path)
    await pad.start()
    threads: list[int] = []
    _record_threads(monkeypatch=monkeypatch, threads=threads)

    try:
        await pad.reset()
    finally:
        await pad.cleanup()

    assert threads, "reset() and cleanup() neither checked nor deleted the venv"
    assert threading.get_ident() not in threads
    assert not (tmp_path / "probe").exists()


def _record_rmtree_threads(*, monkeypatch, threads: list[int]) -> None:
    """Record the thread of every rmtree call the runtime makes."""
    real_rmtree = local.shutil.rmtree

    def recording(*args, **kwargs):
        threads.append(threading.get_ident())
        return real_rmtree(*args, **kwargs)

    monkeypatch.setattr(local.shutil, "rmtree", recording)


async def test_reset_deletes_a_broken_venv_off_the_event_loop(tmp_path, monkeypatch):
    # reset() deletes a venv whose Python no longer runs, then builds a new one.
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: None))
    pad = make_pad(tmp_path)
    await pad.start()
    os.remove(pad._venv_python)
    deleted_on: list[int] = []
    _record_rmtree_threads(monkeypatch=monkeypatch, threads=deleted_on)

    try:
        await pad.reset()
        assert pad._verify_venv_python()
    finally:
        await pad.cleanup()

    assert deleted_on, "reset() kept the broken venv"
    assert threading.get_ident() not in deleted_on


async def test_a_failed_spawn_deletes_the_venv_off_the_event_loop(tmp_path, monkeypatch):
    # When the pad's process can't start, start() deletes the venv so the next
    # attempt builds a fresh one.
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: None))
    pad = make_pad(tmp_path)

    async def refuse_spawn(*_args, **_kwargs):
        raise OSError("spawn refused")

    monkeypatch.setattr(local.asyncio, "create_subprocess_exec", refuse_spawn)
    deleted_on: list[int] = []
    _record_rmtree_threads(monkeypatch=monkeypatch, threads=deleted_on)

    try:
        with pytest.raises(RuntimeError, match="Failed to start scratchpad: spawn refused"):
            await pad.start()
    finally:
        await pad.close()

    assert not (tmp_path / "probe").exists()
    assert deleted_on, "the failed spawn kept the venv"
    assert threading.get_ident() not in deleted_on


async def test_install_packages_provisions_the_venv_off_the_event_loop(tmp_path, monkeypatch):
    class _Provisioned(Exception):
        pass

    threads: list[int] = []

    def fake_ensure_venv():
        threads.append(threading.get_ident())
        raise _Provisioned

    pad = make_pad(tmp_path)
    monkeypatch.setattr(pad, "_ensure_venv", fake_ensure_venv)

    with pytest.raises(_Provisioned):
        await pad.install_packages(["requests"])

    assert threads and threads[0] != threading.get_ident()


def test_concurrent_provisions_of_one_venv_path_create_it_once(tmp_path, monkeypatch):
    # Concurrent turns in one project share a pad's venv directory and
    # provision it on worker threads. Two first starts must not both build it,
    # each deleting the other's half-built venv.
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: None))
    real_create = LocalScratchpadRuntime._create_venv
    creates: list[str] = []

    def slow_create(self):
        creates.append(self.name)
        # Long enough that a second thread without the lock also reaches here.
        time.sleep(0.3)
        real_create(self)

    monkeypatch.setattr(LocalScratchpadRuntime, "_create_venv", slow_create)
    pads = [make_pad(tmp_path, name="shared"), make_pad(tmp_path, name="shared")]
    start_together = threading.Barrier(len(pads))
    errors: list[Exception] = []

    def provision(*, pad):
        start_together.wait()
        try:
            pad._ensure_venv()
        except Exception as exc:
            errors.append(exc)

    # Daemon threads, so a deadlocked lock fails this test without hanging the run.
    workers = [
        threading.Thread(target=provision, kwargs={"pad": pad}, daemon=True) for pad in pads
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=60)

    assert not any(worker.is_alive() for worker in workers)
    assert not errors
    assert creates == ["shared"]
    assert pads[0]._venv_python == pads[1]._venv_python
    assert all(pad._verify_venv_python() for pad in pads)


def test_deleting_a_venv_waits_for_a_build_of_the_same_directory(tmp_path, monkeypatch):
    # cleanup(), reset() and a failed spawn delete the pad's venv directory.
    # While another pad with the same name builds that directory, the delete
    # waits for the build to finish instead of removing a half-built venv.
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: None))
    real_create = LocalScratchpadRuntime._create_venv
    building = threading.Event()
    events: list[str] = []

    def slow_create(self):
        building.set()
        # Long enough that a delete without the lock lands inside the build.
        time.sleep(0.5)
        real_create(self)
        events.append("built")

    real_rmtree = local.shutil.rmtree

    def recording_rmtree(*args, **kwargs):
        events.append("deleted")
        return real_rmtree(*args, **kwargs)

    monkeypatch.setattr(LocalScratchpadRuntime, "_create_venv", slow_create)
    monkeypatch.setattr(local.shutil, "rmtree", recording_rmtree)
    builder, deleter = make_pad(tmp_path), make_pad(tmp_path)
    # As after the deleter's own earlier start of the same venv.
    deleter._venv_dir = str(tmp_path / "probe")

    build = threading.Thread(target=builder._ensure_venv, daemon=True)
    build.start()
    assert building.wait(timeout=30), "the build never started"
    delete = threading.Thread(target=deleter._nuke_venv, daemon=True)
    delete.start()
    build.join(timeout=60)
    delete.join(timeout=60)

    assert not build.is_alive() and not delete.is_alive()
    assert events == ["built", "deleted"]
    assert not (tmp_path / "probe").exists()


def test_builds_of_different_venv_directories_run_at_the_same_time(tmp_path, monkeypatch):
    # The lock is per directory: a pad's first start never waits for another
    # pad's build. Each create waits until the other one is running too.
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: None))
    real_create = LocalScratchpadRuntime._create_venv
    both_building = threading.Barrier(2, timeout=10)

    def create_alongside_the_other(self):
        both_building.wait()
        real_create(self)

    monkeypatch.setattr(LocalScratchpadRuntime, "_create_venv", create_alongside_the_other)
    pads = [make_pad(tmp_path, name="first"), make_pad(tmp_path, name="second")]
    errors: list[Exception] = []

    def provision(*, pad):
        try:
            pad._ensure_venv()
        except Exception as exc:
            errors.append(exc)

    workers = [
        threading.Thread(target=provision, kwargs={"pad": pad}, daemon=True) for pad in pads
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=90)

    assert not any(worker.is_alive() for worker in workers)
    assert not errors, errors
    assert all(pad._verify_venv_python() for pad in pads)


async def test_reset_checks_health_after_a_shared_directory_rebuild(tmp_path, monkeypatch):
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: None))
    resetting = make_pad(tmp_path)
    await resetting.start()
    os.remove(resetting._venv_python)
    builder = make_pad(tmp_path)
    building, release_build = threading.Event(), threading.Event()
    real_create = local.venv.create
    creates = 0

    def gated_create(*args, **kwargs):
        nonlocal creates
        creates += 1
        building.set()
        assert release_build.wait(10), "the test never released the build"
        return real_create(*args, **kwargs)

    checked_during_build = False
    real_verify = resetting._verify_venv_python

    def record_verification():
        nonlocal checked_during_build
        checked_during_build |= not release_build.is_set()
        return real_verify()

    monkeypatch.setattr(local.venv, "create", gated_create)
    build = asyncio.create_task(asyncio.to_thread(builder._ensure_venv))
    assert await asyncio.to_thread(building.wait, 10)
    monkeypatch.setattr(resetting, "_verify_venv_python", record_verification)
    reset = asyncio.create_task(resetting.reset())
    try:
        # The rebuild holds the directory lock. A reset must wait before it
        # decides whether the interpreter is broken, not queue a stale delete.
        await asyncio.sleep(0.1)
        release_build.set()
        await asyncio.wait_for(asyncio.gather(build, reset), timeout=20)
        assert not checked_during_build
        assert creates == 1
        assert builder._verify_venv_python()
    finally:
        release_build.set()
        await asyncio.gather(build, reset, return_exceptions=True)
        await resetting.close()


@pytest.mark.parametrize("action", ["reset", "install"])
@pytest.mark.parametrize("cancel_teardown", [False, True])
async def test_cancelled_provisioning_finishes_before_teardown(
    tmp_path, monkeypatch, action, cancel_teardown
):
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: None))
    pad = make_pad(tmp_path)
    await pad.start()
    os.remove(pad._venv_python)
    building, release_build, provisioned = (
        threading.Event(), threading.Event(), threading.Event()
    )
    real_create = local.venv.create
    creates = 0

    def gated_create(*args, **kwargs):
        nonlocal creates
        creates += 1
        building.set()
        assert release_build.wait(10), "the test never released the build"
        return real_create(*args, **kwargs)

    real_ensure = pad._ensure_venv

    def record_provisioning():
        try:
            return real_ensure()
        finally:
            provisioned.set()

    monkeypatch.setattr(local.venv, "create", gated_create)
    monkeypatch.setattr(pad, "_ensure_venv", record_provisioning)
    operation = asyncio.create_task(
        pad.reset() if action == "reset" else pad.install_packages(["requests"])
    )
    assert await asyncio.to_thread(building.wait, 10)
    operation.cancel()
    await asyncio.sleep(0.05)
    operation.cancel()  # A second Stop must not abandon the same worker.
    close = asyncio.create_task(pad.close())
    try:
        done, _ = await asyncio.wait({operation, close}, timeout=0.1)
        finished_early = bool(done)
        if cancel_teardown:
            close.cancel()
            await asyncio.sleep(0.05)
            close.cancel()
        release_build.set()
        assert await asyncio.to_thread(provisioned.wait, 10)
        await asyncio.wait_for(asyncio.gather(operation, close, return_exceptions=True), 10)
        assert not finished_early, "cancellation or teardown abandoned venv work"
        assert operation.cancelled()
        assert close.cancelled() is cancel_teardown
        assert creates == 1, "teardown cleared a field the builder still used"
        assert pad._venv_dir is None
        assert pad._venv_python is None
    finally:
        release_build.set()
        await asyncio.gather(operation, close, return_exceptions=True)
        await pad.close()
