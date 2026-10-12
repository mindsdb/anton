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
from pathlib import Path
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


@pytest.fixture(autouse=True)
def _forget_policy_blocks():
    """A recorded policy block lasts for the process, so isolate each test."""
    local._POLICY_BLOCKED_BASES.clear()
    yield
    local._POLICY_BLOCKED_BASES.clear()


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


def _policy_error(code: int) -> OSError:
    """An ``OSError`` shaped like Windows refusing a launch with ``code``."""
    exc = OSError(13, "An Application Control policy has blocked this file")
    exc.winerror = code
    return exc


def _raise_on_run(exc: OSError):
    def run(*args, **kwargs):
        raise exc

    return run


def _refuse_every_run(code: int):
    """A subprocess.run that Windows refuses, with a new error per launch as CPython raises."""

    def run(*args, **kwargs):
        raise _policy_error(code)

    return run


@pytest.mark.parametrize("code", [4551, 1260])
def test_verify_records_a_launch_refused_by_policy(tmp_path, monkeypatch, code):
    pad = make_pad(tmp_path)
    pad._venv_python = sys.executable
    blocked = _policy_error(code)
    monkeypatch.setattr(subprocess, "run", _raise_on_run(blocked))

    assert pad._verify_venv_python() is False
    assert pad._policy_block is blocked
    assert "Application Control policy" in pad._last_verify_error


def test_verify_does_not_record_other_windows_errors_as_policy(tmp_path, monkeypatch):
    pad = make_pad(tmp_path)
    pad._venv_python = sys.executable
    monkeypatch.setattr(subprocess, "run", _raise_on_run(_policy_error(5)))

    assert pad._verify_venv_python() is False
    assert pad._policy_block is None


def test_verify_clears_a_recorded_policy_block_on_success(tmp_path):
    pad = make_pad(tmp_path)
    pad._policy_block = _policy_error(4551)
    pad._venv_python = sys.executable

    assert pad._verify_venv_python() is True
    assert pad._policy_block is None


def test_a_venv_build_refused_by_policy_is_recorded(tmp_path, monkeypatch):
    # uv.exe can sit in a folder the same policy covers, so the build itself
    # may be the launch Windows refuses.
    pad = make_pad(tmp_path / "venvs")
    blocked = _policy_error(4551)

    def blocked_build():
        raise blocked

    monkeypatch.setattr(pad, "_create_venv", blocked_build)

    pad._ensure_venv()
    assert pad._base_policy_block() is blocked


def test_recycling_an_existing_venv_records_a_policy_block(tmp_path, monkeypatch):
    pad = make_pad(tmp_path)
    existing = tmp_path / "probe"
    existing.mkdir()
    monkeypatch.setattr(local.os.path, "exists", lambda p: True)
    blocked = _policy_error(4551)
    monkeypatch.setattr(subprocess, "run", _raise_on_run(blocked))

    assert pad._try_recycle_venv(existing) is False
    assert pad._policy_block is blocked


def _count_builds(pad, monkeypatch, *, verify):
    """Replace the venv build with a counter and verification with ``verify``."""
    builds: list[str] = []
    monkeypatch.setattr(pad, "_create_venv", lambda: builds.append(pad.name))
    monkeypatch.setattr(pad, "_verify_venv_python", verify)
    return builds


def _blocked_verify(pad):
    def verify():
        pad._last_verify_error = "[WinError 4551] An Application Control policy has blocked this file"
        pad._record_policy_block(_policy_error(4551), program=pad._venv_python)
        return False

    return verify


def test_a_policy_block_stops_the_build_after_one_attempt(tmp_path, monkeypatch):
    pad = make_pad(tmp_path)
    builds = _count_builds(pad, monkeypatch, verify=_blocked_verify(pad))

    with pytest.raises(RuntimeError, match="4551"):
        pad._ensure_venv()
    assert builds == ["probe"]


def test_a_renamed_pad_under_a_blocked_base_skips_the_build(tmp_path, monkeypatch):
    first = make_pad(tmp_path / "venvs", name="probe")
    _count_builds(first, monkeypatch, verify=_blocked_verify(first))
    with pytest.raises(RuntimeError):
        first._ensure_venv()

    renamed = make_pad(tmp_path / "venvs", name="probe_retry")
    builds = _count_builds(renamed, monkeypatch, verify=lambda: True)

    renamed._ensure_venv()
    assert builds == []
    assert renamed._in_fallback


def test_a_policy_block_leaves_other_venvs_bases_building(tmp_path, monkeypatch):
    blocked_pad = make_pad(tmp_path / "blocked")
    _count_builds(blocked_pad, monkeypatch, verify=_blocked_verify(blocked_pad))
    with pytest.raises(RuntimeError):
        blocked_pad._ensure_venv()

    other = make_pad(tmp_path / "other")
    builds = _count_builds(other, monkeypatch, verify=lambda: True)
    monkeypatch.setattr(other, "_setup_parent_site_packages", lambda: None)
    monkeypatch.setattr(other, "_save_python_version", lambda: None)

    other._ensure_venv()
    assert builds == ["probe"]


def test_a_failure_that_is_not_a_policy_block_still_retries(tmp_path, monkeypatch):
    pad = make_pad(tmp_path)
    builds = _count_builds(pad, monkeypatch, verify=lambda: False)

    with pytest.raises(RuntimeError):
        pad._ensure_venv()
    assert len(builds) == LocalScratchpadRuntime._MAX_VENV_RETRIES


def test_a_policy_block_names_the_blocker_and_a_recovery_path(tmp_path, monkeypatch):
    # The model relays this message; the ticket's failure ended with the user
    # told to re-upload files, which cannot start a blocked interpreter.
    pad = make_pad(tmp_path)
    blocked_python = str(tmp_path / "probe" / "Scripts" / "python.exe")

    def build():
        pad._venv_dir = str(tmp_path / "probe")
        pad._venv_python = blocked_python

    monkeypatch.setattr(pad, "_create_venv", build)
    monkeypatch.setattr(local.os.path, "exists", lambda p: True)
    monkeypatch.setattr(subprocess, "run", _refuse_every_run(4551))

    with pytest.raises(RuntimeError) as exc_info:
        pad._ensure_venv()

    message = str(exc_info.value)
    assert f"Windows blocked {blocked_python}" in message
    assert "WinError 4551" in message
    assert "Application Control policy" in message
    assert "re-uploading files will not help" in message
    assert f"allow programs in {tmp_path}" in message
    assert "restarted" in message
    assert "after 3 attempts" not in message


async def test_a_spawn_refused_by_policy_reports_it_and_keeps_the_venv(tmp_path, monkeypatch):
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: None))
    pad = make_pad(tmp_path / "venvs")

    async def refuse_spawn(*_args, **_kwargs):
        raise _policy_error(4551)

    monkeypatch.setattr(local.asyncio, "create_subprocess_exec", refuse_spawn)

    try:
        with pytest.raises(RuntimeError) as exc_info:
            await pad.start()
    finally:
        await pad.close()

    message = str(exc_info.value)
    venv_python = tmp_path / "venvs" / "probe" / "bin" / "python"
    assert f"Windows blocked {venv_python}" in message
    assert f"Running on the app's own Python ({sys.executable}) failed as well" in message
    assert "will be recreated" not in message
    assert (tmp_path / "venvs" / "probe").is_dir(), "a rebuild cannot help, so the venv stays"
    renamed = make_pad(tmp_path / "venvs", name="probe_retry")
    builds = _count_builds(renamed, monkeypatch, verify=lambda: True)
    renamed._ensure_venv()
    assert builds == []
    assert renamed._in_fallback


def _fallback_pad(tmp_path):
    """A pad whose venvs base Windows has refused, so it runs on sys.executable."""
    pad = make_pad(tmp_path / "venvs")
    pad._record_policy_block(_policy_error(4551), program=None)
    pad._ensure_venv()
    return pad


def _record_rmtree_paths(monkeypatch) -> list[Path]:
    """Record the resolved target of every rmtree call the runtime makes."""
    paths: list[Path] = []
    real_rmtree = local.shutil.rmtree

    def recording(path, *args, **kwargs):
        paths.append(Path(path).resolve())
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(local.shutil, "rmtree", recording)
    return paths


def _under_app_environment(paths: list[Path]) -> list[Path]:
    prefix = Path(sys.prefix).resolve()
    return [p for p in paths if p.is_relative_to(prefix)]


def test_fallback_runs_on_the_app_interpreter_with_a_packages_folder(tmp_path):
    pad = _fallback_pad(tmp_path)

    assert pad._venv_python == sys.executable
    assert pad._venv_dir == str(tmp_path / "scratchpad-packages" / "probe")
    assert (tmp_path / "scratchpad-packages" / "probe").is_dir()
    assert pad.venv_python() is None


def test_fallback_packages_folders_follow_the_venv_names(tmp_path):
    # Sanitized names would put "a b" and "a_b" in one folder under two
    # different venv locks, so one pad's cleanup could delete the other's installs.
    spaced = make_pad(tmp_path / "venvs", name="a b")
    spaced._record_policy_block(_policy_error(4551), program=None)
    spaced._ensure_venv()
    underscored = make_pad(tmp_path / "venvs", name="a_b")
    underscored._ensure_venv()

    assert spaced._venv_dir == str(tmp_path / "scratchpad-packages" / "a b")
    assert underscored._venv_dir == str(tmp_path / "scratchpad-packages" / "a_b")


@pytest.mark.parametrize("name", ["../escape", ".", ""])
def test_fallback_refuses_a_pad_name_that_is_not_a_folder_of_its_own(tmp_path, name):
    pad = make_pad(tmp_path / "venvs", name=name)
    pad._record_policy_block(_policy_error(4551), program=None)

    with pytest.raises(RuntimeError, match="points outside"):
        pad._ensure_venv()
    assert pad._venv_dir is None
    assert not (tmp_path / "escape").exists()


def test_fallback_is_refused_when_the_app_interpreter_is_blocked_too(tmp_path, monkeypatch):
    pad = make_pad(tmp_path / "venvs")
    venv_python = str(tmp_path / "venvs" / "probe" / "Scripts" / "python.exe")
    pad._record_policy_block(_policy_error(4551), program=venv_python)
    monkeypatch.setattr(subprocess, "run", _refuse_every_run(4551))

    with pytest.raises(RuntimeError) as exc_info:
        pad._ensure_venv()

    message = str(exc_info.value)
    assert f"Windows blocked {venv_python}" in message
    assert f"Running on the app's own Python ({sys.executable}) failed as well" in message
    assert not pad._in_fallback
    assert not (tmp_path / "scratchpad-packages").exists()


def _refuse_uv_only(uv: str):
    """A subprocess.run where Windows refuses uv and runs everything else."""
    real_run = subprocess.run

    def run(cmd, *args, **kwargs):
        if cmd[0] == uv:
            raise _policy_error(4551)
        return real_run(cmd, *args, **kwargs)

    return run


def test_a_blocked_uv_is_named_in_the_fallback_note(tmp_path, monkeypatch):
    uv = str(tmp_path / "bin" / "uv.exe")
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: uv))
    monkeypatch.setattr(subprocess, "run", _refuse_uv_only(uv))
    pad = make_pad(tmp_path / "venvs")

    pad._ensure_venv()

    assert pad._in_fallback
    assert f"Windows blocked {uv}" in pad._pending_recovery_note


def test_a_blocked_uv_is_named_in_the_recovery_path(tmp_path, monkeypatch):
    # Allowing the venvs folder alone would leave uv blocked.
    uv = str(tmp_path / "bin" / "uv.exe")
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: uv))
    monkeypatch.setattr(subprocess, "run", _refuse_every_run(4551))
    pad = make_pad(tmp_path / "venvs")

    with pytest.raises(RuntimeError) as exc_info:
        pad._ensure_venv()

    message = str(exc_info.value)
    assert f"Windows blocked {uv}" in message
    assert f"allow {uv} and programs in {tmp_path / 'venvs'}" in message


async def test_an_install_refused_by_policy_names_the_installer(tmp_path, monkeypatch):
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: "/usr/bin/uv"))
    pad = _fallback_pad(tmp_path)

    async def refuse(*_args, **_kwargs):
        raise _policy_error(4551)

    monkeypatch.setattr(local.asyncio, "create_subprocess_exec", refuse)

    result = await pad.install_packages(["requests"])

    assert result.startswith("Install failed: Windows blocked /usr/bin/uv")
    assert "WinError 4551" in result


async def test_a_spawn_refused_by_policy_starts_on_the_app_interpreter(tmp_path, monkeypatch):
    # The venv's Python passed its check but Windows refused the spawn; the
    # same start moves to the app's own Python instead of reporting a block.
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: None))
    real_exec = local.asyncio.create_subprocess_exec

    async def refuse_the_venv(program, *args, **kwargs):
        if program != sys.executable:
            raise _policy_error(4551)
        return await real_exec(program, *args, **kwargs)

    monkeypatch.setattr(local.asyncio, "create_subprocess_exec", refuse_the_venv)
    pad = make_pad(tmp_path / "venvs")
    try:
        await pad.start()
        cell = await pad.execute("import sys; print(sys.executable)")
    finally:
        await pad.close()

    assert pad._in_fallback
    assert cell.stdout.strip() == sys.executable
    assert "WinError 4551" in cell.logs


async def test_fallback_cleanup_deletes_only_the_packages_folder(tmp_path, monkeypatch):
    pad = _fallback_pad(tmp_path)
    await pad.start()
    deleted = _record_rmtree_paths(monkeypatch)

    await pad.cleanup()

    assert deleted == [(tmp_path / "scratchpad-packages" / "probe").resolve()]
    assert not (tmp_path / "scratchpad-packages" / "probe").exists()


async def test_fallback_reset_restarts_on_the_app_interpreter(tmp_path, monkeypatch):
    # The check fails once, so reset() takes its delete path for a broken
    # interpreter; in fallback mode that may only remove the packages folder.
    pad = _fallback_pad(tmp_path)
    await pad.start()
    deleted = _record_rmtree_paths(monkeypatch)
    real_verify = pad._verify_venv_python
    checks: list[bool] = []

    def fails_once() -> bool:
        checks.append(True)
        return len(checks) > 1 and real_verify()

    monkeypatch.setattr(pad, "_verify_venv_python", fails_once)
    try:
        await pad.reset()
        cell = await pad.execute("import sys; print(sys.executable)")
    finally:
        await pad.close()

    assert cell.stdout.strip() == sys.executable
    assert deleted == [(tmp_path / "scratchpad-packages" / "probe").resolve()]
    assert _under_app_environment(deleted) == []


async def test_a_failed_fallback_spawn_never_deletes_the_app_environment(
    tmp_path, monkeypatch
):
    pad = _fallback_pad(tmp_path)
    deleted = _record_rmtree_paths(monkeypatch)

    async def refuse_spawn(*_args, **_kwargs):
        raise OSError("spawn refused")

    monkeypatch.setattr(local.asyncio, "create_subprocess_exec", refuse_spawn)
    try:
        with pytest.raises(RuntimeError, match="spawn refused"):
            await pad.start()
    finally:
        await pad.close()

    assert deleted
    assert _under_app_environment(deleted) == []
    assert Path(sys.executable).exists()


async def test_a_fallback_pad_reads_a_zip_of_workbooks(tmp_path):
    # The ticket's upload: a ZIP of Excel reports. zipfile and openpyxl ship
    # with the app's own Python, so no install is needed to read it.
    pad = _fallback_pad(tmp_path)
    archive = tmp_path / "reports.zip"
    code = (
        "import io, zipfile, openpyxl\n"
        "book = openpyxl.Workbook(); book.active['A1'] = 'March'\n"
        "data = io.BytesIO(); book.save(data)\n"
        f"with zipfile.ZipFile({str(archive)!r}, 'w') as z: z.writestr('march.xlsx', data.getvalue())\n"
        f"with zipfile.ZipFile({str(archive)!r}) as z:\n"
        "    sheet = openpyxl.load_workbook(io.BytesIO(z.read('march.xlsx'))).active\n"
        "print(sheet['A1'].value)\n"
    )
    await pad.start()
    try:
        cell = await pad.execute(code)
    finally:
        await pad.close()

    assert cell.error is None, cell.error
    assert cell.stdout.strip() == "March"


async def test_fallback_child_env_keeps_the_anton_root_first(tmp_path, monkeypatch):
    pad = _fallback_pad(tmp_path)
    envs: list[dict[str, str]] = []
    real_exec = local.asyncio.create_subprocess_exec

    async def recording_exec(*args, **kwargs):
        envs.append(kwargs["env"])
        return await real_exec(*args, **kwargs)

    monkeypatch.setattr(local.asyncio, "create_subprocess_exec", recording_exec)
    await pad.start()
    await pad.close()

    packages = str(tmp_path / "scratchpad-packages" / "probe")
    parts = envs[0]["PYTHONPATH"].split(os.pathsep)
    anton_root = str(Path(local.__file__).resolve().parents[3])
    assert parts.index(anton_root) < parts.index(packages)
    assert envs[0]["PIP_TARGET"] == packages


async def test_the_fallback_note_reaches_the_model_once(tmp_path):
    pad = _fallback_pad(tmp_path)
    await pad.start()
    try:
        first = await pad.execute("print(1)")
        await pad.resume()
        after_resume = await pad.execute("print(2)")
        await pad.reset()
        after_reset = await pad.execute("print(3)")
        await pad.cancel()
        after_cancel = await pad.execute("print(4)")
    finally:
        await pad.close()

    assert "runs on the app's own Python" in first.logs
    assert "WinError 4551" in first.logs
    assert "keeps the app's version" in first.logs
    assert "runs on the app's own Python" not in after_resume.logs
    assert "runs on the app's own Python" not in after_reset.logs
    assert after_cancel.stdout.strip() == "4"
    assert "runs on the app's own Python" not in after_cancel.logs


def _record_install_commands(monkeypatch) -> list[tuple[str, ...]]:
    """Capture install commands and answer each with a successful fake process."""
    commands: list[tuple[str, ...]] = []

    class _Done:
        returncode = 0

        async def communicate(self):
            return b"installed", None

    async def fake_exec(*cmd, **_kwargs):
        commands.append(cmd)
        return _Done()

    monkeypatch.setattr(local.asyncio, "create_subprocess_exec", fake_exec)
    return commands


async def test_fallback_installs_go_to_the_packages_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: "/usr/bin/uv"))
    pad = _fallback_pad(tmp_path)
    commands = _record_install_commands(monkeypatch)

    await pad.install_packages(["requests"])

    packages = str(tmp_path / "scratchpad-packages" / "probe")
    assert commands == [
        ("/usr/bin/uv", "pip", "install", "--python", sys.executable, "--target", packages, "requests")
    ]


async def test_fallback_installs_without_uv_use_pip_with_a_target(tmp_path, monkeypatch):
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: None))
    real_find_spec = local.importlib.util.find_spec
    monkeypatch.setattr(
        local.importlib.util, "find_spec",
        lambda name, *a: object() if name == "pip" else real_find_spec(name, *a),
    )
    pad = _fallback_pad(tmp_path)
    commands = _record_install_commands(monkeypatch)

    await pad.install_packages(["requests"])

    packages = str(tmp_path / "scratchpad-packages" / "probe")
    assert commands == [
        (sys.executable, "-m", "pip", "install", "--no-input", "--target", packages, "requests")
    ]


async def test_fallback_without_any_installer_names_the_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: None))
    real_find_spec = local.importlib.util.find_spec
    monkeypatch.setattr(
        local.importlib.util, "find_spec",
        lambda name, *a: None if name == "pip" else real_find_spec(name, *a),
    )
    pad = _fallback_pad(tmp_path)
    commands = _record_install_commands(monkeypatch)

    result = await pad.install_packages(["requests"])

    assert commands == []
    assert result.startswith("Install failed")
    assert "no installer is available" in result


async def test_fallback_installs_are_remembered_in_the_packages_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: "/usr/bin/uv"))
    pad = _fallback_pad(tmp_path)
    _record_install_commands(monkeypatch)
    await pad.install_packages(["requests"])
    await pad.close()

    assert (tmp_path / "scratchpad-packages" / "probe" / "requirements.txt").read_text() == "requests\n"
    again = _fallback_pad(tmp_path)
    assert await again.install_packages(["requests"]) == "All packages already installed."


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


@pytest.mark.parametrize(
    ("error", "expected_spawns"), [(OSError, 1), (FileNotFoundError, 2)]
)
async def test_a_failed_spawn_deletes_the_venv_off_the_event_loop(
    tmp_path, monkeypatch, error, expected_spawns
):
    # When the pad's process can't start, start() deletes the venv so the next
    # attempt builds a fresh one. A missing interpreter gets one more check and
    # spawn first, and only one.
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: None))
    pad = make_pad(tmp_path)
    spawns = 0

    async def refuse_spawn(*_args, **_kwargs):
        nonlocal spawns
        spawns += 1
        raise error("spawn refused")

    monkeypatch.setattr(local.asyncio, "create_subprocess_exec", refuse_spawn)
    deleted_on: list[int] = []
    _record_rmtree_threads(monkeypatch=monkeypatch, threads=deleted_on)

    try:
        with pytest.raises(RuntimeError, match="Failed to start scratchpad: spawn refused"):
            await pad.start()
    finally:
        await pad.close()

    assert spawns == expected_spawns
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


def test_a_start_that_finds_a_healthy_venv_takes_no_lock(tmp_path, monkeypatch):
    # Every start on a shared venv runs a health check. Only a rebuild or a
    # delete takes the directory lock, so starts that find the venv healthy
    # never queue behind each other or behind another pad's lock holder.
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: None))
    started = make_pad(tmp_path)
    started._ensure_venv()
    held, release = threading.Event(), threading.Event()

    def hold_the_lock():
        with local._venv_lock(venv_path=tmp_path / "probe"):
            held.set()
            release.wait(30)

    threading.Thread(target=hold_the_lock, daemon=True).start()
    assert held.wait(10), "the lock holder never started"
    try:
        # A pad that already has its venv checks its own interpreter, and a new
        # pad recycles the directory. Neither waits for the held lock.
        _in_daemon_thread(fn=started._ensure_venv, timeout=10)
        _in_daemon_thread(fn=make_pad(tmp_path)._ensure_venv, timeout=10)
    finally:
        release.set()


async def test_a_start_survives_a_remove_between_its_check_and_its_spawn(
    tmp_path, monkeypatch
):
    # Pads with one name in one workspace share a venv, and a healthy start
    # checks it without the directory lock. Another pad's cleanup() can then
    # delete the venv after this start's check and before its spawn.
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: None))
    remover = make_pad(tmp_path)
    await remover.start()
    starting = make_pad(tmp_path)
    checked, removed = threading.Event(), threading.Event()
    real_ensure = starting._ensure_venv

    def ensure_then_wait_for_the_remove():
        real_ensure()
        if not checked.is_set():
            checked.set()
            assert removed.wait(10), "the test never removed the venv"

    real_create = local.venv.create
    creates = 0

    def counting_create(*args, **kwargs):
        nonlocal creates
        creates += 1
        return real_create(*args, **kwargs)

    monkeypatch.setattr(local.venv, "create", counting_create)
    monkeypatch.setattr(starting, "_ensure_venv", ensure_then_wait_for_the_remove)
    start = asyncio.create_task(starting.start())
    try:
        assert await asyncio.to_thread(checked.wait, 10)
        await remover.cleanup()
        assert not (tmp_path / "probe").exists()
        removed.set()
        await asyncio.wait_for(start, timeout=60)
        assert starting._proc is not None and starting._proc.returncode is None
        assert starting._verify_venv_python()
        assert creates == 1
    finally:
        removed.set()
        await asyncio.gather(start, return_exceptions=True)
        await starting.close()


@pytest.mark.skipif(
    sys.platform not in ("darwin", "win32"), reason="case-insensitive hosts only"
)
def test_case_variant_pad_names_share_one_venv_lock(tmp_path):
    # `Report` and `report` name one venv directory on macOS and Windows, so a
    # build under one name must wait for a build or delete under the other.
    assert local._venv_lock(venv_path=tmp_path / "Report") is local._venv_lock(
        venv_path=tmp_path / "report"
    )


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


async def test_a_stop_during_cleanup_still_discards_the_namespace_snapshot(
    tmp_path, monkeypatch
):
    # cleanup() deletes the venv on a worker thread. A Stop that lands during
    # the delete must still drop the saved namespace, or the next pad with
    # this name restores the variables of the pad that was removed.
    monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "true")
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: None))

    def session_pad():
        return LocalScratchpadRuntime(
            name="probe", session_id="conv1", _venvs_base=tmp_path / "venvs", **_DEFAULTS
        )

    pad = session_pad()
    cell = await pad.execute("secret = 'value-from-the-removed-pad'")
    assert cell.error is None, cell.error
    snapshot = pad._session_snapshot_path()
    assert snapshot is not None and snapshot.exists()
    deleting, release_delete = threading.Event(), threading.Event()
    real_rmtree = local.shutil.rmtree

    def gated_rmtree(*args, **kwargs):
        deleting.set()
        assert release_delete.wait(10), "the test never released the delete"
        return real_rmtree(*args, **kwargs)

    monkeypatch.setattr(local.shutil, "rmtree", gated_rmtree)
    cleanup = asyncio.create_task(pad.cleanup())
    try:
        assert await asyncio.to_thread(deleting.wait, 10)
        cleanup.cancel()
        release_delete.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(cleanup, timeout=10)
    finally:
        release_delete.set()
        await asyncio.gather(cleanup, return_exceptions=True)
        monkeypatch.setattr(local.shutil, "rmtree", real_rmtree)

    assert not (tmp_path / "venvs" / "probe").exists()
    assert not snapshot.exists(), "the cancelled cleanup kept the namespace snapshot"
    again = session_pad()
    try:
        cell = await again.execute("print(globals().get('secret', 'GONE'))")
        assert cell.stdout.strip() == "GONE"
    finally:
        await again.cleanup()
