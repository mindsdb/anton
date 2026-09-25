"""LocalScratchpadRuntime venv provisioning.

`_find_uv()` only checked a couple of hardcoded directories with no live PATH
search beyond `shutil.which`, so a uv installed via Homebrew/MacPorts/Linuxbrew
was invisible whenever the parent process's PATH didn't happen to include it
(e.g. cowork's Electron app launching cowork-server with a minimal PATH on
macOS). These tests pin the widened candidate list.
"""
from __future__ import annotations

import os
import sys
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


def test_ensure_venv_failure_message_includes_the_verify_detail(tmp_path, monkeypatch):
    pad = make_pad(tmp_path)

    def fake_verify():
        pad._last_verify_error = "exit 1: dyld: Library not loaded"
        return False

    monkeypatch.setattr(pad, "_create_venv", lambda **_: None)
    monkeypatch.setattr(pad, "_verify_venv_python", fake_verify)

    with pytest.raises(RuntimeError, match="dyld: Library not loaded"):
        pad._ensure_venv()


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


# --- ENG-1646: uv sometimes writes bin/python as a copied, merely ad-hoc
# signed launcher (missing its own libpythonX.Y.dylib) instead of a symlink
# to the base interpreter. macOS's AMFI then refuses to execute it. A plain
# symlink to the same interpreter has proven reliable everywhere we've seen
# this fail, so `_create_venv()` now repairs the copy into a symlink, and
# `_ensure_venv()`'s last retry escalates to the stdlib fallback instead of
# repeating the same doomed `uv venv` call a third time.


def _make_uv_venv_layout(tmp_path, venv_dir, *, python_is_symlink, real_interpreter=None):
    """Lay out a bin/ dir the way `uv venv` would, with bin/python either a
    symlink (the healthy/expected form) or a plain copied file (the observed
    broken form)."""
    bin_dir = os.path.join(venv_dir, "bin")
    os.makedirs(bin_dir, exist_ok=True)
    py = os.path.join(bin_dir, "python")
    if python_is_symlink:
        os.symlink(real_interpreter or sys.executable, py)
    else:
        with open(py, "wb") as f:
            f.write(b"\xfa\xde\x0c\xfe fake copied launcher, not a real Mach-O")
        os.chmod(py, 0o755)
    return bin_dir


def test_repair_copied_launcher_replaces_a_copy_with_a_symlink(tmp_path):
    if sys.platform == "win32":
        pytest.skip("posix-only: symlink semantics differ on Windows")
    pad = make_pad(tmp_path)
    bin_dir = _make_uv_venv_layout(tmp_path, str(tmp_path / "v"), python_is_symlink=False)

    pad._repair_copied_launcher(bin_dir)

    py = os.path.join(bin_dir, "python")
    assert os.path.islink(py)
    assert os.path.realpath(py) == os.path.realpath(sys.executable)


def test_repair_copied_launcher_leaves_an_existing_symlink_alone(tmp_path):
    if sys.platform == "win32":
        pytest.skip("posix-only: symlink semantics differ on Windows")
    pad = make_pad(tmp_path)
    bin_dir = _make_uv_venv_layout(
        tmp_path, str(tmp_path / "v"), python_is_symlink=True, real_interpreter="/some/other/python"
    )

    pad._repair_copied_launcher(bin_dir)

    # Untouched — still points at whatever it originally symlinked to, not
    # silently repointed at this process's own interpreter.
    assert os.readlink(os.path.join(bin_dir, "python")) == "/some/other/python"


def test_create_venv_repairs_a_copied_launcher_when_uv_is_used(tmp_path, monkeypatch):
    if sys.platform == "win32":
        pytest.skip("posix-only: symlink semantics differ on Windows")
    import subprocess

    pad = make_pad(tmp_path)
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: "/fake/uv"))

    # _create_venv sets self._venv_dir = self._venvs_base / self.name before
    # invoking uv, so build the venv dir path the same way it does.
    venv_dir = str(tmp_path / pad.name)

    def fake_run(args, **kwargs):
        # Simulate `uv venv` "succeeding" but leaving bin/python as a copy,
        # matching what was actually observed via macOS's AMFI/XProtect logs.
        _make_uv_venv_layout(tmp_path, venv_dir, python_is_symlink=False)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    pad._create_venv()

    assert os.path.islink(pad._venv_python)
    assert os.path.realpath(pad._venv_python) == os.path.realpath(sys.executable)


def test_ensure_venv_escalates_to_stdlib_on_the_last_attempt_even_with_uv(tmp_path, monkeypatch):
    # The failure this fix targets is deterministic per environment, not
    # transient — repeating the identical `uv venv` call 3 times bought
    # nothing. The last attempt must actually try something different.
    pad = make_pad(tmp_path)
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: "/fake/uv"))
    calls = []

    def fake_create(*, force_stdlib=False):
        calls.append(force_stdlib)
        pad._venv_python = str(tmp_path / "never-verifies")

    monkeypatch.setattr(pad, "_create_venv", fake_create)
    monkeypatch.setattr(pad, "_verify_venv_python", lambda: False)

    with pytest.raises(RuntimeError):
        pad._ensure_venv()

    assert calls == [False, False, True]


def test_ensure_venv_preserves_the_venv_dir_after_the_final_failure(tmp_path, monkeypatch):
    # Nuking the directory after every attempt, including the last, erased
    # the only evidence of what actually went wrong before anyone could look.
    pad = make_pad(tmp_path)

    def fake_create(*, force_stdlib=False):
        pad._venv_dir = str(tmp_path / pad.name)
        os.makedirs(pad._venv_dir, exist_ok=True)
        pad._venv_python = str(tmp_path / "never-verifies")

    monkeypatch.setattr(pad, "_create_venv", fake_create)
    monkeypatch.setattr(pad, "_verify_venv_python", lambda: False)

    with pytest.raises(RuntimeError):
        pad._ensure_venv()

    assert os.path.isdir(pad._venv_dir)


def test_diagnose_broken_interpreter_reports_symlink_shape(tmp_path):
    if sys.platform == "win32":
        pytest.skip("posix-only: symlink semantics differ on Windows")
    pad = make_pad(tmp_path)
    copied = tmp_path / "copied_binary"
    copied.write_bytes(b"not a real interpreter, just needs to exist")
    pad._venv_python = str(copied)

    detail = pad._diagnose_broken_interpreter()

    assert "is_symlink=False" in detail
