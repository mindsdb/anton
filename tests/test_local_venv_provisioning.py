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


def test_find_uv_checks_homebrew_apple_silicon(monkeypatch):
    monkeypatch.setattr(local.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        local.os.path, "isfile", lambda p: p == "/opt/homebrew/bin/uv"
    )
    monkeypatch.setattr(local.os, "access", lambda p, mode: True)

    assert local.LocalScratchpadRuntime._find_uv() == "/opt/homebrew/bin/uv"


def test_find_uv_checks_homebrew_intel(monkeypatch):
    monkeypatch.setattr(local.shutil, "which", lambda _: None)
    monkeypatch.setattr(local.os.path, "isfile", lambda p: p == "/usr/local/bin/uv")
    monkeypatch.setattr(local.os, "access", lambda p, mode: True)

    assert local.LocalScratchpadRuntime._find_uv() == "/usr/local/bin/uv"


def test_find_uv_checks_macports(monkeypatch):
    monkeypatch.setattr(local.shutil, "which", lambda _: None)
    monkeypatch.setattr(local.os.path, "isfile", lambda p: p == "/opt/local/bin/uv")
    monkeypatch.setattr(local.os, "access", lambda p, mode: True)

    assert local.LocalScratchpadRuntime._find_uv() == "/opt/local/bin/uv"


def test_find_uv_checks_linuxbrew(monkeypatch):
    monkeypatch.setattr(local.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        local.os.path,
        "isfile",
        lambda p: p == "/home/linuxbrew/.linuxbrew/bin/uv",
    )
    monkeypatch.setattr(local.os, "access", lambda p, mode: True)

    assert local.LocalScratchpadRuntime._find_uv() == "/home/linuxbrew/.linuxbrew/bin/uv"


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


def test_verify_clears_the_error_on_success(tmp_path, monkeypatch):
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


def test_verify_captures_an_exception_reason(tmp_path, monkeypatch):
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

    monkeypatch.setattr(pad, "_create_venv", lambda: None)
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
