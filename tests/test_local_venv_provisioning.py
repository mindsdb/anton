"""LocalScratchpadRuntime venv provisioning (mindshub#12484).

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
