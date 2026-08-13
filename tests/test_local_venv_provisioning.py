"""LocalScratchpadRuntime venv provisioning (mindshub#12484).

`_find_uv()` only checked a couple of hardcoded directories with no live PATH
search beyond `shutil.which`, so a uv installed via Homebrew/MacPorts/Linuxbrew
was invisible whenever the parent process's PATH didn't happen to include it
(e.g. cowork's Electron app launching cowork-server with a minimal PATH on
macOS). These tests pin the widened candidate list.
"""
from __future__ import annotations

import anton.core.backends.local as local


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
