from __future__ import annotations

import io
import urllib.request

import anton
from anton import updater
from rich.console import Console


class _ImmediateThread:
    def __init__(self, *, target=None, daemon=None):
        self._target = target

    def start(self) -> None:
        if self._target is not None:
            self._target()

    def join(self, timeout=None) -> None:
        return None


def test_check_and_update_returns_true_when_update_applied(monkeypatch):
    monkeypatch.setattr(updater.threading, "Thread", _ImmediateThread)

    def fake_check(result, settings, *, deadline) -> None:
        result["messages"] = []
        result["new_version"] = "9.9.9"

    old_version = anton.__version__
    monkeypatch.setattr(updater, "_check_and_update", fake_check)
    console = Console(file=io.StringIO(), force_terminal=False, color_system=None)
    settings = type("Settings", (), {"disable_autoupdates": False})()

    try:
        assert updater.check_and_update(console, settings) is True
        assert anton.__version__ == "9.9.9"
    finally:
        anton.__version__ = old_version


def test_check_and_update_returns_false_when_autoupdates_disabled(monkeypatch):
    called = False

    def fake_check(result, settings, *, deadline) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(updater, "_check_and_update", fake_check)
    console = Console(file=io.StringIO(), force_terminal=False, color_system=None)
    settings = type("Settings", (), {"disable_autoupdates": True})()

    assert updater.check_and_update(console, settings) is False
    assert called is False


def test_check_and_update_uses_remaining_budget_for_upgrade(monkeypatch):
    monkeypatch.setattr(updater.shutil, "which", lambda name: "/usr/bin/uv")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b'__version__ = "9.9.9"'

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: _Response())

    captured: dict[str, float] = {}

    class _Proc:
        returncode = 0

    def fake_run(cmd, capture_output, timeout):
        captured["timeout"] = timeout
        return _Proc()

    monkeypatch.setattr(updater.subprocess, "run", fake_run)

    old_version = anton.__version__
    monkeypatch.setattr(anton, "__version__", "0.6.9")
    try:
        result: dict = {}
        deadline = updater.time.monotonic() + 3.5
        updater._check_and_update(result, settings=object(), deadline=deadline)
    finally:
        anton.__version__ = old_version

    assert 0 < captured["timeout"] <= 3.5
    assert result["new_version"] == "9.9.9"
