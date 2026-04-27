"""Tests for the auto-updater."""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from anton import updater


@pytest.fixture
def updater_dirs(tmp_path, monkeypatch):
    """Redirect updater state files into a temp dir and stub external probes."""
    state_dir = tmp_path / ".anton"
    monkeypatch.setattr(updater, "_STATE_DIR", state_dir)
    monkeypatch.setattr(updater, "_STATE_FILE", state_dir / "updater_state.json")
    monkeypatch.setattr(updater, "_LOCK_FILE", state_dir / "updater.lock")
    monkeypatch.setattr(updater, "_LOG_FILE", state_dir / "updater.log")
    monkeypatch.setattr(updater, "_is_editable_dev_install", lambda: False)
    monkeypatch.setattr(updater, "_resolve_uv", lambda: "/usr/bin/uv")
    monkeypatch.delenv("_ANTON_UPDATED", raising=False)
    return state_dir


def _settings(disable_autoupdates=False):
    s = MagicMock()
    s.disable_autoupdates = disable_autoupdates
    return s


def _console():
    return MagicMock()


def _patch_urlopen(payload):
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return patch("anton.updater.urllib.request.urlopen", return_value=resp)


def _patch_fetch(tag):
    return patch.object(updater, "_fetch_latest_tag", return_value=tag)


def _patch_subprocess(returncode=0, stdout=b"", stderr=b""):
    completed = MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)
    return patch("anton.updater.subprocess.run", return_value=completed)


class TestIsEditableDevInstall:
    def test_detects_git_dir(self, tmp_path, monkeypatch):
        pkg = tmp_path / "anton"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (tmp_path / ".git").mkdir()
        monkeypatch.setattr(updater.anton, "__file__", str(pkg / "__init__.py"))
        assert updater._is_editable_dev_install() is True

    def test_detects_pyproject(self, tmp_path, monkeypatch):
        pkg = tmp_path / "anton"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (tmp_path / "pyproject.toml").write_text("")
        monkeypatch.setattr(updater.anton, "__file__", str(pkg / "__init__.py"))
        assert updater._is_editable_dev_install() is True

    def test_wheel_layout_is_not_editable(self, tmp_path, monkeypatch):
        pkg = tmp_path / "site-packages" / "anton"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        monkeypatch.setattr(updater.anton, "__file__", str(pkg / "__init__.py"))
        assert updater._is_editable_dev_install() is False

    def test_missing_file_attr_is_safe(self, monkeypatch):
        monkeypatch.setattr(updater.anton, "__file__", None)
        assert updater._is_editable_dev_install() is False


class TestState:
    def test_load_missing_file_returns_empty(self, updater_dirs):
        assert updater._load_state() == {}

    def test_save_then_load_roundtrip(self, updater_dirs):
        updater._save_state({"last_check_at": 123, "last_known_tag": "v1.2.3"})
        assert updater._load_state() == {"last_check_at": 123, "last_known_tag": "v1.2.3"}

    def test_load_malformed_returns_empty(self, updater_dirs):
        updater_dirs.mkdir(parents=True, exist_ok=True)
        (updater_dirs / "updater_state.json").write_text("not json {{{")
        assert updater._load_state() == {}

    def test_load_non_dict_returns_empty(self, updater_dirs):
        updater_dirs.mkdir(parents=True, exist_ok=True)
        (updater_dirs / "updater_state.json").write_text('["nope"]')
        assert updater._load_state() == {}

    def test_save_creates_parent_dir(self, tmp_path, monkeypatch):
        nested = tmp_path / "deeply" / "nested"
        monkeypatch.setattr(updater, "_STATE_DIR", nested)
        monkeypatch.setattr(updater, "_STATE_FILE", nested / "s.json")
        updater._save_state({"a": 1})
        assert (nested / "s.json").exists()


class TestFetchLatestTag:
    def test_returns_tag_on_success(self):
        with _patch_urlopen({"tag_name": "v2.0.5"}):
            assert updater._fetch_latest_tag(deadline=2.0) == "v2.0.5"

    def test_strips_whitespace(self):
        with _patch_urlopen({"tag_name": "  v2.0.5\n"}):
            assert updater._fetch_latest_tag(deadline=2.0) == "v2.0.5"

    def test_missing_tag_field_returns_empty(self):
        with _patch_urlopen({"name": "no tag here"}):
            assert updater._fetch_latest_tag(deadline=2.0) == ""

    def test_network_error_returns_empty(self):
        with patch("anton.updater.urllib.request.urlopen", side_effect=OSError("dns")):
            assert updater._fetch_latest_tag(deadline=2.0) == ""

    def test_malformed_json_returns_empty(self):
        resp = MagicMock()
        resp.read.return_value = b"not json"
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        with patch("anton.updater.urllib.request.urlopen", return_value=resp):
            assert updater._fetch_latest_tag(deadline=2.0) == ""

    def test_deadline_respected_when_worker_hangs(self):
        proceed = threading.Event()

        def hang(*args, **kwargs):
            proceed.wait(timeout=10)

        start = time.monotonic()
        with patch("anton.updater.urllib.request.urlopen", side_effect=hang):
            result = updater._fetch_latest_tag(deadline=0.2)
            proceed.set()
        elapsed = time.monotonic() - start
        assert result == ""
        assert elapsed < 1.5


class TestInstallLock:
    def test_fresh_lock_acquired_and_released(self, updater_dirs):
        with updater._install_lock() as got:
            assert got is True
            assert (updater_dirs / "updater.lock").exists()
        assert not (updater_dirs / "updater.lock").exists()

    def test_busy_lock_not_acquired(self, updater_dirs):
        updater_dirs.mkdir(parents=True, exist_ok=True)
        lock = updater_dirs / "updater.lock"
        lock.write_text("9999 1\n")
        with updater._install_lock() as got:
            assert got is False
        assert lock.exists()

    def test_stale_lock_reclaimed(self, updater_dirs):
        updater_dirs.mkdir(parents=True, exist_ok=True)
        lock = updater_dirs / "updater.lock"
        lock.write_text("9999 1\n")
        old = time.time() - updater._LOCK_STALE_AFTER - 60
        os.utime(lock, (old, old))
        with updater._install_lock() as got:
            assert got is True
        assert not lock.exists()

    def test_lock_released_on_exception(self, updater_dirs):
        with pytest.raises(RuntimeError):
            with updater._install_lock() as got:
                assert got is True
                raise RuntimeError("boom")
        assert not (updater_dirs / "updater.lock").exists()


class TestCheckAndUpdate:
    def test_disabled_setting_skips(self, updater_dirs):
        assert updater.check_and_update(_console(), _settings(disable_autoupdates=True)) is False

    def test_env_var_skips(self, updater_dirs, monkeypatch):
        monkeypatch.setenv("_ANTON_UPDATED", "1")
        assert updater.check_and_update(_console(), _settings()) is False

    def test_editable_install_skips(self, updater_dirs, monkeypatch):
        monkeypatch.setattr(updater, "_is_editable_dev_install", lambda: True)
        assert updater.check_and_update(_console(), _settings()) is False

    def test_no_uv_skips(self, updater_dirs, monkeypatch):
        monkeypatch.setattr(updater, "_resolve_uv", lambda: None)
        assert updater.check_and_update(_console(), _settings()) is False

    def test_no_remote_tag_skips(self, updater_dirs):
        with _patch_fetch(""):
            assert updater.check_and_update(_console(), _settings()) is False

    def test_already_up_to_date_skips(self, updater_dirs, monkeypatch):
        monkeypatch.setattr(updater.anton, "__version__", "9.9.9")
        with _patch_fetch("v9.9.9"), _patch_subprocess() as run:
            assert updater.check_and_update(_console(), _settings()) is False
            run.assert_not_called()

    def test_remote_older_skips(self, updater_dirs, monkeypatch):
        monkeypatch.setattr(updater.anton, "__version__", "9.9.9")
        with _patch_fetch("v1.0.0"), _patch_subprocess() as run:
            assert updater.check_and_update(_console(), _settings()) is False
            run.assert_not_called()

    def test_prerelease_remote_skips(self, updater_dirs, monkeypatch):
        monkeypatch.setattr(updater.anton, "__version__", "1.0.0")
        with _patch_fetch("v2.0.0a1"), _patch_subprocess() as run:
            assert updater.check_and_update(_console(), _settings()) is False
            run.assert_not_called()

    def test_invalid_remote_version_skips(self, updater_dirs, monkeypatch):
        monkeypatch.setattr(updater.anton, "__version__", "1.0.0")
        with _patch_fetch("not-a-version"), _patch_subprocess() as run:
            assert updater.check_and_update(_console(), _settings()) is False
            run.assert_not_called()

    def test_happy_path_installs_and_returns_true(self, updater_dirs, monkeypatch):
        monkeypatch.setattr(updater.anton, "__version__", "1.0.0")
        with _patch_fetch("v2.0.0"), _patch_subprocess(returncode=0) as run:
            assert updater.check_and_update(_console(), _settings()) is True
            run.assert_called_once()
            args = run.call_args[0][0]
            assert args[0] == "/usr/bin/uv"
            assert args[1:3] == ["tool", "install"]
            assert "git+https://github.com/mindsdb/anton.git@v2.0.0" in args
            assert "--force" in args

    def test_install_failure_returns_false_and_logs(self, updater_dirs, monkeypatch):
        monkeypatch.setattr(updater.anton, "__version__", "1.0.0")
        with _patch_fetch("v2.0.0"), _patch_subprocess(returncode=1, stderr=b"oh no"):
            assert updater.check_and_update(_console(), _settings()) is False
        log_path = updater_dirs / "updater.log"
        assert log_path.exists()
        assert "oh no" in log_path.read_text()

    def test_install_timeout_returns_false(self, updater_dirs, monkeypatch):
        monkeypatch.setattr(updater.anton, "__version__", "1.0.0")
        with _patch_fetch("v2.0.0"), patch(
            "anton.updater.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="uv", timeout=1),
        ):
            assert updater.check_and_update(_console(), _settings()) is False

    def test_throttle_skips_api_call(self, updater_dirs, monkeypatch):
        monkeypatch.setattr(updater.anton, "__version__", "1.0.0")
        updater._save_state({"last_check_at": time.time(), "last_known_tag": "v1.0.0"})
        with patch.object(updater, "_fetch_latest_tag") as fetch:
            assert updater.check_and_update(_console(), _settings()) is False
            fetch.assert_not_called()

    def test_throttle_uses_cached_tag_to_trigger_install(self, updater_dirs, monkeypatch):
        monkeypatch.setattr(updater.anton, "__version__", "1.0.0")
        updater._save_state({"last_check_at": time.time(), "last_known_tag": "v2.0.0"})
        with patch.object(updater, "_fetch_latest_tag") as fetch, _patch_subprocess() as run:
            assert updater.check_and_update(_console(), _settings()) is True
            fetch.assert_not_called()
            run.assert_called_once()

    def test_expired_throttle_calls_api_and_persists_tag(self, updater_dirs, monkeypatch):
        monkeypatch.setattr(updater.anton, "__version__", "1.0.0")
        updater._save_state({
            "last_check_at": time.time() - updater._CHECK_TTL - 60,
            "last_known_tag": "v1.0.0",
        })
        with _patch_fetch("v2.0.0"), _patch_subprocess() as run:
            assert updater.check_and_update(_console(), _settings()) is True
            run.assert_called_once()
        assert updater._load_state()["last_known_tag"] == "v2.0.0"

    def test_api_failure_keeps_old_throttle_stamp(self, updater_dirs, monkeypatch):
        monkeypatch.setattr(updater.anton, "__version__", "1.0.0")
        updater._save_state({"last_check_at": 1.0, "last_known_tag": "v1.0.0"})
        with _patch_fetch(""), _patch_subprocess() as run:
            assert updater.check_and_update(_console(), _settings()) is False
            run.assert_not_called()
        # Stamp NOT updated because the fetch failed; next start will retry.
        assert updater._load_state()["last_check_at"] == 1.0

    def test_lock_contention_skips_install(self, updater_dirs, monkeypatch):
        monkeypatch.setattr(updater.anton, "__version__", "1.0.0")
        updater_dirs.mkdir(parents=True, exist_ok=True)
        (updater_dirs / "updater.lock").write_text("9999 1\n")
        with _patch_fetch("v2.0.0"), _patch_subprocess() as run:
            assert updater.check_and_update(_console(), _settings()) is False
            run.assert_not_called()
