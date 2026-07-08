"""Anti-thrash guard for the CLI self-updater (ENG-655).

When a release tag's installed version never matches the tag (version drift),
the updater must NOT re-run `uv tool install --force` for that same tag on every
launch — that repeated reinstall of the running tool corrupts the env (Windows).
"""
from __future__ import annotations

from packaging.version import Version

import anton.updater as updater


class _FailProc:
    # returncode != 0 → updater stops right after the install attempt, so tests
    # that only need to observe "did we attempt?" don't have to stub `uv tool list`.
    returncode = 1
    stdout = b""


def _stub_env(monkeypatch, tmp_path, *, latest_tag, local_version, proc=None):
    """Stub network/uv so only the guard logic runs; record install calls."""
    calls = {"install": 0}
    result_proc = proc or _FailProc()

    monkeypatch.setattr(updater.shutil, "which", lambda _: "/usr/bin/uv")
    monkeypatch.setattr(updater, "_fetch_latest_release_tag", lambda: latest_tag)
    monkeypatch.setattr(updater, "_SKIP_MARKER", tmp_path / ".update_skip_tag")

    import anton
    monkeypatch.setattr(anton, "__version__", local_version)

    def _rec_run(*a, **k):
        calls["install"] += 1
        return result_proc

    monkeypatch.setattr(updater.subprocess, "run", _rec_run)
    return calls


def test_marked_tag_is_not_reinstalled(monkeypatch, tmp_path):
    # A newer tag exists (would normally trigger an update)...
    calls = _stub_env(monkeypatch, tmp_path, latest_tag="v9.9.9.9.9", local_version="2.0.0")
    # ...but we already recorded it as a failed attempt.
    (tmp_path / ".update_skip_tag").write_text("v9.9.9.9.9")

    result: dict = {}
    updater._check_and_update(result, settings=None)

    assert calls["install"] == 0  # guard short-circuited before the reinstall
    assert not any("Updating" in m for m in result.get("messages", []))


def test_different_tag_is_not_suppressed(monkeypatch, tmp_path):
    # Marker holds an OLD tag; a genuinely newer tag must still be attempted.
    calls = _stub_env(monkeypatch, tmp_path, latest_tag="v9.9.9.9.9", local_version="2.0.0")
    (tmp_path / ".update_skip_tag").write_text("v1.1.1.1.1")

    result: dict = {}
    updater._check_and_update(result, settings=None)

    assert calls["install"] == 1  # newer tag → install attempted


def test_skip_marker_written_on_version_mismatch(monkeypatch, tmp_path):
    # Install "succeeds" but the verified version still diverges from the tag →
    # record the tag so the next launch doesn't reinstall it again.
    class _OkProc:
        returncode = 0
        stdout = b""

    _stub_env(monkeypatch, tmp_path, latest_tag="v9.9.9.9.9", local_version="2.0.0", proc=_OkProc())
    # Verified installed version != remote tag (the drift case).
    monkeypatch.setattr(updater, "_read_installed_anton_version", lambda: Version("2.0.0"))

    result: dict = {}
    updater._check_and_update(result, settings=None)

    marker = tmp_path / ".update_skip_tag"
    assert marker.is_file() and marker.read_text().strip() == "v9.9.9.9.9"
    assert "new_version" not in result
