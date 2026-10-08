"""_handle_publish passes the selected access through to publisher.publish."""

import json
from pathlib import Path
from unittest import mock

import pytest
from rich.console import Console

import anton.chat as chat


def _make_artifact(tmp_path: Path) -> Path:
    root = tmp_path / "artifacts"
    art = root / "sales"
    art.mkdir(parents=True)
    (art / "metadata.json").write_text(json.dumps({
        "schemaVersion": 1,
        "id": "abcd1234",
        "slug": "sales",
        "createdAt": "2026-01-01T00:00:00Z",
        "updatedAt": "2026-01-01T00:00:00Z",
        "name": "Sales",
        "description": "Sales report",
        "type": "html-app",
        "primary": "report.html",
    }))
    (art / "report.html").write_text("<html><title>Sales</title></html>")
    return root


@pytest.mark.asyncio
async def test_publish_passes_password_access(tmp_path):
    root = _make_artifact(tmp_path)

    settings = mock.Mock()
    settings.minds_api_key = "key"
    settings.artifacts_dir = str(root)
    settings.workspace_path = str(tmp_path)
    settings.publish_url = "https://view.test"
    settings.minds_ssl_verify = True

    fake_publish = mock.Mock(return_value={
        "view_url": "https://view.test/r/abc", "report_id": "abc",
        "md5": "m", "version": 1,
    })

    async def fake_prompt_access(*a, **k):
        return {"mode": "password", "password": "hunter2"}

    # publish and prompt_access are imported inside _handle_publish at the
    # function level, so we patch the SOURCE modules — a function-level import
    # resolves the name at call time and picks up the patch.
    with mock.patch("anton.publisher.publish", fake_publish), \
         mock.patch("anton.publish_access.prompt_access", side_effect=fake_prompt_access), \
         mock.patch("webbrowser.open"):
        # _make_candidate only publishes a directory for fullstack artifacts;
        # for an html-report we address the file (the file branch of _make_candidate).
        await chat._handle_publish(Console(), settings, file_arg="sales/report.html")

    _, kwargs = fake_publish.call_args
    assert kwargs["access"] == {"mode": "password", "password": "hunter2"}

    # owner-side persisted next to the primary, keyed by file name
    published = json.loads((root / "sales" / ".published.json").read_text())
    assert published["report.html"]["mode"] == "password"
    assert published["report.html"]["access_password"] == "hunter2"


# ---------------------------------------------------------------------------
# Password hash kept owner-side, and the 409 for another owner
# ---------------------------------------------------------------------------

_STORED_HASH = "pbkdf2_sha256$200000$c2FsdHNhbHRzYWx0c2FsdA==$ZGVyaXZlZGtleQ=="


def _settings(root: Path, tmp_path: Path):
    settings = mock.Mock()
    settings.minds_api_key = "key"
    settings.artifacts_dir = str(root)
    settings.workspace_path = str(tmp_path)
    settings.publish_url = "https://view.test"
    settings.minds_ssl_verify = True
    return settings


def _write_entry(root: Path, entry: dict) -> Path:
    published = root / "sales" / ".published.json"
    published.write_text(json.dumps({"report.html": entry}))
    return published


def _password_entry() -> dict:
    return {
        "mode": "password", "requires_password": True,
        "access_password": "hunter2", "pwd_version": 1, "password_hash": _STORED_HASH,
        "report_id": "abc", "url": "https://view.test/r/abc", "last_md5": "m",
    }


async def _publish_update(root, tmp_path, fake_publish, access, console=None):
    async def fake_prompt_access(*a, **k):
        return access

    settings = _settings(root, tmp_path)
    with mock.patch("anton.publisher.publish", fake_publish), \
         mock.patch("anton.publish_access.prompt_access", side_effect=fake_prompt_access), \
         mock.patch("anton.chat.prompt_or_cancel", new=mock.AsyncMock(return_value="update")), \
         mock.patch("webbrowser.open"):
        await chat._handle_publish(console or Console(), settings, file_arg="sales/report.html")
    return settings


@pytest.mark.asyncio
async def test_publish_hands_back_the_previous_entry_and_keeps_the_sent_hash(tmp_path):
    root = _make_artifact(tmp_path)
    _write_entry(root, _password_entry())
    fake_publish = mock.Mock(return_value={
        "view_url": "https://view.test/r/abc", "report_id": "abc", "md5": "m2", "version": 2,
        "password_hash": _STORED_HASH,
    })

    await _publish_update(root, tmp_path, fake_publish, {"mode": "password", "password": "hunter2"})

    _, kwargs = fake_publish.call_args
    assert kwargs["report_id"] == "abc"
    assert kwargs["previous_access"]["password_hash"] == _STORED_HASH
    entry = json.loads((root / "sales" / ".published.json").read_text())["report.html"]
    assert entry["password_hash"] == _STORED_HASH
    assert entry["access_password"] == "hunter2"


@pytest.mark.asyncio
async def test_switching_to_public_drops_the_stored_hash(tmp_path):
    root = _make_artifact(tmp_path)
    _write_entry(root, _password_entry())
    fake_publish = mock.Mock(return_value={
        "view_url": "https://view.test/r/abc", "report_id": "abc", "md5": "m2", "version": 2,
    })

    await _publish_update(root, tmp_path, fake_publish, {"mode": "public"})

    entry = json.loads((root / "sales" / ".published.json").read_text())["report.html"]
    assert entry["mode"] == "public"
    assert "password_hash" not in entry
    assert "access_password" not in entry


@pytest.mark.asyncio
async def test_publish_reports_an_artifact_owned_by_another_account(tmp_path):
    from anton.publisher import ArtifactOwnedByOtherUserError

    root = _make_artifact(tmp_path)
    published = _write_entry(root, {
        "mode": "public", "requires_password": False,
        "report_id": "abc", "url": "https://view.test/r/abc", "last_md5": "m",
    })
    before = published.read_text()
    fake_publish = mock.Mock(side_effect=ArtifactOwnedByOtherUserError())
    console = Console(record=True, width=200)

    settings = await _publish_update(root, tmp_path, fake_publish, {"mode": "public"}, console=console)

    assert fake_publish.call_count == 1
    text = console.export_text()
    assert "This artifact was published by another owner." in text
    assert "the existing link belongs to another account" in text
    assert "choose 'new'" in text
    assert published.read_text() == before
    # Not an auth failure: the key stays in place.
    assert settings.minds_api_key == "key"
