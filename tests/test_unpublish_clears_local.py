"""/unpublish must drop the local .published.json entry so a later /publish
does not offer to 'update' a report that no longer exists."""

import json
from pathlib import Path
from unittest import mock

import pytest
from rich.console import Console

import anton.chat as chat
from anton.publisher import _STATE_SNAPSHOT


def _make_published_artifact(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "artifacts"
    art = root / "server-time-display-2"
    art.mkdir(parents=True)
    (art / "report.html").write_text("<html></html>")
    pub = art / ".published.json"
    pub.write_text(json.dumps({
        "report.html": {
            "report_id": "247c4983", "url": "https://view/a/247c4983",
            "last_md5": "m", "mode": "password", "requires_password": True,
            "access_password": "s3cret", "pwd_version": 1,
        },
    }))
    return root, pub


@pytest.mark.asyncio
async def test_unpublish_removes_local_entry(tmp_path):
    root, pub = _make_published_artifact(tmp_path)

    settings = mock.Mock()
    settings.minds_api_key = "key"
    settings.artifacts_dir = str(root)
    settings.publish_url = "https://view.test"
    settings.minds_ssl_verify = True

    reports = [{"title": "server-time-display-2", "report_id": "247c4983",
                "md5": "m", "view_url": "https://view/a/247c4983"}]

    with mock.patch("anton.publisher.list_published", return_value=reports), \
         mock.patch("anton.publisher.unpublish", return_value={}), \
         mock.patch("anton.chat.prompt_or_cancel",
                    new=mock.AsyncMock(side_effect=["1", "y"])):
        await chat._handle_unpublish(Console(), settings, mock.Mock())

    # The stale entry must be gone so /publish treats it as fresh.
    data = json.loads(pub.read_text())
    assert "report.html" not in data


@pytest.mark.asyncio
async def test_unpublish_removes_state_snapshot(tmp_path):
    root, pub = _make_published_artifact(tmp_path)
    snap = pub.parent / _STATE_SNAPSHOT
    snap.write_text(json.dumps({"collections": ["comments"]}), encoding="utf-8")

    settings = mock.Mock()
    settings.minds_api_key = "key"
    settings.artifacts_dir = str(root)
    settings.publish_url = "https://view.test"
    settings.minds_ssl_verify = True

    reports = [{"title": "server-time-display-2", "report_id": "247c4983",
                "md5": "m", "view_url": "https://view/a/247c4983"}]

    with mock.patch("anton.publisher.list_published", return_value=reports), \
         mock.patch("anton.publisher.unpublish", return_value={}), \
         mock.patch("anton.chat.prompt_or_cancel",
                    new=mock.AsyncMock(side_effect=["1", "y"])):
        await chat._handle_unpublish(Console(), settings, mock.Mock())

    assert not snap.exists()  # snapshot cleared → next publish is treated as fresh
