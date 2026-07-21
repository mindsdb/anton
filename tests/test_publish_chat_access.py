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
        await chat._handle_publish(Console(), settings, mock.Mock(), file_arg="sales/report.html")

    _, kwargs = fake_publish.call_args
    assert kwargs["access"] == {"mode": "password", "password": "hunter2"}

    # owner-side persisted next to the primary, keyed by file name
    published = json.loads((root / "sales" / ".published.json").read_text())
    assert published["report.html"]["mode"] == "password"
    assert published["report.html"]["access_password"] == "hunter2"
