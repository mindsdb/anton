"""publish_or_preview tool: access fields + preserve-previous default."""

import json
from pathlib import Path
from unittest import mock

import pytest
from rich.console import Console

import anton.tools as tools


def _artifact(tmp_path: Path) -> Path:
    root = tmp_path / "artifacts"
    art = root / "sales"
    art.mkdir(parents=True)
    (art / "metadata.json").write_text('{"type": "html-report", "primary": "report.html"}')
    f = art / "report.html"
    f.write_text("<html></html>")
    return f


def _fullstack_artifact(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "artifacts"
    art = root / "server-time-display-2"
    (art / "static").mkdir(parents=True)
    (art / "metadata.json").write_text(
        '{"type": "fullstack-stateless-app", "primary": "static/index.html"}'
    )
    (art / "backend.py").write_text("print(1)")
    (art / "static" / "index.html").write_text("<html></html>")
    return root, art


def _session(tmp_path):
    s = mock.Mock()
    s._console = Console()
    ws = mock.Mock()
    ws.base = str(tmp_path)
    s._workspace = ws
    return s


def _settings(root):
    s = mock.Mock()
    s.minds_api_key = "key"
    s.publish_url = "https://view.test"
    s.minds_ssl_verify = True
    s.artifacts_dir = str(root)
    return s


@pytest.mark.asyncio
async def test_tool_explicit_password(tmp_path):
    f = _artifact(tmp_path)
    fake_publish = mock.Mock(return_value={"view_url": "u", "report_id": "r", "md5": "m", "version": 1})
    with mock.patch("anton.publisher.publish", fake_publish), \
         mock.patch("anton.config.settings.AntonSettings", return_value=_settings(f.parent.parent)), \
         mock.patch("webbrowser.open"):
        out = await tools.handle_publish_or_preview(
            _session(tmp_path),
            {"file_path": str(f), "action": "publish",
             "access_mode": "password", "password": "hunter2"},
        )
    _, kwargs = fake_publish.call_args
    assert kwargs["access"] == {"mode": "password", "password": "hunter2"}
    assert "Published" in out


@pytest.mark.asyncio
async def test_tool_preserves_previous_when_no_fields(tmp_path):
    f = _artifact(tmp_path)
    (f.parent / ".published.json").write_text(json.dumps({
        "report.html": {"report_id": "r", "url": "u", "last_md5": "m",
                          "mode": "password", "requires_password": True,
                          "access_password": "old", "pwd_version": 1},
    }))
    fake_publish = mock.Mock(return_value={"view_url": "u", "report_id": "r", "md5": "m2", "version": 2})
    with mock.patch("anton.publisher.publish", fake_publish), \
         mock.patch("anton.config.settings.AntonSettings", return_value=_settings(f.parent.parent)), \
         mock.patch("webbrowser.open"):
        await tools.handle_publish_or_preview(
            _session(tmp_path), {"file_path": str(f), "action": "publish"},
        )
    _, kwargs = fake_publish.call_args
    assert kwargs["access"] == {"mode": "password", "password": "old"}  # NOT public


@pytest.mark.asyncio
async def test_tool_password_no_value_non_tty_cancels(tmp_path):
    f = _artifact(tmp_path)
    fake_publish = mock.Mock()
    with mock.patch("anton.publisher.publish", fake_publish), \
         mock.patch("anton.config.settings.AntonSettings", return_value=_settings(f.parent.parent)), \
         mock.patch("sys.stdin") as stdin:
        stdin.isatty.return_value = False
        out = await tools.handle_publish_or_preview(
            _session(tmp_path),
            {"file_path": str(f), "action": "publish", "access_mode": "password"},
        )
    assert "CANCELLED" in out
    fake_publish.assert_not_called()


@pytest.mark.asyncio
async def test_tool_fullstack_publishes_folder(tmp_path):
    """Given the artifact folder, publish() receives the folder (fullstack bundle)."""
    root, art = _fullstack_artifact(tmp_path)
    fake_publish = mock.Mock(return_value={"view_url": "u", "report_id": "r", "md5": "m", "version": 1})
    with mock.patch("anton.publisher.publish", fake_publish), \
         mock.patch("anton.config.settings.AntonSettings", return_value=_settings(root)), \
         mock.patch("webbrowser.open"):
        await tools.handle_publish_or_preview(
            _session(tmp_path), {"file_path": str(art), "action": "publish"},
        )
    args, _ = fake_publish.call_args
    assert Path(args[0]) == art  # the folder, not an inner file


@pytest.mark.asyncio
async def test_tool_fullstack_from_inner_file_publishes_folder(tmp_path):
    """Even if the model points at an inner HTML file, publish() gets the folder."""
    root, art = _fullstack_artifact(tmp_path)
    fake_publish = mock.Mock(return_value={"view_url": "u", "report_id": "r", "md5": "m", "version": 1})
    with mock.patch("anton.publisher.publish", fake_publish), \
         mock.patch("anton.config.settings.AntonSettings", return_value=_settings(root)), \
         mock.patch("webbrowser.open"):
        await tools.handle_publish_or_preview(
            _session(tmp_path),
            {"file_path": str(art / "static" / "index.html"), "action": "publish"},
        )
    args, _ = fake_publish.call_args
    assert Path(args[0]) == art  # normalized up to the fullstack folder
