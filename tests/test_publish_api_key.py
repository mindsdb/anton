"""Tests for /publish API key handling — specifically the 401 fix.

Covers:
- Bad key entered on first /publish: 401 clears the key so user is re-prompted
- Good key: persisted to workspace only after successful publish
"""
from __future__ import annotations

import urllib.error
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_settings(tmp_path: Path, api_key: str | None = None) -> MagicMock:
    settings = MagicMock()
    settings.minds_api_key = api_key
    settings.workspace_path = tmp_path
    settings.publish_url = "https://4nton.ai"
    settings.minds_ssl_verify = True
    return settings


def _make_workspace() -> MagicMock:
    ws = MagicMock()
    ws.set_secret = MagicMock()
    return ws


def _make_console() -> MagicMock:
    console = MagicMock()
    console.print = MagicMock()
    return console


def _make_html_file(tmp_path: Path) -> Path:
    output_dir = tmp_path / ".anton" / "output"
    output_dir.mkdir(parents=True)
    html = output_dir / "report.html"
    html.write_text("<html><title>Test</title></html>")
    return html


@pytest.mark.asyncio
async def test_401_clears_api_key(tmp_path):
    """When publish returns 401, the key is cleared so the user can re-enter it."""
    from anton.chat import _handle_publish

    html = _make_html_file(tmp_path)
    settings = _make_settings(tmp_path, api_key=None)
    workspace = _make_workspace()
    console = _make_console()

    http_401 = urllib.error.HTTPError(
        url="https://4nton.ai/upload", code=401, msg="Unauthorized", hdrs=None, fp=None
    )

    with (
        patch("anton.chat.prompt_or_cancel", new=AsyncMock(side_effect=["y", "wrongkey", "1"])),
        patch("anton.publisher.publish", side_effect=http_401),
    ):
        await _handle_publish(console, settings, workspace, file_arg=str(html))

    # Key must be cleared after 401
    assert settings.minds_api_key is None
    # Workspace must have the key blanked out
    workspace.set_secret.assert_called_with("ANTON_MINDS_API_KEY", "")
    # User must see a helpful message, not a raw exception
    error_calls = [str(c) for c in console.print.call_args_list]
    assert any("Invalid API key" in c for c in error_calls)


@pytest.mark.asyncio
async def test_successful_publish_persists_key(tmp_path):
    """When publish succeeds, the key is saved to the workspace."""
    from anton.chat import _handle_publish

    html = _make_html_file(tmp_path)
    settings = _make_settings(tmp_path, api_key=None)
    workspace = _make_workspace()
    console = _make_console()

    publish_result = {
        "view_url": "https://4nton.ai/r/abc123",
        "report_id": "abc123",
        "md5": "deadbeef",
        "version": 1,
        "unchanged": False,
    }

    with (
        patch("anton.chat.prompt_or_cancel", new=AsyncMock(side_effect=["y", "goodkey", "1"])),
        patch("anton.publisher.publish", return_value=publish_result),
        patch("webbrowser.open"),
    ):
        await _handle_publish(console, settings, workspace, file_arg=str(html))

    # Key must be persisted after success
    workspace.set_secret.assert_called_with("ANTON_MINDS_API_KEY", "goodkey")


@pytest.mark.asyncio
async def test_401_with_existing_key_clears_it(tmp_path):
    """If a bad key was already saved (e.g. from a previous failed attempt),
    a new 401 clears it so /publish re-prompts next time."""
    from anton.chat import _handle_publish

    html = _make_html_file(tmp_path)
    settings = _make_settings(tmp_path, api_key="stale-bad-key")
    workspace = _make_workspace()
    console = _make_console()

    http_401 = urllib.error.HTTPError(
        url="https://4nton.ai/upload", code=401, msg="Unauthorized", hdrs=None, fp=None
    )

    with (
        patch("anton.publisher.publish", side_effect=http_401),
    ):
        await _handle_publish(console, settings, workspace, file_arg=str(html))

    assert settings.minds_api_key is None
    workspace.set_secret.assert_called_with("ANTON_MINDS_API_KEY", "")
