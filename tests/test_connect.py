"""Tests for /connect and /disconnect handlers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anton.workspace import Workspace


@pytest.fixture()
def workspace(tmp_path):
    ws = Workspace(tmp_path)
    ws.initialize()
    return ws


@pytest.fixture()
def console():
    c = MagicMock()
    c.print = MagicMock()
    return c


class TestHandleDisconnect:
    @pytest.mark.asyncio
    async def test_removes_secrets(self, console, workspace):
        from anton.chat import _handle_disconnect

        workspace.set_secret("MINDS_CONNECTION", '{"url":"https://mdb.ai"}')
        workspace.set_secret("MINDS_API_KEY", "test-key-123")

        await _handle_disconnect(console, workspace)

        assert workspace.has_secret("MINDS_CONNECTION") is False
        assert workspace.has_secret("MINDS_API_KEY") is False
        # Should print success message
        console.print.assert_any_call("[anton.success]MindsDB connection removed.[/]")

    @pytest.mark.asyncio
    async def test_no_existing_connection(self, console, workspace):
        from anton.chat import _handle_disconnect

        await _handle_disconnect(console, workspace)

        # Should print "no connection" message
        console.print.assert_any_call("[anton.muted]No MindsDB connection found.[/]")


class TestHandleConnect:
    @pytest.mark.asyncio
    async def test_stores_connection(self, console, workspace):
        from anton.chat import _handle_connect

        minds_response = [
            {
                "name": "my_mind",
                "model_name": "gpt-4",
                "provider": "openai",
                "datasources": ["my_db"],
            },
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = minds_response

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client), \
             patch("rich.prompt.Prompt") as mock_prompt_cls:
            mock_prompt_cls.ask = MagicMock(
                side_effect=["https://mdb.ai", "test-key", "1", "1"]
            )
            await _handle_connect(console, workspace)

        # Verify secrets were stored
        assert workspace.has_secret("MINDS_API_KEY") is True
        assert workspace.get_secret("MINDS_API_KEY") == "test-key"

        conn_raw = workspace.get_secret("MINDS_CONNECTION")
        assert conn_raw is not None
        conn = json.loads(conn_raw)
        assert conn["url"] == "https://mdb.ai"
        assert conn["mind_name"] == "my_mind"
        assert conn["datasource"] == "my_db"
        assert conn["model_name"] == "gpt-4"
        assert conn["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_connection_failure(self, console, workspace):
        from anton.chat import _handle_connect

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client), \
             patch("rich.prompt.Prompt") as mock_prompt_cls:
            mock_prompt_cls.ask = MagicMock(
                side_effect=["https://mdb.ai", "bad-key"]
            )
            await _handle_connect(console, workspace)

        # Should NOT store anything
        assert workspace.has_secret("MINDS_CONNECTION") is False
        assert workspace.has_secret("MINDS_API_KEY") is False

    @pytest.mark.asyncio
    async def test_empty_api_key_aborts(self, console, workspace):
        from anton.chat import _handle_connect

        with patch("rich.prompt.Prompt") as mock_prompt_cls:
            mock_prompt_cls.ask = MagicMock(
                side_effect=["https://mdb.ai", ""]
            )
            await _handle_connect(console, workspace)

        assert workspace.has_secret("MINDS_CONNECTION") is False
        console.print.assert_any_call(
            "[anton.error]No API key provided. Aborting.[/]"
        )

    @pytest.mark.asyncio
    async def test_no_minds_found(self, console, workspace):
        from anton.chat import _handle_connect

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = []

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client), \
             patch("rich.prompt.Prompt") as mock_prompt_cls:
            mock_prompt_cls.ask = MagicMock(
                side_effect=["https://mdb.ai", "some-key"]
            )
            await _handle_connect(console, workspace)

        assert workspace.has_secret("MINDS_CONNECTION") is False
        console.print.assert_any_call(
            "[anton.error]No Minds found on this server.[/]"
        )

    @pytest.mark.asyncio
    async def test_url_without_protocol_gets_https(self, console, workspace):
        from anton.chat import _handle_connect

        minds_response = [
            {
                "name": "test_mind",
                "model_name": "gpt-4",
                "provider": "openai",
                "datasources": ["ds1"],
            },
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = minds_response

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client), \
             patch("rich.prompt.Prompt") as mock_prompt_cls:
            mock_prompt_cls.ask = MagicMock(
                side_effect=["terbase.dev.mdb.ai", "test-key", "1", "1"]
            )
            await _handle_connect(console, workspace)

        conn_raw = workspace.get_secret("MINDS_CONNECTION")
        assert conn_raw is not None
        conn = json.loads(conn_raw)
        assert conn["url"] == "https://terbase.dev.mdb.ai"

        # Verify the GET was called with the https:// prefixed URL
        mock_client.get.assert_called_once_with(
            "https://terbase.dev.mdb.ai/api/v1/minds",
            headers={"Authorization": "Bearer test-key"},
        )
