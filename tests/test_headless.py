"""Tests for headless mode (--prompt flag)."""
from __future__ import annotations

import json
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anton.chat import _headless
from anton.llm.provider import (
    LLMResponse,
    StreamComplete,
    StreamTextDelta,
    StreamToolUseEnd,
    StreamToolUseStart,
    ToolCall,
    Usage,
)


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="end_turn",
    )


def _mock_settings():
    settings = MagicMock()
    settings.workspace_path = MagicMock()
    settings.workspace_path.__truediv__ = lambda self, other: MagicMock()
    settings.context_dir = "/tmp/test-anton-context"
    settings.memory_mode = "off"
    settings.episodic_memory = False
    settings.coding_provider = "anthropic"
    settings.anthropic_api_key = "test-key"
    settings.openai_api_key = None
    settings.openai_base_url = None
    settings.proactive_dashboards = False
    return settings


def _patches(mock_session):
    """Common patches for headless tests. Returns context manager stack."""
    mock_dv = MagicMock()
    mock_dv.return_value.list_connections.return_value = []
    mock_ep = MagicMock()
    mock_ep.return_value.enabled = False
    mock_ep.return_value._session_id = None

    return (
        patch("anton.llm.client.LLMClient.from_settings", return_value=MagicMock()),
        patch("anton.context.self_awareness.SelfAwarenessContext"),
        patch("anton.workspace.Workspace"),
        patch("anton.data_vault.DataVault", mock_dv),
        patch("anton.datasource_registry.DatasourceRegistry"),
        patch("anton.memory.cortex.Cortex"),
        patch("anton.memory.episodes.EpisodicMemory", mock_ep),
        patch("anton.memory.history_store.HistoryStore"),
        patch("anton.chat_session.build_runtime_context", return_value=""),
        patch("anton.chat.ChatSession", return_value=mock_session),
    )


class TestHeadlessTextOutput:
    @pytest.mark.asyncio
    async def test_basic_text_response(self, capsys):
        mock_session = AsyncMock()

        async def fake_stream(prompt):
            yield StreamTextDelta("The answer is 42.")
            yield StreamComplete(_text_response("The answer is 42."))

        mock_session.turn_stream = fake_stream

        patches = _patches(mock_session)
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6], patches[7], patches[8], patches[9]:
            from rich.console import Console
            console = Console(file=StringIO())
            await _headless(console, _mock_settings(), prompt="question", output_format="text")

        captured = capsys.readouterr()
        assert "The answer is 42." in captured.out


class TestHeadlessJsonOutput:
    @pytest.mark.asyncio
    async def test_json_response(self, capsys):
        mock_session = AsyncMock()

        async def fake_stream(prompt):
            yield StreamTextDelta("Hello world")
            yield StreamComplete(_text_response("Hello world"))

        mock_session.turn_stream = fake_stream

        patches = _patches(mock_session)
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6], patches[7], patches[8], patches[9]:
            from rich.console import Console
            console = Console(file=StringIO())
            await _headless(console, _mock_settings(), prompt="say hello", output_format="json")

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["response"] == "Hello world"
        assert isinstance(result["tool_calls"], list)
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 20


class TestHeadlessToolCalls:
    @pytest.mark.asyncio
    async def test_tool_calls_in_json_output(self, capsys):
        mock_session = AsyncMock()

        async def fake_stream(prompt):
            yield StreamToolUseStart(id="tc_1", name="scratchpad")
            yield StreamToolUseEnd(id="tc_1")
            yield StreamTextDelta("Result: 55")
            yield StreamComplete(LLMResponse(
                content="Result: 55",
                tool_calls=[ToolCall(id="tc_1", name="scratchpad", input={"action": "exec"})],
                usage=Usage(input_tokens=50, output_tokens=100),
                stop_reason="end_turn",
            ))

        mock_session.turn_stream = fake_stream

        patches = _patches(mock_session)
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6], patches[7], patches[8], patches[9]:
            from rich.console import Console
            console = Console(file=StringIO())
            await _headless(console, _mock_settings(), prompt="fibonacci", output_format="json")

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "scratchpad"
        assert "55" in result["response"]


class TestHeadlessNoInteractive:
    @pytest.mark.asyncio
    async def test_completes_without_interactive_input(self):
        """Headless mode completes without prompt_toolkit or interactive console."""
        mock_session = AsyncMock()

        async def fake_stream(prompt):
            yield StreamTextDelta("ok")
            yield StreamComplete(_text_response("ok"))

        mock_session.turn_stream = fake_stream

        patches = _patches(mock_session)
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6], patches[7], patches[8], patches[9]:
            from rich.console import Console
            console = Console(file=StringIO())
            # Should complete without hanging on interactive input
            await _headless(console, _mock_settings(), prompt="test", output_format="text")
