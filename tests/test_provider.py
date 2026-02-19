from __future__ import annotations

from dataclasses import fields
from unittest.mock import AsyncMock, MagicMock, patch

from anton.llm.anthropic import AnthropicProvider
from anton.llm.provider import LLMResponse, ToolCall, Usage


class TestDataclasses:
    def test_tool_call(self):
        tc = ToolCall(id="tc_1", name="create_plan", input={"key": "val"})
        assert tc.id == "tc_1"
        assert tc.name == "create_plan"
        assert tc.input == {"key": "val"}

    def test_usage_defaults(self):
        u = Usage()
        assert u.input_tokens == 0
        assert u.output_tokens == 0

    def test_llm_response_defaults(self):
        r = LLMResponse(content="hi")
        assert r.content == "hi"
        assert r.tool_calls == []
        assert r.usage.input_tokens == 0
        assert r.stop_reason is None

    def test_llm_response_with_tool_calls(self):
        tc = ToolCall(id="1", name="test", input={})
        r = LLMResponse(content="", tool_calls=[tc], stop_reason="tool_use")
        assert len(r.tool_calls) == 1
        assert r.stop_reason == "tool_use"


class TestAnthropicProvider:
    async def test_complete_text_response(self):
        with patch("anton.llm.anthropic.anthropic") as mock_anthropic:
            mock_client = AsyncMock()
            mock_anthropic.AsyncAnthropic.return_value = mock_client

            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = "Hello world"

            mock_response = MagicMock()
            mock_response.content = [text_block]
            mock_response.usage.input_tokens = 5
            mock_response.usage.output_tokens = 10
            mock_response.stop_reason = "end_turn"

            mock_client.messages.create = AsyncMock(return_value=mock_response)

            provider = AnthropicProvider(api_key="test-key")
            result = await provider.complete(
                model="claude-sonnet-4-6",
                system="be helpful",
                messages=[{"role": "user", "content": "hi"}],
            )

            assert result.content == "Hello world"
            assert result.tool_calls == []
            assert result.usage.input_tokens == 5
            assert result.stop_reason == "end_turn"

    async def test_complete_tool_use_response(self):
        with patch("anton.llm.anthropic.anthropic") as mock_anthropic:
            mock_client = AsyncMock()
            mock_anthropic.AsyncAnthropic.return_value = mock_client

            tool_block = MagicMock()
            tool_block.type = "tool_use"
            tool_block.id = "tool_1"
            tool_block.name = "create_plan"
            tool_block.input = {"reasoning": "test"}

            mock_response = MagicMock()
            mock_response.content = [tool_block]
            mock_response.usage.input_tokens = 15
            mock_response.usage.output_tokens = 25
            mock_response.stop_reason = "tool_use"

            mock_client.messages.create = AsyncMock(return_value=mock_response)

            provider = AnthropicProvider(api_key="test-key")
            result = await provider.complete(
                model="claude-sonnet-4-6",
                system="plan",
                messages=[{"role": "user", "content": "do something"}],
                tools=[{"name": "create_plan", "description": "plan", "input_schema": {}}],
            )

            assert result.content == ""
            assert len(result.tool_calls) == 1
            assert result.tool_calls[0].name == "create_plan"
            assert result.tool_calls[0].input == {"reasoning": "test"}
            assert result.stop_reason == "tool_use"

    async def test_provider_without_api_key(self):
        with patch("anton.llm.anthropic.anthropic") as mock_anthropic:
            mock_anthropic.AsyncAnthropic.return_value = AsyncMock()
            provider = AnthropicProvider()
            mock_anthropic.AsyncAnthropic.assert_called_once_with()
