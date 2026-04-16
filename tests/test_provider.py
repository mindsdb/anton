from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from anton.core.llm.anthropic import AnthropicProvider
from anton.core.llm.provider import LLMResponse, ToolCall


class TestDataclasses:
    def test_llm_response_with_tool_calls(self):
        tc = ToolCall(id="1", name="test", input={})
        r = LLMResponse(content="", tool_calls=[tc], stop_reason="tool_use")
        assert len(r.tool_calls) == 1
        assert r.stop_reason == "tool_use"


class TestAnthropicProvider:
    async def test_complete_text_response(self):
        with patch("anton.core.llm.anthropic.anthropic") as mock_anthropic:
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
        with patch("anton.core.llm.anthropic.anthropic") as mock_anthropic:
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

    async def test_complete_passes_tool_choice(self):
        with patch("anton.core.llm.anthropic.anthropic") as mock_anthropic:
            mock_client = AsyncMock()
            mock_anthropic.AsyncAnthropic.return_value = mock_client

            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = "ok"

            mock_response = MagicMock()
            mock_response.content = [text_block]
            mock_response.usage.input_tokens = 5
            mock_response.usage.output_tokens = 10
            mock_response.stop_reason = "end_turn"

            mock_client.messages.create = AsyncMock(return_value=mock_response)

            provider = AnthropicProvider(api_key="test-key")
            tool_choice = {"type": "tool", "name": "my_tool"}
            tools = [{"name": "my_tool", "description": "d", "input_schema": {"type": "object"}}]
            await provider.complete(
                model="claude-sonnet-4-6",
                system="sys",
                messages=[{"role": "user", "content": "hi"}],
                tools=tools,
                tool_choice=tool_choice,
            )

            call_kwargs = mock_client.messages.create.call_args[1]
            assert call_kwargs["tool_choice"] == tool_choice
            assert call_kwargs["tools"] == tools

    async def test_complete_omits_tool_choice_when_none(self):
        with patch("anton.core.llm.anthropic.anthropic") as mock_anthropic:
            mock_client = AsyncMock()
            mock_anthropic.AsyncAnthropic.return_value = mock_client

            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = "ok"

            mock_response = MagicMock()
            mock_response.content = [text_block]
            mock_response.usage.input_tokens = 5
            mock_response.usage.output_tokens = 10
            mock_response.stop_reason = "end_turn"

            mock_client.messages.create = AsyncMock(return_value=mock_response)

            provider = AnthropicProvider(api_key="test-key")
            await provider.complete(
                model="claude-sonnet-4-6",
                system="sys",
                messages=[{"role": "user", "content": "hi"}],
            )

            call_kwargs = mock_client.messages.create.call_args[1]
            assert "tool_choice" not in call_kwargs

    async def test_provider_without_api_key(self):
        with patch("anton.core.llm.anthropic.anthropic") as mock_anthropic:
            mock_anthropic.AsyncAnthropic.return_value = AsyncMock()
            provider = AnthropicProvider()
            mock_anthropic.AsyncAnthropic.assert_called_once_with()
