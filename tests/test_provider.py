from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from anton.core.llm.anthropic import AnthropicProvider
from anton.core.llm.openai import _parse_response_object
from anton.core.llm.provider import LLMResponse, ToolCall, compute_context_pressure


class TestComputeContextPressure:
    def test_none_input_tokens_is_zero_not_crash(self):
        # The MindsHub passthrough returns usage.input_tokens=None on
        # web-search responses; compute_context_pressure must not raise
        # `unsupported operand type(s) for /: 'NoneType' and 'int'`.
        assert compute_context_pressure("claude-sonnet-4-6", None) == 0.0

    def test_zero_input_tokens_is_zero(self):
        assert compute_context_pressure("claude-sonnet-4-6", 0) == 0.0

    def test_normal_ratio(self):
        # 100k tokens against a 200k window → 0.5.
        assert compute_context_pressure("claude-sonnet-4-6", 100_000) == 0.5

    def test_clamps_at_one(self):
        assert compute_context_pressure("claude-3", 10_000_000) == 1.0

    def test_parse_response_object_coerces_none_usage_tokens(self):
        # End-to-end at the crash site: a web-search Responses object comes
        # back with usage.input_tokens/output_tokens = None. _parse_response_object
        # must coerce them to 0 (not pass None into compute_context_pressure)
        # and must not raise.
        response = SimpleNamespace(
            output=[],
            usage=SimpleNamespace(input_tokens=None, output_tokens=None),
        )
        result = _parse_response_object(response, "claude-sonnet-4-6")
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0
        assert result.usage.context_pressure == 0.0

    def test_parse_response_object_keeps_real_usage_tokens(self):
        # Sanity: valid counts are preserved unchanged.
        response = SimpleNamespace(
            output=[],
            usage=SimpleNamespace(input_tokens=100_000, output_tokens=250),
        )
        result = _parse_response_object(response, "claude-sonnet-4-6")
        assert result.usage.input_tokens == 100_000
        assert result.usage.output_tokens == 250
        assert result.usage.context_pressure == 0.5


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


# ─────────────────────────────────────────────────────────────────────────────
# Native server-side web tools (web_search / web_fetch)
# ─────────────────────────────────────────────────────────────────────────────


def _stub_text_response(text: str = "ok"):
    """Build a MagicMock response that looks like a plain text Anthropic reply."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    response.usage.input_tokens = 1
    response.usage.output_tokens = 1
    response.stop_reason = "end_turn"
    return response


class TestAnthropicNativeWebTools:
    def test_native_web_tools_advertises_search_and_fetch(self):
        with patch("anton.core.llm.anthropic.anthropic") as mock_anthropic:
            mock_anthropic.AsyncAnthropic.return_value = AsyncMock()
            provider = AnthropicProvider(api_key="k")
        assert provider.native_web_tools() == {"web_search", "web_fetch"}

    async def test_complete_appends_web_search_server_tool(self):
        from anton.core.llm.anthropic import ANTHROPIC_WEB_SEARCH_TOOL_TYPE

        with patch("anton.core.llm.anthropic.anthropic") as mock_anthropic:
            mock_client = AsyncMock()
            mock_anthropic.AsyncAnthropic.return_value = mock_client
            mock_client.messages.create = AsyncMock(return_value=_stub_text_response())

            provider = AnthropicProvider(api_key="k")
            await provider.complete(
                model="claude-sonnet-4-6",
                system="sys",
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"name": "scratchpad", "description": "x", "input_schema": {}}],
                native_web_tools={"web_search"},
            )

            kwargs = mock_client.messages.create.call_args[1]
            tools = kwargs["tools"]
            # Existing function tool is preserved
            assert any(t.get("name") == "scratchpad" for t in tools)
            # Server tool entry is appended in the right shape
            assert {"type": ANTHROPIC_WEB_SEARCH_TOOL_TYPE, "name": "web_search"} in tools
            # web_search is GA — no beta header should be set
            assert "extra_headers" not in kwargs

    async def test_complete_appends_web_fetch_with_beta_header(self):
        from anton.core.llm.anthropic import (
            ANTHROPIC_WEB_FETCH_BETA_HEADER,
            ANTHROPIC_WEB_FETCH_TOOL_TYPE,
        )

        with patch("anton.core.llm.anthropic.anthropic") as mock_anthropic:
            mock_client = AsyncMock()
            mock_anthropic.AsyncAnthropic.return_value = mock_client
            mock_client.messages.create = AsyncMock(return_value=_stub_text_response())

            provider = AnthropicProvider(api_key="k")
            await provider.complete(
                model="claude-sonnet-4-6",
                system="sys",
                messages=[{"role": "user", "content": "hi"}],
                native_web_tools={"web_fetch"},
            )

            kwargs = mock_client.messages.create.call_args[1]
            assert {"type": ANTHROPIC_WEB_FETCH_TOOL_TYPE, "name": "web_fetch"} in kwargs["tools"]
            # web_fetch is beta — header must be present
            assert kwargs["extra_headers"] == {
                "anthropic-beta": ANTHROPIC_WEB_FETCH_BETA_HEADER
            }

    async def test_complete_appends_both_server_tools(self):
        with patch("anton.core.llm.anthropic.anthropic") as mock_anthropic:
            mock_client = AsyncMock()
            mock_anthropic.AsyncAnthropic.return_value = mock_client
            mock_client.messages.create = AsyncMock(return_value=_stub_text_response())

            provider = AnthropicProvider(api_key="k")
            await provider.complete(
                model="claude-sonnet-4-6",
                system="sys",
                messages=[{"role": "user", "content": "hi"}],
                native_web_tools={"web_search", "web_fetch"},
            )

            kwargs = mock_client.messages.create.call_args[1]
            names = [t.get("name") for t in kwargs["tools"]]
            assert "web_search" in names and "web_fetch" in names
            # web_fetch always brings the beta header along
            assert "anthropic-beta" in kwargs["extra_headers"]

    async def test_complete_omits_web_tools_when_set_is_empty(self):
        with patch("anton.core.llm.anthropic.anthropic") as mock_anthropic:
            mock_client = AsyncMock()
            mock_anthropic.AsyncAnthropic.return_value = mock_client
            mock_client.messages.create = AsyncMock(return_value=_stub_text_response())

            provider = AnthropicProvider(api_key="k")
            await provider.complete(
                model="claude-sonnet-4-6",
                system="sys",
                messages=[{"role": "user", "content": "hi"}],
                native_web_tools=None,
            )

            kwargs = mock_client.messages.create.call_args[1]
            # No tools array at all — backward-compatible with the no-tools case
            assert "tools" not in kwargs
            assert "extra_headers" not in kwargs
