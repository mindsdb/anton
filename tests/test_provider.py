from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anton.core.llm.anthropic import AnthropicProvider
from anton.core.llm.openai import _parse_response_object
from anton.core.llm.provider import (
    LLMResponse,
    ToolCall,
    TransientProviderError,
    compute_context_pressure,
    raise_on_empty_response,
)


def _output_text_item(text: str) -> SimpleNamespace:
    """A minimal Responses API ``message`` item carrying one output_text block."""
    return SimpleNamespace(
        type="message",
        content=[SimpleNamespace(type="output_text", text=text)],
    )


class TestRaiseOnEmptyResponse:
    def test_raises_on_empty_200(self):
        # No content, no tool calls, no stop reason → the empty-200 failure mode.
        with pytest.raises(TransientProviderError) as exc:
            raise_on_empty_response(content="", tool_calls=[], stop_reason=None)
        assert exc.value.code == "empty_response"
        # Fail fast, don't loop the retry budget on a broken endpoint.
        assert exc.value.session_backoff is False

    def test_passthrough_with_content(self):
        raise_on_empty_response(content="hi", tool_calls=[], stop_reason=None)

    def test_passthrough_with_tool_calls(self):
        raise_on_empty_response(
            content="", tool_calls=[ToolCall(id="1", name="t", input={})], stop_reason=None
        )

    def test_passthrough_with_stop_reason(self):
        # A real stop_reason means the provider terminated deliberately (e.g. a
        # legitimately empty turn) — not a truncated/empty 200.
        raise_on_empty_response(content="", tool_calls=[], stop_reason="stop")

    def test_empty_string_stop_reason_counts_as_absent(self):
        # An empty-string stop_reason is treated as absent (no real provider
        # sends ""), so an otherwise-empty response still raises. Pins the
        # truthiness predicate against a revert to `stop_reason is not None`.
        with pytest.raises(TransientProviderError):
            raise_on_empty_response(content="", tool_calls=[], stop_reason="")

    def test_parse_response_object_raises_on_empty_200(self):
        # End-to-end: an empty output with no status is the silent-empty 200 the
        # guard exists to catch — it must raise, not return an empty LLMResponse.
        response = SimpleNamespace(output=[], usage=None)
        with pytest.raises(TransientProviderError):
            _parse_response_object(response, "claude-sonnet-4-6")


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
            output=[_output_text_item("ok")],
            status="completed",
            usage=SimpleNamespace(input_tokens=None, output_tokens=None),
        )
        result = _parse_response_object(response, "claude-sonnet-4-6")
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0
        assert result.usage.context_pressure == 0.0

    def test_parse_response_object_keeps_real_usage_tokens(self):
        # Sanity: valid counts are preserved unchanged.
        response = SimpleNamespace(
            output=[_output_text_item("ok")],
            status="completed",
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


class _FakeAnthropicStream:
    """Minimal async-context-manager + async-iterator stand-in for the
    object `client.messages.stream(**kwargs)` returns (NOT awaited itself
    — used via `async with ... as stream: async for event in stream:`)."""

    def __init__(self, events):
        self._events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for event in self._events:
            yield event


class TestAnthropicProviderReasoningStream:
    """ENG-1109: extended-thinking content blocks (triggered server-side by
    `output_config.effort`, already sent whenever reasoning_effort is set)
    must surface as StreamReasoningDelta, not get misclassified as text or
    silently dropped."""

    async def test_thinking_delta_becomes_stream_reasoning_delta(self):
        from anton.core.llm.provider import StreamReasoningDelta, StreamTextDelta

        events = [
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(usage=SimpleNamespace(input_tokens=5, output_tokens=0)),
            ),
            SimpleNamespace(
                type="content_block_start", index=0,
                content_block=SimpleNamespace(type="thinking"),
            ),
            SimpleNamespace(
                type="content_block_delta", index=0,
                delta=SimpleNamespace(type="thinking_delta", thinking="Let me check that first."),
            ),
            SimpleNamespace(
                type="content_block_delta", index=0,
                delta=SimpleNamespace(type="signature_delta", signature="sig-abc"),
            ),
            SimpleNamespace(type="content_block_stop", index=0),
            SimpleNamespace(
                type="content_block_start", index=1,
                content_block=SimpleNamespace(type="text"),
            ),
            SimpleNamespace(
                type="content_block_delta", index=1,
                delta=SimpleNamespace(type="text_delta", text="The real answer."),
            ),
            SimpleNamespace(type="content_block_stop", index=1),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="end_turn"),
                usage=SimpleNamespace(output_tokens=12),
            ),
        ]

        with patch("anton.core.llm.anthropic.anthropic") as mock_anthropic:
            mock_client = AsyncMock()
            mock_client.messages.stream = MagicMock(return_value=_FakeAnthropicStream(events))
            mock_anthropic.AsyncAnthropic.return_value = mock_client

            provider = AnthropicProvider(api_key="test-key", reasoning_effort="medium")
            yielded = [
                e async for e in provider.stream(
                    model="claude-sonnet-4-6",
                    system="be helpful",
                    messages=[{"role": "user", "content": "hi"}],
                )
            ]

        reasoning_events = [e for e in yielded if isinstance(e, StreamReasoningDelta)]
        text_events = [e for e in yielded if isinstance(e, StreamTextDelta)]
        assert reasoning_events == [StreamReasoningDelta(text="Let me check that first.")]
        assert text_events == [StreamTextDelta(text="The real answer.")]

    async def test_stream_passes_effort_via_extra_body(self):
        with patch("anton.core.llm.anthropic.anthropic") as mock_anthropic:
            mock_client = AsyncMock()
            mock_client.messages.stream = MagicMock(
                return_value=_FakeAnthropicStream([
                    SimpleNamespace(
                        type="message_delta",
                        delta=SimpleNamespace(stop_reason="end_turn"),
                        usage=SimpleNamespace(output_tokens=0),
                    ),
                ])
            )
            mock_anthropic.AsyncAnthropic.return_value = mock_client

            provider = AnthropicProvider(api_key="k", reasoning_effort="high")
            async for _ in provider.stream(
                model="claude-sonnet-4-6", system="s", messages=[{"role": "user", "content": "hi"}],
            ):
                pass

            call_kwargs = mock_client.messages.stream.call_args[1]
            assert call_kwargs["extra_body"] == {"output_config": {"effort": "high"}}


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
