"""Responses API transport (``flavor="openai"``) at parity with chat.completions.

The gaps that kept hosts on chat.completions for direct OpenAI: truncation was
never reported (no ``response.incomplete`` handling), failed responses and
stream ``error`` events ended the stream silently, images a tool returned were
dropped, and Langfuse trace headers were never attached.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from anton.core.llm.openai import OpenAIProvider, _translate_user_blocks_to_responses
from anton.core.llm.provider import StreamComplete, TransientProviderError
from anton.core.llm.structured import looks_truncated
from anton.core.llm.tracing import TraceContext, reset_trace_context, set_trace_context

_PNG = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}}


async def _aiter(items):
    for item in items:
        yield item


def _provider(client, **kw):
    # Only the client is faked: the real `openai` exception classes must stay
    # in place, since failed responses are raised as `openai.APIError`.
    with patch("anton.core.llm.openai.openai.AsyncOpenAI", return_value=client):
        return OpenAIProvider(api_key="k", flavor=OpenAIProvider.FLAVOR_OPENAI, **kw)


async def _stream(events, **kw):
    client = AsyncMock()
    client.responses.create = AsyncMock(return_value=_aiter(events))
    provider = _provider(client, **kw)
    out = [e async for e in provider.stream(
        model="gpt-5", system="s", messages=[{"role": "user", "content": "hi"}])]
    return out, client


def _final(events):
    return [e for e in events if isinstance(e, StreamComplete)][-1].response


def _usage(inp=100, out=50):
    return SimpleNamespace(input_tokens=inp, output_tokens=out, input_tokens_details=None)


class TestTruncation:
    async def test_incomplete_stream_reports_length_and_usage(self):
        events, _ = await _stream([
            SimpleNamespace(type="response.output_text.delta", delta="partial"),
            SimpleNamespace(type="response.incomplete", response=SimpleNamespace(
                status="incomplete", usage=_usage(out=40),
                incomplete_details=SimpleNamespace(reason="max_output_tokens"))),
        ])
        response = _final(events)
        assert response.stop_reason == "length"
        assert response.usage.output_tokens == 40
        assert looks_truncated(response, budget=4096)

    async def test_other_incomplete_reasons_pass_through_by_name(self):
        events, _ = await _stream([
            SimpleNamespace(type="response.output_text.delta", delta="x"),
            SimpleNamespace(type="response.incomplete", response=SimpleNamespace(
                status="incomplete", usage=_usage(),
                incomplete_details=SimpleNamespace(reason="content_filter"))),
        ])
        assert _final(events).stop_reason == "content_filter"

    async def test_completed_stays_completed(self):
        events, _ = await _stream([
            SimpleNamespace(type="response.output_text.delta", delta="done"),
            SimpleNamespace(type="response.completed",
                            response=SimpleNamespace(status="completed", usage=_usage())),
        ])
        response = _final(events)
        assert response.stop_reason == "completed"
        assert not looks_truncated(response, budget=4096)

    async def test_incomplete_non_streaming_reports_length(self):
        client = AsyncMock()
        client.responses.create = AsyncMock(return_value=SimpleNamespace(
            status="incomplete", usage=_usage(out=4096), model="gpt-5",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            output=[SimpleNamespace(type="message", content=[
                SimpleNamespace(type="output_text", text="cut")])]))
        result = await _provider(client).complete(
            model="gpt-5", system="s", messages=[{"role": "user", "content": "hi"}])
        assert result.stop_reason == "length"
        assert result.content == "cut"


class TestFailures:
    async def test_failed_stream_raises_classified_transient(self):
        with pytest.raises(TransientProviderError) as info:
            await _stream([
                SimpleNamespace(type="response.failed", response=SimpleNamespace(
                    status="failed", error=SimpleNamespace(code="server_error", message="boom"))),
            ])
        assert info.value.code == "server_error"
        assert info.value.session_backoff is True  # mid-stream: never SDK-retried

    async def test_error_event_raises_instead_of_an_empty_answer(self):
        # An unmapped code, so the stream_error backoff applies.
        with pytest.raises(TransientProviderError) as info:
            await _stream([
                SimpleNamespace(type="error", code="vector_store_timeout",
                                message="the vector store timed out",
                                param=None, sequence_number=1),
            ])
        assert info.value.code == "stream_error"
        assert info.value.session_backoff is True

    async def test_stream_with_no_terminal_event_and_no_output_is_truncated(self):
        with pytest.raises(TransientProviderError) as info:
            await _stream([])
        assert info.value.code == "truncated_stream"

    async def test_failed_non_streaming_response_raises(self):
        client = AsyncMock()
        client.responses.create = AsyncMock(return_value=SimpleNamespace(
            status="failed", output=[], usage=None, model="gpt-5",
            error=SimpleNamespace(code="server_error", message="boom")))
        with pytest.raises(TransientProviderError) as info:
            await _provider(client).complete(
                model="gpt-5", system="s", messages=[{"role": "user", "content": "hi"}])
        assert info.value.code == "server_error"


class TestToolResultImages:
    def test_images_are_deferred_after_every_function_call_output(self):
        items = _translate_user_blocks_to_responses([
            {"type": "tool_result", "tool_use_id": "a", "content": [_PNG]},
            {"type": "tool_result", "tool_use_id": "b", "content": [{"type": "text", "text": "ok"}]},
        ])
        assert [i.get("type") for i in items] == [
            "function_call_output", "function_call_output", "message"]
        assert items[0]["output"] == "Image attached in next user message."
        assert items[1]["output"] == "ok"
        assert items[2]["role"] == "user"
        assert items[2]["content"] == [
            {"type": "input_text", "text": "Image(s) returned by previous tool call:"},
            {"type": "input_image", "image_url": "data:image/png;base64,QUJD", "detail": "auto"},
        ]

    def test_text_beside_an_image_stays_in_the_output(self):
        items = _translate_user_blocks_to_responses([
            {"type": "tool_result", "tool_use_id": "a",
             "content": [{"type": "text", "text": "chart rendered"}, _PNG]},
        ])
        assert items[0]["output"] == "chart rendered"
        assert items[1]["content"][1]["type"] == "input_image"

    def test_no_vision_drops_images_but_keeps_the_output(self):
        items = _translate_user_blocks_to_responses(
            [{"type": "tool_result", "tool_use_id": "a", "content": [_PNG]}],
            supports_vision=False)
        assert items == [{"type": "function_call_output", "call_id": "a", "output": ""}]

    def test_openai_shaped_image_url_blocks_reach_the_model(self):
        items = _translate_user_blocks_to_responses([
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,Zm9v"}},
        ])
        assert items == [{"role": "user", "type": "message", "content": [
            {"type": "input_text", "text": "look"},
            {"type": "input_image", "image_url": "data:image/jpeg;base64,Zm9v", "detail": "auto"},
        ]}]


class TestRequestShape:
    async def test_trace_headers_ride_on_the_responses_request(self, monkeypatch):
        monkeypatch.setenv("ANTON_LANGFUSE_HEADERS", "1")
        token = set_trace_context(TraceContext(session_id="s1", turn_id=2, harness="cowork"))
        try:
            _, client = await _stream([
                SimpleNamespace(type="response.output_text.delta", delta="hi"),
                SimpleNamespace(type="response.completed",
                                response=SimpleNamespace(status="completed", usage=_usage())),
            ])
        finally:
            reset_trace_context(token)
        headers = client.responses.create.call_args.kwargs["extra_headers"]
        assert headers["Langfuse-Session-Id"] == "s1"

    async def test_no_trace_headers_when_emission_is_off(self, monkeypatch):
        monkeypatch.delenv("ANTON_LANGFUSE_HEADERS", raising=False)
        token = set_trace_context(TraceContext(session_id="s1"))
        try:
            _, client = await _stream([
                SimpleNamespace(type="response.output_text.delta", delta="hi"),
                SimpleNamespace(type="response.completed",
                                response=SimpleNamespace(status="completed", usage=_usage())),
            ])
        finally:
            reset_trace_context(token)
        assert "extra_headers" not in client.responses.create.call_args.kwargs

    async def test_store_is_left_at_the_provider_default(self):
        # store=False made later prompt-cache hits measurably slower; see
        # _build_responses_kwargs before changing this.
        _, client = await _stream([
            SimpleNamespace(type="response.output_text.delta", delta="hi"),
            SimpleNamespace(type="response.completed",
                            response=SimpleNamespace(status="completed", usage=_usage())),
        ])
        assert "store" not in client.responses.create.call_args.kwargs


def test_hosts_can_detect_the_ready_responses_transport():
    assert OpenAIProvider.RESPONSES_TRANSPORT_READY is True
