"""Responses transport through the real OpenAI SDK against a local scripted endpoint.

test_responses_parity.py fakes the SDK's event objects. These tests serve real
SSE/JSON bytes over HTTP so the SDK's own parsing, typed events and error
classes are in the path: what the provider sees here is what it sees from
api.openai.com.
"""
from __future__ import annotations

import json
import threading
import typing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import openai
import pytest
from openai.types.responses import ResponseError

from anton.core.llm import provider as provider_mod
from anton.core.llm.openai import OpenAIProvider
from anton.core.llm.provider import (
    ContentTooLargeError,
    ContentValidationError,
    ContextOverflowError,
    RequestRefusedError,
    StreamComplete,
    StreamToolUseEnd,
    StreamToolUseStart,
    TransientProviderError,
    provider_failure_kind,
)
from anton.core.llm.structured import looks_truncated
from anton.core.llm.tracing import TraceContext, reset_trace_context, set_trace_context

USAGE = {"input_tokens": 120, "input_tokens_details": {"cached_tokens": 100}, "output_tokens": 16,
         "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 136}
TOOLS = [{"name": "get_stock", "description": "Stock for a part.",
          "input_schema": {"type": "object", "properties": {"part": {"type": "string"}}}}]


def _response(status, output=(), **extra):
    return {"id": "resp_1", "object": "response", "created_at": 0, "model": "gpt-test", "status": status,
            "output": list(output), "usage": USAGE, "parallel_tool_calls": True, "tool_choice": "auto",
            "tools": [], **extra}


def _sse(*events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps({'sequence_number': i, **e})}\n\n"
                   for i, e in enumerate(events)).encode()


def _text_events(text="Partial answer"):
    return [
        {"type": "response.created", "response": _response("in_progress", usage=None)},
        {"type": "response.output_item.added", "output_index": 0,
         "item": {"type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": []}},
        {"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0, "content_index": 0,
         "delta": text, "logprobs": []},
    ]


def _call_events(arguments):
    return [
        {"type": "response.output_item.added", "output_index": 1,
         "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_stock",
                  "arguments": "", "status": "in_progress"}},
        {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "output_index": 1, "delta": arguments},
        {"type": "response.function_call_arguments.done", "item_id": "fc_1", "output_index": 1, "arguments": arguments},
    ]


class _Endpoint:
    """Serves one scripted reply per POST /v1/responses and records each request."""

    def __init__(self):
        self.replies, self.requests = [], []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append({"body": body, "headers": {k.lower(): v for k, v in self.headers.items()}})
                status, kind, payload = owner.replies.pop(0)
                data = payload if kind == "sse" else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "text/event-stream" if kind == "sse" else "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def provider(self, **kw):
        return OpenAIProvider(api_key="k", base_url=self.base_url, flavor=OpenAIProvider.FLAVOR_OPENAI, **kw)


@pytest.fixture
def endpoint():
    e = _Endpoint()
    yield e
    e.server.shutdown()


async def _stream(provider, **kw):
    kw.setdefault("messages", [{"role": "user", "content": "hi"}])
    events = [e async for e in provider.stream(model="gpt-test", system="s", max_tokens=16, **kw)]
    done = [e for e in events if isinstance(e, StreamComplete)]
    return events, (done[-1].response if done else None)


async def test_incomplete_text_reports_length_and_usage(endpoint):
    endpoint.replies.append((200, "sse", _sse(*_text_events(), {
        "type": "response.incomplete",
        "response": _response("incomplete", incomplete_details={"reason": "max_output_tokens"})})))
    _, r = await _stream(endpoint.provider())
    assert (r.content, r.stop_reason) == ("Partial answer", "length")
    assert (r.usage.input_tokens, r.usage.cache_read_tokens, r.usage.output_tokens) == (20, 100, 16)
    assert looks_truncated(r, budget=4096)


async def test_incomplete_inside_tool_arguments_is_a_damaged_call_and_length(endpoint):
    endpoint.replies.append((200, "sse", _sse(*_text_events("Checking."), *_call_events('{"part": "P'), {
        "type": "response.incomplete",
        "response": _response("incomplete", incomplete_details={"reason": "max_output_tokens"})})))
    events, r = await _stream(endpoint.provider(), tools=TOOLS)
    assert r.stop_reason == "length"
    (call,) = r.tool_calls
    assert call.name == "get_stock" and (call.parse_error or call.repaired)
    assert sum(isinstance(e, StreamToolUseStart) for e in events) == sum(isinstance(e, StreamToolUseEnd) for e in events) == 1


async def test_failed_response_raises_a_backoff_transient(endpoint):
    endpoint.replies.append((200, "sse", _sse(*_text_events(""), {
        "type": "response.failed",
        "response": _response("failed", error={"code": "server_error", "message": "The server had an error"})})))
    with pytest.raises(TransientProviderError) as info:
        await _stream(endpoint.provider())
    assert (info.value.code, info.value.session_backoff) == ("server_error", True)


async def test_error_event_raises_instead_of_ending_quietly(endpoint):
    # An unmapped code, so the stream_error backoff applies. The codes that name
    # their cause have their own tests below.
    endpoint.replies.append((200, "sse", _sse(*_text_events(""), {
        "type": "error", "code": "vector_store_timeout", "message": "The vector store timed out",
        "param": None})))
    with pytest.raises(TransientProviderError) as info:
        await _stream(endpoint.provider())
    assert info.value.code == "stream_error" and info.value.session_backoff is True


async def test_stream_cut_before_any_output_is_truncated_stream(endpoint):
    endpoint.replies.append((200, "sse", _sse({"type": "response.created", "response": _response("in_progress", usage=None)})))
    with pytest.raises(TransientProviderError) as info:
        await _stream(endpoint.provider())
    assert info.value.code == "truncated_stream"


async def test_stream_cut_after_text_passes_the_text_through(endpoint):
    endpoint.replies.append((200, "sse", _sse(*_text_events("Complete enough"))))
    _, r = await _stream(endpoint.provider())
    assert r.content == "Complete enough" and r.stop_reason is None


async def test_non_streaming_incomplete_and_failed(endpoint):
    message = {"type": "message", "id": "msg_1", "role": "assistant", "status": "incomplete",
               "content": [{"type": "output_text", "text": "cut", "annotations": []}]}
    endpoint.replies.append((200, "json", _response("incomplete", [message], incomplete_details={"reason": "max_output_tokens"})))
    endpoint.replies.append((200, "json", _response("failed", error={"code": "server_error", "message": "boom"})))
    p = endpoint.provider()
    r = await p.complete(model="gpt-test", system="s", messages=[{"role": "user", "content": "hi"}], max_tokens=16)
    assert (r.content, r.stop_reason) == ("cut", "length")
    with pytest.raises(TransientProviderError) as info:
        await p.complete(model="gpt-test", system="s", messages=[{"role": "user", "content": "hi"}], max_tokens=16)
    assert info.value.code == "server_error"


@pytest.mark.parametrize("parameter", [{"param": "reasoning.summary"}, {"param": None}, {"param": ""}, {}])
async def test_refused_reasoning_summary_is_dropped_and_not_asked_again(endpoint, parameter):
    refusal = {"error": {"message": "Your organization must be verified to generate reasoning summaries.",
                         "type": "invalid_request_error", **parameter, "code": "unsupported_value"}}
    done = {"type": "response.completed", "response": _response("completed")}
    endpoint.replies += [(400, "json", refusal), (200, "sse", _sse(*_text_events("ok"), done)),
                         (200, "sse", _sse(*_text_events("again"), done))]
    p = endpoint.provider(reasoning_effort="low")
    _, first = await _stream(p)
    _, second = await _stream(p)
    assert (first.content, second.content) == ("ok", "again")
    sent = [r["body"]["reasoning"] for r in endpoint.requests]
    assert sent == [{"effort": "low", "summary": "auto"}, {"effort": "low"}, {"effort": "low"}]


@pytest.mark.parametrize("message,param", [
    ("Invalid schema for function 'get_stock'.", "tools[0]"),
    ("Invalid schema for function 'summarize_reasoning'.", "tools[0]"),
    ("Invalid schema for function 'summarize_reasoning'.", None),
    ("Your organization must be verified to generate reasoning summaries.", "tools[0]"),
])
async def test_other_bad_requests_are_not_retried(endpoint, message, param):
    endpoint.replies.append((400, "json", {"error": {"message": message,
                                                     "type": "invalid_request_error", "param": param, "code": None}}))
    done = {"type": "response.completed", "response": _response("completed")}
    endpoint.replies.append((200, "sse", _sse(*_text_events("ok"), done)))
    p = endpoint.provider(reasoning_effort="low")
    with pytest.raises(openai.BadRequestError):
        await _stream(p, tools=TOOLS)
    assert len(endpoint.requests) == 1
    _, result = await _stream(p, tools=TOOLS)
    assert result.content == "ok"
    assert endpoint.requests[-1]["body"]["reasoning"] == {"effort": "low", "summary": "auto"}


@pytest.mark.parametrize("error", [None, {"message": "Failed"}, {"code": None, "message": "Failed"}])
async def test_non_streaming_failure_without_a_code_maps_to_bad_response(endpoint, error):
    endpoint.replies.append((200, "json", _response("failed", error=error)))
    with pytest.raises(TransientProviderError) as info:
        await endpoint.provider().complete(
            model="gpt-test", system="s", messages=[{"role": "user", "content": "hi"}],
        )
    assert info.value.code == "response_failed"
    assert provider_failure_kind(info.value.code) == "bad_response"


async def test_wire_body_defers_tool_images_and_carries_trace_headers(endpoint, monkeypatch):
    monkeypatch.setenv("ANTON_LANGFUSE_HEADERS", "1")
    endpoint.replies.append((200, "sse", _sse(*_text_events("Red"), {"type": "response.completed", "response": _response("completed")})))
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}}
    messages = [{"role": "user", "content": "Render both."},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "call_a", "name": "get_stock", "input": {}},
                                                  {"type": "tool_use", "id": "call_b", "name": "get_stock", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_a", "content": [image]},
                                             {"type": "tool_result", "tool_use_id": "call_b", "content": "B done"}]}]
    token = set_trace_context(TraceContext(session_id="sess-1", harness="cowork"))
    try:
        await _stream(endpoint.provider(), messages=messages, tools=TOOLS)
    finally:
        reset_trace_context(token)
    req = endpoint.requests[0]
    kinds = [(i.get("type"), i.get("role")) for i in req["body"]["input"]]
    assert kinds[-3:] == [("function_call_output", None), ("function_call_output", None), ("message", "user")]
    assert req["body"]["input"][-1]["content"][1] == {"type": "input_image", "image_url": "data:image/png;base64,QUJD", "detail": "auto"}
    assert "store" not in req["body"]
    assert req["headers"].get("langfuse-session-id") == "sess-1"


def _failed(code, message):
    """A stream that fails in-band with a `response.failed` naming its cause."""
    return _sse(*_text_events(""), {
        "type": "response.failed", "response": _response("failed", error={"code": code, "message": message})})


async def test_failed_context_length_exceeded_raises_context_overflow(endpoint):
    """The session compacts the history on this type and re-sends, as it does
    for the request-time 400."""
    endpoint.replies.append((200, "sse", _failed(
        "context_length_exceeded", "Your input exceeds the context window of this model.")))
    with pytest.raises(ContextOverflowError):
        await _stream(endpoint.provider())


async def test_rate_limit_error_event_takes_the_rate_limit_path(endpoint):
    """`rate_limited` on the count path, as an unconfirmed 429 gets, not the
    incident backoff: the code does not say whether a per-minute or a per-day
    limit was hit, so it earns no session wait. No HTTP status, because the
    stream itself was a 200."""
    endpoint.replies.append((200, "sse", _sse(*_text_events(""), {
        "type": "error", "code": "rate_limit_exceeded", "message": "Slow down", "param": None})))
    with pytest.raises(TransientProviderError) as info:
        await _stream(endpoint.provider())
    assert (info.value.code, info.value.session_backoff, info.value.status_code) == ("rate_limited", False, None)
    assert info.value.retry_after is None


async def test_failed_invalid_prompt_raises_the_refusal(endpoint):
    endpoint.replies.append((200, "sse", _failed(
        "invalid_prompt", "Invalid prompt: your prompt was flagged as potentially violating our usage policy.")))
    with pytest.raises(RequestRefusedError) as info:
        await _stream(endpoint.provider())
    assert (info.value.code, info.value.status_code) == ("invalid_prompt", None)
    assert "flagged as potentially violating our usage policy" in str(info.value)


async def test_failed_prompt_content_filter_raises_the_refusal(endpoint):
    endpoint.replies.append((200, "sse", _failed(
        "content_filter", "The prompt was filtered due to triggering the content management policy.")))
    with pytest.raises(RequestRefusedError) as info:
        await _stream(endpoint.provider())
    assert info.value.code == "content_filter"


# The SDK's in-band image codes, written out here rather than imported, so a
# code dropped from the classifier fails its own test. The size codes also ask
# the user for a smaller copy.
_IMAGE_TOO_LARGE_CODES = ["image_too_large", "image_file_too_large"]
_IMAGE_REJECTED_CODES = [
    "invalid_image", "invalid_image_format", "invalid_base64_image", "invalid_image_url",
    "invalid_image_mode", "image_too_small", "image_parse_error",
    "image_content_policy_violation", "unsupported_image_media_type", "empty_image_file",
    "failed_to_download_image", "image_file_not_found",
]


def test_the_image_codes_are_the_sdks_own():
    """A misspelled code never matches a real failure. Both the lists above and
    the classifier's are checked against the SDK's `ResponseError.code`."""
    sdk_codes = set(typing.get_args(ResponseError.model_fields["code"].annotation))
    assert set(_IMAGE_TOO_LARGE_CODES + _IMAGE_REJECTED_CODES) <= sdk_codes
    mapped = provider_mod._RESPONSES_IMAGE_TOO_LARGE_CODES + provider_mod._RESPONSES_IMAGE_REJECTED_CODES
    assert set(mapped) <= sdk_codes


@pytest.mark.parametrize("code", _IMAGE_TOO_LARGE_CODES)
async def test_failed_image_size_code_raises_content_too_large(endpoint, code):
    endpoint.replies.append((200, "sse", _failed(code, "The image is larger than the 20 MB limit.")))
    with pytest.raises(ContentTooLargeError) as info:
        await _stream(endpoint.provider())
    assert "The image is larger than the 20 MB limit." in str(info.value)


@pytest.mark.parametrize("code", _IMAGE_REJECTED_CODES)
async def test_failed_image_code_raises_content_validation(endpoint, code):
    endpoint.replies.append((200, "sse", _failed(code, "The image could not be used.")))
    with pytest.raises(ContentValidationError) as info:
        await _stream(endpoint.provider())
    assert not isinstance(info.value, ContentTooLargeError)


async def test_a_filtered_output_still_ends_as_a_stop_reason(endpoint):
    """A filtered PROMPT fails the turn (above); a filtered OUTPUT does not. The
    Responses API documents it as `response.incomplete`, which stays an answer
    that stops early. Not yet observed on Azure."""
    endpoint.replies.append((200, "sse", _sse(*_text_events("Part of"), {
        "type": "response.incomplete",
        "response": _response("incomplete", incomplete_details={"reason": "content_filter"})})))
    _, r = await _stream(endpoint.provider())
    assert (r.content, r.stop_reason) == ("Part of", "content_filter")


async def test_a_nested_error_event_is_mapped_after_the_sdk_raises_it(endpoint):
    """An `error` event that nests its error object is raised by the SDK itself,
    before the reader sees the event, so the mapping must cover that path too."""
    endpoint.replies.append((200, "sse", _sse(*_text_events(""), {
        "type": "error", "error": {"type": "invalid_request_error", "code": "context_length_exceeded",
                                   "message": "Your input exceeds the context window of this model."}})))
    with pytest.raises(ContextOverflowError):
        await _stream(endpoint.provider())


@pytest.mark.parametrize("code,expected", [
    ("context_length_exceeded", ContextOverflowError),
    ("rate_limit_exceeded", TransientProviderError),
    ("invalid_prompt", RequestRefusedError),
    ("image_too_large", ContentTooLargeError),
])
async def test_a_failed_non_streamed_response_maps_the_same_codes(endpoint, code, expected):
    endpoint.replies.append((200, "json", _response("failed", error={"code": code, "message": "Refused."})))
    with pytest.raises(expected) as info:
        await endpoint.provider().complete(
            model="gpt-test", system="s", messages=[{"role": "user", "content": "hi"}], max_tokens=16)
    if expected is TransientProviderError:
        assert (info.value.code, info.value.session_backoff) == ("rate_limited", False)


async def test_function_tools_are_sent_non_strict(endpoint):
    """chat.completions reads an omitted `strict` as non-strict, while the
    Responses API normalizes an omitted one into strict mode. Sending False
    keeps both transports on the same schema rules. A hosted tool takes none."""
    endpoint.replies.append((200, "sse", _sse(*_text_events("ok"), {
        "type": "response.completed", "response": _response("completed")})))
    await _stream(endpoint.provider(), tools=TOOLS, native_web_tools={"web_search"})
    assert endpoint.requests[0]["body"]["tools"] == [
        {"type": "function", "name": "get_stock", "description": "Stock for a part.",
         "parameters": TOOLS[0]["input_schema"], "strict": False},
        {"type": "web_search"},
    ]
