"""Provider failures retain retry metadata without persisting echoed content."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable

import anthropic
import httpx2 as httpx
import openai
import pytest

from anton.core.llm.anthropic import AnthropicProvider
from anton.core.llm.openai import OpenAIProvider
from anton.core.llm.provider import (
    LLMProvider,
    StreamComplete,
    TransientProviderError,
    log_transient_provider_error,
    safe_parse_tool_input,
)


_PRIVATE_VALUES = (
    "unrecognized-provider-password-3304",
    "connector-form-body-3304",
    "sql-bind-value-3304",
    "tool-arguments-3304",
    "model-request-body-3304",
)


def _error_body(error_type: str) -> dict[str, object]:
    return {"error": {
        "type": error_type,
        "code": error_type,
        "message": _PRIVATE_VALUES[0],
        "form": _PRIVATE_VALUES[1],
        "sql_parameters": _PRIVATE_VALUES[2],
        "tool_arguments": _PRIVATE_VALUES[3],
        "request_body": _PRIVATE_VALUES[4],
    }}


def _provider(
    monkeypatch: pytest.MonkeyPatch, lane: str,
    handler: Callable[[httpx.Request], httpx.Response],
) -> LLMProvider:
    # Keep the real SDK parser and error types; only replace its HTTP transport.
    sdk = anthropic if lane == "anthropic" else openai
    client_name = "AsyncAnthropic" if lane == "anthropic" else "AsyncOpenAI"
    client_class = getattr(sdk, client_name)

    def client(**kwargs: object):
        return client_class(
            **kwargs, max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

    monkeypatch.setattr(sdk, client_name, client)
    if lane == "anthropic":
        return AnthropicProvider(api_key="test")
    flavor = (OpenAIProvider.FLAVOR_OPENAI if lane == "responses"
              else OpenAIProvider.FLAVOR_OPENAI_COMPATIBLE_GENERIC)
    return OpenAIProvider(api_key="test", base_url="https://provider.example/v1", flavor=flavor)


def _assert_log(
    caplog: pytest.LogCaptureFixture, lane: str, error: TransientProviderError,
) -> None:
    logger_name = "anton.core.llm.anthropic" if lane == "anthropic" else "anton.core.llm.openai"
    records = [record for record in caplog.records
               if record.name == logger_name and record.levelno == logging.WARNING]
    assert len(records) == 1
    record = records[0]
    message = record.getMessage()
    assert record.exc_info is None
    assert record.stack_info is None
    # Check arguments too: rendering or downstream formatting must never recover
    # the body from a record that merely hid it in its message template.
    for private_value in _PRIVATE_VALUES:
        assert private_value not in message
        assert private_value not in repr(record.args)
    assert f"error_code={error.code}" in message
    status = "unknown" if error.status_code is None else error.status_code
    assert f"status={status}" in message
    assert f"retry_after={error.retry_after}" in message
    assert f"session_backoff={error.session_backoff}" in message


@pytest.mark.parametrize("lane", ["chat", "anthropic"])
@pytest.mark.parametrize("status,error_type,code,backoff,retry_after", [
    (503, "overloaded_error", "overloaded_error", False, None),
    (429, "rate_limited", "rate_limited", True, 7.5),
    # A provider-controlled type is not safe metadata. The classifier replaces
    # an unrecognized type with its HTTP code rather than logging that value.
    (503, _PRIVATE_VALUES[0], "http_503", False, None),
])
async def test_http_error_logs_metadata_without_body(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    lane: str, status: int, error_type: str, code: str, backoff: bool,
    retry_after: float | None,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=_error_body(error_type), headers={"Retry-After": "7.5"})

    provider = _provider(monkeypatch, lane, handler)
    try:
        with pytest.raises(TransientProviderError) as raised:
            await provider.complete(
                model="test-model", system="system",
                messages=[{"role": "user", "content": "question"}],
            )
        error = raised.value
        assert error.code == code
        assert error.status_code == status
        assert error.session_backoff is backoff
        assert error.retry_after == retry_after
        assert error.model == "test-model"
        assert error.__cause__ is not None  # error propagation is unchanged
        _assert_log(caplog, lane, error)
    finally:
        await provider.aclose()


@pytest.mark.parametrize("lane", ["chat", "responses", "anthropic"])
async def test_stream_error_logs_metadata_without_body(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, lane: str,
) -> None:
    body = _error_body("overloaded_error")
    if lane == "anthropic":
        body["type"] = "error"
    payload = (("event: error\n" if lane == "anthropic" else "")
               + "data: " + json.dumps(body) + "\n\n").encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload, headers={"Content-Type": "text/event-stream"})

    provider = _provider(monkeypatch, lane, handler)
    try:
        with pytest.raises(TransientProviderError) as raised:
            _ = [event async for event in provider.stream(
                model="test-model", system="system",
                messages=[{"role": "user", "content": "question"}],
            )]
        error = raised.value
        assert error.code == "overloaded_error"
        assert error.status_code == (200 if lane == "anthropic" else None)
        assert error.session_backoff is True
        assert error.retry_after is None
        assert error.model == "test-model"
        assert error.__cause__ is not None
        _assert_log(caplog, lane, error)
    finally:
        await provider.aclose()


class _TextStatus(int):
    """An in-range int whose rendering carries private text."""

    def __str__(self) -> str:
        return _PRIVATE_VALUES[0]

    __repr__ = __str__


class _TextCode(str):
    """A known code whose rendering carries private text."""

    def __str__(self) -> str:
        return _PRIVATE_VALUES[0]

    __repr__ = __str__


class _MatchingCode(str):
    """Private text that claims to equal every code."""

    def __eq__(self, other: object) -> bool:
        return True

    __hash__ = str.__hash__


@pytest.mark.parametrize("code,status,expected", [
    ("rate_limited", 429, "error_code=rate_limited status=429"),
    ("overloaded_error", None, "error_code=overloaded_error status=unknown"),
    ("http_503", 503, "error_code=http_503 status=503"),
    # A Responses failure takes its code from the provider's body.
    (_PRIVATE_VALUES[0], None, "error_code=unrecognized status=unknown"),
    # Only the exact `http_5xx` shape classify_transient mints is an HTTP code.
    ("http_\n503", 503, "error_code=unrecognized status=503"),
    ("http_5_03", 503, "error_code=unrecognized status=503"),
    # A str subclass can render anything or match anything, so it is not a code.
    (_TextCode("rate_limited"), 429, "error_code=unrecognized status=429"),
    (_MatchingCode(_PRIVATE_VALUES[0]), 429, "error_code=unrecognized status=429"),
    # Some providers send an integer code.
    (503, 503, "error_code=unrecognized status=503"),
    (None, "503 " + _PRIVATE_VALUES[1], "error_code=unrecognized status=unknown"),
    # Only an exact int HTTP status is logged; bool is an int subclass.
    ("rate_limited", True, "error_code=rate_limited status=unknown"),
    ("rate_limited", 99, "error_code=rate_limited status=unknown"),
    ("rate_limited", 600, "error_code=rate_limited status=unknown"),
    # An int subclass can render anything, so it is not a status.
    ("rate_limited", _TextStatus(429), "error_code=rate_limited status=unknown"),
])
def test_transient_warning_logs_only_known_codes_and_real_statuses(
    caplog: pytest.LogCaptureFixture, code: object, status: object, expected: str,
) -> None:
    logger = logging.getLogger("anton.core.llm.openai")
    error = TransientProviderError("The model provider failed.", code=code, status_code=status)
    with caplog.at_level(logging.WARNING, logger=logger.name):
        log_transient_provider_error(logger=logger, error=error)
    records = [record for record in caplog.records
               if record.name == logger.name and record.levelno == logging.WARNING]
    assert len(records) == 1
    assert records[0].getMessage() == (
        f"transient provider error {expected} retry_after=None session_backoff=True"
    )
    for private_value in _PRIVATE_VALUES:
        assert private_value not in repr(records[0].args)


# The model sends credentials in tool arguments, for example a datasource's
# known variables. A missing comma leaves them unrepairable. The non-ASCII name
# makes the UTF-8 byte count differ from the character count.
_TOOL_SECRET = "private-key-passphrase-3304"
_MALFORMED_TOOL_ARGUMENTS = (
    '{"known_variables": {"private_key_passphrase": "%s" "user": "andré"}}' % _TOOL_SECRET
)
_TOOLS = [{"name": "connect_datasource", "description": "Connect.", "input_schema": {"type": "object"}}]


def _sse(*, events: list[dict[str, object]], named: bool) -> bytes:
    lines = []
    for event in events:
        if named:
            lines.append(f"event: {event['type']}\n")
        lines.append("data: " + json.dumps(event) + "\n\n")
    return "".join(lines).encode()


def _tool_call_response(*, lane: str) -> httpx.Response:
    if lane == "chat_complete":
        return httpx.Response(200, json={
            "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "test-model",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {
                    "name": "connect_datasource", "arguments": _MALFORMED_TOOL_ARGUMENTS}}],
            }}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })
    if lane == "chat":
        chunk = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 0, "model": "test-model"}
        body = _sse(events=[
            {**chunk, "choices": [{"index": 0, "finish_reason": None, "delta": {
                "role": "assistant",
                "tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {
                    "name": "connect_datasource", "arguments": _MALFORMED_TOOL_ARGUMENTS}}],
            }}]},
            {**chunk, "choices": [{"index": 0, "finish_reason": "tool_calls", "delta": {}}]},
        ], named=False) + b"data: [DONE]\n\n"
    elif lane == "responses":
        body = _sse(events=[
            {"type": "response.output_item.added", "sequence_number": 1, "output_index": 0,
             "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1",
                      "name": "connect_datasource", "arguments": "", "status": "in_progress"}},
            {"type": "response.function_call_arguments.delta", "sequence_number": 2,
             "output_index": 0, "item_id": "fc_1", "delta": _MALFORMED_TOOL_ARGUMENTS},
            {"type": "response.function_call_arguments.done", "sequence_number": 3,
             "output_index": 0, "item_id": "fc_1", "name": "connect_datasource",
             "arguments": _MALFORMED_TOOL_ARGUMENTS},
            {"type": "response.completed", "sequence_number": 4, "response": {
                "id": "resp_1", "object": "response", "created_at": 0, "model": "test-model",
                "status": "completed", "output": [],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}},
        ], named=False)
    else:
        body = _sse(events=[
            {"type": "message_start", "message": {
                "id": "msg_1", "type": "message", "role": "assistant", "model": "test-model",
                "content": [], "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1}}},
            {"type": "content_block_start", "index": 0, "content_block": {
                "type": "tool_use", "id": "toolu_1", "name": "connect_datasource", "input": {}}},
            {"type": "content_block_delta", "index": 0, "delta": {
                "type": "input_json_delta", "partial_json": _MALFORMED_TOOL_ARGUMENTS}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
             "usage": {"output_tokens": 1}},
            {"type": "message_stop"},
        ], named=True)
    return httpx.Response(200, content=body, headers={"Content-Type": "text/event-stream"})


@pytest.mark.parametrize("lane", ["chat_complete", "chat", "responses", "anthropic"])
async def test_unrecoverable_tool_arguments_log_position_without_text(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, lane: str,
) -> None:
    provider = _provider(monkeypatch, "chat" if lane == "chat_complete" else lane,
                         lambda request: _tool_call_response(lane=lane))
    request = {"model": "test-model", "system": "system", "tools": _TOOLS,
               "messages": [{"role": "user", "content": "question"}]}
    try:
        with caplog.at_level(logging.INFO, logger="anton.core.llm.provider"):
            if lane == "chat_complete":
                tool_calls = (await provider.complete(**request)).tool_calls
            else:
                events = [event async for event in provider.stream(**request)]
                tool_calls = [event for event in events
                              if isinstance(event, StreamComplete)][-1].response.tool_calls
    finally:
        await provider.aclose()
    assert [call.parse_error for call in tool_calls] == ["Expecting ',' delimiter: line 1 column 78 (char 77)"]
    records = [record for record in caplog.records
               if record.name == "anton.core.llm.provider" and record.levelno == logging.WARNING]
    assert len(records) == 1
    assert records[0].getMessage() == (
        "Tool-use input JSON was malformed and unrecoverable "
        "(Expecting ',' delimiter: line 1 column 78 (char 77)). "
        f"Raw bytes: {len(_MALFORMED_TOOL_ARGUMENTS.encode('utf-8'))}"
    )
    assert _TOOL_SECRET not in repr(records[0].args)


def test_repaired_tool_arguments_log_position_without_text(caplog: pytest.LogCaptureFixture) -> None:
    # Cut off before the closing brace.
    raw = '{"user": "andré", "private_key_passphrase": "%s"' % _TOOL_SECRET
    with caplog.at_level(logging.INFO, logger="anton.core.llm.provider"):
        assert safe_parse_tool_input(raw) == (
            {"user": "andré", "private_key_passphrase": _TOOL_SECRET}, None, True)
    records = [record for record in caplog.records
               if record.name == "anton.core.llm.provider" and record.levelno == logging.INFO]
    assert len(records) == 1
    assert _TOOL_SECRET not in records[0].getMessage()
    assert records[0].getMessage().endswith(
        f"Raw bytes: {len(raw.encode('utf-8'))}, truncated: True.")
    # The decoder's exception keeps the whole raw body in `doc`, so it must not ride in the args.
    assert all(isinstance(arg, (str, int, bool)) for arg in records[0].args)


def test_tool_arguments_with_a_split_surrogate_pair_log_without_raising(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Streamed deltas joined as str can hold one half of a surrogate pair.
    raw = '{"note": "\ud83d" "user": "analyst"}'
    with caplog.at_level(logging.WARNING, logger="anton.core.llm.provider"):
        parsed, parse_error, truncated = safe_parse_tool_input(raw)
    assert (parsed, truncated) == ({}, False)
    assert parse_error is not None
    records = [record for record in caplog.records
               if record.name == "anton.core.llm.provider" and record.levelno == logging.WARNING]
    assert len(records) == 1
    assert records[0].getMessage().endswith(
        f"Raw bytes: {len(raw.encode('utf-8', 'surrogatepass'))}")
