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
from anton.core.llm.provider import LLMProvider, TransientProviderError


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
    assert f"code={error.code}" in message
    assert f"status={error.status_code}" in message
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
