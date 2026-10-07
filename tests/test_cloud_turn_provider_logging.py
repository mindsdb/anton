"""Cloud-turn diagnostics omit provider bodies while keeping the wire contract."""

from __future__ import annotations

import logging
from pathlib import Path

import anthropic
import httpx2 as httpx
import openai
import pytest

from anton.cloud_turn.__main__ import _scrub, stream_turn
from anton.core.llm.anthropic import AnthropicProvider
from anton.core.llm.openai import OpenAIProvider
from anton.core.llm.provider import ProviderOverloadedError, TransientProviderError


PRIVATE = "provider-body-private-marker-3304"
RAW = '{"protocol_version":1,"conversation_id":"privacy","input":"hi"}'
CLOUD_LOGGER = "anton.cloud_turn.__main__"


@pytest.mark.parametrize("lane", ["chat", "responses", "anthropic"])
@pytest.mark.parametrize("status", [400, 503])
async def test_cloud_turn_filters_real_sdk_bodies_and_terminal_cause_chains(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    tmp_path: Path, lane: str, status: int,
) -> None:
    """Drive the real adapter and cloud catch; adapter-only safe warnings are insufficient."""
    body = {"error": {"type": "overloaded_error" if status == 503 else PRIVATE,
                       "message": PRIVATE, "tool_arguments": PRIVATE}}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    sdk = anthropic if lane == "anthropic" else openai
    client_name = "AsyncAnthropic" if lane == "anthropic" else "AsyncOpenAI"
    original_client = getattr(sdk, client_name)

    def client(**kwargs: object):
        return original_client(
            **kwargs, max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

    monkeypatch.setattr(sdk, client_name, client)
    if lane == "anthropic":
        provider = AnthropicProvider(api_key="test")
    else:
        flavor = (OpenAIProvider.FLAVOR_OPENAI if lane == "responses"
                  else OpenAIProvider.FLAVOR_OPENAI_COMPATIBLE_GENERIC)
        provider = OpenAIProvider(api_key="test", base_url="https://provider.example/v1", flavor=flavor)

    class Session:
        error: Exception | None = None
        sdk_error: Exception | None = None
        closed = False

        async def turn_stream(self, *args, **kwargs):
            try:
                await provider.complete(model="test-model", system="system",
                                        messages=[{"role": "user", "content": "question"}])
            except Exception as error:
                self.sdk_error = error.__cause__ if isinstance(error, TransientProviderError) else error
                self.error = error
                if isinstance(error, TransientProviderError):
                    # ChatSession raises this terminal wrapper after exhausting retries.
                    self.error = ProviderOverloadedError("The provider did not recover.", model="test-model")
                    raise self.error from error
                raise
            yield  # An async stream whose failure is handled by stream_turn.

        def close(self) -> None:
            self.closed = True

    session = Session()
    events: list[dict] = []
    monkeypatch.setenv("ANTON_CLOUD_WORKSPACE_PATH", str(tmp_path))
    try:
        with caplog.at_level(logging.ERROR, logger=CLOUD_LOGGER):
            await stream_turn(RAW, events.append, session_builder=lambda req: session)
        assert session.error is not None and session.sdk_error is not None
        assert session.closed
        assert PRIVATE in str(session.sdk_error)
        if status == 503:
            assert session.error.__cause__.__cause__ is session.sdk_error
        # Logging must leave the same short wire error and original SDK body intact.
        assert events[-1] == {"kind": "turn_failed", "error": _scrub(session.error)}
        records = [record for record in caplog.records
                   if record.name == CLOUD_LOGGER and record.levelno == logging.ERROR]
        assert len(records) == 1
        record = records[0]
        assert PRIVATE not in record.getMessage()
        assert PRIVATE not in repr(record.args)
        assert PRIVATE not in logging.Formatter().format(record)
        assert record.exc_info is None and record.exc_text is None and record.stack_info is None
        assert f"status={status}" in record.getMessage()
        expected_type = "ProviderOverloadedError" if status == 503 else "BadRequestError"
        assert f"error_type={expected_type}" in record.getMessage()
    finally:
        await provider.aclose()


@pytest.mark.parametrize("chain", ["cause", "context", "group"])
def test_provider_filter_finds_wrapped_errors_without_mutating_them(chain: str) -> None:
    from anton.core.llm.provider import ProviderErrorFilter

    provider_error = ProviderOverloadedError(PRIVATE)
    provider_error.status_code = PRIVATE  # Arbitrary strings are never safe HTTP metadata.
    wrapper = RuntimeError(PRIVATE)
    if chain == "group":
        wrapper = ExceptionGroup(PRIVATE, [wrapper, provider_error])
    else:
        setattr(wrapper, "__cause__" if chain == "cause" else "__context__", provider_error)
        provider_error.__context__ = wrapper  # Cycles must not hang the logger.
    record = logging.LogRecord(CLOUD_LOGGER, logging.ERROR, __file__, 1,
                               f"Failed: {PRIVATE}", (), (type(wrapper), wrapper, None))
    record.exc_text = PRIVATE
    record.stack_info = PRIVATE
    record.request_id = "request-123"
    assert ProviderErrorFilter().filter(record)
    assert record.getMessage() == "Provider operation failed: error_type=ProviderOverloadedError status=None"
    assert record.exc_info is None and record.exc_text is None and record.stack_info is None
    assert record.request_id == "request-123"
    assert str(provider_error) == PRIVATE


def test_cloud_turn_filter_preserves_unrelated_error_tracebacks(
    caplog: pytest.LogCaptureFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    error = RuntimeError("ordinary runtime failure")

    class Session:
        async def turn_stream(self, *args, **kwargs):
            raise error
            yield

    monkeypatch.setenv("ANTON_CLOUD_WORKSPACE_PATH", str(tmp_path))
    events = []
    with caplog.at_level(logging.ERROR, logger=CLOUD_LOGGER):
        asyncio.run(stream_turn(RAW, events.append, session_builder=lambda req: Session()))
    records = [record for record in caplog.records
               if record.name == CLOUD_LOGGER and record.levelno == logging.ERROR]
    assert len(records) == 1
    record = records[0]
    assert record.getMessage() == "cloud turn failed"
    assert record.exc_info[1] is error
    assert "ordinary runtime failure" in logging.Formatter().format(record)
    assert events[-1] == {"kind": "turn_failed", "error": _scrub(error)}
