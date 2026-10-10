"""Cloud-turn diagnostics omit provider bodies while keeping the wire contract."""

from __future__ import annotations

import io
import logging
from collections.abc import Iterator
from pathlib import Path

import anthropic
import httpx2 as httpx
import openai
import pytest
from openai.types.chat import ChatCompletion

from anton.cloud_turn.__main__ import _scrub, _stderr_log_handler, stream_turn
from anton.core.llm.anthropic import AnthropicProvider
from anton.core.llm.openai import OpenAIProvider
from anton.core.llm.provider import (
    ProviderErrorFilter,
    ProviderOverloadedError,
    TransientProviderError,
)


PRIVATE = "provider-body-private-marker-3304"
RAW = '{"protocol_version":1,"conversation_id":"privacy","input":"hi"}'
CLOUD_LOGGER = "anton.cloud_turn.__main__"


@pytest.fixture
def pod_stderr() -> Iterator[io.StringIO]:
    """The pod's root stderr handler, writing to a buffer the test can read."""
    stream = io.StringIO()
    handler = _stderr_log_handler(stream=stream)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield stream
    finally:
        root.removeHandler(handler)


@pytest.mark.parametrize("lane", ["chat", "responses", "anthropic"])
@pytest.mark.parametrize("status", [400, 503])
async def test_cloud_turn_filters_real_sdk_bodies_and_terminal_cause_chains(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, pod_stderr: io.StringIO,
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
        # The pod logs every logger at INFO to stderr, so check everything the turn writes.
        with caplog.at_level(logging.INFO):
            await stream_turn(RAW, events.append, session_builder=lambda req: session)
        assert session.error is not None and session.sdk_error is not None
        assert session.closed
        assert PRIVATE in str(session.sdk_error)
        if status == 503:
            assert session.error.__cause__.__cause__ is session.sdk_error
        # Logging must leave the same short wire error and original SDK body intact.
        assert events[-1] == {"kind": "turn_failed", "error": _scrub(session.error)}
        assert PRIVATE not in pod_stderr.getvalue()
        assert caplog.records
        for record in caplog.records:
            assert PRIVATE not in logging.Formatter().format(record)
        records = [record for record in caplog.records
                   if record.name == CLOUD_LOGGER and record.levelno == logging.ERROR]
        assert len(records) == 1
        record = records[0]
        assert PRIVATE not in repr(record.args)
        assert record.exc_info is None and record.exc_text is None and record.stack_info is None
        # A terminal ProviderOverloadedError carries no status of its own; the
        # adapter's transient warning logged the 503 earlier in the turn.
        expected = ("error_type=ProviderOverloadedError provider_error=ProviderOverloadedError status=unknown"
                    if status == 503 else
                    "error_type=BadRequestError provider_error=BadRequestError status=400")
        assert record.getMessage() == f"Provider operation failed: {expected}"
    finally:
        await provider.aclose()


@pytest.mark.parametrize("chain", ["cause", "context", "suppressed_context", "group"])
def test_provider_filter_finds_wrapped_errors_without_mutating_them(chain: str) -> None:
    provider_error = ProviderOverloadedError(PRIVATE)
    provider_error.status_code = PRIVATE  # Arbitrary strings are never safe HTTP metadata.
    wrapper = RuntimeError(PRIVATE)
    if chain == "group":
        wrapper = ExceptionGroup(PRIVATE, [wrapper, provider_error])
    else:
        setattr(wrapper, "__cause__" if chain == "cause" else "__context__", provider_error)
        # `raise ... from None` hides the context from tracebacks, but the
        # wrapper's own message can still repeat the provider's text.
        wrapper.__suppress_context__ = chain == "suppressed_context"
        provider_error.__context__ = wrapper  # Cycles must not hang the logger.
    record = logging.LogRecord(CLOUD_LOGGER, logging.ERROR, __file__, 1,
                               f"Failed: {PRIVATE}", (), (type(wrapper), wrapper, None))
    record.exc_text = PRIVATE
    record.stack_info = PRIVATE
    record.request_id = "request-123"
    assert ProviderErrorFilter().filter(record)
    assert record.getMessage() == (
        f"Provider operation failed: error_type={type(wrapper).__name__} "
        "provider_error=ProviderOverloadedError status=unknown"
    )
    assert record.exc_info is None and record.exc_text is None and record.stack_info is None
    assert record.request_id == "request-123"
    assert str(provider_error) == PRIVATE


def test_cloud_turn_filter_preserves_unrelated_error_tracebacks(
    caplog: pytest.LogCaptureFixture, pod_stderr: io.StringIO,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
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
    assert "RuntimeError: ordinary runtime failure" in pod_stderr.getvalue()
    assert events[-1] == {"kind": "turn_failed", "error": _scrub(error)}


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://provider.example/v1/chat/completions")


def _rate_limit_error() -> openai.RateLimitError:
    return openai.RateLimitError(
        PRIVATE, response=httpx.Response(429, request=_request()), body={"message": PRIVATE},
    )


def _raised(*, error: BaseException) -> BaseException:
    try:
        raise error
    except BaseException as caught:
        return caught


def _wrapped_with_its_own_status() -> BaseException:
    try:
        raise _rate_limit_error()
    except openai.RateLimitError as provider_error:
        wrapper = RuntimeError(PRIVATE)
        wrapper.status_code = 502  # A gateway status on the wrapper is not the provider's.
        try:
            raise wrapper from provider_error
        except RuntimeError as caught:
            return caught


def _raised_while_handling() -> BaseException:
    try:
        try:
            raise _rate_limit_error()
        except openai.RateLimitError:
            raise KeyError(PRIVATE)
    except KeyError as caught:
        return caught


def _cause_subtree_before_context() -> BaseException:
    # The provider error sits one level down the cause, and another sits on the
    # wrapper's own context. A depth-first walk reaches the cause's context first.
    cause = RuntimeError(PRIVATE)
    cause.__context__ = _rate_limit_error()
    wrapper = RuntimeError(PRIVATE)
    wrapper.__cause__ = cause
    wrapper.__context__ = openai.APIConnectionError(message=PRIVATE, request=_request())
    return wrapper


# cowork-server's copy of ProviderErrorFilter runs these same cases and
# expects the same lines, so the two copies stay aligned.
@pytest.mark.parametrize("build,expected", [
    (lambda: _raised(error=_rate_limit_error()),
     "error_type=RateLimitError provider_error=RateLimitError status=429"),
    (_wrapped_with_its_own_status,
     "error_type=RuntimeError provider_error=RateLimitError status=429"),
    (lambda: _raised(error=openai.APIConnectionError(message=PRIVATE, request=_request())),
     "error_type=APIConnectionError provider_error=APIConnectionError status=unknown"),
    (_raised_while_handling,
     "error_type=KeyError provider_error=RateLimitError status=429"),
    (lambda: _raised(error=openai.LengthFinishReasonError(completion=ChatCompletion.model_construct(usage=None))),
     "error_type=LengthFinishReasonError provider_error=LengthFinishReasonError status=unknown"),
    (_cause_subtree_before_context,
     "error_type=RuntimeError provider_error=RateLimitError status=429"),
], ids=["provider_429", "wrapper_status", "no_status", "raised_while_handling", "not_an_api_error",
        "cause_subtree_before_context"])
def test_provider_filter_parity_cases(build, expected: str) -> None:
    error = build()
    record = logging.LogRecord(CLOUD_LOGGER, logging.ERROR, __file__, 1,
                               f"Failed: {PRIVATE}", (), (type(error), error, error.__traceback__))
    assert ProviderErrorFilter().filter(record)
    assert record.getMessage() == f"Provider operation failed: {expected}"
    assert record.exc_info is None and record.exc_text is None and record.stack_info is None
    assert PRIVATE not in logging.Formatter().format(record)


@pytest.mark.parametrize("shape", ["message", "argument", "mapping"])
def test_provider_filter_reads_errors_from_the_message_and_arguments(shape: str) -> None:
    error = _raised(error=_rate_limit_error())
    msg, args = {
        "message": (error, ()),
        "argument": ("failed: %s", (error,)),
        "mapping": ("failed: %(error)s", ({"error": error},)),
    }[shape]
    record = logging.LogRecord(CLOUD_LOGGER, logging.WARNING, __file__, 1, msg, args, None)
    assert ProviderErrorFilter().filter(record)
    assert record.getMessage() == (
        "Provider operation failed: error_type=RateLimitError provider_error=RateLimitError status=429"
    )
    assert PRIVATE not in logging.Formatter().format(record)
