"""ENG-673 — transient mid-stream provider errors: classification + backoff.

Covers the anton-side of the fix without a live provider:
  * `classify_transient` (Arm A) — the shared body/status classifier.
  * both provider mappers (`anthropic`/`openai` `_raise_for_status_error`) — the
    right typed exception for each failure, incl. the mid-stream HTTP-200 case.
  * the session backoff helpers — cadence, `retry_after`, and cancel-awareness.

The real-SDK wire path (a genuine SSE `error` inside a 200 is parsed and
classified on BOTH providers) is exercised by the mock-provider harness in
`test_transient_retry_e2e.py`, which drives the real anthropic/openai SDKs
against an `httpx.MockTransport`.
"""

from __future__ import annotations

import asyncio
import json
import types

import pytest

from anton.core.llm.provider import (
    ModelUnavailableError,
    ProviderOverloadedError,
    TokenLimitExceeded,
    TransientProviderError,
    classify_transient,
)


class _FakeStatusError(Exception):
    """Minimal stand-in for anthropic/openai APIStatusError — the mappers only
    read `.status_code` and `.body`, and `raise X from exc` needs a real exc."""

    def __init__(self, status_code, body):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.body = body


# --------------------------------------------------------------------------- #
# classify_transient (Arm A)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "status,body",
    [
        (200, {"type": "error", "error": {"type": "overloaded_error"}}),  # mid-stream
        (200, {"error": {"code": "api_error"}}),
        (503, {}),
        (500, {}),
        (429, {}),                       # plain rate-limit (no quota detail)
    ],
)
def test_classify_transient_retryable(status, body):
    result = classify_transient(status, body, provider="Anthropic")
    assert isinstance(result, TransientProviderError)
    assert result.provider == "Anthropic"


def test_session_backoff_flag_splits_midstream_from_requesttime():
    # Mid-stream (overload smuggled into a 200) had NO prior retry → session
    # backs off. Request-time errors (real 5xx/429/529) were SDK-retried already
    # → fail fast, no extra session budget.
    assert classify_transient(
        200, {"error": {"type": "overloaded_error"}}, provider="X"
    ).session_backoff is True
    assert classify_transient(503, {}, provider="X").session_backoff is False
    assert classify_transient(429, {}, provider="X").session_backoff is False
    assert classify_transient(
        529, {"error": {"type": "overloaded_error"}}, provider="X"
    ).session_backoff is False


@pytest.mark.parametrize(
    "status,body",
    [
        (200, {}),                                   # clean stream, no error
        (401, {}),                                   # auth
        (403, {"error": {"code": "model_disabled"}}),      # model gate
        (403, {"error": {"code": "model_access_denied"}}),
        (429, {"detail": "out of tokens"}),          # quota → TokenLimitExceeded upstream
        (400, {"error": {"type": "invalid_request_error"}}),
    ],
)
def test_classify_transient_not_retryable(status, body):
    assert classify_transient(status, body, provider="X") is None


# --------------------------------------------------------------------------- #
# provider mappers — the same input must map identically on both providers
# --------------------------------------------------------------------------- #

def _anthropic_map(status, body):
    from anton.core.llm.anthropic import _raise_for_status_error
    _raise_for_status_error(_FakeStatusError(status, body))


def _openai_map(status, body):
    from anton.core.llm.openai import _raise_for_status_error
    _raise_for_status_error(_FakeStatusError(status, body), "latest:sonnet")


@pytest.mark.parametrize("mapper", [_anthropic_map, _openai_map])
def test_mapper_midstream_200_is_transient_not_confusing_200(mapper):
    """The core ENG-673 fix: a mid-stream overload arriving inside an HTTP-200
    stream must become TransientProviderError, never 'Server returned 200'."""
    with pytest.raises(TransientProviderError) as ei:
        mapper(200, {"type": "error", "error": {"type": "overloaded_error"}})
    assert "200" not in str(ei.value)          # no misleading status echo
    assert "overloaded" in str(ei.value).lower()


@pytest.mark.parametrize("mapper", [_anthropic_map, _openai_map])
def test_mapper_5xx_is_transient(mapper):
    with pytest.raises(TransientProviderError):
        mapper(503, {})


@pytest.mark.parametrize("mapper", [_anthropic_map, _openai_map])
def test_mapper_quota_429_is_token_limit_not_transient(mapper):
    with pytest.raises(TokenLimitExceeded):
        mapper(429, {"detail": "You have run out of tokens."})


@pytest.mark.parametrize("mapper", [_anthropic_map, _openai_map])
def test_mapper_401_stays_connectionerror_fail_fast(mapper):
    # 401 must NOT be transient — retrying resends the same bad key forever.
    with pytest.raises(ConnectionError) as ei:
        mapper(401, {})
    assert not isinstance(ei.value, TransientProviderError)


def test_openai_mapper_model_gate_still_model_unavailable():
    # ENG-598 behavior must be preserved: structured 403 → ModelUnavailableError.
    with pytest.raises(ModelUnavailableError):
        _openai_map(403, {"error": {"code": "model_access_denied"}})


# --------------------------------------------------------------------------- #
# session backoff helpers
# --------------------------------------------------------------------------- #

def _delay(attempt, retry_after=None):
    from anton.core.session import ChatSession
    return ChatSession._transient_backoff_delay(attempt, retry_after)


def test_backoff_cadence_grows_and_jitters():
    # Monotone-ish base cadence ~2 → ~10 → ~18, each within its ±20% jitter band.
    assert 1.6 <= _delay(0) <= 2.4
    assert 8.0 <= _delay(1) <= 12.0
    assert 14.4 <= _delay(2) <= 21.6
    # Beyond the table, clamps to the last step (still jittered).
    assert 14.4 <= _delay(9) <= 21.6


def test_backoff_honors_retry_after():
    assert _delay(0, retry_after=5) == 5.0
    assert _delay(0, retry_after=999) == 30.0   # capped at the budget


async def test_backoff_sleep_returns_false_on_full_delay():
    from anton.core.session import ChatSession
    ns = types.SimpleNamespace(_cancel_event=asyncio.Event())
    assert await ChatSession._backoff_sleep(ns, 0.01) is False


async def test_backoff_sleep_wakes_on_cancel():
    from anton.core.session import ChatSession
    ev = asyncio.Event()
    ns = types.SimpleNamespace(_cancel_event=ev)
    ev.set()  # already cancelled → must return immediately, not wait out the delay
    assert await asyncio.wait_for(ChatSession._backoff_sleep(ns, 30.0), timeout=1.0) is True


def test_provider_overloaded_error_carries_model_and_provider():
    exc = ProviderOverloadedError("boom", provider="Anthropic", model="latest:sonnet")
    assert exc.code == "provider_overloaded"
    assert exc.provider == "Anthropic"
    assert exc.model == "latest:sonnet"
    assert isinstance(exc, ConnectionError)  # legacy call sites still catch it


# --------------------------------------------------------------------------- #
# session turn loop — recovery, budget exhaustion, no-replay (idempotency)
# --------------------------------------------------------------------------- #

from unittest.mock import AsyncMock, MagicMock, patch  # noqa: E402

import httpx  # noqa: E402
import openai  # noqa: E402

from anton.chat import ChatSession  # noqa: E402
from anton.core.llm.openai import OpenAIProvider  # noqa: E402
from anton.core.llm.provider import StreamComplete, StreamTextDelta  # noqa: E402
from anton.core.session import ChatSessionConfig  # noqa: E402
from tests.conftest import make_mock_llm  # noqa: E402


# --------------------------------------------------------------------------- #
# provider-level truncation — #1 fix regression guard
# --------------------------------------------------------------------------- #

def _chunk(content=None, finish_reason=None):
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = None
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason
    ch = MagicMock()
    ch.choices = [choice]
    ch.usage = None
    return ch


async def _run_openai_stream(chunks):
    async def _aiter():
        for c in chunks:
            yield c
    with patch("anton.core.llm.openai.openai") as mock_openai:
        client = AsyncMock()
        mock_openai.AsyncOpenAI.return_value = client
        client.chat.completions.create = AsyncMock(return_value=_aiter())
        prov = OpenAIProvider(api_key="k")
        return [
            ev async for ev in prov.stream(
                model="latest:sonnet", system="s",
                messages=[{"role": "user", "content": "hi"}],
            )
        ]


async def test_stream_content_without_finish_reason_passes_through():
    # #1 regression: a COMPLETE answer with no finish_reason (common on
    # OpenAI-compatible endpoints) must complete, NOT be discarded as truncated.
    events = await _run_openai_stream([_chunk(content="Hello "), _chunk(content="world")])
    completes = [e for e in events if isinstance(e, StreamComplete)]
    assert len(completes) == 1
    assert completes[0].response.content == "Hello world"


async def test_stream_empty_without_finish_reason_is_truncated():
    # Only the truly-empty no-terminal-marker stream is a genuine truncation.
    with pytest.raises(TransientProviderError) as ei:
        await _run_openai_stream([])
    assert ei.value.code == "truncated_stream"
    # Deliberate carve-out (ENG-673): an empty-from-start 200 is a broken-endpoint
    # signal, not a mid-incident blip — it fails fast rather than looping the
    # backoff budget. Genuine mid-incident silence arrives as a connection drop /
    # read timeout / SSE error, which DO back off (tested above/below).
    assert ei.value.session_backoff is False


# --------------------------------------------------------------------------- #
# mid-stream vs request-time boundary — the ENG-673 session_backoff flag
# (Sam's review): a failure AFTER the 200 was never SDK-retried → session backs
# off; a failure DURING establishment was already SDK-retried → fail fast.
# Drive a REAL OpenAIProvider whose client we replace, so the genuine openai.*
# exception classes match the provider's except clauses (no module patching).
# --------------------------------------------------------------------------- #

def _req() -> httpx.Request:
    return httpx.Request("POST", "http://mock/v1/chat/completions")


async def _drain_openai_with_create(create_mock) -> TransientProviderError:
    prov = OpenAIProvider(api_key="k")
    prov._client = AsyncMock()
    prov._client.chat.completions.create = create_mock
    with pytest.raises(TransientProviderError) as ei:
        _ = [
            ev async for ev in prov.stream(
                model="latest:sonnet", system="s",
                messages=[{"role": "user", "content": "hi"}],
            )
        ]
    return ei.value


async def test_stream_connection_drop_during_establishment_is_fail_fast():
    # create() itself raises → the SDK already retried at the transport layer.
    exc = await _drain_openai_with_create(
        AsyncMock(side_effect=openai.APIConnectionError(request=_req()))
    )
    assert exc.code == "connection_error"
    assert exc.session_backoff is False


async def test_stream_connection_drop_midstream_backs_off():
    # A 200 was established (first chunk arrived) then the connection dropped —
    # the SDK never retried this, so the session must.
    async def _aiter():
        yield _chunk(content="partial")
        raise openai.APIConnectionError(request=_req())

    exc = await _drain_openai_with_create(AsyncMock(return_value=_aiter()))
    assert exc.code == "connection_error"
    assert exc.session_backoff is True


async def test_stream_midstream_bare_apierror_is_classified_and_backs_off():
    # The OpenAI-shaped gap: a mid-stream SSE `error` surfaces as a bare
    # openai.APIError (no status_code, body type at top level), which is NOT an
    # APIStatusError. It must still be caught, classified from the body, and
    # backed off — otherwise it escapes as an opaque generic error (ENG-673).
    async def _aiter():
        yield _chunk(content="Hel")
        raise openai.APIError(
            "overloaded", request=_req(),
            body={"type": "server_error", "message": "overloaded"},
        )

    exc = await _drain_openai_with_create(AsyncMock(return_value=_aiter()))
    assert exc.session_backoff is True
    assert exc.code in ("server_error", "stream_error")


def _session() -> ChatSession:
    s = ChatSession(ChatSessionConfig(llm_client=make_mock_llm()))
    s._llm.planning_model = "latest:sonnet"
    return s


def _stream_that_fails_then_succeeds(fail_n: int, calls: dict):
    """Async-generator factory: raise a transient error `fail_n` times, then
    yield a successful text delta."""
    async def _gen(user_msg):
        calls["n"] += 1
        if calls["n"] <= fail_n:
            raise TransientProviderError(
                "Anthropic is momentarily overloaded.",
                provider="Anthropic", code="overloaded_error",
            )
        yield StreamTextDelta(text="recovered-and-done")
    return _gen


async def test_turn_recovers_after_transient_retries():
    s = _session()
    calls = {"n": 0}
    s._stream_and_handle_tools = _stream_that_fails_then_succeeds(2, calls)
    s._backoff_sleep = AsyncMock(return_value=False)  # don't really sleep

    events = [e async for e in s.turn_stream("build me a dashboard")]

    text = "".join(e.text for e in events if isinstance(e, StreamTextDelta))
    assert "recovered-and-done" in text          # turn completed
    assert calls["n"] == 3                        # 2 failures + 1 success
    assert s._backoff_sleep.await_count == 2      # backed off before each retry


async def test_turn_transient_retry_does_not_inject_recovery_note():
    """Idempotency/no-replay: a transient blip must NOT inject a 'you errored,
    change approach' SYSTEM note (which would alter behavior / prompt a replay);
    it simply retries the same step from unchanged history."""
    s = _session()
    calls = {"n": 0}
    s._stream_and_handle_tools = _stream_that_fails_then_succeeds(2, calls)
    s._backoff_sleep = AsyncMock(return_value=False)

    _ = [e async for e in s.turn_stream("do it")]

    joined = json.dumps(s._history)
    assert "An error interrupted execution" not in joined
    assert "transient provider error" not in joined  # seal reason not injected either


async def test_turn_exhausts_budget_to_provider_overloaded():
    s = _session()
    s._transient_budget_s = 0.05   # exhaust almost immediately
    calls = {"n": 0}

    async def _always_fail(user_msg):
        calls["n"] += 1
        raise TransientProviderError(
            "Anthropic is momentarily overloaded.",
            provider="Anthropic", code="overloaded_error",
        )
        yield  # pragma: no cover  (makes this an async generator)

    s._stream_and_handle_tools = _always_fail

    with pytest.raises(ProviderOverloadedError) as ei:
        _ = [e async for e in s.turn_stream("do it")]

    assert ei.value.code == "provider_overloaded"
    assert ei.value.provider == "Anthropic"
    assert ei.value.model == "latest:sonnet"


async def test_provider_overloaded_names_the_failing_model_not_planning():
    # ENG-673 #4: the card must name the model that actually failed (which may be
    # the coding model), not always the session's planning model.
    s = _session()  # planning_model = latest:sonnet
    s._transient_budget_s = 0.05

    async def _always_fail(user_msg):
        raise TransientProviderError(
            "overloaded", provider="Anthropic", code="overloaded_error", model="latest:haiku",
        )
        yield  # pragma: no cover

    s._stream_and_handle_tools = _always_fail
    with pytest.raises(ProviderOverloadedError) as ei:
        _ = [e async for e in s.turn_stream("do it")]
    assert ei.value.model == "latest:haiku"   # the failing model, not planning


async def test_request_time_transient_does_not_tell_model_to_adjust_approach():
    # ENG-673 #6: a request-time provider blip (session_backoff=False) recovers
    # via the count-based path, but must NOT inject "adjust your approach" — that
    # misattributes a service hiccup to the model.
    s = _session()
    calls = {"n": 0}

    async def _gen(user_msg):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TransientProviderError(
                "Server returned 500.", provider="X", code="http_500", session_backoff=False,
            )
        yield StreamTextDelta(text="ok done")

    s._stream_and_handle_tools = _gen
    events = [e async for e in s.turn_stream("do it")]

    assert "ok done" in "".join(e.text for e in events if isinstance(e, StreamTextDelta))
    history = json.dumps(s._history).lower()
    assert "transient service issue" in history
    assert "adjust your approach" not in history


def test_classify_transient_propagates_model():
    err = classify_transient(200, {"error": {"type": "overloaded_error"}}, provider="X", model="latest:opus")
    assert err.model == "latest:opus"


async def test_turn_transient_backoff_cancels_on_stop():
    """User-stop during backoff aborts immediately rather than waiting out the
    incident — and stops CLEANLY (like a normal stop), not with a provider-error
    card surfaced to the user."""
    s = _session()
    calls = {"n": 0}
    s._stream_and_handle_tools = _stream_that_fails_then_succeeds(5, calls)
    s._backoff_sleep = AsyncMock(return_value=True)  # cancelled during backoff

    events = [e async for e in s.turn_stream("do it")]  # no exception raised

    assert calls["n"] == 1                    # failed once, then cancelled — no further retries
    assert s._backoff_sleep.await_count == 1
    # Clean cancel: no transient-error prose leaks into the transcript.
    text = "".join(e.text for e in events if isinstance(e, StreamTextDelta))
    assert "overloaded" not in text.lower()
