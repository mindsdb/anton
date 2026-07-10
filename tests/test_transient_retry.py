"""ENG-673 — transient mid-stream provider errors: classification + backoff.

Covers the anton-side of the fix without a live provider:
  * `classify_transient` (Arm A) — the shared body/status classifier.
  * both provider mappers (`anthropic`/`openai` `_raise_for_status_error`) — the
    right typed exception for each failure, incl. the mid-stream HTTP-200 case.
  * the session backoff helpers — cadence, `retry_after`, and cancel-awareness.

The end-to-end turn behavior (retry recovers, budget exhausts to a
`provider_overloaded` card, completed tools aren't re-run) is exercised by the
mock-provider harness in `test_transient_retry_e2e` / the fixture under
`tests/fixtures/`.
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

from unittest.mock import AsyncMock  # noqa: E402

from anton.chat import ChatSession  # noqa: E402
from anton.core.llm.provider import StreamTextDelta  # noqa: E402
from anton.core.session import ChatSessionConfig  # noqa: E402
from tests.conftest import make_mock_llm  # noqa: E402


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


async def test_turn_transient_backoff_cancels_on_stop():
    """User-stop during backoff aborts immediately rather than waiting out the
    incident: _backoff_sleep reports cancellation and the turn stops retrying."""
    s = _session()
    calls = {"n": 0}
    s._stream_and_handle_tools = _stream_that_fails_then_succeeds(5, calls)
    s._backoff_sleep = AsyncMock(return_value=True)  # cancelled during backoff

    with pytest.raises(TransientProviderError):
        _ = [e async for e in s.turn_stream("do it")]

    assert calls["n"] == 1                    # failed once, then cancelled — no further retries
    assert s._backoff_sleep.await_count == 1
