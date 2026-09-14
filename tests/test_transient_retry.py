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
    # backs off. Request-time errors (real 5xx/529) were SDK-retried already
    # → fail fast, no extra session budget.
    #
    # The velocity 429 is the documented exception and has its own test below
    # (ENG-1537): the SDK's retries fire seconds apart, which is the right
    # answer for a 5xx that recovers instantly and useless against a
    # per-minute token ceiling.
    assert classify_transient(
        200, {"error": {"type": "overloaded_error"}}, provider="X"
    ).session_backoff is True
    assert classify_transient(503, {}, provider="X").session_backoff is False
    assert classify_transient(
        529, {"error": {"type": "overloaded_error"}}, provider="X"
    ).session_backoff is False


def test_an_unconfirmed_429_does_not_earn_a_session_wait():
    # ENG-1537 finding 5. The session wait needs POSITIVE evidence of a velocity
    # limit, not merely the absence of billing carriers. Both fail-fast guards
    # are string-exact, so a provider quota in an unrecognised dialect — Gemini
    # sends an INTEGER `code` with status RESOURCE_EXHAUSTED — slips past them.
    # Waiting there burns the whole budget on a daily quota that resets at
    # midnight, then tells the user it isn't a credits problem.
    gemini_quota = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}}
    t = classify_transient(429, gemini_quota, provider="Gemini", retry_after=30.0)
    assert t.session_backoff is False
    assert t.retry_after is None  # not carried, so nothing downstream waits on it
    # A bare 429 from any BYOK provider keeps the pre-ENG-1537 behaviour too.
    assert classify_transient(429, {}, provider="X", retry_after=1.0).session_backoff is False


def test_velocity_429_backs_off_in_session_and_carries_retry_after():
    # ENG-1537. A velocity rate-limit is the one request-time status the SESSION
    # must wait on: it is the only failure class where waiting is both necessary
    # and sufficient, and the server hands us the interval. With
    # session_backoff=False this fell to the count-based path, which re-issued
    # the request twice with NO delay and a recovery note appended each time —
    # told "too many tokens per minute", anton immediately sent more.
    t = classify_transient(429, {}, provider="X", retry_after=30.0, velocity_confirmed=True)
    assert isinstance(t, TransientProviderError)
    assert t.code == "rate_limited"
    assert t.session_backoff is True
    assert t.retry_after == 30.0


def test_velocity_429_without_a_hint_still_backs_off():
    # No Retry-After (a provider that omits it) → still waits, on the jittered
    # curve rather than a named interval.
    t = classify_transient(429, {}, provider="X", velocity_confirmed=True)
    assert t.session_backoff is True
    assert t.retry_after is None


@pytest.mark.parametrize(
    "status,body",
    [
        # The M3 gate's out-of-credits denials, in both body dialects.
        (429, {"error": {"code": "included_allowance_exhausted"}}),
        (429, {"code": "included_allowance_exhausted"}),
        (402, {"error": {"code": "wallet_empty"}}),
        (402, {"code": "wallet_empty"}),
        # OpenAI's own quota dialect.
        (429, {"error": {"code": "insufficient_quota"}}),
        (429, {"code": "insufficient_quota"}),
    ],
)
def test_billing_denials_never_enter_the_retry_loop(status, body):
    # ENG-1169 regression guard, re-asserted because ENG-1537 made the sibling
    # 429 branch retryable. A spent allowance and an empty wallet share the 429
    # status with the velocity limit but are PERMANENT for the identical
    # request: the allowance resets monthly, so waiting can never help and a
    # wait would only delay the out-of-credits card the user needs.
    # A Retry-After is passed deliberately — even with a hint present, these
    # must not become retryable.
    assert classify_transient(status, body, provider="X", retry_after=30.0) is None


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

import httpx2 as httpx  # noqa: E402
import openai  # noqa: E402

from anton.chat import ChatSession  # noqa: E402
from anton.core.llm.openai import OpenAIProvider  # noqa: E402
from anton.core.llm.provider import StreamComplete, StreamTaskProgress, StreamTextDelta  # noqa: E402
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


# ── Session-side wiring (ENG-1537 review finding 3) ────────────────────────
# The 85-line session.py half of ENG-1537 reverted with the suite green — the
# wait itself is covered via `session_backoff`, but the budgets, the cap, the
# progress event and the exhaustion code were not. Each of these was watched
# failing against the merge-base.

def test_rate_limits_get_their_own_budget_and_cap():
    # Separate from the incident budget so neither starves the other in a turn,
    # and a larger hint cap because a velocity hint is a real end time whereas
    # an incident's is a guess. Reverting the cap silently clamps a 45s hint to
    # 30s — retrying into a window that is still open.
    from anton.core.session import ChatSession

    assert ChatSession._rate_limit_budget_s == 90.0
    assert ChatSession._transient_budget_s == 30.0  # unchanged
    assert ChatSession._rate_limit_max_delay_s == 60.0


def test_a_hint_never_retries_faster_than_the_curve():
    # ENG-1537 finding 1. Obeying a small hint verbatim ignores `attempt`, and
    # the gateway's small hints are the COMMON case: an RPM denial always
    # computes exactly 1, and a concurrency denial sends a flat 1 that its own
    # config calls a fake end time. Taken literally inside a 90s budget that is
    # ~90 retries of a single denial.
    from anton.core.session import ChatSession

    first = ChatSession._transient_backoff_delay(0, 1.0, 60.0)
    second = ChatSession._transient_backoff_delay(1, 1.0, 60.0)
    assert first >= 1.5, first          # the curve's floor wins over the 1s hint
    assert second > first               # and it still escalates
    # A hint LARGER than the curve is still honoured — that's the TPM case.
    assert ChatSession._transient_backoff_delay(0, 45.0, 60.0) == 45.0
    # And never above the cap.
    assert ChatSession._transient_backoff_delay(0, 3600.0, 60.0) == 60.0


def test_incident_backoff_is_untouched_by_the_rate_limit_cap():
    # "No change to 5xx incident handling" — the incident path keeps its own
    # 30s clamp even though the rate-limit path now passes 60.
    from anton.core.session import ChatSession

    assert ChatSession._transient_backoff_delay(0, 3600.0) == 30.0


def test_exhaustion_carries_the_rate_limited_code_and_the_interval():
    # cowork-server maps this code to the wait-and-retry card instead of the
    # out-of-credits one, and the card gates its Retry on the interval — so
    # both must survive the raise.
    from anton.core.llm.provider import ProviderOverloadedError

    exc = ProviderOverloadedError(
        "x", provider="P", model="sonnet", code="rate_limited", retry_after=30.0,
    )
    assert exc.code == "rate_limited"
    assert exc.retry_after == 30.0
    # The incident default is unchanged and carries no interval.
    plain = ProviderOverloadedError("y")
    assert plain.code == "provider_overloaded"
    assert plain.retry_after is None


async def test_the_wait_emits_the_progress_event_the_other_two_repos_key_on():
    # cowork-server's never-throttle set and PHASE_LABELS, and cowork's reducer
    # branch, all key on this exact phase string. A silent wait is
    # indistinguishable from a hang, which is the failure this event prevents.
    #
    # Asserted as an EMITTED EVENT, not a source literal: the previous version
    # grepped `inspect.getsource` and survived moving the yield into dead code
    # — the event never fired and the whole suite stayed green.
    s = _session()
    s._backoff_sleep = AsyncMock(return_value=False)
    calls = {"n": 0}

    async def _fail_then_succeed(user_msg):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TransientProviderError(
                "The model provider is rate-limiting requests.",
                provider="P", code="rate_limited",
                session_backoff=True, retry_after=30.0,
            )
        yield StreamTextDelta(text="ok")

    s._stream_and_handle_tools = _fail_then_succeed
    events = [e async for e in s.turn_stream("do it")]

    phases = [e.phase for e in events if isinstance(e, StreamTaskProgress)]
    assert "rate_limited" in phases, phases
    notice = next(e for e in events
                  if isinstance(e, StreamTaskProgress) and e.phase == "rate_limited")
    assert "30" in notice.message           # names the wait
    assert notice.eta_seconds == 30.0       # and carries it structurally


async def test_a_hint_above_the_cap_cards_immediately_and_names_the_interval():
    """ENG-1537 How #3, and the item most likely to regress.

    A hint longer than we are willing to sleep must NOT be absorbed: card at
    once and name the real number, rather than sleeping the cap twice and then
    telling the user to wait "a moment". This is pinned specifically because
    the branch was absent for a whole revision while both the ticket and the PR
    body asserted it shipped — and disabling it left the entire suite green.
    """
    s = _session()
    sleeps = []

    async def _record_sleep(delay):
        sleeps.append(delay)
        return False

    s._backoff_sleep = _record_sleep

    async def _always_rate_limited(user_msg):
        raise TransientProviderError(
            "The model provider is rate-limiting requests.",
            provider="The model provider", code="rate_limited",
            session_backoff=True, retry_after=3600.0,
        )
        yield  # pragma: no cover  (async generator)

    s._stream_and_handle_tools = _always_rate_limited

    with pytest.raises(ProviderOverloadedError) as ei:
        _ = [e async for e in s.turn_stream("do it")]

    assert sleeps == [], f"must not sleep at all, slept {sleeps}"
    assert ei.value.code == "rate_limited"
    assert ei.value.retry_after == 3600.0
    assert "3600s" in str(ei.value)
    # And never the out-of-credits framing.
    assert "credits" in str(ei.value).lower()  # as a denial: "isn't a credits problem"
    assert "add credits" not in str(ei.value).lower()


async def test_a_hint_within_the_cap_still_waits_rather_than_carding():
    """The complement — otherwise the branch above could swallow every case."""
    s = _session()
    sleeps = []

    async def _record_sleep(delay):
        sleeps.append(delay)
        return False

    s._backoff_sleep = _record_sleep
    calls = {"n": 0}

    async def _fail_then_succeed(user_msg):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TransientProviderError(
                "The model provider is rate-limiting requests.",
                provider="The model provider", code="rate_limited",
                session_backoff=True, retry_after=30.0,
            )
        yield StreamTextDelta("done")

    s._stream_and_handle_tools = _fail_then_succeed

    _ = [e async for e in s.turn_stream("do it")]

    assert sleeps == [30.0], sleeps          # the hint, honoured
    assert calls["n"] == 2                   # and the SAME step resumed


async def test_the_wait_budget_is_finite_across_multiple_denials():
    """The loop's ONLY termination guarantee is `_rate_limit_slept += delay`.

    Nothing pinned it: deleting that line makes every velocity 429 that does
    not clear retry forever — sleeping tens of seconds between attempts, with
    no exhaustion card and no way out — and the suite stayed green. The two
    adjacent `if _rate_limited:` blocks actively invite that merge.

    A single-denial test cannot catch it; this needs repeated denials so the
    accumulator has to advance.
    """
    s = _session()
    sleeps = []

    async def _record(delay):
        sleeps.append(delay)
        # Bail loudly rather than hanging. With the accumulator removed the
        # loop is genuinely infinite and `_backoff_sleep` is mocked, so it
        # spins at full speed — a plain assertion at the end would never be
        # reached and CI would sit on a 10-minute timeout with no diagnosis
        # (observed while mutation-testing this very test).
        if len(sleeps) > 20:
            raise AssertionError(
                "the rate-limit wait never terminates — `_rate_limit_slept` "
                "is not advancing, so the budget can never be spent"
            )
        return False

    s._backoff_sleep = _record

    async def _always(user_msg):
        raise TransientProviderError(
            "The model provider is rate-limiting requests.",
            provider="P", code="rate_limited", session_backoff=True, retry_after=30.0,
        )
        yield  # pragma: no cover

    s._stream_and_handle_tools = _always

    with pytest.raises(ProviderOverloadedError) as ei:
        _ = [e async for e in s.turn_stream("do it")]

    assert ei.value.code == "rate_limited"
    # Terminates, and within the stated budget rather than merely "eventually".
    assert sum(sleeps) <= s._rate_limit_budget_s + 0.01, sleeps
    assert len(sleeps) <= 6, sleeps


async def test_a_rate_limit_wait_does_not_advance_the_incident_curve():
    """The split attempt index — the headline of the commit that added it, and
    unpinned until now: reverting to the shared counter kept the suite green.

    An interleaved turn is the common shape when a gateway is degraded. With
    one counter, the incident that follows a rate-limit wait starts partway
    down a curve it never walked.
    """
    s = _session()
    sleeps = []

    async def _record(delay):
        sleeps.append(delay)
        return False

    s._backoff_sleep = _record
    calls = {"n": 0}

    async def _rate_then_incident(user_msg):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TransientProviderError(
                "rate-limiting", provider="P", code="rate_limited",
                session_backoff=True, retry_after=45.0,
            )
        if calls["n"] == 2:
            raise TransientProviderError(
                "overloaded", provider="P", code="overloaded_error",
                session_backoff=True,
            )
        yield StreamTextDelta(text="ok")

    s._stream_and_handle_tools = _rate_then_incident
    _ = [e async for e in s.turn_stream("do it")]

    assert len(sleeps) == 2, sleeps
    assert sleeps[0] == 45.0                 # the rate limit honoured its hint
    # The incident starts at ITS OWN curve position 0 (~2s ±20%), not position
    # 1 (~10s) as a shared counter would give.
    assert sleeps[1] < 5.0, sleeps
