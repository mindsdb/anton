"""ENG-1361 — curated provider failures must fail the turn, never become prose.

Three guarantees:
  * the count-based retry path has a terminal that produces the
    `provider_overloaded` card (it previously had none, ever);
  * no member of `CURATED_PROVIDER_ERRORS` can be wrapped into assistant text
    on the summarize path;
  * the curated set stays exhaustive as new exception classes are added — the
    check that fails at authoring time rather than in production.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from unittest.mock import AsyncMock, patch

import pytest

from anton.chat import ChatSession
from anton.core.llm import provider as provider_mod
from anton.core.llm.provider import (
    CURATED_PROVIDER_ERRORS,
    PROVIDER_FAILURE_KINDS,
    ContentValidationError,
    ContextOverflowError,
    EndpointConfigurationError,
    ModelUnavailableError,
    ProviderAuthError,
    ProviderOverloadedError,
    StreamTextDelta,
    StructuredOutputError,
    TokenLimitExceeded,
    TransientProviderError,
    classify_transient,
    provider_failure_kind,
)
from anton.core.session import ChatSessionConfig
from tests.conftest import make_mock_llm


def _session() -> ChatSession:
    # A session_id, deliberately: a turn with none and zero LLM calls is dropped
    # before the analytics sink as script traffic (ENG-1692), and the telemetry
    # tests below assert on the emitted event.
    s = ChatSession(ChatSessionConfig(llm_client=make_mock_llm(), session_id="conv-1361"))
    s._llm.planning_model = "latest:sonnet"
    return s


def _fields(send_event_mock) -> dict:
    """ENG-1361's properties on the last emitted `turn_completed`."""
    assert send_event_mock.called, "no turn_completed event emitted"
    return send_event_mock.call_args.kwargs


def _unreachable(**kw):
    """The failure from the ENG-1361 incident: request-time, SDK already retried."""
    kw.setdefault("code", "connection_error")
    kw.setdefault("provider", "The model provider")
    kw.setdefault("model", "latest:sonnet")
    return TransientProviderError(
        "Could not reach the model provider — check your connection or try again in a moment.",
        session_backoff=False, **kw,
    )


def _always_raise(exc_factory):
    async def _gen(user_msg):
        raise exc_factory()
        yield  # pragma: no cover  (makes this an async generator)
    return _gen


def _summarize_raises(exc_factory):
    async def _gen(*a, **kw):
        raise exc_factory()
        yield  # pragma: no cover
    return _gen


# --------------------------------------------------------------------------- #
# Fix 1 — the count path now has a card terminal
# --------------------------------------------------------------------------- #

async def test_count_exhausted_transient_raises_the_card_not_prose():
    """The reported bug: provider unreachable at request time, still unreachable
    when we ask the model to explain it. Used to end as assistant text telling
    the user to rephrase; must now fail the turn so the card renders."""
    s = _session()
    s._stream_and_handle_tools = _always_raise(_unreachable)
    s._llm.plan_stream = _summarize_raises(_unreachable)

    with pytest.raises(ProviderOverloadedError) as ei:
        _ = [e async for e in s.turn_stream("research local AI coding models")]

    assert ei.value.code == "provider_overloaded"
    assert "rephrase" not in str(ei.value).lower()


async def test_the_card_names_the_failing_model_not_the_planning_one():
    """cowork-server maps this model back to its provider to decide
    `reconnectable` (responses.py), so mis-attribution suppresses the BYOK
    failover nudge. The two candidate sources must differ or the assertion is
    vacuous — which it was until the #433 review."""
    s = _session()                      # planning_model = latest:sonnet
    s._stream_and_handle_tools = _always_raise(
        lambda: _unreachable(model="latest:haiku")   # the CODING model failed
    )
    s._llm.plan_stream = _summarize_raises(_unreachable)

    with pytest.raises(ProviderOverloadedError) as ei:
        _ = [e async for e in s.turn_stream("do it")]

    assert ei.value.model == "latest:haiku"   # not the session's planning model


async def test_the_card_terminal_does_not_need_the_summarize_call_to_fail():
    """Even when the provider recovers in time to answer the summarize call, the
    turn must card — otherwise the same failure has two different UIs depending
    on timing."""
    s = _session()
    s._stream_and_handle_tools = _always_raise(_unreachable)

    async def _summarize_succeeds(*a, **kw):
        yield StreamTextDelta(text="here is what I managed to do")

    s._llm.plan_stream = _summarize_succeeds

    with pytest.raises(ProviderOverloadedError):
        _ = [e async for e in s.turn_stream("do it")]


async def test_an_unconfirmed_429_makes_no_claim_about_credits_or_waiting():
    """A 429 on the COUNT path is one `classify_transient` could NOT confirm was
    a velocity limit — the population its docstring warns about, where a daily
    quota in an unrecognised dialect "would otherwise spend the whole budget
    waiting out a daily quota that resets at midnight — then be told it is not a
    credits problem." So this branch must promise nothing about waiting and deny
    nothing about credits. Guards against reintroducing that copy (#433 review)."""
    s = _session()
    s._stream_and_handle_tools = _always_raise(
        lambda: _unreachable(code="rate_limited", status_code=429)
    )
    s._llm.plan_stream = _summarize_raises(lambda: _unreachable(code="rate_limited"))

    with pytest.raises(ProviderOverloadedError) as ei:
        _ = [e async for e in s.turn_stream("do it")]

    body = str(ei.value).lower()
    assert "credits" not in body      # we do not know that it isn't one
    assert "should work" not in body  # nor that waiting will help
    assert "incident" not in body     # nor that anything is broken
    # Falls to the generic branch, so the card is the plain overloaded one.
    assert ei.value.code == "provider_overloaded"


async def test_the_card_keeps_antons_own_classification():
    """The typed message says WHICH failure this was ("returned 500"), which
    ENG-673 put there on purpose. Converting to the card must add the attempt
    count, not replace the diagnosis with a generic "could not be reached"."""
    s = _session()
    s._stream_and_handle_tools = _always_raise(
        lambda: TransientProviderError(
            "The model provider returned 500.", provider="The model provider",
            code="http_500", session_backoff=False, model="latest:sonnet",
            status_code=500,
        )
    )
    s._llm.plan_stream = _summarize_raises(_unreachable)

    with pytest.raises(ProviderOverloadedError) as ei:
        _ = [e async for e in s.turn_stream("do it")]

    assert "returned 500" in str(ei.value)
    assert "after 3 attempts" in str(ei.value)


async def test_the_card_copy_never_doubles_a_period():
    """The fingerprint of the bug this ticket came from: concatenating a curated
    message that already ends in '.' with a template that adds its own."""
    s = _session()
    s._stream_and_handle_tools = _always_raise(_unreachable)   # message ends in '.'
    s._llm.plan_stream = _summarize_raises(_unreachable)

    with pytest.raises(ProviderOverloadedError) as ei:
        _ = [e async for e in s.turn_stream("do it")]

    assert ".." not in str(ei.value)


async def test_the_raise_leaves_no_dangling_system_message_in_history():
    """Raising AFTER appending the summarize prompt would poison the next turn
    with an orphan "The task has failed N times"."""
    s = _session()
    s._stream_and_handle_tools = _always_raise(_unreachable)
    s._llm.plan_stream = _summarize_raises(_unreachable)

    with pytest.raises(ProviderOverloadedError):
        _ = [e async for e in s.turn_stream("do it")]

    assert "The task has failed" not in json.dumps(s._history)


async def test_a_non_transient_failure_still_summarizes():
    """Fix 1 is scoped to transients. A 400 has no card to route to and the model
    can genuinely explain it, so that path must be unchanged."""
    s = _session()

    class _BadRequest(Exception):
        pass

    s._stream_and_handle_tools = _always_raise(lambda: _BadRequest("bad tool schema"))

    async def _summarize_succeeds(*a, **kw):
        yield StreamTextDelta(text="I hit a bad request; here is what happened")

    s._llm.plan_stream = _summarize_succeeds

    with patch("anton.analytics.send_event") as send:
        events = [e async for e in s.turn_stream("do it")]

    text = "".join(e.text for e in events if isinstance(e, StreamTextDelta))
    assert "here is what happened" in text
    assert _fields(send)["ended_by"] == "retry_exhausted"


# --------------------------------------------------------------------------- #
# Fix 2 — no curated failure becomes prose on the summarize path
# --------------------------------------------------------------------------- #

_CURATED_SAMPLES = {
    ContextOverflowError: lambda: ContextOverflowError("too long"),
    TokenLimitExceeded: lambda: TokenLimitExceeded("out of credits"),
    ProviderAuthError: lambda: ProviderAuthError("Invalid API key"),
    StructuredOutputError: lambda: StructuredOutputError("no tool call"),
    TransientProviderError: _unreachable,
    ProviderOverloadedError: lambda: ProviderOverloadedError("overloaded"),
    ModelUnavailableError: lambda: ModelUnavailableError(
        "no such model", code="model_not_found", model="x"
    ),
    ContentValidationError: lambda: ContentValidationError("bad image block"),
    EndpointConfigurationError: lambda: EndpointConfigurationError("bad base url"),
}


def test_every_curated_error_has_a_sample():
    """Keeps the parametrised test below honest as the set grows."""
    assert set(_CURATED_SAMPLES) == set(CURATED_PROVIDER_ERRORS)


@pytest.mark.parametrize("exc_type", list(CURATED_PROVIDER_ERRORS), ids=lambda t: t.__name__)
async def test_a_curated_error_on_the_summarize_path_fails_the_turn(exc_type):
    """Table-driven over the whole set on purpose. Written per-type, this test
    would have passed for the two members the old allowlist happened to hold
    while three others were missing — which is how each was found by a user
    instead of by CI."""
    s = _session()

    class _Boom(Exception):
        pass

    # A NON-transient first failure, so we reach the summarize call rather than
    # Fix 1's card terminal — this is testing the wrap-up guard specifically.
    s._stream_and_handle_tools = _always_raise(lambda: _Boom("something broke"))
    s._llm.plan_stream = _summarize_raises(_CURATED_SAMPLES[exc_type])

    with pytest.raises(exc_type):
        _ = [e async for e in s.turn_stream("do it")]


async def test_an_uncurated_error_still_becomes_prose_without_rephrase_advice():
    """The fallback still exists for genuinely unexpected failures — it just no
    longer claims the user's wording was the problem."""
    s = _session()

    class _Boom(Exception):
        pass

    s._stream_and_handle_tools = _always_raise(lambda: _Boom("first failure"))
    s._llm.plan_stream = _summarize_raises(lambda: _Boom("and again"))

    events = [e async for e in s.turn_stream("do it")]
    text = "".join(e.text for e in events if isinstance(e, StreamTextDelta))
    assert "An unexpected error occurred" in text
    assert "rephrase" not in text.lower()


# --------------------------------------------------------------------------- #
# Fix 2 — the set stays exhaustive as classes are added
# --------------------------------------------------------------------------- #

# Exception classes DEFINED in provider.py that deliberately stay uncurated.
# Empty today; an entry here is a decision with a reason, not an oversight.
_DELIBERATELY_UNCURATED: frozenset[str] = frozenset()


def test_every_exception_defined_in_a_provider_module_is_triaged():
    """Fails on the NEXT exception class, at authoring time.

    Covers EVERY module that defines a provider-facing exception, not just
    `provider.py` — scoping it to one file was the gap found reviewing #433:
    adding `class ProviderRegionBlockedError(ConnectionError)` to `openai.py`
    left the whole suite green and the type uncurated, which is exactly the
    drift this test exists to stop. The `__module__` check keeps each module
    responsible only for what it defines, so an import doesn't double-count.
    """
    from anton.core.llm import anthropic as anthropic_mod
    from anton.core.llm import openai as openai_mod

    defined = {
        name
        for mod in (provider_mod, openai_mod, anthropic_mod)
        for name, obj in vars(mod).items()
        if inspect.isclass(obj)
        and issubclass(obj, BaseException)
        and obj.__module__ == mod.__name__
    }
    curated = {t.__name__ for t in CURATED_PROVIDER_ERRORS}
    assert defined == curated | _DELIBERATELY_UNCURATED, (
        "New exception class in a provider module — add it to "
        "CURATED_PROVIDER_ERRORS, or to _DELIBERATELY_UNCURATED with a reason."
    )


def test_builtin_connectionerror_is_not_curated():
    """It is the base of half the set AND of unrelated socket failures, so
    curating it would silently curate everything. Typing the status mapper's
    catch-all is ENG-1283."""
    assert ConnectionError not in CURATED_PROVIDER_ERRORS
    assert not isinstance(ConnectionError("boom"), CURATED_PROVIDER_ERRORS)


# --------------------------------------------------------------------------- #
# classify_transient must PLUMB the status, not just accept one
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "status,body,expected",
    [
        (503, {}, 503),                                              # 5xx branch
        (503, {"error": {"type": "service_unavailable"}}, 503),       # body-type branch
        (429, {}, 429),                                              # rate-limit branch
        (200, {"error": {"type": "overloaded_error"}}, 200),          # mid-stream
    ],
)
def test_classify_transient_carries_the_status_it_classified_from(status, body, expected):
    """`provider_failure_kind`/`provider_http_status` have exactly one production
    writer — this function. Every field test above hand-builds the exception, so
    a mutant deleting the plumbing survived the whole suite until the #433
    review, and `test_status_error_mapper.py` never asserts `.status_code`."""
    exc = classify_transient(status, body, provider="p", model="m")
    assert exc is not None
    assert exc.status_code == expected


# --------------------------------------------------------------------------- #
# Fix 5 — the terminal stays measurable
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "code,expected_kind",
    [
        ("overloaded_error", "overload_signal"),
        ("overloaded", "overload_signal"),
        ("api_error", "overload_signal"),
        ("server_error", "overload_signal"),
        ("service_unavailable", "overload_signal"),
        ("stream_error", "bad_response"),
        ("truncated_stream", "bad_response"),
        ("empty_response", "bad_response"),
        ("rate_limited", "rate_limit"),
        ("connection_error", "connection_failure"),
        ("http_500", "http_5xx"),
        ("http_503", "http_5xx"),
    ],
)
def test_every_code_anton_mints_maps_to_a_kind(code, expected_kind):
    """Exhaustive over the codes anton actually constructs. An unmapped code
    would silently emit "" and hide a whole population."""
    assert provider_failure_kind(code) == expected_kind
    assert expected_kind in PROVIDER_FAILURE_KINDS


@pytest.mark.parametrize("code", [None, "", "something_new", "http_404", "http_abc"])
def test_an_unknown_code_is_blank_not_guessed(code):
    """Blank is a prompt to extend the vocabulary; a wrong value is invisible.
    `http_404` specifically must NOT read as a server fault."""
    assert provider_failure_kind(code) == ""


async def test_the_count_terminal_records_reason_kind_and_status():
    s = _session()
    s._stream_and_handle_tools = _always_raise(
        lambda: _unreachable(code="http_503", status_code=503)
    )
    s._llm.plan_stream = _summarize_raises(_unreachable)

    with patch("anton.analytics.send_event") as send:
        with pytest.raises(ProviderOverloadedError):
            _ = [e async for e in s.turn_stream("do it")]

    k = _fields(send)
    assert k["retry_terminal_reason"] == "request_attempt_limit"
    assert k["provider_failure_kind"] == "http_5xx"
    assert k["provider_http_status"] == "503"


async def test_a_connection_failure_records_no_status_rather_than_a_placeholder():
    s = _session()
    s._stream_and_handle_tools = _always_raise(_unreachable)
    s._llm.plan_stream = _summarize_raises(_unreachable)

    with patch("anton.analytics.send_event") as send:
        with pytest.raises(ProviderOverloadedError):
            _ = [e async for e in s.turn_stream("do it")]

    k = _fields(send)
    assert k["provider_failure_kind"] == "connection_failure"
    # OMITTED, not "" — an empty string would sort and group beside real
    # statuses and make the column unusable.
    assert "provider_http_status" not in k


async def test_the_time_budget_terminal_is_distinguishable_from_the_count_one():
    """The whole point of the field: after Fix 1 both terminals raise the same
    exception type, so nothing else can tell them apart."""
    s = _session()
    s._transient_budget_s = 0.05
    s._stream_and_handle_tools = _always_raise(
        lambda: TransientProviderError(
            "momentarily overloaded", provider="Anthropic",
            code="overloaded_error", model="latest:sonnet",
        )
    )

    with patch("anton.analytics.send_event") as send:
        with pytest.raises(ProviderOverloadedError):
            _ = [e async for e in s.turn_stream("do it")]

    k = _fields(send)
    assert k["retry_terminal_reason"] == "provider_recovery_timeout"
    assert k["provider_failure_kind"] == "overload_signal"


async def test_a_long_retry_after_records_the_decline_not_an_exhaustion():
    """Nothing ran out here — we declined to wait. `retry_terminal_reason` is
    named for termination precisely so this value isn't a lie."""
    s = _session()
    s._stream_and_handle_tools = _always_raise(
        lambda: TransientProviderError(
            "rate-limiting requests", provider="MindsHub", code="rate_limited",
            retry_after=600.0, session_backoff=True, model="latest:sonnet",
            status_code=429,
        )
    )

    with patch("anton.analytics.send_event") as send:
        with pytest.raises(ProviderOverloadedError) as ei:
            _ = [e async for e in s.turn_stream("do it")]

    assert ei.value.code == "rate_limited"
    k = _fields(send)
    assert k["retry_terminal_reason"] == "rate_limit_wait_too_long"
    assert k["provider_failure_kind"] == "rate_limit"
    assert k["provider_http_status"] == "429"


async def test_a_non_transient_terminal_still_books_its_http_status():
    """`provider_http_status` is INDEPENDENT of `provider_failure_kind`: the kind
    classifies provider failures and is empty here, but the status is a plain
    fact about the exception and is the only signal these turns carry. Pins the
    asymmetry so nobody "tidies" it away — self-review finding 1."""
    import httpx
    import openai

    s = _session()
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(400, request=req, json={"error": {"code": "invalid_request_error"}})

    s._stream_and_handle_tools = _always_raise(
        lambda: openai.BadRequestError("bad tool schema", response=resp, body=None)
    )

    async def _summarize_succeeds(*a, **kw):
        yield StreamTextDelta(text="explaining the 400")

    s._llm.plan_stream = _summarize_succeeds

    with patch("anton.analytics.send_event") as send:
        _ = [e async for e in s.turn_stream("do it")]

    k = _fields(send)
    assert k["ended_by"] == "retry_exhausted"
    assert k["retry_terminal_reason"] == "request_attempt_limit"
    assert k["provider_failure_kind"] == ""     # not a classified PROVIDER failure
    assert k["provider_http_status"] == "400"   # ...but the status is still real


async def test_a_stop_after_a_retry_terminal_books_no_terminal_reason():
    """A user who has sat through failed retries is exactly the user who presses
    Stop. Without the clearing, a plain cancel would appear in an error-cause
    breakdown carrying a terminal it never reached — self-review finding 3."""
    s = _session()

    async def _fail_then_hang(user_msg):
        raise _unreachable()
        yield  # pragma: no cover

    s._stream_and_handle_tools = _fail_then_hang

    async def _summarize_hangs(*a, **kw):
        await asyncio.sleep(30)   # cancelled here, after the terminal was stamped
        yield StreamTextDelta(text="never reached")  # pragma: no cover

    # A NON-transient first failure would reach summarize; use one so the stamp
    # happens and the cancel lands after it.
    class _Boom(Exception):
        pass

    s._stream_and_handle_tools = _always_raise(lambda: _Boom("first"))
    s._llm.plan_stream = _summarize_hangs

    with patch("anton.analytics.send_event") as send:
        async def _consume():
            async for _ in s.turn_stream("do it"):
                pass

        task = asyncio.create_task(_consume())
        await asyncio.sleep(0.2)   # let it reach the hanging summarize call
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    k = _fields(send)
    assert k["ended_by"] == "cancelled"
    assert k["retry_terminal_reason"] == ""
    assert k["provider_failure_kind"] == ""
    assert "provider_http_status" not in k


async def test_the_summarize_prompt_does_not_teach_the_model_to_say_rephrase():
    """Half of fix 3, and the half that governs the path SURVIVING this change:
    a non-transient failure still summarizes, and the model writes that reply.
    If the SYSTEM prompt asks for "rephrase / simplify", the advice comes back
    out of the model's mouth instead of the template's. Untested until the #433
    review."""
    s = _session()

    class _Boom(Exception):
        pass

    s._stream_and_handle_tools = _always_raise(lambda: _Boom("nope"))

    async def _summarize_succeeds(*a, **kw):
        yield StreamTextDelta(text="ok")

    s._llm.plan_stream = _summarize_succeeds
    _ = [e async for e in s.turn_stream("do it")]

    prompt = json.dumps(s._history).lower()
    assert "the task has failed" in prompt          # we did reach the wrap-up
    assert "rephrase" not in prompt
    assert "simplify the request" not in prompt


async def test_the_rate_limit_wait_budget_terminal_is_labelled():
    """The one row of the five-terminal table with no telemetry test."""
    s = _session()
    s._rate_limit_budget_s = 0.05
    s._stream_and_handle_tools = _always_raise(
        lambda: TransientProviderError(
            "rate-limiting requests", provider="MindsHub", code="rate_limited",
            session_backoff=True, model="latest:sonnet", status_code=429,
        )
    )
    s._backoff_sleep = AsyncMock(return_value=False)

    with patch("anton.analytics.send_event") as send:
        with pytest.raises(ProviderOverloadedError):
            _ = [e async for e in s.turn_stream("do it")]

    k = _fields(send)
    assert k["retry_terminal_reason"] == "rate_limit_wait_limit"
    assert k["provider_failure_kind"] == "rate_limit"
    assert k["provider_http_status"] == "429"


async def test_a_turn_that_never_retried_leaves_the_fields_empty():
    s = _session()

    async def _ok(user_msg):
        yield StreamTextDelta(text="done")

    s._stream_and_handle_tools = _ok
    with patch("anton.analytics.send_event") as send:
        _ = [e async for e in s.turn_stream("do it")]

    k = _fields(send)
    assert k["retry_terminal_reason"] == ""
    assert k["provider_failure_kind"] == ""
    assert "provider_http_status" not in k
