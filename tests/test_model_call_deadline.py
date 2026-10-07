"""The idle deadline LLMClient puts on every model call.

A model call that sends nothing for the deadline ends with
``ModelCallTimeoutError`` (code ``model_timeout``). Output resets the
deadline, so a slow model that keeps writing is never cut. A non-streamed call
is bounded over its whole duration. An outside cancel stays a cancel.

The deadline is set as a plain attribute on the client, as the session does,
so these run in well under a second. The error type is checked by name and
read from the module lazily, so a client without the deadline fails these on
behavior (the guard below trips) rather than on an import.
"""

from __future__ import annotations

import asyncio
import time
from typing import NamedTuple

import pytest

from anton.core.llm import provider as provider_mod
from anton.core.llm.client import LLMClient
from anton.core.llm.provider import (
    LLMProvider,
    LLMResponse,
    ProviderAuthError,
    StreamComplete,
    StreamReasoningDelta,
    StreamTextDelta,
    StreamToolUseDelta,
    StreamToolUseStart,
    Usage,
)
from pydantic import BaseModel

DEADLINE_S = 0.3
# Wall-clock guard on every call: well past the deadline, far short of a hang.
GUARD_S = 2.0


def _response(text: str = "ok") -> LLMResponse:
    return LLMResponse(
        content=text, tool_calls=[], usage=Usage(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
    )


class _ScriptedProvider(LLMProvider):
    """A provider whose stream and complete run the coroutines a test gives it."""

    name = "scripted"

    def __init__(self, *, stream_events=None, complete=None) -> None:
        self.stream_events = stream_events
        self.complete_fn = complete
        self.complete_calls = 0
        self.stream_calls = 0

    async def complete(self, **kwargs) -> LLMResponse:
        self.complete_calls += 1
        return await self.complete_fn(**kwargs)

    async def stream(self, **kwargs):
        self.stream_calls += 1
        async for event in self.stream_events(**kwargs):
            yield event


async def _hang(**_kwargs):
    await asyncio.Event().wait()


async def _silent_stream(**_kwargs):
    await asyncio.Event().wait()
    yield StreamTextDelta(text="never")  # pragma: no cover


def _client(provider: LLMProvider, *, idle_s: float = DEADLINE_S) -> LLMClient:
    client = LLMClient(
        planning_provider=provider, planning_model="planner",
        coding_provider=provider, coding_model="coder",
    )
    client.model_call_idle_timeout_s = idle_s
    return client


async def _drain(stream) -> list:
    return [event async for event in stream]


class _Outcome(NamedTuple):
    exc: BaseException | None
    elapsed: float


async def _outcome(awaitable) -> _Outcome:
    """Run under the guard; report what it raised (or None) and how long it took."""
    started = time.monotonic()
    try:
        await asyncio.wait_for(awaitable, timeout=GUARD_S)
    except Exception as exc:  # noqa: BLE001 - the type is the assertion
        return _Outcome(exc=exc, elapsed=time.monotonic() - started)
    return _Outcome(exc=None, elapsed=time.monotonic() - started)


def _assert_deadline_error(exc: BaseException | None, elapsed: float, *, role: str) -> None:
    assert exc is not None, "the silent call returned"
    assert type(exc).__name__ == "ModelCallTimeoutError", (
        f"expected the deadline error, got {type(exc).__name__}: {exc}"
    )
    assert getattr(exc, "code", None) == "model_timeout"
    assert getattr(exc, "role", None) == role
    assert elapsed < DEADLINE_S + 0.2, f"took {elapsed:.2f}s against a {DEADLINE_S}s deadline"


class _Verdict(BaseModel):
    status: str


@pytest.mark.parametrize(
    "entry,role",
    [
        ("plan", "planning"),
        ("code", "coding"),
        ("summarize", "router"),
        ("generate_object", "planning"),
        ("generate_object_code", "coding"),
    ],
)
async def test_a_hung_non_streamed_call_ends_with_the_deadline_error(entry, role):
    client = _client(_ScriptedProvider(complete=_hang))
    if entry.startswith("generate_object"):
        call = getattr(client, entry)(_Verdict, system="s", messages=[])
    else:
        call = getattr(client, entry)(system="s", messages=[])

    exc, elapsed = await _outcome(call)

    _assert_deadline_error(exc, elapsed, role=role)


@pytest.mark.parametrize("entry,role", [("plan_stream", "planning"), ("code_stream", "coding")])
async def test_a_silent_stream_ends_with_the_deadline_error(entry, role):
    client = _client(_ScriptedProvider(stream_events=_silent_stream))

    exc, elapsed = await _outcome(_drain(getattr(client, entry)(system="s", messages=[])))

    _assert_deadline_error(exc, elapsed, role=role)


async def test_a_stream_silent_after_some_output_is_cut_too():
    """The deadline restarts on output, so a stream that writes and then goes
    quiet is cut one deadline after its last event."""

    async def _then_silent(**_kwargs):
        yield StreamTextDelta(text="partial")
        await asyncio.Event().wait()

    client = _client(_ScriptedProvider(stream_events=_then_silent))
    seen: list = []

    async def _consume():
        async for event in client.plan_stream(system="s", messages=[]):
            seen.append(event)

    exc, elapsed = await _outcome(_consume())

    _assert_deadline_error(exc, elapsed, role="planning")
    assert [e.text for e in seen] == ["partial"]


@pytest.mark.parametrize(
    "make_event",
    [
        lambda i: StreamTextDelta(text=f"t{i}"),
        lambda i: StreamReasoningDelta(text=f"r{i}"),
        lambda i: StreamToolUseDelta(id="call_1", json_delta=f'"{i}'),
    ],
    ids=["text", "reasoning", "tool_args"],
)
async def test_a_stream_that_keeps_writing_is_never_cut(make_event):
    """Output every 0.05 s for twice the deadline: nothing ends the call."""

    async def _steady(**_kwargs):
        yield StreamToolUseStart(id="call_1", name="scratchpad")
        for i in range(12):
            await asyncio.sleep(0.05)
            yield make_event(i)
        yield StreamComplete(response=_response())

    client = _client(_ScriptedProvider(stream_events=_steady))

    exc, elapsed = await _outcome(_drain(client.plan_stream(system="s", messages=[])))

    assert exc is None, f"a writing stream was cut: {exc!r}"
    assert elapsed > DEADLINE_S, "the stream must outlast one deadline for this to mean anything"


async def test_a_slow_consumer_is_not_charged_to_the_model():
    """The deadline never spans a yield: time the caller spends between events
    (a host writing to its wire) is not silence from the model."""

    async def _two_events(**_kwargs):
        yield StreamTextDelta(text="a")
        yield StreamComplete(response=_response())

    client = _client(_ScriptedProvider(stream_events=_two_events))

    async def _slow():
        async for _event in client.plan_stream(system="s", messages=[]):
            await asyncio.sleep(DEADLINE_S * 2)

    exc, _ = await _outcome(_slow())

    assert exc is None, f"consumer time was charged to the model: {exc!r}"


@pytest.mark.parametrize("entry", ["plan", "plan_stream", "code_stream"])
async def test_an_outside_cancel_stays_a_cancel(entry):
    """Stop, a watchdog or a shutdown cancels the task: that must stay a
    CancelledError, never become a deadline error the host shows as a card.
    The streams matter most: the main turn loop calls the model through them."""
    from anton.core.llm.liveness import ModelCallTracker, arm_turn_tracker

    client = _client(_ScriptedProvider(stream_events=_silent_stream, complete=_hang), idle_s=5.0)
    tracker = ModelCallTracker()
    arm_turn_tracker(tracker)
    calls = {
        "plan": lambda: client.plan(system="s", messages=[]),
        "plan_stream": lambda: _drain(client.plan_stream(system="s", messages=[])),
        "code_stream": lambda: _drain(client.code_stream(system="s", messages=[])),
    }
    task = asyncio.ensure_future(calls[entry]())
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=GUARD_S)
    tracker.raise_if_expired()  # a cancel must not latch a deadline


async def test_the_deadline_runs_across_the_auth_confirmation_retry():
    """The bound is on the whole call: a refused first attempt and a hung retry
    together still end at one deadline, not two. A per-attempt deadline would
    end at 0.35 + 0.5 s; the margin on both sides is at least 150 ms."""
    deadline_s = 0.5
    attempts = 0

    async def _refuse_then_hang(**_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            await asyncio.sleep(0.35)
            raise ProviderAuthError("Invalid API key")
        await asyncio.Event().wait()

    client = _client(_ScriptedProvider(complete=_refuse_then_hang), idle_s=deadline_s)

    exc, elapsed = await _outcome(client.plan(system="s", messages=[]))

    assert type(exc).__name__ == "ModelCallTimeoutError", repr(exc)
    assert attempts == 2
    assert elapsed < deadline_s + 0.2, f"took {elapsed:.2f}s: the deadline restarted on the retry"


async def test_zero_turns_the_deadline_off():
    async def _slow(**_kwargs):
        await asyncio.sleep(DEADLINE_S * 2)
        return _response("late")

    client = _client(_ScriptedProvider(complete=_slow), idle_s=0)

    exc, _ = await _outcome(client.plan(system="s", messages=[]))

    assert exc is None


async def test_the_message_names_the_deadline_in_words():
    from anton.core.llm.provider import ModelCallTimeoutError

    assert str(ModelCallTimeoutError(idle_timeout_s=600.0)) == (
        "The model sent no output for 10 minutes, so the call was stopped."
    )
    assert "1 second," in str(ModelCallTimeoutError(idle_timeout_s=1.0))
    assert "0.5 seconds" in str(ModelCallTimeoutError(idle_timeout_s=0.5))


async def test_a_deadline_latches_and_fails_the_next_call_at_once():
    """Side paths (compaction, tool dispatch) swallow the first expiry. The
    turn's tracker latches it, so the next call fails without waiting out a
    second deadline or reaching the provider."""
    from anton.core.llm.liveness import ModelCallTracker, arm_turn_tracker

    provider = _ScriptedProvider(complete=_hang)
    client = _client(provider)
    arm_turn_tracker(ModelCallTracker())

    first, _ = await _outcome(client.code(system="s", messages=[]))
    assert type(first).__name__ == "ModelCallTimeoutError"
    calls_before = provider.complete_calls

    second, elapsed = await _outcome(client.plan(system="s", messages=[]))

    assert type(second).__name__ == "ModelCallTimeoutError"
    assert second is not first, "each call gets its own exception"
    assert elapsed < 0.05
    assert provider.complete_calls == calls_before, "the latched call reached the provider"


async def test_a_closed_tracker_latches_nothing():
    """Work that outlives the turn must not be failed by that turn's expiry."""
    from anton.core.llm.liveness import ModelCallTracker, arm_turn_tracker

    client = _client(_ScriptedProvider(complete=_hang))
    tracker = ModelCallTracker()
    arm_turn_tracker(tracker)
    await _outcome(client.code(system="s", messages=[]))
    tracker.close()

    async def _ok(**_kwargs):
        return _response("fine")

    client._planning_provider.complete_fn = _ok
    exc, _ = await _outcome(client.plan(system="s", messages=[]))

    assert exc is None


def test_the_setting_reaches_the_client(monkeypatch):
    from anton.config.settings import AntonSettings

    monkeypatch.setenv("ANTON_MODEL_CALL_IDLE_TIMEOUT_S", "42")
    monkeypatch.setenv("ANTON_PLANNING_PROVIDER", "openai-compatible")
    monkeypatch.setenv("ANTON_CODING_PROVIDER", "openai-compatible")
    monkeypatch.setenv("ANTON_OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    settings = AntonSettings(_env_file=None, openai_api_key="x")

    client = LLMClient.from_settings(settings)

    assert client.model_call_idle_timeout_s == 42.0
    assert client._idle_timeout_for(max_tokens=8192) == 42.0


def test_the_default_is_ten_minutes():
    from anton.core.settings import CoreSettings

    assert CoreSettings().model_call_idle_timeout_s == 600.0
    client = LLMClient(
        planning_provider=_ScriptedProvider(), planning_model="p",
        coding_provider=_ScriptedProvider(), coding_model="c",
    )
    assert client._idle_timeout_for(max_tokens=8192) == 600.0


@pytest.mark.parametrize("value", ["nan", "inf"])
def test_a_deadline_that_is_not_a_finite_number_is_refused(monkeypatch, value):
    """NaN would fire every call at once and then fail building the message,
    so every turn would end on a ValueError instead of the deadline."""
    from pydantic import ValidationError

    from anton.core.settings import CoreSettings

    monkeypatch.setenv("ANTON_MODEL_CALL_IDLE_TIMEOUT_S", value)
    with pytest.raises(ValidationError):
        CoreSettings()


def test_a_nan_deadline_set_on_the_client_reads_as_off():
    """Hosts can set the attribute directly, past the settings validator."""
    client = LLMClient(
        planning_provider=_ScriptedProvider(), planning_model="p",
        coding_provider=_ScriptedProvider(), coding_model="c",
    )
    client.model_call_idle_timeout_s = float("nan")
    assert client._idle_timeout_for(max_tokens=8192) is None


def test_the_error_is_curated_and_not_a_network_blip():
    exc_type = getattr(provider_mod, "ModelCallTimeoutError", None)
    assert exc_type is not None
    assert exc_type in provider_mod.CURATED_PROVIDER_ERRORS
    # Not an OSError: the verifier's transient catch and generate_artifact's
    # stream-drop retry would otherwise read it as a retryable network blip.
    assert not issubclass(exc_type, OSError)
    assert provider_mod.provider_failure_kind("model_timeout") == "no_output"
