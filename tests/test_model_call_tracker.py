"""ModelCallTracker: which model calls a turn has open, as a host polls it.

A host keeps a quiet turn alive only while ``snapshot()`` reports a call
waiting on the provider. So the snapshot must be set for every LLMClient entry
point while it waits, and None everywhere a turn can hang for another reason:
a stream parked at its ``yield`` while the consumer works, an open question,
and any time after the turn closed the tracker.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, ValidationError

from anton.core.llm import liveness
from anton.core.llm.client import LLMClient
from anton.core.llm.liveness import (
    MODEL_WAIT_PHASE,
    ModelCallSnapshot,
    ModelCallTracker,
)
from anton.core.llm.provider import (
    LLMProvider,
    LLMResponse,
    StreamComplete,
    StreamTextDelta,
    ToolCall,
    Usage,
)


class _Verdict(BaseModel):
    status: str


class _GatedProvider(LLMProvider):
    """Every call waits on ``release`` before it answers."""

    name = "gated"

    def __init__(self) -> None:
        self.release = asyncio.Event()

    async def complete(self, *, tools=None, tool_choice=None, **_kwargs) -> LLMResponse:
        await self.release.wait()
        calls = []
        if tool_choice:
            calls = [ToolCall(id="t1", name=tool_choice["name"], input={"status": "ok"})]
        return LLMResponse(
            content="ok", tool_calls=calls,
            usage=Usage(input_tokens=1, output_tokens=1), stop_reason="end_turn",
        )

    async def stream(self, **_kwargs):
        await self.release.wait()
        yield StreamTextDelta(text="ok")
        yield StreamComplete(response=LLMResponse(
            content="ok", usage=Usage(input_tokens=1, output_tokens=1), stop_reason="end_turn",
        ))


def _client(provider: LLMProvider) -> LLMClient:
    return LLMClient(
        planning_provider=provider, planning_model="planner",
        coding_provider=provider, coding_model="coder",
    )


async def _drain(stream) -> None:
    async for _event in stream:
        pass


_ENTRY_POINTS = {
    "plan": lambda c: c.plan(system="s", messages=[]),
    "plan_stream": lambda c: _drain(c.plan_stream(system="s", messages=[])),
    "code": lambda c: c.code(system="s", messages=[]),
    "code_stream": lambda c: _drain(c.code_stream(system="s", messages=[])),
    "summarize": lambda c: c.summarize(system="s", messages=[]),
    "generate_object": lambda c: c.generate_object(_Verdict, system="s", messages=[]),
    "generate_object_code": lambda c: c.generate_object_code(_Verdict, system="s", messages=[]),
}


class _FailingProvider(_GatedProvider):
    """Every call waits on ``release``, then fails the way a gateway 500 does."""

    async def complete(self, **_kwargs) -> LLMResponse:
        await self.release.wait()
        raise RuntimeError("gateway 500")

    async def stream(self, **_kwargs):
        await self.release.wait()
        raise RuntimeError("gateway 500")
        yield  # pragma: no cover


def test_the_names_hosts_read_stay_put():
    """cowork-server reads these by attribute and by class name, never by
    import, so a rename here passes its CI and silently stops its ticks or
    its no-response card: ModelWaitTicker._snapshot and model_wait_sse
    (cowork/streaming/liveness.py), and is_model_timeout_error and the
    ``ModelCallTimeoutError`` row of _REMOTE_TYPE_MAPPINGS
    (cowork/handlers/turn_errors.py). The cloud pod's _model_wait_line reads
    the same snapshot fields."""
    from anton.core.llm.provider import ModelCallTimeoutError

    assert MODEL_WAIT_PHASE == "model_wait"
    assert {"message", "open_for_s"} <= set(ModelCallSnapshot.model_fields)
    assert ModelCallTimeoutError.__name__ == "ModelCallTimeoutError"
    assert ModelCallTimeoutError.code == "model_timeout"
    exc = ModelCallTimeoutError(role="planning", model="m", idle_timeout_s=600.0)
    assert not hasattr(exc, "response") and not hasattr(exc, "request")


@pytest.mark.parametrize("entry", list(_ENTRY_POINTS))
async def test_every_entry_point_fails_at_once_once_the_turn_latched(entry):
    """After one call ran out its deadline, every later call in the turn fails
    as soon as it is issued, without reaching the provider."""
    from anton.core.llm.provider import ModelCallTimeoutError

    client = _client(_GatedProvider())  # would wait forever if it were reached
    tracker = ModelCallTracker()
    liveness.arm_turn_tracker(tracker)
    tracker.record_expiry(exc=ModelCallTimeoutError(role="router", model="m", idle_timeout_s=600.0))

    with pytest.raises(ModelCallTimeoutError):
        await asyncio.wait_for(_ENTRY_POINTS[entry](client), timeout=1)
    assert tracker._calls == [], f"{entry} registered a call after the latch"


@pytest.mark.parametrize("entry", list(_ENTRY_POINTS))
async def test_a_call_that_fails_closes_too(entry):
    """Side paths (compaction, the rule filter) swallow a failed call. A call
    left open would keep reporting a wait, so a tool that hangs afterwards
    would get still-working lines instead of meeting the host's bounds."""
    provider = _FailingProvider()
    client = _client(provider)
    tracker = ModelCallTracker()
    liveness.arm_turn_tracker(tracker)

    task = asyncio.ensure_future(_ENTRY_POINTS[entry](client))
    await asyncio.sleep(0.02)
    assert tracker.snapshot() is not None, f"{entry} reported no wait"

    provider.release.set()
    with pytest.raises(RuntimeError):
        await asyncio.wait_for(task, timeout=2)

    assert tracker._calls == [], f"{entry} left its failed call open"
    assert tracker.snapshot() is None


@pytest.mark.parametrize("entry", list(_ENTRY_POINTS))
async def test_every_entry_point_reports_while_it_waits_and_closes_after(entry):
    provider = _GatedProvider()
    client = _client(provider)
    tracker = ModelCallTracker()
    liveness.arm_turn_tracker(tracker)

    task = asyncio.ensure_future(_ENTRY_POINTS[entry](client))
    await asyncio.sleep(0.02)

    snap = tracker.snapshot()
    assert isinstance(snap, ModelCallSnapshot), f"{entry} reported no wait"
    assert snap.message.startswith("Waiting for the model (")

    provider.release.set()
    await asyncio.wait_for(task, timeout=2)

    assert tracker.snapshot() is None, f"{entry} left its call open"
    assert tracker._calls == []


async def test_a_stream_parked_at_its_yield_is_not_waiting_on_the_provider():
    """Between events the stream waits on its consumer (a tool round, a host
    writing to its wire). Reporting a wait then would keep a hung consumer
    alive."""
    provider = _GatedProvider()
    provider.release.set()
    client = _client(provider)
    tracker = ModelCallTracker()
    liveness.arm_turn_tracker(tracker)

    stream = client.plan_stream(system="s", messages=[])
    first = await anext(stream)
    assert isinstance(first, StreamTextDelta)

    assert tracker.snapshot() is None
    assert len(tracker._calls) == 1, "the call is still open, just not waiting"
    await stream.aclose()
    assert tracker._calls == []


def test_no_snapshot_while_a_question_is_open():
    tracker = ModelCallTracker()
    call = tracker.open(role="planning", idle_timeout_s=600.0)
    call.awaiting = True
    assert tracker.snapshot() is not None

    tracker.question_opened()
    assert tracker.snapshot() is None

    tracker.question_closed()
    assert tracker.snapshot() is not None


def test_a_closed_tracker_reports_and_opens_nothing():
    tracker = ModelCallTracker()
    call = tracker.open(role="planning", idle_timeout_s=600.0)
    call.awaiting = True

    tracker.close()

    assert tracker.snapshot() is None
    assert tracker.open(role="planning", idle_timeout_s=600.0) is None


def test_the_oldest_waiting_call_wins(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(liveness.time, "monotonic", lambda: now[0])
    tracker = ModelCallTracker()
    old = tracker.open(role="planning", idle_timeout_s=600.0)
    now[0] = 130.0
    young = tracker.open(role="coding", idle_timeout_s=600.0)
    old.awaiting = young.awaiting = True
    now[0] = 260.0

    snap = tracker.snapshot()

    assert snap.role == "planning"
    assert snap.open_for_s == pytest.approx(160.0)
    assert snap.message == "Waiting for the model (2m 40s)"


def test_recent_unrelayed_output_reads_as_still_writing(monkeypatch):
    """Tool-argument deltas and generate_artifact text never reach the wire,
    so the line says the model is writing rather than waiting."""
    now = [0.0]
    monkeypatch.setattr(liveness.time, "monotonic", lambda: now[0])
    tracker = ModelCallTracker()
    call = tracker.open(role="planning", idle_timeout_s=600.0)
    call.awaiting = True
    now[0] = 50.0
    tracker.output(call=call)
    now[0] = 60.0

    assert tracker.snapshot().message == "The model is still writing (1m 0s)"

    now[0] = 50.0 + liveness.MODEL_WAIT_TICK_S + 1
    assert tracker.snapshot().message.startswith("Waiting for the model (")


def test_short_waits_read_in_seconds(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(liveness.time, "monotonic", lambda: now[0])
    tracker = ModelCallTracker()
    tracker.open(role="planning", idle_timeout_s=None).awaiting = True
    now[0] = 45.4

    assert tracker.snapshot().message == "Waiting for the model (45s)"


def test_a_call_past_its_deadline_reports_nothing(monkeypatch):
    """Only a call still inside its idle timeout counts as waiting."""
    now = [0.0]
    monkeypatch.setattr(liveness.time, "monotonic", lambda: now[0])
    tracker = ModelCallTracker()
    tracker.open(role="planning", idle_timeout_s=10.0).awaiting = True
    now[0] = 10.5

    assert tracker.snapshot() is None


def test_the_snapshot_is_a_frozen_model():
    snap = ModelCallSnapshot(role="planning", open_for_s=1.0, quiet_for_s=1.0, message="m")
    with pytest.raises(ValidationError):
        snap.message = "changed"


def test_the_phase_string_is_the_cross_repo_contract():
    assert MODEL_WAIT_PHASE == "model_wait"
    assert liveness.MODEL_WAIT_TICK_S == 20.0
