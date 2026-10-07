"""The pod's still-working line while a model call is silent.

cowork-server ends a cloud turn whose reply stream goes quiet, and the UI ends
any turn that sends no `data:` event for 300 s. A model thinking in silence
sends nothing, so the pod's heartbeat task writes a `progress` line with phase
`model_wait` when the session reports a call waiting on the provider and the
wire has been quiet for `MODEL_WAIT_TICK_S`. A hung tool reports no wait, so it
still goes quiet and still ends.
"""

from __future__ import annotations

import asyncio
import time

import pytest

import anton.cloud_turn.__main__ as m
from anton.core.llm.provider import (
    StreamTaskProgress,
    StreamTextDelta,
    StreamToolResult,
    StreamToolUseEnd,
    StreamToolUseStart,
)

RAW = '{"protocol_version":1,"conversation_id":"c","input":"hi"}'


class _Snapshot:
    """Stands in for ModelCallSnapshot: the pod reads only these fields."""

    def __init__(self, message: str, open_for_s: float) -> None:
        self.role = "planning"
        self.message = message
        self.open_for_s = open_for_s
        self.quiet_for_s = open_for_s


class _Calls:
    def __init__(self, snapshot) -> None:
        self._snapshot = snapshot
        self.polls = 0

    def snapshot(self):
        self.polls += 1
        if isinstance(self._snapshot, Exception):
            raise self._snapshot
        return self._snapshot


class _SilentSession:
    """A turn that stays silent on the wire for ``quiet_s`` and then answers,
    while ``model_calls`` reports what the test gives it."""

    def __init__(self, calls, *, quiet_s: float = 0.4, lead=None) -> None:
        self.model_calls = calls
        self._quiet_s = quiet_s
        self._lead = lead or []

    async def turn_stream(self, user_input, **kwargs):
        for event in self._lead:
            yield event
        await asyncio.sleep(self._quiet_s)
        yield StreamTextDelta(text="done")

    def close(self): ...


@pytest.fixture
def fast_ticks(monkeypatch):
    monkeypatch.setenv("ANTON_CLOUD_TURN_HEARTBEAT_SECONDS", "0.02")
    # raising=False: a pod without the tick still runs these and fails on
    # what it writes, not on the patch.
    monkeypatch.setattr(m, "MODEL_WAIT_TICK_S", 0.05, raising=False)


async def _run(session) -> list[dict]:
    events: list[dict] = []
    await m.stream_turn(RAW, emit=events.append, session_builder=lambda req: session)
    return events


def _waits(events: list[dict]) -> list[dict]:
    return [e for e in events if e.get("kind") == "progress" and e.get("phase") == "model_wait"]


async def test_a_silent_model_call_puts_wait_lines_on_the_wire(fast_ticks):
    calls = _Calls(_Snapshot("Waiting for the model (2m 40s)", 160.0))

    events = await _run(_SilentSession(calls))

    waits = _waits(events)
    assert waits, f"no model_wait line in {[e['kind'] for e in events]}"
    assert waits[0] == {
        "kind": "progress", "phase": "model_wait",
        "message": "Waiting for the model (2m 40s)", "eta_seconds": 160.0,
        "id": None, "ok": None,
    }
    assert events[-1] == {"kind": "turn_completed"}


async def test_wait_lines_are_spaced_by_the_quiet_window(fast_ticks):
    """Each wait line resets the window, so they come no faster than the
    window even though the heartbeat ticks four times as often."""
    calls = _Calls(_Snapshot("Waiting for the model (1s)", 1.0))
    timed: list[tuple[float, dict]] = []

    def _timed_emit(event: dict) -> None:
        timed.append((time.monotonic(), event))

    await m.stream_turn(
        RAW, emit=_timed_emit,
        session_builder=lambda req: _SilentSession(calls, quiet_s=0.5),
    )

    wait_times = [t for t, e in timed if e.get("phase") == "model_wait"]
    assert len(wait_times) >= 2, f"{len(wait_times)} wait lines"
    gaps = [later - earlier for earlier, later in zip(wait_times, wait_times[1:])]
    assert min(gaps) >= m.MODEL_WAIT_TICK_S * 0.9, f"wait lines {gaps} s apart"


async def test_no_wait_line_when_nothing_is_waiting_on_the_model(fast_ticks):
    """A hung tool or cell reports no wait, so the turn stays quiet and the
    host's idle bounds still end it."""
    calls = _Calls(None)

    events = await _run(_SilentSession(calls))

    assert calls.polls > 0, "the heartbeat never asked"
    assert not _waits(events)
    assert any(e["kind"] == "heartbeat" for e in events)


class _QuietLeadQuiet:
    """Quiet, then one ordinary line, then quiet again, then the answer."""

    def __init__(self, calls, *, lead) -> None:
        self.model_calls = calls
        self._lead = lead

    async def turn_stream(self, user_input, **kwargs):
        await asyncio.sleep(0.1)
        yield self._lead
        await asyncio.sleep(0.3)
        yield StreamTextDelta(text="done")

    def close(self): ...


@pytest.mark.parametrize("lead", [
    StreamTaskProgress(phase="reasoning_start", message=""),
    StreamTextDelta(text="partial"),
    StreamToolUseStart(id="t1", name="probe"),
    StreamToolUseEnd(id="t1"),
    StreamToolResult(name="probe", content="ok", id="t1"),
], ids=["progress", "delta", "tool_start", "tool_end", "tool_result"])
async def test_any_other_line_restarts_the_quiet_window(fast_ticks, monkeypatch, lead):
    """Any other line already tells the host the turn is alive, so the next
    wait line waits a full window after it. Each lead is its own case, so a
    line written past the wire clock fails here."""
    monkeypatch.setattr(m, "MODEL_WAIT_TICK_S", 0.15, raising=False)
    calls = _Calls(_Snapshot("Waiting for the model (1s)", 1.0))
    timed: list[tuple[float, dict]] = []

    def _timed_emit(event: dict) -> None:
        timed.append((time.monotonic(), event))

    await m.stream_turn(
        RAW, emit=_timed_emit,
        session_builder=lambda req: _QuietLeadQuiet(calls, lead=lead),
    )

    lines = [(t, e) for t, e in timed if e["kind"] not in ("heartbeat", "turn_completed")]
    waits = [t for t, e in lines if e.get("phase") == "model_wait"]
    others = [t for t, e in lines if e.get("phase") != "model_wait"]
    assert waits, "no wait line at all, so the spacing below proves nothing"
    assert others, "the lead never reached the wire"
    window = m.MODEL_WAIT_TICK_S * 0.9
    early = [w - o for w in waits for o in others if 0 <= w - o < window]
    assert not early, f"wait lines {early} s after another line"


async def test_a_broken_snapshot_does_not_kill_the_heartbeat(fast_ticks):
    """A dead heartbeat trips the controller's stall window, so a failing
    check must cost only the wait line."""
    calls = _Calls(RuntimeError("boom"))

    events = await _run(_SilentSession(calls))

    beats = [i for i, e in enumerate(events) if e["kind"] == "heartbeat"]
    assert len(beats) >= 5
    assert events[-1] == {"kind": "turn_completed"}


async def test_a_session_without_model_calls_still_works(fast_ticks):
    """An older session (or a test double) has no tracker: heartbeats only."""

    class _Old:
        async def turn_stream(self, user_input, **kwargs):
            await asyncio.sleep(0.2)
            yield StreamTextDelta(text="done")

        def close(self): ...

    events = await _run(_Old())

    assert not _waits(events)
    assert events[-1] == {"kind": "turn_completed"}


async def test_a_real_session_puts_wait_lines_out_and_ends_on_the_deadline(fast_ticks):
    """The tests above use a stand-in snapshot. This drives a real ChatSession
    whose first model call never answers, so a rename of the snapshot's fields
    or of the deadline error fails here, on what the pod writes."""
    from tests.test_session_model_calls import _ScriptedProvider, _session

    async def _silent():
        await asyncio.Event().wait()
        yield  # pragma: no cover

    session = _session(_ScriptedProvider(streams=[_silent]))
    session._llm.model_call_idle_timeout_s = 0.3

    events = await asyncio.wait_for(_run(session), timeout=5)

    waits = _waits(events)
    assert waits, f"no model_wait line in {[e['kind'] for e in events]}"
    assert waits[0]["message"].startswith("Waiting for the model (")
    assert isinstance(waits[0]["eta_seconds"], float)
    assert events[-1]["kind"] == "turn_failed"
    assert events[-1]["error"].startswith("ModelCallTimeoutError: The model sent no output for ")


async def test_the_deadline_error_reaches_the_wire_by_its_type_name(fast_ticks):
    """cowork-server maps a remote failure by the class name `_scrub` keeps."""
    from anton.core.llm.provider import ModelCallTimeoutError

    class _TimesOut:
        model_calls = None

        async def turn_stream(self, user_input, **kwargs):
            raise ModelCallTimeoutError(idle_timeout_s=600.0)
            yield  # pragma: no cover

        def close(self): ...

    events = await _run(_TimesOut())

    assert events[-1] == {
        "kind": "turn_failed",
        "error": "ModelCallTimeoutError: The model sent no output for 10 minutes, "
        "so the call was stopped.",
    }
