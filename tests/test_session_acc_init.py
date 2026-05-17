"""Wiring tests for the ACC integration in ChatSession.

These tests don't construct a full ChatSession (it requires a live LLM
client and many other dependencies). Instead they verify the contract
points where ChatSession and the ACC meet:

  - `_acc_observe` is a safe-emit wrapper (silent on unknown kinds,
    silent when the cortex is disabled).
  - `_schedule_acc_flush` drains the ACC buffer, converts each Lesson
    into an Engram with the kind the detector tagged, and routes them
    through `cortex.encode()`.

We exercise the methods by binding them onto a `SimpleNamespace` with
the minimum attributes they read. This is the same approach
test_session_skills_init.py uses for the skill-store contract — keeps
the tests fast and pinned to the *seam* under test instead of dragging
in workspaces, LLM clients, and storage backends.
"""

from __future__ import annotations

import asyncio
import types
from types import SimpleNamespace

import pytest

from anton.core.memory.acc import AnteriorCingulate
from anton.core.memory.base import Engram
from anton.core.session import ChatSession


class FakeCortex:
    """Minimal stand-in for Cortex used by _schedule_acc_flush.

    Records every batch of engrams passed to `encode()` so the test can
    inspect exactly what would land in long-term memory.
    """

    def __init__(self, mode: str = "autopilot"):
        self.mode = mode
        self.encoded: list[list[Engram]] = []
        self.global_hc = SimpleNamespace(recall_rules=lambda: "")

    async def encode(self, engrams: list[Engram]) -> list[str]:
        self.encoded.append(list(engrams))
        return [f"encoded:{e.kind}" for e in engrams]


def _make_session_stub(cortex: FakeCortex, *, acc_mode: str = "passive") -> SimpleNamespace:
    """Build the smallest object that satisfies _acc_observe,
    _schedule_acc_flush, and _acc_maybe_nudge. Binds the real
    ChatSession methods onto it so the test exercises production
    code, not a re-implementation."""
    stub = SimpleNamespace(
        _cortex=cortex,
        _acc=AnteriorCingulate(),  # default predicate accepts everything
        _acc_mode=acc_mode,
    )
    stub._acc_observe = types.MethodType(ChatSession._acc_observe, stub)
    stub._schedule_acc_flush = types.MethodType(ChatSession._schedule_acc_flush, stub)
    stub._acc_maybe_nudge = types.MethodType(ChatSession._acc_maybe_nudge, stub)
    return stub


class TestAccObserveWrapper:
    def test_silent_when_cortex_disabled(self):
        cortex = FakeCortex(mode="off")
        s = _make_session_stub(cortex)
        s._acc_observe("scratchpad_call", {"name": "x", "code_len": 10})
        # Mode=="off" → emit suppressed entirely so we don't accumulate
        # events the flush will throw away anyway.
        assert s._acc.events == ()

    def test_silent_on_unknown_kind(self):
        cortex = FakeCortex()
        s = _make_session_stub(cortex)
        # No exception even though "made_up_kind" isn't in EVENT_KINDS.
        # Emit-site drift must never break a turn.
        s._acc_observe("made_up_kind", {})
        assert s._acc.events == ()

    def test_records_known_kind(self):
        cortex = FakeCortex()
        s = _make_session_stub(cortex)
        s._acc_observe("scratchpad_call", {"name": "x", "code_len": 10}, round_idx=2)
        assert len(s._acc.events) == 1
        assert s._acc.events[0].kind == "scratchpad_call"
        assert s._acc.events[0].round_idx == 2


class TestScheduleAccFlush:
    @pytest.mark.asyncio
    async def test_drains_fires_and_encodes_engrams(self):
        cortex = FakeCortex()
        s = _make_session_stub(cortex)
        # Pump events that fire detect_name_switch (always) and
        # detect_repeated_tool_error (when). Two different Lesson
        # kinds — verifies the wiring preserves the detector's tag.
        s._acc_observe("scratchpad_call", {"name": "a", "code_len": 100}, round_idx=1)
        s._acc_observe("scratchpad_call", {"name": "b", "code_len": 100}, round_idx=2)
        s._acc_observe(
            "tool_result",
            {"name": "t", "success": False, "error": "boom"},
            severity=5,
            round_idx=3,
        )
        s._acc_observe(
            "tool_result",
            {"name": "t", "success": False, "error": "boom"},
            severity=5,
            round_idx=4,
        )

        s._schedule_acc_flush()
        # Give the create_task'd coroutine a turn to run.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        # Buffer cleared.
        assert s._acc.events == ()

        # Exactly one encode batch.
        assert len(cortex.encoded) == 1
        engrams = cortex.encoded[0]
        kinds = {e.kind for e in engrams}
        # name_switch lands as "always", repeated_tool_error as "when".
        assert "always" in kinds
        assert "when" in kinds
        # All engrams should be global-scope, high-confidence, sourced
        # as consolidation — same envelope cerebellum uses.
        for e in engrams:
            assert e.scope == "global"
            assert e.confidence == "high"
            assert e.source == "consolidation"

    @pytest.mark.asyncio
    async def test_noop_when_no_lessons_fire(self):
        cortex = FakeCortex()
        s = _make_session_stub(cortex)
        # One scratchpad_call → name_switch needs ≥2 names, no other
        # detector fires on a single benign event.
        s._acc_observe("scratchpad_call", {"name": "solo", "code_len": 50})
        s._schedule_acc_flush()
        await asyncio.sleep(0)
        assert cortex.encoded == []
        # Buffer still cleared regardless.
        assert s._acc.events == ()

    @pytest.mark.asyncio
    async def test_clears_without_encoding_when_cortex_off(self):
        cortex = FakeCortex(mode="off")
        s = _make_session_stub(cortex)
        # Direct ACC.observe() bypasses the _acc_observe gate so we
        # can verify the flush itself respects mode=="off". (In
        # production this state can't be reached — _acc_observe would
        # have refused the input — but the flush has its own guard
        # for defense-in-depth and we want to exercise it.)
        s._acc.observe("scratchpad_call", {"name": "a", "code_len": 100})
        s._acc.observe("scratchpad_call", {"name": "b", "code_len": 100})
        s._schedule_acc_flush()
        await asyncio.sleep(0)
        assert cortex.encoded == []
        assert s._acc.events == ()

    def test_acc_observe_silent_in_off_mode(self):
        # In off mode the ACC observes nothing — events drop on the
        # floor at the safe-emit wrapper. Cheaper than passive when
        # the user wants the feature entirely disabled.
        cortex = FakeCortex()
        s = _make_session_stub(cortex, acc_mode="off")
        s._acc_observe("scratchpad_call", {"name": "a", "code_len": 100})
        assert s._acc.events == ()

    def test_clears_buffer_when_no_event_loop(self):
        # Synchronous context — asyncio.create_task() inside the flush
        # will raise RuntimeError. The flush must catch that and still
        # clear the buffer rather than leak events into the next turn.
        cortex = FakeCortex()
        s = _make_session_stub(cortex)
        s._acc.observe("scratchpad_call", {"name": "a", "code_len": 100})
        s._acc.observe("scratchpad_call", {"name": "b", "code_len": 100})
        s._schedule_acc_flush()
        assert s._acc.events == ()
        # And no encode happened because we never got the loop.
        assert cortex.encoded == []


class TestAccMaybeNudge:
    """Layer 2 — mid-turn nudge wiring contract."""

    def test_passive_mode_appends_nothing(self):
        cortex = FakeCortex()
        s = _make_session_stub(cortex, acc_mode="passive")
        s._acc_observe("scratchpad_call", {"name": "a", "code_len": 100})
        s._acc_observe("scratchpad_call", {"name": "b", "code_len": 100})
        tool_results: list[dict] = [
            {"type": "tool_result", "tool_use_id": "x", "content": "ok"},
        ]
        n = s._acc_maybe_nudge(tool_results)
        # Passive mode: the nudge is a no-op even though a pattern fired.
        # End-of-turn drain still writes the lesson to memory; mid-turn
        # injection is skipped to keep the turn loop unchanged.
        assert n == 0
        assert len(tool_results) == 1

    def test_active_mode_appends_text_block(self):
        cortex = FakeCortex()
        s = _make_session_stub(cortex, acc_mode="active")
        s._acc_observe("scratchpad_call", {"name": "a", "code_len": 100})
        s._acc_observe("scratchpad_call", {"name": "b", "code_len": 100})
        tool_results: list[dict] = [
            {"type": "tool_result", "tool_use_id": "x", "content": "ok"},
        ]
        n = s._acc_maybe_nudge(tool_results)
        assert n == 1
        # Original tool_result preserved; nudge appended after it.
        assert tool_results[0]["type"] == "tool_result"
        assert tool_results[1]["type"] == "text"
        assert "Anton self-check" in tool_results[1]["text"]
        assert "detect_name_switch" in tool_results[1]["text"]

    def test_active_mode_one_nudge_per_detector_per_turn(self):
        # Same pattern firing on two consecutive rounds should produce
        # ONE nudge total (not one per round). The dlPFC has been told;
        # re-stating the same alarm round after round would just spam
        # the history.
        cortex = FakeCortex()
        s = _make_session_stub(cortex, acc_mode="active")
        s._acc_observe("scratchpad_call", {"name": "a", "code_len": 100})
        s._acc_observe("scratchpad_call", {"name": "b", "code_len": 100})
        tr1: list[dict] = []
        n1 = s._acc_maybe_nudge(tr1)
        assert n1 == 1

        # Round 2 — more events of the same kind, no new pattern.
        s._acc_observe("scratchpad_call", {"name": "c", "code_len": 100})
        tr2: list[dict] = []
        n2 = s._acc_maybe_nudge(tr2)
        assert n2 == 0

    def test_active_mode_silent_when_no_new_pattern(self):
        cortex = FakeCortex()
        s = _make_session_stub(cortex, acc_mode="active")
        s._acc_observe("scratchpad_call", {"name": "solo", "code_len": 50})
        tool_results: list[dict] = []
        n = s._acc_maybe_nudge(tool_results)
        assert n == 0
        assert tool_results == []

    def test_off_mode_skips_nudge_entirely(self):
        # Off mode short-circuits the safe-emit wrapper, so the ACC
        # never sees the events in the first place. Even if a detector
        # had something to say, there's nothing in the buffer to read.
        cortex = FakeCortex()
        s = _make_session_stub(cortex, acc_mode="off")
        s._acc_observe("scratchpad_call", {"name": "a", "code_len": 100})
        s._acc_observe("scratchpad_call", {"name": "b", "code_len": 100})
        n = s._acc_maybe_nudge([])
        assert n == 0
        assert s._acc.events == ()

    def test_detector_exception_does_not_crash_turn(self):
        # Wire a broken detector and verify the nudge wrapper swallows
        # the exception. Layer 1's end-of-turn flush still runs
        # independently — we don't want a buggy detector blocking the
        # whole turn loop.
        cortex = FakeCortex()
        stub = SimpleNamespace(
            _cortex=cortex,
            _acc_mode="active",
        )
        def boom(_events):
            raise RuntimeError("detector bug")
        stub._acc = AnteriorCingulate(detectors=(boom,))
        stub._acc_observe = types.MethodType(ChatSession._acc_observe, stub)
        stub._acc_maybe_nudge = types.MethodType(ChatSession._acc_maybe_nudge, stub)
        stub._acc_observe("scratchpad_call", {"name": "x", "code_len": 1})
        n = stub._acc_maybe_nudge([])
        assert n == 0  # graceful degradation
