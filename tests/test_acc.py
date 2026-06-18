"""Tests for `anton.core.memory.acc` — the ACC error-detection module.

Layout mirrors the design contract documented in `acc.py`:

  * Layer 1 — pure-function detector tests. Each `detect_*` is
    exercised with one positive case (the pattern fires) and one
    negative case (the pattern does not fire). No state, no I/O.

  * Layer 2 — `AnteriorCingulate` unit tests. State management
    (`observe`, `clear`), vocabulary discipline (unknown kinds
    raise), and the dedupe seam (`has_similar_lesson`).

  * Layer 3 — replay tests. Real captured event traces from
    antontron sessions live as JSON in `tests/fixtures/acc/`. Each
    fixture asserts the exact lesson the ACC should produce. This
    layer is the living regression suite — every new failure mode
    Anton encounters in the wild can be captured here once and
    locked in forever after.

  * Layer 4 — vocabulary sanity. A single test that walks every
    `kind` in `EVENT_KINDS` and asserts at least one detector
    reads it. Prevents the event taxonomy from drifting away from
    the detectors that consume it.

No LLM calls in any test. Detectors are pure; dedupe is stubbed.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from anton.core.memory.acc import (
    DETECTORS,
    EVENT_KINDS,
    AnteriorCingulate,
    Event,
    Lesson,
    detect_cap_exhausted,
    detect_kill_loop,
    detect_name_switch,
    detect_oversized_cell,
    detect_repair_churn,
    detect_repeated_error_signature,
    detect_repeated_tool_error,
    detect_reset_churn,
    detect_severity_climb,
)


FIXTURES = Path(__file__).parent / "fixtures" / "acc"


def _load_fixture(name: str) -> list[Event]:
    """Load a fixture JSON and return Event objects."""
    data = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return [
        Event(
            kind=item["kind"],
            severity=int(item.get("severity", 1)),
            detail=dict(item.get("detail", {})),
            round_idx=int(item.get("round_idx", 0)),
        )
        for item in data
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — pure-function detector tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDetectNameSwitch:
    def test_fires_on_two_distinct_names(self):
        events = [
            Event("scratchpad_call", 1, {"name": "build_pres"}, 3),
            Event("scratchpad_call", 1, {"name": "write_html"}, 7),
        ]
        lesson = detect_name_switch(events)
        assert lesson is not None
        assert lesson.detector == "detect_name_switch"
        # The phrase the LLM is supposed to internalise.
        assert "ONE scratchpad" in lesson.rule

    def test_silent_on_single_name(self):
        events = [
            Event("scratchpad_call", 1, {"name": "dash"}, i) for i in range(5)
        ]
        assert detect_name_switch(events) is None

    def test_silent_on_empty_event_list(self):
        assert detect_name_switch([]) is None

    def test_ignores_other_event_kinds(self):
        # tool_call has no `name` field for *scratchpad*; the detector
        # should not be tricked into firing on tool names.
        events = [
            Event("tool_call", 1, {"name": "scratchpad", "args_summary": "..."}, 1),
            Event("tool_call", 1, {"name": "publish_or_preview", "args_summary": "..."}, 2),
        ]
        assert detect_name_switch(events) is None


class TestDetectOversizedCell:
    def test_fires_on_observed_empty_code(self):
        # One big cell plus the empty-code failure mode IS the smoking
        # gun. Should fire even though only one big cell is present.
        events = [
            Event("scratchpad_call", 1, {"name": "dash", "code_len": 18000}, 4),
            Event("scratchpad_empty_code", 7, {"name": "dash"}, 4),
        ]
        lesson = detect_oversized_cell(events)
        assert lesson is not None
        assert "5 KB" in lesson.rule or "5 KB" in lesson.rule.replace(" ", " ")

    def test_fires_on_two_or_more_big_cells_even_without_empty_code(self):
        events = [
            Event("scratchpad_call", 1, {"name": "dash", "code_len": 9000}, 2),
            Event("scratchpad_call", 1, {"name": "dash", "code_len": 11000}, 4),
        ]
        assert detect_oversized_cell(events) is not None

    def test_silent_on_single_big_cell(self):
        # One isolated big cell isn't pattern enough. Detector only
        # fires when the pattern is repeated OR the failure mode
        # actually manifested.
        events = [Event("scratchpad_call", 1, {"name": "dash", "code_len": 10000}, 2)]
        assert detect_oversized_cell(events) is None

    def test_silent_when_all_cells_are_small(self):
        events = [
            Event("scratchpad_call", 1, {"name": "dash", "code_len": n}, i)
            for i, n in enumerate([200, 500, 1200, 800])
        ]
        assert detect_oversized_cell(events) is None


class TestDetectRepeatedToolError:
    def test_fires_on_two_consecutive_failures_of_same_tool(self):
        events = [
            Event("tool_result", 1, {"name": "publish_or_preview", "success": False, "error": "x"}, 6),
            Event("tool_result", 1, {"name": "publish_or_preview", "success": False, "error": "x"}, 8),
        ]
        lesson = detect_repeated_tool_error(events)
        assert lesson is not None
        assert "don't retry" in lesson.rule.lower()

    def test_silent_when_failures_are_for_different_tools(self):
        events = [
            Event("tool_result", 1, {"name": "publish_or_preview", "success": False}, 6),
            Event("tool_result", 1, {"name": "lookup_connector",   "success": False}, 8),
        ]
        assert detect_repeated_tool_error(events) is None

    def test_success_resets_the_run(self):
        events = [
            Event("tool_result", 1, {"name": "publish_or_preview", "success": False}, 6),
            Event("tool_result", 1, {"name": "publish_or_preview", "success": True},  8),
            Event("tool_result", 1, {"name": "publish_or_preview", "success": False}, 10),
        ]
        # Longest consecutive failure run is 1 → does not fire.
        assert detect_repeated_tool_error(events) is None

    def test_silent_on_single_failure(self):
        events = [
            Event("tool_result", 1, {"name": "publish_or_preview", "success": False}, 6),
        ]
        assert detect_repeated_tool_error(events) is None


class TestDetectRepeatedErrorSignature:
    def test_fires_on_same_error_three_times_across_tools(self):
        # The unifier — same normalised signature across DIFFERENT tools
        # should fire, where detect_repeated_tool_error would not.
        events = [
            Event("tool_result", 5, {"name": "a", "success": False, "error": "Refusing to save record for engine='gmail-1'"}, 3),
            Event("tool_result", 5, {"name": "b", "success": False, "error": "Refusing to save record for engine='gmail-2'"}, 4),
            Event("tool_result", 5, {"name": "c", "success": False, "error": "Refusing to save record for engine='gmail-3'"}, 5),
        ]
        lesson = detect_repeated_error_signature(events)
        assert lesson is not None
        assert lesson.detector == "detect_repeated_error_signature"
        assert "same error message" in lesson.rule.lower()

    def test_silent_on_two_same_errors(self):
        # Threshold is 3 — two same-errors is one legitimate retry.
        events = [
            Event("tool_result", 5, {"name": "x", "success": False, "error": "boom"}, 1),
            Event("tool_result", 5, {"name": "x", "success": False, "error": "boom"}, 2),
        ]
        assert detect_repeated_error_signature(events) is None

    def test_silent_when_errors_are_distinct(self):
        events = [
            Event("tool_result", 5, {"name": "x", "success": False, "error": "auth failed"}, 1),
            Event("tool_result", 5, {"name": "x", "success": False, "error": "rate limited"}, 2),
            Event("tool_result", 5, {"name": "x", "success": False, "error": "schema invalid"}, 3),
        ]
        assert detect_repeated_error_signature(events) is None

    def test_reads_scratchpad_results_too(self):
        events = [
            Event("scratchpad_result", 5, {"name": "p", "success": False, "error": "NameError: 'data' undefined"}, 1),
            Event("scratchpad_result", 5, {"name": "p", "success": False, "error": "NameError: 'data' undefined"}, 2),
            Event("scratchpad_result", 5, {"name": "p", "success": False, "error": "NameError: 'data' undefined"}, 3),
        ]
        assert detect_repeated_error_signature(events) is not None

    def test_success_events_dont_count(self):
        events = [
            Event("tool_result", 1, {"name": "x", "success": True}, 1),
            Event("tool_result", 1, {"name": "x", "success": True}, 2),
            Event("tool_result", 1, {"name": "x", "success": True}, 3),
        ]
        assert detect_repeated_error_signature(events) is None


class TestDetectResetChurn:
    def test_fires_on_two_resets(self):
        events = [
            Event("scratchpad_reset", 5, {"name": "dash", "reason": "manual"}, 4),
            Event("scratchpad_reset", 5, {"name": "dash", "reason": "manual"}, 8),
        ]
        lesson = detect_reset_churn(events)
        assert lesson is not None
        assert lesson.detector == "detect_reset_churn"
        assert "reset" in lesson.rule.lower()

    def test_silent_on_single_reset(self):
        events = [Event("scratchpad_reset", 5, {"name": "dash"}, 4)]
        assert detect_reset_churn(events) is None


class TestDetectKillLoop:
    def test_fires_on_two_kills_same_name(self):
        events = [
            Event("scratchpad_killed", 6, {"name": "compute", "reason": "timeout"}, 3),
            Event("scratchpad_killed", 6, {"name": "compute", "reason": "timeout"}, 6),
        ]
        lesson = detect_kill_loop(events)
        assert lesson is not None
        assert lesson.detector == "detect_kill_loop"

    def test_fires_on_kills_across_different_names(self):
        # Renaming the scratchpad between failed attempts must NOT hide the
        # loop — two kills in a turn fire regardless of name.
        events = [
            Event("scratchpad_killed", 6, {"name": "a", "reason": "timeout"}, 1),
            Event("scratchpad_killed", 6, {"name": "b", "reason": "timeout"}, 2),
        ]
        lesson = detect_kill_loop(events)
        assert lesson is not None
        assert lesson.detector == "detect_kill_loop"

    def test_silent_on_single_kill(self):
        events = [Event("scratchpad_killed", 6, {"name": "compute"}, 3)]
        assert detect_kill_loop(events) is None


class TestDetectSeverityClimb:
    def test_fires_on_strictly_increasing_run_to_high_severity(self):
        events = [
            Event("tool_result", 1, {"name": "publish", "success": True}, 1),
            Event("tool_result", 3, {"name": "publish", "success": False, "error": "minor"}, 2),
            Event("tool_result", 5, {"name": "publish", "success": False, "error": "worse"}, 3),
            Event("tool_result", 7, {"name": "publish", "success": False, "error": "very bad"}, 4),
        ]
        lesson = detect_severity_climb(events)
        assert lesson is not None
        assert lesson.detector == "detect_severity_climb"

    def test_silent_on_flat_severities(self):
        events = [
            Event("tool_result", 5, {"name": "x", "success": False}, 1),
            Event("tool_result", 5, {"name": "x", "success": False}, 2),
            Event("tool_result", 5, {"name": "x", "success": False}, 3),
        ]
        assert detect_severity_climb(events) is None

    def test_silent_when_peak_below_threshold(self):
        # Strictly increasing but never reaches _SEVERITY_CLIMB_PEAK.
        events = [
            Event("tool_result", 1, {"name": "x", "success": True}, 1),
            Event("tool_result", 2, {"name": "x", "success": True}, 2),
            Event("tool_result", 3, {"name": "x", "success": True}, 3),
        ]
        assert detect_severity_climb(events) is None

    def test_silent_when_climb_is_split_across_producers(self):
        # Climbing-severity events for DIFFERENT names → not a single
        # producer's deteriorating sequence.
        events = [
            Event("tool_result", 1, {"name": "a", "success": True}, 1),
            Event("tool_result", 5, {"name": "b", "success": False, "error": "x"}, 2),
            Event("tool_result", 7, {"name": "c", "success": False, "error": "y"}, 3),
        ]
        assert detect_severity_climb(events) is None


class TestDetectRepairChurn:
    def test_fires_on_three_repairs(self):
        events = [
            Event("history_repair", 5, {"reason": "orphan_tool_use"}, 2),
            Event("history_repair", 5, {"reason": "orphan_tool_use"}, 4),
            Event("history_repair", 5, {"reason": "orphan_tool_use"}, 7),
        ]
        lesson = detect_repair_churn(events)
        assert lesson is not None
        assert lesson.detector == "detect_repair_churn"

    def test_silent_on_two_repairs(self):
        events = [
            Event("history_repair", 5, {"reason": "x"}, 2),
            Event("history_repair", 5, {"reason": "x"}, 4),
        ]
        assert detect_repair_churn(events) is None


class TestDetectCapExhausted:
    def test_fires_on_single_occurrence(self):
        events = [Event("cap_exhausted", 9, {}, 25)]
        lesson = detect_cap_exhausted(events)
        assert lesson is not None
        assert lesson.detector == "detect_cap_exhausted"
        assert "round cap" in lesson.rule.lower()

    def test_silent_when_absent(self):
        events = [Event("tool_result", 1, {"name": "x", "success": True}, 1)]
        assert detect_cap_exhausted(events) is None


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — AnteriorCingulate state tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAnteriorCingulate:
    def test_observe_appends_to_buffer(self):
        acc = AnteriorCingulate()
        acc.observe("scratchpad_call", {"name": "x", "code_len": 100})
        assert len(acc.events) == 1
        assert acc.events[0].kind == "scratchpad_call"

    def test_observe_rejects_unknown_kind(self):
        acc = AnteriorCingulate()
        with pytest.raises(ValueError, match="Unknown ACC event kind"):
            acc.observe("scratchpad_explosion", {})

    def test_clear_drops_buffer(self):
        acc = AnteriorCingulate()
        acc.observe("scratchpad_call", {"name": "x", "code_len": 100})
        acc.clear()
        assert len(acc.events) == 0

    def test_at_end_of_turn_returns_lessons_in_detector_order(self):
        acc = AnteriorCingulate()
        # Trigger both name_switch and repeated_tool_error.
        acc.observe("scratchpad_call", {"name": "a", "code_len": 200}, round_idx=1)
        acc.observe("scratchpad_call", {"name": "b", "code_len": 200}, round_idx=2)
        acc.observe("tool_result", {"name": "t", "success": False}, round_idx=3)
        acc.observe("tool_result", {"name": "t", "success": False}, round_idx=4)
        lessons = acc.at_end_of_turn()
        # DETECTORS order: name_switch, oversized_cell (silent here),
        # repeated_tool_error. So name_switch first, then tool_error.
        assert [l.detector for l in lessons] == [
            "detect_name_switch",
            "detect_repeated_tool_error",
        ]

    def test_has_similar_lesson_blocks_persistence(self):
        # Stub: pretend memory already knows the name-switch rule.
        def already_known(rule: str) -> bool:
            return "ONE scratchpad" in rule

        acc = AnteriorCingulate(has_similar_lesson=already_known)
        acc.observe("scratchpad_call", {"name": "a", "code_len": 100}, round_idx=1)
        acc.observe("scratchpad_call", {"name": "b", "code_len": 100}, round_idx=2)
        lessons = acc.at_end_of_turn()
        assert lessons == []  # de-duped against existing memory

    def test_cross_detector_dedupe(self):
        # Belt-and-suspenders: if two detectors ever produce the same
        # rule string, we don't write it twice in one turn.
        called = {"count": 0}

        def fake_detector(events):
            called["count"] += 1
            return Lesson(rule="duplicate rule", kind="when", triggers=(), detector=f"fake_{called['count']}")

        acc = AnteriorCingulate(detectors=(fake_detector, fake_detector))
        acc.observe("scratchpad_call", {"name": "x", "code_len": 100})
        lessons = acc.at_end_of_turn()
        assert len(lessons) == 1
        assert lessons[0].rule == "duplicate rule"

    def test_event_kind_counts(self):
        acc = AnteriorCingulate()
        acc.observe("scratchpad_call", {"name": "a", "code_len": 100})
        acc.observe("scratchpad_call", {"name": "b", "code_len": 100})
        acc.observe("tool_result",     {"name": "t", "success": False})
        assert acc.event_kind_counts == {"scratchpad_call": 2, "tool_result": 1}


class TestAtRoundN:
    """Layer 2 — mid-turn nudging contract.

    Each detector is allowed to nudge AT MOST ONCE per turn.
    Subsequent calls during the same turn only return lessons from
    detectors that haven't nudged yet. `clear()` resets the nudge
    tracking so the next turn starts fresh.
    """

    def test_returns_lessons_for_newly_fired_detectors(self):
        acc = AnteriorCingulate()
        acc.observe("scratchpad_call", {"name": "a", "code_len": 100}, round_idx=1)
        acc.observe("scratchpad_call", {"name": "b", "code_len": 100}, round_idx=2)
        lessons = acc.at_round_n()
        assert len(lessons) == 1
        assert lessons[0].detector == "detect_name_switch"

    def test_second_call_same_turn_returns_empty_when_no_new_pattern(self):
        acc = AnteriorCingulate()
        acc.observe("scratchpad_call", {"name": "a", "code_len": 100}, round_idx=1)
        acc.observe("scratchpad_call", {"name": "b", "code_len": 100}, round_idx=2)
        first = acc.at_round_n()
        second = acc.at_round_n()
        assert len(first) == 1
        assert second == []

    def test_second_call_returns_new_pattern_when_one_emerges(self):
        # Round 1: name_switch fires. Round 2: tool-error pattern emerges.
        # Second call should return ONLY the tool-error lesson, not the
        # already-nudged name-switch lesson.
        acc = AnteriorCingulate()
        acc.observe("scratchpad_call", {"name": "a", "code_len": 100}, round_idx=1)
        acc.observe("scratchpad_call", {"name": "b", "code_len": 100}, round_idx=2)
        first = acc.at_round_n()
        assert [l.detector for l in first] == ["detect_name_switch"]

        acc.observe("tool_result", {"name": "t", "success": False}, round_idx=3)
        acc.observe("tool_result", {"name": "t", "success": False}, round_idx=4)
        second = acc.at_round_n()
        assert [l.detector for l in second] == ["detect_repeated_tool_error"]

    def test_ignores_has_similar_lesson(self):
        """Mid-turn nudges should fire even when the rule already
        lives in memory. Skipping based on memory dedupe would mean
        the LLM gets no in-context reminder of a rule it's actively
        violating right now."""
        def always_known(_rule):
            return True

        acc = AnteriorCingulate(has_similar_lesson=always_known)
        acc.observe("scratchpad_call", {"name": "a", "code_len": 100}, round_idx=1)
        acc.observe("scratchpad_call", {"name": "b", "code_len": 100}, round_idx=2)
        assert acc.at_round_n() != []  # fires even though rule "is in memory"

    def test_clear_resets_nudge_tracking(self):
        acc = AnteriorCingulate()
        acc.observe("scratchpad_call", {"name": "a", "code_len": 100}, round_idx=1)
        acc.observe("scratchpad_call", {"name": "b", "code_len": 100}, round_idx=2)
        first = acc.at_round_n()
        assert len(first) == 1
        # Simulate a turn boundary.
        acc.clear()
        acc.observe("scratchpad_call", {"name": "a", "code_len": 100}, round_idx=1)
        acc.observe("scratchpad_call", {"name": "b", "code_len": 100}, round_idx=2)
        again = acc.at_round_n()
        # Fresh turn: same pattern fires again.
        assert len(again) == 1

    def test_silent_when_no_patterns(self):
        acc = AnteriorCingulate()
        acc.observe("scratchpad_call", {"name": "solo", "code_len": 50}, round_idx=1)
        assert acc.at_round_n() == []


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — replay tests
#
# Each fixture is a real captured failure mode. The assertions lock
# in the exact lesson ACC should produce. To add a new fixture:
#
#   1. Drop a JSON file in tests/fixtures/acc/.
#   2. Add a method below.
#   3. If no existing detector catches it, write one in acc.py and
#      add unit-test coverage in Layer 1.
# ─────────────────────────────────────────────────────────────────────────────


class TestReplay:
    def _run(self, fixture: str) -> list[Lesson]:
        acc = AnteriorCingulate()
        for e in _load_fixture(fixture):
            acc.observe(e.kind, e.detail, severity=e.severity, round_idx=e.round_idx)
        return acc.at_end_of_turn()

    def test_replay_name_switch(self):
        lessons = self._run("name_switch.json")
        rules = {l.rule for l in lessons}
        assert any("ONE scratchpad" in r for r in rules), (
            f"name_switch fixture should produce a name-switch lesson, got: {rules}"
        )

    def test_replay_oversized_cell(self):
        lessons = self._run("oversized_cell.json")
        rules = {l.rule for l in lessons}
        assert any("5 KB" in r for r in rules), (
            f"oversized_cell fixture should produce the cell-size lesson, got: {rules}"
        )

    def test_replay_publish_failure_loop(self):
        lessons = self._run("publish_failure_loop.json")
        rules = {l.rule for l in lessons}
        # The fixture has three identical failures of the same tool with
        # the same error message — it should fire BOTH the tool-level
        # retry detector AND the broader error-signature detector.
        assert any("don't retry" in r.lower() for r in rules), (
            f"publish_failure_loop fixture should produce the retry-loop lesson, got: {rules}"
        )
        assert any("same error message" in r.lower() for r in rules), (
            f"publish_failure_loop fixture should also produce the repeated-signature lesson, got: {rules}"
        )

    def test_replay_reset_churn(self):
        lessons = self._run("reset_churn.json")
        rules = {l.rule for l in lessons}
        assert any("reset" in r.lower() and "scratchpad" in r.lower() for r in rules), (
            f"reset_churn fixture should produce the state-abandonment lesson, got: {rules}"
        )

    def test_replay_kill_loop(self):
        lessons = self._run("kill_loop.json")
        rules = {l.rule for l in lessons}
        assert any("killed" in r.lower() for r in rules), (
            f"kill_loop fixture should produce the kill-loop lesson, got: {rules}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Layer 4 — vocabulary discipline
# ─────────────────────────────────────────────────────────────────────────────


def test_every_event_kind_is_read_by_at_least_one_detector():
    """Every value in EVENT_KINDS should appear as a string literal in
    at least one detector's source. Drift between the vocabulary and
    the detectors is the rot we're guarding against — when a new kind
    is added without a detector that consumes it, this test fails
    loudly and the contributor either adds a detector or deletes the
    kind from the vocabulary.

    `KNOWN_PRODUCER_ONLY` is the tight, justified allowlist for kinds
    that are emitted today but not yet consumed by a detector. Every
    entry needs an explicit reason. Adding to this set is a smell —
    the bar for a new entry is "we genuinely need this emitted for
    correlation/telemetry today AND a detector will land within a
    sprint." Otherwise, drop the kind from EVENT_KINDS.
    """
    KNOWN_PRODUCER_ONLY = {
        # tool_call carries `args_summary`, paired with `tool_result`
        # by round_idx. No current detector reads it (the actionable
        # signal is the result, not the call). Reserved for a future
        # `detect_orphaned_tool_call` (call with no matching result =
        # transport dropped the response). Keeping the emit point is
        # cheap; adding it later would require re-instrumenting every
        # tool dispatch site.
        "tool_call",
    }
    detector_sources = "\n".join(inspect.getsource(d) for d in DETECTORS)
    missing = []
    for kind in sorted(EVENT_KINDS):
        if kind in KNOWN_PRODUCER_ONLY:
            continue
        if f'"{kind}"' not in detector_sources:
            missing.append(kind)
    assert not missing, (
        f"Event kinds in EVENT_KINDS but not read by any detector: {missing}. "
        f"Either add a detector that reads them or remove them from EVENT_KINDS. "
        f"Producer-only kinds require an entry in KNOWN_PRODUCER_ONLY with a reason."
    )


def test_every_lesson_has_a_valid_kind():
    """Every detector must tag its Lesson with a `kind` so the wiring
    layer can route it to the right Engram kind (always/never/when)
    without parsing the rule text. Detectors that fire here drive the
    smoke check; detectors that don't fire in this synthetic mix are
    indirectly covered by their own positive unit tests."""
    # Craft an event stream that fires every detector. Some need
    # specific shapes — these were copied from the per-detector tests.
    events = [
        # name_switch + oversized_cell
        Event("scratchpad_call", 1, {"name": "a", "code_len": 9000}, 1),
        Event("scratchpad_call", 1, {"name": "b", "code_len": 11000}, 2),
        # tool repeats
        Event("tool_result", 5, {"name": "t", "success": False, "error": "same err"}, 3),
        Event("tool_result", 5, {"name": "t", "success": False, "error": "same err"}, 4),
        Event("tool_result", 5, {"name": "t", "success": False, "error": "same err"}, 5),
        # reset + kill
        Event("scratchpad_reset",  5, {"name": "a"}, 6),
        Event("scratchpad_reset",  5, {"name": "a"}, 7),
        Event("scratchpad_killed", 6, {"name": "c"}, 8),
        Event("scratchpad_killed", 6, {"name": "c"}, 9),
        # severity climb on producer "z"
        Event("tool_result", 1, {"name": "z", "success": True}, 10),
        Event("tool_result", 3, {"name": "z", "success": False, "error": "small"}, 11),
        Event("tool_result", 7, {"name": "z", "success": False, "error": "big"}, 12),
        # repair churn
        Event("history_repair", 5, {"reason": "x"}, 13),
        Event("history_repair", 5, {"reason": "x"}, 14),
        Event("history_repair", 5, {"reason": "x"}, 15),
        # cap exhausted
        Event("cap_exhausted", 9, {}, 25),
    ]
    acc = AnteriorCingulate()
    for e in events:
        acc.observe(e.kind, e.detail, severity=e.severity, round_idx=e.round_idx)
    lessons = acc.at_end_of_turn()
    assert lessons, "Crafted event stream should fire multiple detectors"
    for l in lessons:
        assert l.kind in ("always", "never", "when"), (
            f"Detector {l.detector} produced invalid kind {l.kind!r}; "
            f"must be one of always/never/when so Cortex.encode() routes correctly."
        )


def test_no_dropped_kinds_lingering_in_event_kinds():
    """Guard against zombies — kinds we explicitly dropped should not
    reappear in EVENT_KINDS. If a future reviewer adds them back, this
    test forces a conversation about *why* (and probably about adding
    a real detector this time).
    """
    DROPPED = {"context_compaction", "round_milestone"}
    leaked = DROPPED & EVENT_KINDS
    assert not leaked, (
        f"These kinds were intentionally dropped from the ACC vocabulary "
        f"because no detector consumed them: {leaked}. If you need to "
        f"re-add one, add a detector that reads it in the same change."
    )
