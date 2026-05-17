"""Anterior Cingulate Cortex (ACC) — pattern-level error detection.

Brain analogue
==============

The anterior cingulate cortex is the brain's error-detector. It fires
the *error-related negativity* (ERN) ~80 ms after the brain notices
that an actual outcome diverged from an expected one. The signal it
emits flows downstream to the dopaminergic midbrain (computing the
*reward prediction error*) and onward to the striatum (updating
action policies) and the dorsolateral prefrontal cortex (adjusting
strategy on the next attempt).

Anton's analogue
================

Anton's ACC watches a turn unfold, classifies events as they arrive,
and at end-of-turn extracts *actionable lessons* from the patterns
that fired more than once. The cerebellum (in this directory)
already handles the per-cell forward-model error — predicted vs
actual outcome of a single scratchpad cell. The ACC handles a
larger time scale: patterns *across* cells within a single task.
Examples we've seen in real sessions:

  - The LLM switched scratchpad names mid-task (`build_pres` →
    `write_html` → `pres1`). Each name is a separate isolated
    environment; variables in one don't exist in another. Burned 8
    rounds on recovery.

  - Repeated scratchpad calls with empty `code` parameters because a
    large HTML string serialised to "" in the tool-call schema.
    Burned 10 rounds before anyone noticed.

  - The same tool failed three times in a row with the same args.

These are not per-cell prediction errors — they're patterns over
multiple events. ACC's job is to notice them and write one-sentence
rules that the next turn's system prompt picks up via the existing
cortex memory path.

Storage and retrieval
=====================

The ACC is a *producer* only. It does not own its own storage.
Lessons it extracts flow into the same `Engram` pipeline that the
cerebellum and the consolidator already use — see `cortex.encode()`
and `consolidator.consolidate()`. The ACC does not duplicate the
de-dupe logic; it consults a caller-supplied `has_similar_lesson`
predicate so the wiring layer can use whichever similarity check
fits (substring, embedding, semantic).

Design constraints
==================

  - **Pure where possible.** Each `detect_*` function is a free
    function of `Sequence[Event] → Lesson | None`. Easy to unit-test
    in isolation, easy to add new ones, easy to delete bad ones.

  - **No LLM calls at detect time.** Detectors are deterministic
    pattern matchers. The lesson `rule` strings are templates
    parameterised by what was seen. Keep this property: the cortex
    has its own LLM-based consolidation; ACC's job is to surface
    candidates cheaply.

  - **De-dupe at the seam.** Two detectors might match overlapping
    patterns. Resolution is on the caller — pass a `has_similar_lesson`
    that consults the existing memory store before persisting.

  - **Bounded buffer.** Events are kept in memory for one turn only;
    `clear()` drops them at end-of-turn. The persisted artifact is
    the lesson, not the event log.

Adding a new detector
=====================

  1. Write a function `detect_<thing>(events: Sequence[Event]) ->
     Lesson | None`.
  2. Add it to `DETECTORS`.
  3. Add a positive + negative unit test in `tests/test_acc.py`.
  4. If the pattern was discovered from a real failure, also drop
     the captured event trace into `tests/fixtures/acc/<thing>.json`
     and write a replay test that asserts the lesson fires.

That recipe keeps the regression suite as a living catalogue of
Anton's known failure modes — exactly the post-mortem loop that
inspired this module.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence


# ─────────────────────────────────────────────────────────────────────────────
# Public data types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Event:
    """One observed signal in the current turn.

    Attributes:
        kind: One of the canonical strings in `EVENT_KINDS`. Adding a
            new value requires updating `EVENT_KINDS` and writing at
            least one detector that consumes it.
        severity: 1-10. Detectors use this as a rough filter ("only
            care about ≥ 5 events"). The wiring layer picks the
            value at emit time — we deliberately don't try to score
            here, because severity is context-dependent and the
            emitter has more context than the detector does.
        detail: Free-form payload. Detectors that need a specific
            field document it in their docstring.
        round_idx: The 0-based round inside the current turn. Useful
            for clustering (e.g. "three failures within 4 rounds").
    """

    kind: str
    severity: int
    detail: dict
    round_idx: int


# Canonical kind vocabulary. Add a new value here only when you also
# add (a) the emitter that publishes it and (b) a detector that
# reads it. Drift between these three is what makes event-based
# architectures rot — keeping the list explicit is the brake.
EVENT_KINDS = frozenset({
    # Scratchpad
    "scratchpad_call",          # detail: {name, code_len, one_line_description}
    "scratchpad_result",        # detail: {name, success, stdout_len, error}
    "scratchpad_empty_code",    # detail: {name} — code parameter was empty/missing
    "scratchpad_reset",         # detail: {name, reason} — venv/state cleared mid-turn
    "scratchpad_killed",        # detail: {name, reason} — cell killed (timeout/cancel/OOM)
    # Tools
    "tool_call",                # detail: {name, args_summary} — paired with tool_result; producer-only today
    "tool_result",              # detail: {name, success, error}
    # Session-level
    "history_repair",           # detail: {reason} — tool_use/result mismatch repair fired
    "cap_exhausted",            # detail: {} — terminal: hit max_rounds
})


LessonKind = Literal["always", "never", "when"]


@dataclass(frozen=True)
class Lesson:
    """An actionable rule the ACC wants future-Anton to remember.

    Attributes:
        rule: The one-sentence rule to memorise. Phrase as guidance
            ("Use ONE scratchpad name per task — switching mid-task
            isolates variables in separate environments.") rather
            than judgement of past behaviour.
        kind: Which behavioural section this rule lands in when it
            flows through `Cortex.encode()`. Maps directly to
            `Engram.kind`:
              * `always` — unconditional habit ("Use ONE scratchpad…").
              * `never`  — unconditional prohibition ("Don't reset…").
              * `when`   — conditional rule ("When a tool fails…").
            The detector knows the semantics; routing by string-
            matching the rule at the wiring layer would be brittle.
        triggers: The event kinds that contributed to firing this
            lesson. Used for audit + for de-dupe across detectors.
        detector: Name of the detector function that produced it.
            Helps when an unexpected lesson shows up in memory —
            you can find its origin without a full re-derivation.
    """

    rule: str
    kind: LessonKind
    triggers: tuple[str, ...]
    detector: str


# Type alias for clarity at the seams.
Detector = Callable[[Sequence[Event]], Lesson | None]


# ─────────────────────────────────────────────────────────────────────────────
# Detectors — pure functions of the event sequence
# ─────────────────────────────────────────────────────────────────────────────


# Threshold for "cell was too big for the scratchpad to handle reliably".
# Chosen at 5000 chars based on the empirical "silent empty-code drop"
# pattern seen on payloads above ~4-6 KB. Tuneable.
_OVERSIZED_CELL_CHARS = 5000

# Minimum distinct names seen before we flag fragmentation. Two is
# enough to fire — there is no legitimate reason to spin up a second
# scratchpad mid-task; if there were we'd add an exception, but
# every observed instance has been an error.
_NAME_SWITCH_THRESHOLD = 2

# Minimum consecutive failures of the same tool before we flag a
# "retry-with-same-args" loop. Two is enough for the lesson to be
# actionable; lifts to three would miss the cap-exhaustion case
# where the round budget runs out at attempt 2.
_REPEATED_TOOL_ERROR_THRESHOLD = 2

# Minimum occurrences of the same normalised error signature across
# any producers (tool_result / scratchpad_result) before we flag a
# blind-retry loop. Three is the right floor: two same-errors is the
# LLM doing one legitimate retry after a transient failure; three is
# a real loop. Generalises detect_repeated_tool_error to errors that
# repeat across different tools or with arg tweaks in between.
_REPEATED_ERROR_THRESHOLD = 3

# Number of scratchpad resets in one turn before we flag state-
# abandonment. Two is enough — one reset can be intentional cleanup,
# two means the LLM is using reset as a debugging primitive.
_RESET_CHURN_THRESHOLD = 2

# Number of cells killed (timeout/cancel/OOM) on the same scratchpad
# name before we flag a "writing cells that hang" pattern.
_KILL_LOOP_THRESHOLD = 2

# Number of history_repair events in one turn before we flag a
# derailing conversation. Three is the right floor — one or two
# repairs are normal model-side hiccups; three means the LLM is
# generating malformed tool_use/result pairs structurally.
_REPAIR_CHURN_THRESHOLD = 3

# detect_severity_climb fires when a producer emits a strictly-
# increasing severity sequence of length >= _SEVERITY_CLIMB_LEN whose
# final element is >= _SEVERITY_CLIMB_PEAK. Brain analog: the ERN
# climbs as outcomes worsen — when the climb crosses a threshold,
# the dlPFC is supposed to switch strategy rather than amplify.
_SEVERITY_CLIMB_LEN = 3
_SEVERITY_CLIMB_PEAK = 5


def _normalise_error_signature(text: str) -> str:
    """Collapse the variable parts of an error message into a stable
    signature so that "Refusing to save record for engine='gmail-1'"
    and "Refusing to save record for engine='gmail-2'" hash to the
    same string. Cheap regex pass — paths, integers, hex, quoted
    short tokens all become placeholders.
    """
    s = text or ""
    s = re.sub(r"0x[0-9a-fA-F]+", "0xN", s)
    s = re.sub(r"\b\d+\b", "N", s)
    s = re.sub(r"'[^']{1,80}'", "'X'", s)
    s = re.sub(r'"[^"]{1,80}"', '"X"', s)
    s = re.sub(r"(?:[A-Za-z]:)?(?:/|\\\\)[^\s'\"]+", "/P", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def detect_name_switch(events: Sequence[Event]) -> Lesson | None:
    """The LLM used >1 distinct scratchpad name in one turn.

    Each scratchpad name is a separate isolated environment in
    anton's runtime. Variables in `A` are not visible to code
    running in `B`. The LLM occasionally treats them as one
    namespace and ends up re-defining symbols, re-fetching data,
    and burning rounds.

    Reads `kind == "scratchpad_call"` events; looks at `detail.name`.
    """
    names = {
        e.detail.get("name")
        for e in events
        if e.kind == "scratchpad_call" and e.detail.get("name")
    }
    if len(names) < _NAME_SWITCH_THRESHOLD:
        return None
    return Lesson(
        rule=(
            "Use ONE scratchpad name per task and reuse it for every cell. "
            "Each name is a separate isolated environment — variables in one "
            "are not visible to code running in another. Switching mid-task "
            "(e.g. `build_pres` → `write_html`) forces re-imports, re-fetches, "
            "and burns rounds on recovery."
        ),
        kind="always",
        triggers=("scratchpad_call",),
        detector="detect_name_switch",
    )


def detect_oversized_cell(events: Sequence[Event]) -> Lesson | None:
    """A scratchpad cell exceeded the size we know to be reliable.

    Empirical: scratchpad calls with code payloads above ~5 KB
    occasionally drop the `code` parameter entirely (the tool sees
    an empty string and returns "No code provided"), which still
    counts against the round cap. The lesson: build the output
    incrementally on disk instead of holding it all in one string.

    Reads `kind == "scratchpad_call"` (uses `detail.code_len`) and
    `kind == "scratchpad_empty_code"` (the failure mode itself).
    """
    too_big = [
        e for e in events
        if e.kind == "scratchpad_call" and int(e.detail.get("code_len", 0)) >= _OVERSIZED_CELL_CHARS
    ]
    empty = [e for e in events if e.kind == "scratchpad_empty_code"]
    if not too_big and not empty:
        return None
    # Don't fire on a single one-off big cell; only when there's
    # either a *pattern* of big cells OR we've actually observed the
    # empty-code failure mode that big cells trigger.
    if not empty and len(too_big) < 2:
        return None
    return Lesson(
        rule=(
            "Keep individual scratchpad cells under ~5 KB of string content. "
            "When generating large outputs (HTML, reports), write to disk "
            "incrementally — `open(path, 'w')` once, then `open(path, 'a')` "
            "to append each chunk. Building one large string in memory and "
            "writing it in a single cell occasionally drops the `code` "
            "parameter on the wire and silently burns a round."
        ),
        kind="always",
        triggers=tuple({"scratchpad_call", "scratchpad_empty_code"} & {e.kind for e in (*too_big, *empty)}),
        detector="detect_oversized_cell",
    )


def detect_repeated_tool_error(events: Sequence[Event]) -> Lesson | None:
    """The same tool returned an error >= N times in a row.

    Reads `kind == "tool_result"` (uses `detail.name`, `detail.success`,
    `detail.error`). Looks for the longest consecutive run of failures
    keyed on the tool name. The lesson is generic — the rule reminds
    future-Anton that retrying with no diagnosis is the failure mode;
    the specific tool name lands in `triggers` for audit.
    """
    # Walk in chronological order, track longest run per tool name.
    longest: defaultdict[str, int] = defaultdict(int)
    current_name: str | None = None
    current_run = 0
    for e in events:
        if e.kind != "tool_result":
            continue
        name = e.detail.get("name") or ""
        if not e.detail.get("success", False):
            if name == current_name:
                current_run += 1
            else:
                current_name = name
                current_run = 1
            longest[name] = max(longest[name], current_run)
        else:
            # success breaks the run
            current_name = None
            current_run = 0
    bad = [name for name, n in longest.items() if n >= _REPEATED_TOOL_ERROR_THRESHOLD]
    if not bad:
        return None
    return Lesson(
        rule=(
            "When a tool fails, don't retry with the same arguments. "
            "Inspect the error, change one thing (args, preconditions, "
            "or strategy), and only then retry. A run of identical "
            "failures is signal to escalate — surface the error to the "
            "user or pick a different tool."
        ),
        kind="when",
        triggers=("tool_result",),
        detector="detect_repeated_tool_error",
    )


def detect_repeated_error_signature(events: Sequence[Event]) -> Lesson | None:
    """The same normalised error signature appeared >= N times in one turn.

    Generalises `detect_repeated_tool_error`: the latter only catches
    consecutive failures of the *same tool with the same args*. This
    detector catches the broader pattern where the SAME error message
    keeps appearing across the turn — across different tools, across
    arg tweaks, anywhere. Maps directly to two real bugs we've already
    debugged:

      - publish-from-chat: "PUBLISH FAILED: settings module unavailable"
        appeared three times across `publish_or_preview` retries.
      - gmail datavault: "Refusing to save empty credential record"
        appeared multiple times even as the LLM tweaked arg shapes.

    Reads `kind in {"tool_result", "scratchpad_result"}`; takes
    `detail.success` and `detail.error`.
    """
    sigs: list[str] = []
    for e in events:
        if e.kind not in ("tool_result", "scratchpad_result"):
            continue
        if e.detail.get("success", True):
            continue
        err = e.detail.get("error") or ""
        if not err:
            continue
        sigs.append(_normalise_error_signature(err))
    if not sigs:
        return None
    top, count = Counter(sigs).most_common(1)[0]
    if count < _REPEATED_ERROR_THRESHOLD:
        return None
    return Lesson(
        rule=(
            "When the same error message appears repeatedly in one turn, "
            "the cause hasn't changed between attempts — retrying won't help. "
            "Stop, inspect what produced the error, and either fix the "
            "precondition, pick a different tool, or surface the failure to "
            "the user. The error signature is the signal; the tool name is not."
        ),
        kind="when",
        triggers=("tool_result", "scratchpad_result"),
        detector="detect_repeated_error_signature",
    )


def detect_reset_churn(events: Sequence[Event]) -> Lesson | None:
    """Scratchpad was reset/cleared >= N times in one turn.

    A reset wipes the venv: all imports, all in-memory state, all
    partial results are gone. One reset can be intentional cleanup
    (the user asked for a clean slate). Two or more means the LLM is
    using reset as a debugging primitive — almost always a mistake,
    because the next cell now has to re-import, re-fetch, and re-
    reason from scratch, burning rounds.

    Reads `kind == "scratchpad_reset"`.
    """
    resets = [e for e in events if e.kind == "scratchpad_reset"]
    if len(resets) < _RESET_CHURN_THRESHOLD:
        return None
    return Lesson(
        rule=(
            "Don't reset the scratchpad to recover from errors. A reset "
            "discards every variable, import, and partial result — the next "
            "cell has to re-fetch and re-import everything. Debug the "
            "failing cell in place: print local state, narrow the scope, "
            "comment out the broken bit. Reach for reset only when the venv "
            "is genuinely corrupted, not when one cell raised."
        ),
        kind="never",
        triggers=("scratchpad_reset",),
        detector="detect_reset_churn",
    )


def detect_kill_loop(events: Sequence[Event]) -> Lesson | None:
    """The same scratchpad name had >= N cells killed (timeout/cancel/OOM).

    Reads `kind == "scratchpad_killed"`; looks at `detail.name`.
    """
    by_name: defaultdict[str, int] = defaultdict(int)
    for e in events:
        if e.kind != "scratchpad_killed":
            continue
        n = e.detail.get("name") or ""
        if n:
            by_name[n] += 1
    if not by_name or max(by_name.values()) < _KILL_LOOP_THRESHOLD:
        return None
    return Lesson(
        rule=(
            "When a scratchpad cell is killed (timeout, cancel, OOM), "
            "the next cell on the same scratchpad needs to be smaller — "
            "fewer rows, smaller batch, explicit timeout, narrower scope. "
            "Two kills on the same scratchpad means the approach itself is "
            "too heavy, not that the same cell needs another try."
        ),
        kind="when",
        triggers=("scratchpad_killed",),
        detector="detect_kill_loop",
    )


def detect_severity_climb(events: Sequence[Event]) -> Lesson | None:
    """A producer emits a strictly-increasing severity sequence ending high.

    Per-producer (grouped by `detail.name`), find a strictly increasing
    severity run of length >= _SEVERITY_CLIMB_LEN whose final element is
    >= _SEVERITY_CLIMB_PEAK. That's the "situation is deteriorating"
    pattern: each attempt fails worse than the last and the LLM keeps
    going. Brain analog: the ACC's error signal climbs as outcomes
    worsen — when it crosses threshold the dlPFC is supposed to switch
    strategy rather than amplify.

    Reads severity from every event; uses `detail.name` to bucket per
    producer. No specific kind required — that's intentional, this
    pattern can show up on scratchpad_result OR tool_result.
    """
    by_name: defaultdict[str, list[int]] = defaultdict(list)
    for e in events:
        n = e.detail.get("name") or ""
        if not n:
            continue
        by_name[n].append(int(e.severity))
    for sevs in by_name.values():
        # Longest strictly-increasing suffix ending at each index.
        run = 1
        for i in range(1, len(sevs)):
            if sevs[i] > sevs[i - 1]:
                run += 1
                if run >= _SEVERITY_CLIMB_LEN and sevs[i] >= _SEVERITY_CLIMB_PEAK:
                    return Lesson(
                        rule=(
                            "When successive results on the same target get "
                            "worse rather than better — escalating severity, "
                            "deeper failures — the current strategy is wrong, "
                            "not under-applied. Stop iterating and switch "
                            "approach: different tool, different decomposition, "
                            "or surface the situation to the user."
                        ),
                        kind="when",
                        triggers=("scratchpad_result", "tool_result"),
                        detector="detect_severity_climb",
                    )
            else:
                run = 1
    return None


def detect_repair_churn(events: Sequence[Event]) -> Lesson | None:
    """`history_repair` fired >= N times in one turn.

    The history-repair pass kicks in when the LLM's tool_use / tool_result
    sequence is structurally malformed (orphaned tool_use, mismatched ids,
    out-of-order results). One repair is a hiccup, two is unlucky, three
    means the conversation is structurally derailing and the LLM is
    burning rounds on recovery instead of progress.

    Reads `kind == "history_repair"`.
    """
    n = sum(1 for e in events if e.kind == "history_repair")
    if n < _REPAIR_CHURN_THRESHOLD:
        return None
    return Lesson(
        rule=(
            "Repeated history-repair events in one turn mean the LLM is "
            "generating malformed tool_use/result pairs structurally — "
            "the conversation is derailing. Pause the loop, surface the "
            "situation to the user, and ask for direction instead of "
            "continuing to retry."
        ),
        kind="when",
        triggers=("history_repair",),
        detector="detect_repair_churn",
    )


def detect_cap_exhausted(events: Sequence[Event]) -> Lesson | None:
    """The turn hit the round cap. Single-occurrence — not a pattern
    detector, but a *trigger* for the post-mortem path.

    Reads `kind == "cap_exhausted"`.
    """
    if not any(e.kind == "cap_exhausted" for e in events):
        return None
    return Lesson(
        rule=(
            "Hitting the round cap means a goal was attempted that didn't "
            "fit the available budget. Don't silently declare done — "
            "produce a post-mortem: what was tried, what failed, what the "
            "next turn should try differently. The user gets context; "
            "future turns get a lesson; nothing is lost in the void."
        ),
        kind="when",
        triggers=("cap_exhausted",),
        detector="detect_cap_exhausted",
    )


# Ordered list — earlier entries get first crack at firing. Order
# matters only when two detectors would produce overlapping lessons;
# `at_end_of_turn` de-dupes anyway, but a stable order makes test
# assertions less brittle.
DETECTORS: tuple[Detector, ...] = (
    detect_name_switch,
    detect_oversized_cell,
    detect_repeated_tool_error,
    detect_repeated_error_signature,
    detect_reset_churn,
    detect_kill_loop,
    detect_severity_climb,
    detect_repair_churn,
    detect_cap_exhausted,
)


# ─────────────────────────────────────────────────────────────────────────────
# The ACC itself — per-session error-detection coordinator
# ─────────────────────────────────────────────────────────────────────────────


class AnteriorCingulate:
    """Per-turn error pattern detector.

    Instantiated once per `ChatSession`. The session calls `observe()`
    whenever a noteworthy event happens (scratchpad result, tool
    failure, history repair, …). At end-of-turn the session calls
    `at_end_of_turn()` to extract any lessons that fell out of the
    event sequence; the wiring layer then routes them through the
    existing `cortex.encode()` path to land in long-term memory.

    Thread-safety: not thread-safe. The session is single-threaded
    per turn; if multiple turns share an ACC instance, the buffer
    needs locking. We intentionally don't add locking here because
    the simpler invariant (one ACC per turn, dropped after) is easier
    to reason about. Wiring should follow that.
    """

    def __init__(
        self,
        *,
        has_similar_lesson: Callable[[str], bool] | None = None,
        detectors: Sequence[Detector] = DETECTORS,
    ) -> None:
        self._events: list[Event] = []
        self._has_similar = has_similar_lesson or (lambda _rule: False)
        self._detectors = tuple(detectors)
        # Names of detectors that have already produced a mid-turn
        # nudge (via at_round_n) this turn. Tracks "newly fired" so
        # the same alarm isn't injected into history on every
        # subsequent round. Reset by `clear()` between turns.
        self._nudged_detectors: set[str] = set()

    def observe(
        self,
        kind: str,
        detail: dict | None = None,
        *,
        severity: int = 1,
        round_idx: int = 0,
    ) -> None:
        """Append an event to the turn's buffer.

        Raises `ValueError` for unknown `kind` so the vocabulary
        stays disciplined. If you need a new kind, add it to
        `EVENT_KINDS` and to at least one detector.
        """
        if kind not in EVENT_KINDS:
            raise ValueError(
                f"Unknown ACC event kind: {kind!r}. "
                f"Add it to EVENT_KINDS in anton/core/memory/acc.py "
                f"and to at least one detector before emitting."
            )
        self._events.append(Event(
            kind=kind,
            severity=int(severity),
            detail=dict(detail or {}),
            round_idx=int(round_idx),
        ))

    def at_end_of_turn(self) -> list[Lesson]:
        """Run every detector against the buffered events.

        Returns lessons that:
          - actually fired (detector returned a non-None Lesson), and
          - aren't already in memory (per `has_similar_lesson`).

        The buffer is NOT cleared automatically — callers may want to
        inspect it after extraction (for logging, telemetry, an
        end-of-task post-mortem). Call `clear()` explicitly between
        turns.
        """
        out: list[Lesson] = []
        seen_rules: set[str] = set()
        for d in self._detectors:
            lesson = d(self._events)
            if lesson is None:
                continue
            # Cross-detector de-dupe by rule text. Two detectors
            # shouldn't produce the same rule, but if a future
            # contributor adds an overlapping pattern this keeps the
            # memory store clean.
            if lesson.rule in seen_rules:
                continue
            if self._has_similar(lesson.rule):
                continue
            seen_rules.add(lesson.rule)
            out.append(lesson)
        return out

    def at_round_n(self) -> list[Lesson]:
        """Run detectors against the current buffer and return only
        *newly-fired* lessons since the previous call this turn.

        Layer 2 — mid-turn nudging. Where `at_end_of_turn()` runs once
        per turn to drain lessons into long-term memory, this runs after
        each tool-call round so the LLM sees the alarm right when the
        pattern is happening, not on the next turn.

        Brain analog: the ACC's ERN fires as soon as a divergence is
        detected. The dlPFC reads the alarm and can adjust strategy on
        the very next action — not at end-of-task.

        Implementation notes:
          - Detectors are pure functions, so re-running them on the
            growing event buffer is cheap. No memoisation.
          - We track *which detectors* have already nudged this turn
            (`self._nudged_detectors`) so the same alarm doesn't get
            injected into history on every subsequent round. One nudge
            per detector per turn is enough — if the LLM ignores it,
            re-stating it round after round won't help and would only
            inflate the history.
          - We deliberately do NOT consult `has_similar_lesson` here.
            For mid-turn nudges we want to re-assert the rule inline
            even when it already lives in `rules.md` — the LLM clearly
            isn't following the in-prompt version, so making it visible
            again in immediate context is the whole point.

        Returns the newly-fired lessons (possibly empty).
        """
        fresh: list[Lesson] = []
        for d in self._detectors:
            name = getattr(d, "__name__", "")
            if name in self._nudged_detectors:
                continue
            lesson = d(self._events)
            if lesson is None:
                continue
            self._nudged_detectors.add(name)
            fresh.append(lesson)
        return fresh

    def clear(self) -> None:
        """Drop the event buffer. Call between turns."""
        self._events.clear()
        self._nudged_detectors.clear()

    # ── Introspection helpers (used by tests + telemetry) ────────────

    @property
    def events(self) -> tuple[Event, ...]:
        """Read-only view of the current turn's events."""
        return tuple(self._events)

    @property
    def event_kind_counts(self) -> dict[str, int]:
        """Histogram of event kinds in the current turn. Cheap; used
        by `__repr__` and by any future per-session telemetry."""
        return dict(Counter(e.kind for e in self._events))

    def __repr__(self) -> str:  # pragma: no cover — debug only
        return f"AnteriorCingulate(events={len(self._events)}, by_kind={self.event_kind_counts})"
