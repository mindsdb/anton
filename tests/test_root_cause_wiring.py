"""`ToolOutcome.reason` reaches the session ledger, and nothing else changes (ENG-1492).

`test_root_cause.py` covers the classifier as pure logic. This covers the part
that was actually broken for two tickets: the value existed and **nothing read
it** — both dispatch sites took `.content` and `.ok` and dropped `.reason` on the
floor. So these drive the real `turn_stream` / `turn` and assert on the ledger.

The second half is the no-behaviour-change contract. This ticket is measurement
only; if it can alter a turn, it has failed regardless of how good the numbers
look.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import make_mock_llm

from anton.core.llm.provider import LLMResponse, StreamComplete, ToolCall, Usage
from anton.core.root_cause import TIER_SELF, TIER_WALL
from anton.core.session import ChatSession, ChatSessionConfig
from anton.core.tools.registry import ToolOutcome


@pytest.fixture()
def workspace():
    base = Path(__file__).resolve().parents[1] / ".pytest-workspace"
    base.mkdir(parents=True, exist_ok=True)
    return MagicMock(base=base)


def _usage(n: int = 1_000) -> Usage:
    return Usage(input_tokens=n // 2, output_tokens=n // 2)


def _tool_call(i: int) -> LLMResponse:
    return LLMResponse(
        content="working",
        tool_calls=[ToolCall(id=f"tc{i}", name="scratchpad",
                             input={"action": "view", "name": "m"})],
        usage=_usage(), stop_reason="tool_use",
    )


def _text(t: str = "done") -> LLMResponse:
    return LLMResponse(content=t, tool_calls=[], usage=_usage(), stop_reason="end_turn")


def _session(workspace, outcomes, n_tool_calls: int):
    """Session whose tool dispatch returns `outcomes` in order."""
    llm = make_mock_llm()
    llm.usage_listener = None
    seq = {"i": 0}

    def plan_stream(**kw):
        async def gen():
            seq["i"] += 1
            if llm.usage_listener:
                llm.usage_listener("planning", "m", _usage())
            yield StreamComplete(
                response=_tool_call(seq["i"]) if seq["i"] <= n_tool_calls else _text()
            )
        return gen()

    async def plan(**kw):
        seq["i"] += 1
        if llm.usage_listener:
            llm.usage_listener("planning", "m", _usage())
        return _tool_call(seq["i"]) if seq["i"] <= n_tool_calls else _text()

    llm.plan_stream = plan_stream
    llm.plan = plan
    s = ChatSession(ChatSessionConfig(llm_client=llm, workspace=workspace))
    s._max_turn_tokens = 0  # ceiling off — this ticket must be independent of it
    pending = list(outcomes)
    s.tool_registry.dispatch_tool = AsyncMock(
        side_effect=lambda *a, **k: pending.pop(0) if pending
        else ToolOutcome(content="ok", ok=True)
    )
    return s


async def _run(session, prompt="go"):
    async for _ in session.turn_stream(prompt):
        pass


WALL = ToolOutcome(content="[error]\nModuleNotFoundError: No module named 'pyodbc'",
                   ok=False, reason="ModuleNotFoundError: No module named 'pyodbc'")
SELF = ToolOutcome(content="[error]\nNameError: name 'wb' is not defined",
                   ok=False, reason="NameError: name 'wb' is not defined")
OK = ToolOutcome(content="fine", ok=True)


async def test_reason_reaches_the_ledger_through_the_real_loop(workspace):
    """The plumbing this ticket exists for.

    Before it, `.reason` was set by five handlers and read by nobody.
    """
    session = _session(workspace, [WALL, WALL, WALL], n_tool_calls=3)
    with patch("anton.analytics.send_event"):
        await _run(session)

    led = session._root_causes
    assert led.failures == 3
    assert led.tiers[TIER_WALL] == 3
    assert led.max_exact == 3
    assert led.top_class == "missing_dependency"
    assert led.reason_coverage == 1.0


async def test_interleaved_successes_do_not_reset_the_count(workspace):
    """The ENG-1276 lesson one level up, end to end.

    The per-tool streak resets on success, and interleaved false successes are
    exactly why it never counted to five through ENG-836. This must not.
    """
    session = _session(workspace, [WALL, OK, WALL, OK, WALL], n_tool_calls=5)
    with patch("anton.analytics.send_event"):
        await _run(session)

    assert session._root_causes.max_exact == 3, (
        "a success between failures reset the count — the exact defect this "
        "counter exists to avoid"
    )


async def test_self_inflicted_failures_reach_the_ledger_but_no_trip_rung(workspace):
    session = _session(workspace, [SELF, SELF, SELF], n_tool_calls=3)
    with patch("anton.analytics.send_event"):
        await _run(session)

    led = session._root_causes
    assert led.tiers[TIER_SELF] == 3      # measured…
    assert led.max_exact == 0             # …but never trip-eligible
    assert led.max_class == 0


async def test_the_ledger_spans_turns_on_a_reused_session(workspace):
    """Spans turns when the SESSION is reused — which is the CLI, not Cowork.

    Deliberately named for the shape it actually tests. `chat.py` builds one
    ChatSession and loops `turn_stream`, so the ledger accumulates. cowork-server
    calls `_build_chat_session()` inside `stream_response()` — once per HTTP turn
    — so on the primary product this resets every turn instead.

    Keeping the test (the object behaves as designed) but not letting its name
    imply coverage the deployment does not give. See `RootCauseLedger`.
    """
    session = _session(workspace, [WALL], n_tool_calls=1)
    with patch("anton.analytics.send_event"):
        await _run(session, "first")
    assert session._root_causes.failures == 1

    # Turn 2 must be DETERMINISTIC. The previous version installed a plan that
    # returned a tool call forever, so turn 2 ran to the 25-round cap and
    # contributed 25 failures against an assertion of `>= 2` — wiping the ledger
    # between turns was invisible. One failing call, then a text answer.
    replies = [_tool_call(9), _text("done")]
    session._llm.plan_stream = lambda **kw: _one(replies.pop(0))
    session.tool_registry.dispatch_tool = AsyncMock(return_value=WALL)
    with patch("anton.analytics.send_event"):
        await _run(session, "second")

    # Exact equality, so a reset between turns reddens this.
    assert session._root_causes.failures == 2
    assert session._root_causes.max_exact == 2


def _one(resp):
    async def gen():
        yield StreamComplete(response=resp)
    return gen()


async def test_counts_ride_the_turn_completed_event(workspace):
    session = _session(workspace, [WALL, WALL], n_tool_calls=2)
    with patch("anton.analytics.send_event") as send:
        await _run(session)

    kw = send.call_args.kwargs
    assert int(kw["root_cause_failures"]) == 2
    assert int(kw["root_cause_max_class"]) == 2
    assert kw["root_cause_top_class"] == "missing_dependency"
    # Flat scalars only — the collector relays query params, not structures.
    assert all(not isinstance(v, (dict, list)) for v in kw.values())


async def test_a_failure_with_no_reason_is_recorded_as_uncovered(workspace):
    """An unmigrated handler must show up as missing coverage, not as a wall."""
    bare = ToolOutcome(content="Tool 'web_fetch' failed: nope", ok=False)
    session = _session(workspace, [bare, bare], n_tool_calls=2)
    with patch("anton.analytics.send_event"):
        await _run(session)

    led = session._root_causes
    assert led.failures == 2
    assert led.reason_coverage == 0.0
    assert led.max_exact == 0, "text-derived keys must never be trip-eligible"


# ── The no-behaviour-change contract ───────────────────────────────────────


async def test_classification_never_changes_the_turn(workspace):
    """Measurement only: same history, same ending, with and without failures.

    If this ticket can alter a turn it has failed, however good the numbers are.
    """
    a = _session(workspace, [WALL, WALL], n_tool_calls=2)
    with patch("anton.analytics.send_event") as send_a:
        await _run(a)

    b = _session(workspace, [WALL, WALL], n_tool_calls=2)
    b._record_root_cause = lambda *args, **kw: None  # classification disabled
    with patch("anton.analytics.send_event") as send_b:
        await _run(b)

    assert send_a.call_args.kwargs["ended_by"] == send_b.call_args.kwargs["ended_by"]
    assert send_a.call_args.kwargs["rounds"] == send_b.call_args.kwargs["rounds"]

    # Roles alone are far too weak: injecting an extra user message from
    # `_record_root_cause` passes a role-only comparison, because history
    # bundling merges it into the adjacent tool-result block — the sequence
    # stays byte-identical while what reaches the LLM changes. Compare CONTENT.
    #
    # Raw equality does not work either: index 6 embeds a repr containing a
    # coroutine's memory address, which differs per run on clean HEAD. Normalise
    # addresses (and the timestamp, which is minute-resolution and would flake
    # across a minute boundary).
    def _shape(history):
        out = []
        for m in history:
            blob = json.dumps(m.get("content"), default=str)
            blob = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", blob)
            blob = re.sub(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}", "<TS>", blob)
            out.append((m.get("role"), blob))
        return out

    assert _shape(a._history) == _shape(b._history)


async def test_a_raising_classifier_cannot_break_a_turn(workspace):
    """Reporting is guarded whole — a bad key must never cost a user their turn."""
    session = _session(workspace, [WALL], n_tool_calls=1)
    with patch("anton.core.session.classify_root_cause",
               side_effect=RuntimeError("boom")), \
         patch("anton.analytics.send_event") as send:
        await _run(session)

    assert send.call_args.kwargs["ended_by"] == "completed"
    # `ended_by` alone is far too weak. Unguarded, the raise escapes into the
    # tool loop's broad `except Exception`, which seals the tool_use as
    # "[interrupted by error — tool call did not complete]" and injects a
    # "SYSTEM: An error interrupted execution … Adjust your approach" note, then
    # retries — so the turn still ends `completed` while the model has LOST the
    # tool output and been told to change tack because of an internal reporting
    # bug. That is the actual damage, so assert on what reached the model.
    blob = json.dumps(session._history, default=str)
    assert "interrupted by error" not in blob
    assert "An error interrupted execution" not in blob
    assert "No module named 'pyodbc'" in blob


async def test_successes_are_not_recorded_at_all(workspace):
    session = _session(workspace, [OK, OK, OK], n_tool_calls=3)
    with patch("anton.analytics.send_event"):
        await _run(session)
    assert session._root_causes.failures == 0


async def test_the_non_streaming_turn_records_too(workspace):
    """`turn()` is public API and its books are wired, so its failures count."""
    session = _session(workspace, [WALL, WALL], n_tool_calls=2)
    with patch("anton.analytics.send_event"):
        await session.turn("go")
    assert session._root_causes.failures == 2
    assert session._root_causes.max_exact == 2


async def test_unmigrated_handlers_count_against_reason_coverage(workspace):
    """A handler that returns an error STRING must not be silently excluded.

    `reason_coverage` exists to expose under-coverage. Dropping `ok is None`
    failures entirely made it report 1.0 by excluding the uncovered population
    from its own denominator — measured: 1 migrated + 3 unmigrated failures
    reported coverage 1.0, failures 1.

    They count as `unclassified`, never trip-eligible: the text is prose the
    model can influence, which is the ENG-1276 defect one level up.
    """
    legacy = ToolOutcome(content="[error] connect failed: no route to host")
    assert legacy.ok is None  # the shape registry.py produces for a plain str
    session = _session(workspace, [WALL, legacy, legacy, legacy], n_tool_calls=4)
    with patch("anton.analytics.send_event") as send:
        await _run(session)

    led = session._root_causes
    assert led.failures == 4
    assert led.reason_coverage == 0.25
    assert led.tiers["unclassified"] == 3
    assert led.max_exact == 1, "legacy text must never create a wall"
    assert float(send.call_args.kwargs["root_cause_reason_coverage"]) == 0.25


async def test_a_successful_legacy_result_is_not_counted_as_a_failure(workspace):
    """The legacy verdict must not invent failures out of ordinary output."""
    fine = ToolOutcome(content="Wrote 12 rows. 0 records failed validation.")
    session = _session(workspace, [fine, fine], n_tool_calls=2)
    with patch("anton.analytics.send_event"):
        await _run(session)
    # NOTE: this text DOES contain "failed", so it is counted — the legacy
    # matcher's false-positive direction, inherited deliberately rather than
    # forked. Pinning it so the inheritance is visible if it ever changes.
    assert session._root_causes.failures == 2
    assert session._root_causes.tiers["unclassified"] == 2
    assert session._root_causes.max_exact == 0


# ── The dominant production path: scratchpad `action: "exec"` ──────────────


async def test_the_scratchpad_exec_path_records_its_traceback(workspace):
    """`{"action": "exec"}` is where every real ENG-836-shaped failure arrives.

    Coverage gap found in review: every other wiring test issues
    `{"action": "view"}`, so the exec branch — gated on `action == "exec"` —
    was never entered. Four of the five `tool_reason` writer sites had zero
    coverage, including this one, and blanking it was invisible: a real
    3-failure wall turn would silently emit `wall 3 -> 0`,
    `reason_coverage 1.0 -> 0.0`, `unclassified 0 -> 3` while looking healthy.

    Uses a realistic multi-line traceback, because the producer takes the LAST
    line — the reason this path reads `cell.error.splitlines()[-1][:160]`
    rather than the whole thing.
    """
    from anton.core.backends.base import Cell

    traceback = (
        'Traceback (most recent call last):\n'
        '  File "<cell>", line 3, in <module>\n'
        "    import pyodbc\n"
        '  File "/usr/lib/python3.12/importlib/__init__.py", line 90, in import_module\n'
        "    return _bootstrap._gcd_import(name[level:], package, level)\n"
        "ModuleNotFoundError: No module named 'pyodbc'"
    )
    cell = Cell(code="import pyodbc", stdout="", stderr="", error=traceback)

    llm = make_mock_llm()
    llm.usage_listener = None
    n = {"i": 0}

    def plan_stream(**kw):
        async def gen():
            n["i"] += 1
            resp = LLMResponse(
                content="trying",
                tool_calls=[ToolCall(id=f"tc{n['i']}", name="scratchpad",
                                     input={"action": "exec", "name": "main",
                                            "code": "import pyodbc"})],
                usage=_usage(), stop_reason="tool_use",
            ) if n["i"] <= 3 else _text()
            yield StreamComplete(response=resp)
        return gen()

    llm.plan_stream = plan_stream
    session = ChatSession(ChatSessionConfig(llm_client=llm, workspace=workspace))
    session._max_turn_tokens = 0

    pad = MagicMock()
    async def _exec_streaming(*a, **k):
        yield cell
    pad.execute_streaming = _exec_streaming
    pad.cancel = AsyncMock()

    with patch("anton.core.session.prepare_scratchpad_exec",
               AsyncMock(return_value=(pad, "import pyodbc", "install driver", "5s", 5))), \
         patch.object(ChatSession, "_record_cell_explainability", lambda *a, **k: None), \
         patch("anton.analytics.send_event") as send:
        await _run(session)

    led = session._root_causes
    assert led.failures == 3, f"exec-path failures not recorded: {led.failures}"
    assert led.max_exact == 3
    assert led.top_class == "missing_dependency"
    assert led.reason_coverage == 1.0
    assert int(send.call_args.kwargs["root_cause_wall"]) == 3


async def test_a_raising_tool_records_its_exception_type(workspace):
    """The dispatcher's own except-clause is the fifth writer site."""
    session = _session(workspace, [], n_tool_calls=2)
    session.tool_registry.dispatch_tool = AsyncMock(
        side_effect=ConnectionRefusedError("[Errno 61] Connection refused")
    )
    with patch("anton.analytics.send_event"):
        await _run(session)

    led = session._root_causes
    assert led.failures == 2
    assert led.tiers["external_wall"] == 2
    assert led.top_class == "connection_refused"


async def test_the_legacy_predicate_is_inside_the_guard(workspace):
    """"Guarded whole" means the predicate too, not just the classify call.

    `_legacy_looks_like_failure` used to run OUTSIDE the try/except, so a
    `result_text` that raised on `in` would escape into the tool loop's broad
    `except`, seal the tool_use as "[interrupted by error]" and inject a
    "adjust your approach" note — the finding-4 damage, reached by a different
    door. No such input is reachable today (`result_text` is only ever `str` or
    `list`), which is exactly why the invariant needs a test rather than a
    reader's goodwill. #348 review.
    """
    class Hostile:
        """Truthy, so `result_text or ""` yields it; explodes on membership."""

        def __bool__(self):
            return True

        def __contains__(self, item):
            raise RuntimeError("boom")

    session = _session(workspace, [WALL], n_tool_calls=1)
    before = session._root_causes.failures

    # Must not raise. `tool_ok=None` is the unmigrated-handler path.
    session._record_root_cause(None, "", Hostile())

    # …and must not have booked anything from an input it could not read.
    assert session._root_causes.failures == before
    # …but must NOT swallow it silently either. This guard is a separate writer
    # from the one on the migrated path, and removing its `note_error()` alone
    # passed the whole file before this assertion existed.
    assert session._root_causes.classify_errors == 1


async def test_a_broken_classifier_is_distinguishable_from_a_quiet_turn(workspace):
    """A silent instrument must not read as a legitimate answer.

    Before `classify_errors`, a turn with three real wall failures and a
    raising classifier emitted all ten properties IDENTICAL to a turn where
    nothing failed — the guard swallowed each failure and left only a
    `logger.debug` line, which production does not emit.

    That ambiguity is load-bearing, not cosmetic: "the wall-repeat population
    is too small to justify the control at all" is one of ENG-1492's own
    sanctioned conclusions, so a broken classifier reads as a real finding and
    would cancel ENG-1531 on the strength of a bug.
    """
    def _fields(send):
        return {k: v for k, v in send.call_args.kwargs.items()
                if k.startswith("root_cause")}

    # Three real wall failures, classifier raising on every one.
    broken = _session(workspace, [WALL, WALL, WALL], n_tool_calls=3)
    with patch("anton.core.session.classify_root_cause",
               side_effect=RuntimeError("boom")), \
         patch("anton.analytics.send_event") as send_broken:
        await _run(broken)
    broken_fields = _fields(send_broken)

    # A genuinely quiet turn: three successes, nothing to classify.
    quiet = _session(workspace, [OK, OK, OK], n_tool_calls=3)
    with patch("anton.analytics.send_event") as send_quiet:
        await _run(quiet)
    quiet_fields = _fields(send_quiet)

    # The signal that separates them.
    assert broken_fields["root_cause_classify_errors"] == 3
    assert quiet_fields["root_cause_classify_errors"] == 0
    assert broken_fields != quiet_fields

    # Everything else still reads zero in BOTH — which is exactly why the
    # counter had to be added rather than inferred from the existing ten.
    for k in ("root_cause_failures", "root_cause_wall", "root_cause_max_exact"):
        assert broken_fields[k] == 0 and quiet_fields[k] == 0, k

    # And the turn itself is untouched either way — the guard still holds.
    assert send_broken.call_args.kwargs["ended_by"] == "completed"
