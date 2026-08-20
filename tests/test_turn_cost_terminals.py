"""Terminal-path classification for the per-turn cost books (ENG-1288).

Each test here corresponds to a #309 review finding that a real production
exit filed the wrong ``ended_by`` (or emitted nothing at all). They drive the
real ``turn_stream``/``turn`` rather than poking the classifier, because every
one of these bugs lived in the wiring, not the arithmetic.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import make_mock_llm

from anton.core.llm.provider import (
    LLMResponse,
    StreamComplete,
    StreamTextDelta,
    TokenLimitExceeded,
    ToolCall,
    Usage,
)
from anton.core.session import ChatSession, ChatSessionConfig, _VerifierVerdict


@pytest.fixture()
def workspace():
    base = Path(__file__).resolve().parents[1] / ".pytest-workspace"
    base.mkdir(parents=True, exist_ok=True)
    return MagicMock(base=base)


def _text(text: str = "done") -> LLMResponse:
    return LLMResponse(
        content=text, tool_calls=[], usage=Usage(input_tokens=10, output_tokens=5),
        stop_reason="end_turn",
    )


def _tool_call(i: int = 1) -> LLMResponse:
    return LLMResponse(
        content="working",
        tool_calls=[ToolCall(id=f"tc_{i}", name="scratchpad",
                             input={"action": "view", "name": "main"})],
        usage=Usage(input_tokens=10, output_tokens=5),
        stop_reason="tool_use",
    )


class _Iter:
    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


def _stub_tools(session) -> None:
    """Answer every tool call in-process.

    These tests only need the tool LOOP to turn over; letting the real
    scratchpad handler run would spawn a venv/subprocess per test (slow, and
    it leaks event-loop teardown noise).
    """
    from anton.core.tools.registry import ToolOutcome

    session.tool_registry.dispatch_tool = AsyncMock(
        return_value=ToolOutcome(content="stubbed tool result", ok=True)
    )


def _ended_by(send_event_mock) -> str:
    assert send_event_mock.called, "no turn_completed event emitted"
    return send_event_mock.call_args.kwargs["ended_by"]


async def test_task_cancel_reports_cancelled_not_error(workspace):
    """cowork-server's Stop is ``task.cancel()`` → ``asyncio.CancelledError``
    inside the suspended generator, and nothing there sets ``_cancel_event``.
    Before the fix every user Stop on the primary host filed as ``error``.
    """
    mock_llm = make_mock_llm()
    started = asyncio.Event()

    def _hang(**kwargs):
        async def _gen():
            started.set()
            await asyncio.sleep(30)  # cancelled here, mid-await
            yield StreamComplete(response=_text())
        return _gen()

    mock_llm.plan_stream = _hang
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))

    with patch("anton.analytics.send_event") as send:
        async def _consume():
            async for _ in session.turn_stream("hang please"):
                pass

        task = asyncio.create_task(_consume())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert _ended_by(send) == "cancelled"
        assert not session._cancel_event.is_set(), (
            "guard: the fix must not depend on _cancel_event, which the "
            "hosted path never sets"
        )


async def test_abandoned_generator_still_emits_exactly_once(workspace):
    """Abandoning mid-iteration still books the turn exactly once, as
    ``cancelled``.

    Scope note (#309 fix-verification): this is the SAME-TASK abandon path.
    ``aclose()`` here runs the ``finally`` in the original context, so
    ``reset_trace_context`` succeeds and the cross-context ValueError never
    fires — this test therefore does NOT cover the emit-before-reset fix.
    That mechanism is pinned by
    ``test_emit_survives_reset_trace_context_valueerror`` below; keep both.
    """
    mock_llm = make_mock_llm()

    def _stream(**kwargs):
        async def _gen():
            yield StreamTextDelta(text="partial")
            await asyncio.sleep(30)
            yield StreamComplete(response=_text())
        return _gen()

    mock_llm.plan_stream = _stream
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))

    with patch("anton.analytics.send_event") as send:
        gen = session.turn_stream("start then abandon")
        async for _ in gen:
            break  # abandon mid-iteration
        # Same-task close — NOT what a deferred finalizer does (that runs in a
        # fresh task with a copied context); see the scope note above.
        await gen.aclose()

        assert send.call_count == 1
        assert _ended_by(send) == "cancelled"


async def test_emit_survives_reset_trace_context_valueerror(workspace):
    """The real deferred-finalizer condition: ``reset_trace_context`` raises.

    asyncio runs an abandoned async generator's ``finally`` in a fresh task
    with a COPIED context, where resetting a token created elsewhere raises
    ``ValueError: Token ... created in a different Context``. That used to
    abort the finally before emission — so an ESC-cancelled CLI turn was
    never counted at all. Patching the raise is deliberate: the ValueError
    *is* the mechanism, and forcing a genuine deferred finalizer is
    timing-dependent and flaky by comparison (#309 fix-verification).
    """
    mock_llm = make_mock_llm()
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))

    def _boom(_token):
        raise ValueError("Token was created in a different Context")

    with patch("anton.core.session.reset_trace_context", side_effect=_boom), \
         patch("anton.analytics.send_event") as send:
        async for _ in session.turn_stream("hello"):
            pass

        assert send.call_count == 1, "books must emit even when reset raises"


async def test_late_finalizer_cannot_close_a_newer_turns_books(workspace):
    """The books guard: a finalizer arriving after the next turn opened its
    own books must no-op instead of emitting the new turn's partial totals.
    """
    from anton.core.turn_cost import TurnCost

    session = ChatSession.__new__(ChatSession)
    session._llm = MagicMock()
    session._llm.planning_model = "p"
    session._llm.coding_model = "c"
    session._session_id = "s"
    session._harness = "cowork"
    session._turn_count = 1
    session._cancel_event = MagicMock(is_set=lambda: False)
    session._settings = None

    stale, current = TurnCost(), TurnCost()
    stale.add("planning", "sonnet", Usage(input_tokens=777, output_tokens=3))
    session._turn_cost = current
    with patch("anton.analytics.send_event") as send:
        # The invariant is narrow: a late finalizer must not close a NEWER
        # turn's books — NOT that it must emit nothing. The stale books hold a
        # complete, real turn (often the runaway the user just cancelled), so
        # they get reported; only the owning turn may clear the shared slot
        # (#309 review — the earlier version of this test codified the drop).
        session._emit_turn_cost(expected=stale)
        assert send.called, "the abandoned turn must still be counted"
        assert send.call_args.kwargs["tokens_total"] == "780"
        assert session._turn_cost is current, "current turn keeps its books"

        # Re-emitting the same books is still a no-op (now via `emitted`).
        send.reset_mock()
        session._emit_turn_cost(expected=stale)
        assert not send.called

        session._emit_turn_cost(expected=current)
        assert send.called
        assert session._turn_cost is None, "the owner clears the slot"


async def test_exhausted_retries_is_not_reported_as_completed(workspace):
    """Every LLM call failing → the retry loop yields an apology and breaks.
    Nothing is in flight in the finally, so the default ``completed`` used to
    hold — filing the most common failure mode as a clean turn.
    """
    mock_llm = make_mock_llm()
    mock_llm.plan_stream = MagicMock(side_effect=RuntimeError("provider down"))
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))

    with patch("anton.analytics.send_event") as send:
        async for _ in session.turn_stream("do something"):
            pass

    assert _ended_by(send) == "retry_exhausted"


async def test_rounds_accumulate_across_verifier_continuations(workspace):
    """``tool_round`` is loop-local and resets on each continuation, so
    assigning it reported only the last continuation's rounds — breaking the
    rounds x context shape metric for exactly the expensive turns it exists
    to classify.
    """
    mock_llm = make_mock_llm()
    # 2 tool rounds, reply; INCOMPLETE verdict; 1 tool round, reply; COMPLETE.
    plans = [
        _Iter([StreamComplete(response=_tool_call(1))]),
        _Iter([StreamComplete(response=_tool_call(2))]),
        _Iter([StreamComplete(response=_text("first pass"))]),
        _Iter([StreamComplete(response=_tool_call(3))]),
        _Iter([StreamComplete(response=_text("second pass"))]),
    ]
    mock_llm.plan_stream = MagicMock(side_effect=lambda **kw: plans.pop(0))
    verdicts = [
        _VerifierVerdict(status="INCOMPLETE", reason="more to do"),
        _VerifierVerdict(status="COMPLETE", reason="done"),
    ]
    mock_llm.generate_object_code = AsyncMock(side_effect=verdicts)

    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))
    _stub_tools(session)
    with patch("anton.analytics.send_event") as send:
        async for _ in session.turn_stream("multi-step task"):
            pass

    kwargs = send.call_args.kwargs
    assert kwargs["continuations"] == "1"
    assert kwargs["rounds"] == "3", (
        f"rounds must span continuations (2 + 1), got {kwargs['rounds']}"
    )


async def test_round_cap_reports_the_cap_not_cap_plus_one(workspace):
    mock_llm = make_mock_llm()
    mock_llm.plan_stream = MagicMock(
        side_effect=lambda **kw: _Iter([StreamComplete(response=_tool_call())])
    )
    mock_llm.generate_object_code = AsyncMock(
        return_value=_VerifierVerdict(status="COMPLETE", reason="ok")
    )
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))
    _stub_tools(session)
    session._max_tool_rounds = 2

    with patch("anton.analytics.send_event") as send:
        async for _ in session.turn_stream("loop forever"):
            pass

    kwargs = send.call_args.kwargs
    assert kwargs["ended_by"] == "round_cap"
    assert kwargs["rounds"] == "2", f"cap is 2, reported {kwargs['rounds']}"


async def test_non_streaming_turn_marks_round_cap(workspace):
    mock_llm = make_mock_llm()
    mock_llm.plan = AsyncMock(return_value=_tool_call())
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))
    _stub_tools(session)
    session._max_tool_rounds = 1
    session._router_enabled = False

    with patch("anton.analytics.send_event") as send:
        await session.turn("loop forever")

    assert _ended_by(send) == "round_cap"


# ---------------------------------------------------------------------------
# The three hand-back terminals. Every one of these marks survived deletion
# with the suite green (#309 review mutation run) — and they are precisely the
# "expensive turn ended badly" values the `group by ended_by` contract needs.
# ---------------------------------------------------------------------------


def _verdict_llm(status: str, *, tool_rounds: int = 1):
    """A mock client whose turn does `tool_rounds` rounds then gets `status`."""
    mock_llm = make_mock_llm()
    plans = [_Iter([StreamComplete(response=_tool_call(i))]) for i in range(tool_rounds)]
    plans.append(_Iter([StreamComplete(response=_text("reply"))]))
    # Any further re-entries (continuations) just reply again.
    def _next_plan(**kw):
        return plans.pop(0) if plans else _Iter(
            [StreamComplete(response=_text("reply again"))]
        )
    mock_llm.plan_stream = MagicMock(side_effect=_next_plan)
    mock_llm.generate_object_code = AsyncMock(
        return_value=_VerifierVerdict(status=status, reason="because")
    )
    return mock_llm


async def test_stuck_verdict_reports_handback_stuck(workspace):
    session = ChatSession(
        ChatSessionConfig(llm_client=_verdict_llm("STUCK"), workspace=workspace)
    )
    _stub_tools(session)
    with patch("anton.analytics.send_event") as send:
        async for _ in session.turn_stream("connect to the database"):
            pass
    assert _ended_by(send) == "handback_stuck"


async def test_exhausted_continuations_report_handback_budget(workspace):
    # INCOMPLETE forever + no continuation budget = the budget-exhausted
    # hand-back, the path ENG-1155 rewrote.
    session = ChatSession(
        ChatSessionConfig(llm_client=_verdict_llm("INCOMPLETE"), workspace=workspace)
    )
    _stub_tools(session)
    session._max_continuations = 0
    with patch("anton.analytics.send_event") as send:
        async for _ in session.turn_stream("do a multi-part job"):
            pass
    assert _ended_by(send) == "handback_budget"


async def test_verifier_failure_reports_handback_verifier_failure(workspace):
    # The verdict call itself failing (ENG-1079's fail-safe), not a verdict.
    mock_llm = _verdict_llm("COMPLETE")
    mock_llm.generate_object_code = AsyncMock(side_effect=RuntimeError("provider hiccup"))
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))
    _stub_tools(session)
    with patch("anton.analytics.send_event") as send:
        async for _ in session.turn_stream("run my script"):
            pass
    assert _ended_by(send) == "handback_verifier_failure"


async def test_denied_verdict_reports_completed_not_verifier_failure(workspace):
    # ENG-1632: a deterministic denial (wallet 402 → TokenLimitExceeded)
    # latches silently — the turn's work succeeded and the user saw a normal
    # reply, so the terminal is "completed", NOT handback_verifier_failure.
    # This is a deliberate taxonomy decision: `group by ended_by` error-rate
    # queries must not count a priced-out completion check as a broken turn,
    # and the denied probe stays countable from the gateway-side ERROR trace
    # in Langfuse, so no signal is lost by booking the turn as completed.
    mock_llm = _verdict_llm("COMPLETE")
    mock_llm.generate_object_code = AsyncMock(
        side_effect=TokenLimitExceeded(
            "402: Your wallet has no balance to cover the model 'haiku'."
        )
    )
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))
    _stub_tools(session)
    with patch("anton.analytics.send_event") as send:
        async for _ in session.turn_stream("run my script"):
            pass
    assert _ended_by(send) == "completed"
    # ...but never byte-identical to a verified pass: the flag is what lets
    # honest-stop denominators exclude unverified turns without a
    # per-conversation Langfuse hop (ENG-1632 review).
    assert send.call_args.kwargs["verification_skipped"] == "true"


async def test_verified_turn_reports_verification_not_skipped(workspace):
    # The control for the flag above: a turn whose verdict actually ran
    # (COMPLETE) books verification_skipped="false".
    session = ChatSession(
        ChatSessionConfig(llm_client=_verdict_llm("COMPLETE"), workspace=workspace)
    )
    _stub_tools(session)
    with patch("anton.analytics.send_event") as send:
        async for _ in session.turn_stream("run my script"):
            pass
    assert _ended_by(send) == "completed"
    assert send.call_args.kwargs["verification_skipped"] == "false"


async def test_callers_already_handled_exception_is_not_reported_as_error(workspace):
    """`sys.exc_info()` returns whatever the thread is handling — including an
    exception the CALLER already caught. Asking the interpreter instead of
    capturing this turn's own exception reported `error` for a clean turn
    (#309 review). Latent today; `ended_by` is what error-rate queries key on.
    """
    mock_llm = make_mock_llm()
    mock_llm.plan_stream = MagicMock(
        side_effect=lambda **kw: _Iter([StreamComplete(response=_text("hi"))])
    )
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))

    with patch("anton.analytics.send_event") as send:
        try:
            raise ValueError("caller's own, already-handled error")
        except ValueError:
            async for _ in session.turn_stream("hello"):
                pass

    assert _ended_by(send) == "completed"


async def test_abandoned_turn_keeps_its_own_index_not_a_later_turns(workspace):
    """Per-turn facts are stamped at books-open, not read at emit.

    A late finalizer emits books whose turn ended long ago; reading
    ``self._turn_count`` there gave the abandoned turn a LATER, unrelated
    turn's index (#309 review follow-up). Sequence: abandon turn 1 → finish
    turn 2 → finalize turn 1 → finish turn 3.

    Note on uniqueness, which is NOT what this asserts: ``_turn_count``
    advances only after a turn completes, so an abandoned turn and its
    successor legitimately share an index — and the Langfuse trace context
    derives its ``turn_id`` from the *same* expression (``session.py`` where
    ``TraceContext`` is built). So a shared index means "two attempts at one
    conversational turn" on both sides, and the cost→trace hop stays
    consistent. Claiming `(conversation_id, turn_index)` were a unique event
    key would have been wrong; it is a join key.
    """
    mock_llm = make_mock_llm()
    hang = asyncio.Event()

    def _stream(**kwargs):
        async def _gen():
            yield StreamTextDelta(text="partial")
            await hang.wait()
            yield StreamComplete(response=_text())
        return _gen()

    def _quick(**kwargs):
        return _Iter([StreamComplete(response=_text("done"))])

    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))

    with patch("anton.analytics.send_event") as send:
        mock_llm.plan_stream = _stream
        abandoned = session.turn_stream("turn one — abandoned")
        async for _ in abandoned:
            break                                    # turn 1 left open

        mock_llm.plan_stream = _quick
        async for _ in session.turn_stream("turn two"):
            pass                                     # completes -> _turn_count 1

        await abandoned.aclose()                     # turn 1's late finalizer

        async for _ in session.turn_stream("turn three"):
            pass

    events = [(c.kwargs["ended_by"], c.kwargs["turn_index"]) for c in send.call_args_list]
    assert len(events) == 3, f"expected 3 events, got {events}"

    # The abandoned turn is the cancelled one. It opened as turn 1 and must
    # still say 1 — reading live state at emit would report 2 (turn 3's).
    cancelled = [idx for ended, idx in events if ended == "cancelled"]
    assert cancelled == ["1"], f"abandoned turn drifted to a later index: {events}"


async def test_late_emit_duration_is_bounded_by_last_activity(workspace):
    """A late finalizer must not measure duration up to whenever asyncio ran
    it — that inflates the field a runaway query sorts on. With no owning turn
    ending, the last LLM call is the bounded approximation used instead.
    """
    from anton.core.turn_cost import TurnCost

    session = ChatSession.__new__(ChatSession)
    session._llm = MagicMock()
    session._llm.planning_model = "p"
    session._llm.coding_model = "c"
    session._session_id = "conv"
    session._harness = "cli"
    session._turn_count = 5
    session._cancel_event = MagicMock(is_set=lambda: False)
    session._settings = None

    stale = TurnCost(turn_index=2)
    stale.add("planning", "sonnet", Usage(input_tokens=10, output_tokens=1))
    stale.last_activity_monotonic = stale.started_monotonic + 0.010  # ~10ms of work
    session._turn_cost = TurnCost(turn_index=6)                      # newer owner

    await asyncio.sleep(0.05)  # the finalizer fires much later

    with patch("anton.analytics.send_event") as send:
        session._emit_turn_cost(expected=stale)

    kw = send.call_args.kwargs
    assert kw["turn_index"] == "2"
    assert int(kw["duration_ms"]) < 40, (
        f"duration measured to finalizer time, not turn end: {kw['duration_ms']}ms"
    )


async def test_host_supplied_turn_id_is_what_the_event_reports(workspace):
    """The cost event's turn_index must equal the trace's turn_id.

    `turn_stream(turn_id=...)` exists so a host can supply its own identifier
    "so downstream telemetry can correlate" — and the trace context prefers it.
    Deriving the books' index independently from `_turn_count` agreed only
    because no host passes `turn_id` today; if one did, the cost event and the
    Langfuse trace would name different turns — the exact defect the stamping
    exists to prevent (#309 review follow-up, second pass).
    """
    mock_llm = make_mock_llm()
    mock_llm.plan_stream = MagicMock(
        side_effect=lambda **kw: _Iter([StreamComplete(response=_text("hi"))])
    )
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))

    with patch("anton.analytics.send_event") as send:
        async for _ in session.turn_stream("hello", turn_id=4242):
            pass

    assert send.call_args.kwargs["turn_index"] == "4242", (
        "the event must report the host's turn_id, not the internal counter"
    )


async def test_latched_skip_turns_also_stamp_verification_skipped(workspace):
    # The steady-state site: after the denied latch fires (turn 1), every
    # later turn in the session takes the LATCHED-SKIP path — a different
    # stamp site three lines long and visually identical to the tested one,
    # which is exactly the shape that ships unpinned (review on #357; same
    # pattern as anton#348's legacy guard and anton#339's PROGRESS_MARKER
    # writer). Two turns, one session: turn 2's event must carry the flag.
    mock_llm = _verdict_llm("COMPLETE", tool_rounds=1)
    mock_llm.generate_object_code = AsyncMock(
        side_effect=TokenLimitExceeded(
            "402: Your wallet has no balance to cover the model 'haiku'."
        )
    )
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))
    _stub_tools(session)
    rows = []
    with patch("anton.analytics.send_event") as send:
        for i in range(2):
            # _verdict_llm's plan list is consumed by turn 1; re-arm per turn.
            plans = [
                _Iter([StreamComplete(response=_tool_call(1))]),
                _Iter([StreamComplete(response=_text("reply"))]),
            ]
            mock_llm.plan_stream = MagicMock(
                side_effect=lambda **kw: plans.pop(0) if plans else _Iter(
                    [StreamComplete(response=_text("reply again"))]
                )
            )
            async for _ in session.turn_stream(f"step {i}"):
                pass
            k = send.call_args.kwargs
            rows.append((k["ended_by"], k["verification_skipped"]))
    # Turn 1 = the denied-latch site; turn 2 = the latched-skip site.
    assert rows == [("completed", "true"), ("completed", "true")]


async def test_hard_latch_and_failed_reprobe_turns_also_stamp_verification_skipped(workspace):
    # The remaining two stamp sites: the second-hard-failure latch (turn 2)
    # and the failed re-probe (the turn after _VERIFIER_LATCH_REPROBE_TURNS
    # skips). Turn 1 hands back (ended_by=handback_verifier_failure — its
    # terminal already distinguishes it, no flag needed); everything after
    # books "completed" without a verdict and must carry the flag.
    from anton.core.session import _VERIFIER_LATCH_REPROBE_TURNS

    mock_llm = _verdict_llm("COMPLETE", tool_rounds=1)
    mock_llm.generate_object_code = AsyncMock(
        side_effect=RuntimeError("400 tool_choice not supported")
    )
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))
    _stub_tools(session)
    rows = []
    with patch("anton.analytics.send_event") as send:
        for i in range(_VERIFIER_LATCH_REPROBE_TURNS + 2):
            plans = [
                _Iter([StreamComplete(response=_tool_call(1))]),
                _Iter([StreamComplete(response=_text("reply"))]),
            ]
            mock_llm.plan_stream = MagicMock(
                side_effect=lambda **kw: plans.pop(0) if plans else _Iter(
                    [StreamComplete(response=_text("diagnosis or reply"))]
                )
            )
            async for _ in session.turn_stream(f"step {i}"):
                pass
            k = send.call_args.kwargs
            rows.append((k["ended_by"], k["verification_skipped"]))
    # Turn 1: honest diagnosis, distinguished by its own terminal.
    assert rows[0] == ("handback_verifier_failure", "false")
    # Turn 2: the second-hard-failure latch site.
    assert rows[1] == ("completed", "true")
    # Every later turn — the latched skips AND the failed re-probe that falls
    # inside this window — books completed and must carry the flag.
    assert all(r == ("completed", "true") for r in rows[2:]), rows[2:]


# ─── error_type: naming the failure, not just counting it (ENG-1689) ─────────

# Every session here carries a `session_id` deliberately: ENG-1692's guard
# suppresses `turn_completed` for a turn with no session id AND zero LLM
# calls (script traffic), which is exactly the shape of a pre-model failure.
# Without it these tests pass alone and fail the moment anton#379 lands —
# a break neither PR's CI can see, since each is green on its own base.
def _error_type(send_event_mock) -> str:
    assert send_event_mock.called, "no turn_completed event emitted"
    return send_event_mock.call_args.kwargs["error_type"]


async def test_retry_exhausted_names_the_exception_class(workspace):
    """`retry_exhausted` was 6.7% of real turns with a median of ZERO LLM
    calls — dying before the first model call — and said nothing about why.
    The exception was already in hand: the retry loop formats it into a SYSTEM
    message the model and the user both read, then dropped it.
    """
    mock_llm = make_mock_llm()
    mock_llm.plan_stream = MagicMock(side_effect=RuntimeError("provider down"))
    session = ChatSession(
        ChatSessionConfig(
            llm_client=mock_llm, workspace=workspace, session_id="conv-t"
        )
    )

    with patch("anton.analytics.send_event") as send:
        async for _ in session.turn_stream("do something"):
            pass

    assert _ended_by(send) == "retry_exhausted"
    assert _error_type(send) == "RuntimeError"


async def test_retry_exhausted_reports_the_specific_provider_exception(workspace):
    """The value of the field is discriminating between causes, so a second
    exception type must produce a different label rather than a generic one.
    """
    mock_llm = make_mock_llm()
    mock_llm.plan_stream = MagicMock(side_effect=TimeoutError("upstream stalled"))
    session = ChatSession(
        ChatSessionConfig(
            llm_client=mock_llm, workspace=workspace, session_id="conv-t"
        )
    )

    with patch("anton.analytics.send_event") as send:
        async for _ in session.turn_stream("do something"):
            pass

    assert _ended_by(send) == "retry_exhausted"
    assert _error_type(send) == "TimeoutError"


async def test_error_terminal_names_the_exception_class(workspace):
    """The catch-all. `ended_by="error"` covered a parse failure, a tool
    crash, a bad config and an anton bug alike, with the exception object in
    scope at the assignment and discarded.

    A `BaseException` is what reaches this branch — the retry loop catches
    `Exception`, so only something outside that hierarchy escapes to the
    outer handler. `KeyboardInterrupt` is the real-world instance of that
    (Ctrl-C in the CLI) and is not one of the cancel shapes that would file
    as `cancelled`.
    """
    mock_llm = make_mock_llm()
    mock_llm.plan_stream = MagicMock(side_effect=KeyboardInterrupt())
    session = ChatSession(
        ChatSessionConfig(
            llm_client=mock_llm, workspace=workspace, session_id="conv-t"
        )
    )

    with patch("anton.analytics.send_event") as send:
        with pytest.raises(KeyboardInterrupt):
            async for _ in session.turn_stream("do something"):
                pass

    assert _ended_by(send) == "error"
    assert _error_type(send) == "KeyboardInterrupt"


async def test_completed_turn_carries_no_error_type(workspace):
    """Empty, not "none" or "unknown" — a clean turn has no exception, and a
    placeholder would pollute the distribution this field exists to produce.
    """
    mock_llm = make_mock_llm()
    mock_llm.plan_stream = MagicMock(
        side_effect=lambda **kw: _Iter([StreamComplete(response=_text("hi"))])
    )
    session = ChatSession(
        ChatSessionConfig(
            llm_client=mock_llm, workspace=workspace, session_id="conv-t"
        )
    )

    with patch("anton.analytics.send_event") as send:
        async for _ in session.turn_stream("hello"):
            pass

    assert _ended_by(send) == "completed"
    assert _error_type(send) == ""


async def test_user_stop_carries_no_error_type(workspace):
    """A user pressing Stop is not a failure. `CancelledError` takes the
    `cancelled` branch, which deliberately does not stamp `error_type` — so
    an error-cause breakdown cannot be inflated by ordinary stops.
    """
    mock_llm = make_mock_llm()
    started = asyncio.Event()

    def _hang(**kwargs):
        async def _gen():
            started.set()
            await asyncio.sleep(3600)
            yield StreamComplete(response=_text())

        return _gen()

    mock_llm.plan_stream = MagicMock(side_effect=_hang)
    session = ChatSession(
        ChatSessionConfig(
            llm_client=mock_llm, workspace=workspace, session_id="conv-t"
        )
    )

    with patch("anton.analytics.send_event") as send:

        async def _run():
            async for _ in session.turn_stream("long one"):
                pass

        task = asyncio.create_task(_run())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert _ended_by(send) == "cancelled"
    assert _error_type(send) == ""
