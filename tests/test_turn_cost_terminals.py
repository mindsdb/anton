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
    session._turn_cost = current
    with patch("anton.analytics.send_event") as send:
        session._emit_turn_cost(expected=stale)
        assert not send.called
        assert session._turn_cost is current, "current turn keeps its books"

        session._emit_turn_cost(expected=current)
        assert send.called


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
