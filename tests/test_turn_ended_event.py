"""`StreamTurnEnded`: a turn that hands back says why in its last event (ENG-3436).

Drives the real `turn_stream` to each hand-back terminal, reusing the scenario
builders the terminal-classification tests already trust.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import make_mock_llm
from tests.test_spend_ceiling import _REFUSAL, _described_call
from tests.test_spend_ceiling import _session as _ceiling_session
from tests.test_spend_ceiling import _text as _ceiling_text
from tests.test_spend_ceiling import _tool_call as _ceiling_tool_call
from tests.test_turn_cost_terminals import (
    _Iter,
    _stub_tools,
    _text,
    _tool_call,
    _verdict_llm,
)

from anton.core.llm.provider import (
    LLMResponse,
    StreamComplete,
    StreamTaskProgress,
    StreamTextDelta,
    StreamTurnEnded,
    ToolCall,
)
from anton.core.session import (
    ChatSession,
    ChatSessionConfig,
    _handback_report,
    _step_description,
    _TurnStep,
    _VerifierVerdict,
)
from anton.core.tools.registry import ToolOutcome


@pytest.fixture(autouse=True)
def _verify_single_round_turns(monkeypatch):
    monkeypatch.setenv("ANTON_VERIFY_MIN_TOOL_ROUNDS", "1")


@pytest.fixture()
def workspace():
    base = Path(__file__).resolve().parents[1] / ".pytest-workspace"
    base.mkdir(parents=True, exist_ok=True)
    return MagicMock(base=base)


def _plain(workspace, mock_llm) -> ChatSession:
    session = ChatSession(
        ChatSessionConfig(llm_client=mock_llm, workspace=workspace, session_id="conv-t")
    )
    _stub_tools(session)
    return session


def _completed(workspace):
    return _plain(workspace, _verdict_llm("COMPLETE"))


def _retry_exhausted(workspace):
    mock_llm = make_mock_llm()
    calls = {"n": 0}

    def _plan_stream(**kw):
        # Every attempt fails; the summarize call afterwards answers.
        calls["n"] += 1
        if calls["n"] <= 3:
            raise RuntimeError("provider down")
        return _Iter([StreamTextDelta(text="summary"), StreamComplete(response=_text("summary"))])

    mock_llm.plan_stream = MagicMock(side_effect=_plan_stream)
    return _plain(workspace, mock_llm)


def _round_cap(workspace):
    mock_llm = make_mock_llm()
    mock_llm.plan_stream = MagicMock(
        side_effect=lambda **kw: _Iter([StreamComplete(response=_tool_call())])
    )
    mock_llm.generate_object_code = AsyncMock(
        return_value=_VerifierVerdict(status="COMPLETE", reason="ok")
    )
    session = _plain(workspace, mock_llm)
    session._max_tool_rounds = 2
    return session


def _handback_budget(workspace):
    session = _plain(workspace, _verdict_llm("INCOMPLETE"))
    session._max_continuations = 0
    return session


def _handback_stuck(workspace):
    return _plain(workspace, _verdict_llm("STUCK"))


def _handback_verifier_failure(workspace):
    mock_llm = _verdict_llm("COMPLETE")
    mock_llm.generate_object_code = AsyncMock(side_effect=RuntimeError("provider hiccup"))
    return _plain(workspace, mock_llm)


def _spend_ceiling_in_loop(workspace):
    return _ceiling_session(workspace, responses=[_ceiling_tool_call(i) for i in range(1, 40)])


def _spend_ceiling_at_continuation(workspace):
    # Text-only continuations never enter the tool loop; only the gate stops them.
    session = _ceiling_session(
        workspace, per_call=300_000,
        responses=[_ceiling_tool_call(1), _ceiling_text("partial"),
                   _ceiling_text("still going"), _ceiling_text("more")],
    )
    session._llm.generate_object_code = AsyncMock(
        return_value=_VerifierVerdict(status="INCOMPLETE", reason="not done")
    )
    return session


async def _drain(session) -> tuple[list, str]:
    with patch("anton.analytics.send_event") as send:
        events = [e async for e in session.turn_stream("do the thing")]
    return events, send.call_args.kwargs["ended_by"]


@pytest.mark.parametrize(
    "build,ended_by",
    [
        (_retry_exhausted, "retry_exhausted"),
        (_round_cap, "round_cap"),
        (_handback_budget, "handback_budget"),
        (_handback_stuck, "handback_stuck"),
        (_handback_verifier_failure, "handback_verifier_failure"),
        (_spend_ceiling_in_loop, "spend_ceiling"),
        (_spend_ceiling_at_continuation, "spend_ceiling"),
    ],
)
async def test_handback_ends_with_its_reason(workspace, build, ended_by):
    events, booked = await _drain(build(workspace))
    assert booked == ended_by, "scenario must reach the terminal it names"
    assert events[-1] == StreamTurnEnded(ended_by=ended_by)
    assert sum(isinstance(e, StreamTurnEnded) for e in events) == 1


async def test_completed_turn_sends_no_reason(workspace):
    events, booked = await _drain(_completed(workspace))
    assert booked == "completed"
    assert not any(isinstance(e, StreamTurnEnded) for e in events)


async def test_failed_turn_sends_no_reason(workspace):
    # A failed turn ends in an error, which the host maps to response.failed.
    mock_llm = make_mock_llm()
    mock_llm.plan_stream = MagicMock(side_effect=KeyboardInterrupt)
    session = _plain(workspace, mock_llm)
    events = []
    with patch("anton.analytics.send_event"), pytest.raises(KeyboardInterrupt):
        async for e in session.turn_stream("do the thing"):
            events.append(e)
    assert not any(isinstance(e, StreamTurnEnded) for e in events)


async def test_cancelled_turn_sends_no_reason(workspace):
    mock_llm = make_mock_llm()
    started = asyncio.Event()

    def _hang(**kwargs):
        async def _gen():
            started.set()
            await asyncio.sleep(30)
            yield StreamComplete(response=_text())
        return _gen()

    mock_llm.plan_stream = _hang
    session = _plain(workspace, mock_llm)
    events = []

    async def _consume():
        async for e in session.turn_stream("hang please"):
            events.append(e)

    with patch("anton.analytics.send_event"):
        task = asyncio.create_task(_consume())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not any(isinstance(e, StreamTurnEnded) for e in events)


async def test_clean_exit_after_a_cancel_sends_no_reason(workspace):
    # The CLI's cancel flag lets a turn exit without raising, booked "cancelled".
    session = _completed(workspace)
    session._cancel_event.set()
    events, booked = await _drain(session)
    assert booked == "cancelled"
    assert not any(isinstance(e, StreamTurnEnded) for e in events)


def _verifier_failure_with_handback(workspace, handback: LLMResponse, handback_stream=None):
    mock_llm = _handback_verifier_failure(workspace)._llm
    plans = [
        _Iter([StreamComplete(response=_tool_call(1))]),
        _Iter([StreamTextDelta(text="the answer"), StreamComplete(response=_text("the answer"))]),
        handback_stream or _Iter([StreamTextDelta(text=handback.content), StreamComplete(response=handback)]),
    ]
    mock_llm.plan_stream = MagicMock(side_effect=lambda **kw: plans.pop(0))
    return _plain(workspace, mock_llm)


async def test_filtered_handback_is_dropped(workspace):
    # Azure streams the refusal, then marks the reply content_filter (ENG-3478).
    refusal = "I'm sorry, but I cannot assist with that request."
    filtered = LLMResponse(content=refusal, stop_reason="content_filter")
    session = _verifier_failure_with_handback(workspace, filtered)
    events, booked = await _drain(session)
    shown = "".join(e.text for e in events if isinstance(e, StreamTextDelta))
    assert refusal not in shown
    assert "the answer" in shown
    assert refusal not in str(session.history)
    # The note goes with its reply, so the turn ends on the answer.
    assert session.history[-1] == {"role": "assistant", "content": "the answer"}
    # And the next question stays the user's own words, not merged into the note.
    session._append_history({"role": "user", "content": "next question"})
    assert session.history[-1] == {"role": "user", "content": "next question"}
    # The answer stands, so no second try; the reason names the filter.
    assert session._llm.plan_stream.call_count == 3
    assert booked == "content_filter"
    assert events[-1] == StreamTurnEnded(ended_by="content_filter")


async def test_unfiltered_handback_still_streams(workspace):
    session = _verifier_failure_with_handback(workspace, _text("here is what I did"))
    events, _ = await _drain(session)
    shown = "".join(e.text for e in events if isinstance(e, StreamTextDelta))
    assert shown.endswith("here is what I did")
    assert session.history[-1] == {"role": "assistant", "content": "here is what I did"}


async def test_filtered_handback_keeps_the_answer_for_turn(workspace):
    filtered = LLMResponse(content="I'm sorry, but I cannot assist with that request.", stop_reason="content_filter")
    session = _verifier_failure_with_handback(workspace, filtered)
    with patch("anton.analytics.send_event"):
        assert await session.turn("do the thing") == "the answer"


async def test_escape_while_a_handback_is_held_stops_the_call(workspace):
    session = None
    reached: list[str] = []

    async def _handback():
        yield StreamTextDelta(text="first")
        reached.append("first")
        session.escape_watcher.cancelled.set()  # Escape between held deltas
        yield StreamTextDelta(text="second")
        reached.append("second")
        yield StreamComplete(response=_text("first second"))
        reached.append("complete")

    session = _verifier_failure_with_handback(workspace, None, handback_stream=_handback())
    session.escape_watcher = MagicMock(cancelled=asyncio.Event())
    events, booked = await _drain(session)
    shown = "".join(e.text for e in events if isinstance(e, StreamTextDelta))
    assert "complete" not in reached, "the held call ran on to its end after Escape"
    assert "first" not in shown and "second" not in shown
    assert booked == "cancelled"
    assert isinstance(events[-1], StreamTaskProgress), "the CLI needs a last event to see Escape on"
    assert session.history[-1] == {"role": "assistant", "content": "the answer"}


async def test_escape_while_a_handback_is_held_cancels_the_turn_as_the_cli_sees_it(workspace):
    session = None

    async def _handback():
        yield StreamTextDelta(text="first")
        session.escape_watcher.cancelled.set()
        yield StreamTextDelta(text="second")
        yield StreamComplete(response=_text("first second"))

    session = _verifier_failure_with_handback(workspace, None, handback_stream=_handback())
    session.escape_watcher = MagicMock(cancelled=asyncio.Event())
    # As anton/chat.py reads the stream: on the first event after Escape, set the flag and stop.
    with patch("anton.analytics.send_event") as send:
        stream = session.turn_stream("do the thing")
        stopped = False
        try:
            async for _event in stream:
                if session.escape_watcher.cancelled.is_set():
                    session._cancel_event.set()
                    stopped = True
                    break
        finally:
            await stream.aclose()
    assert stopped
    assert send.call_args.kwargs["ended_by"] == "cancelled"


def _filtered_stream():
    return _Iter([
        StreamTextDelta(text=_REFUSAL),
        StreamComplete(response=LLMResponse(content=_REFUSAL, stop_reason="content_filter")),
    ])


def _unanswered_handback(workspace, *handbacks, base=_handback_verifier_failure, first_call=None):
    # The reply before the hand-back has no text, so the hand-back is the whole answer.
    mock_llm = base(workspace)._llm
    plans = [
        _Iter([StreamComplete(response=first_call or _tool_call(1))]),
        _Iter([StreamComplete(response=_text(""))]),
        *handbacks,
    ]
    mock_llm.plan_stream = MagicMock(side_effect=lambda **kw: plans.pop(0))
    return _plain(workspace, mock_llm)


def _handback_calls(session) -> int:
    # A hand-back call sends no tools; every tool-loop call does.
    return sum("tools" not in c.kwargs for c in session._llm.plan_stream.call_args_list)


async def test_filtered_handback_without_an_answer_is_tried_once_more(workspace):
    retried = _Iter([StreamTextDelta(text="here is what I did"), StreamComplete(response=_text("here is what I did"))])
    session = _unanswered_handback(workspace, _filtered_stream(), retried)
    events, booked = await _drain(session)
    shown = "".join(e.text for e in events if isinstance(e, StreamTextDelta))
    assert shown == "here is what I did"
    assert session.history[-1] == {"role": "assistant", "content": "here is what I did"}
    assert booked == "content_filter"


@pytest.mark.parametrize("base", [_handback_verifier_failure, _handback_stuck])
async def test_handback_filtered_twice_sends_anton_s_own_report(workspace, base):
    # No limit stopped these turns, so the report has no limit sentence.
    session = _unanswered_handback(
        workspace, _filtered_stream(), _filtered_stream(),
        base=base, first_call=_described_call(1, "Sum the first 20 Fibonacci numbers"),
    )
    events, booked = await _drain(session)
    shown = "".join(e.text for e in events if isinstance(e, StreamTextDelta))
    assert shown == (
        "The model provider's content filter stopped my summary of this request. "
        "This is what I did before I stopped:\n\n"
        "- Sum the first 20 Fibonacci numbers\n\n"
        'Reply "keep going" to continue from where I stopped.'
    )
    assert session.history[-1] == {"role": "assistant", "content": shown}
    assert _REFUSAL not in str(session.history)
    # Two hand-back tries and no third call to write the report.
    assert _handback_calls(session) == 2
    assert booked == "content_filter"
    assert events[-1] == StreamTurnEnded(ended_by="content_filter")


async def test_report_lists_only_this_turn_s_steps(workspace):
    # A CLI session runs many turns on one ChatSession, so the step list must start empty each turn.
    mock_llm = make_mock_llm()
    plans = [
        _Iter([StreamComplete(response=_described_call(1, "Old step"))]),
        _Iter([StreamComplete(response=_text("done"))]),
        _Iter([StreamComplete(response=_described_call(2, "New step"))]),
        _Iter([StreamComplete(response=_text(""))]),
        _filtered_stream(),
        _filtered_stream(),
    ]
    mock_llm.plan_stream = MagicMock(side_effect=lambda **kw: plans.pop(0))
    mock_llm.generate_object_code = AsyncMock(
        return_value=_VerifierVerdict(status="COMPLETE", reason="ok")
    )
    session = _plain(workspace, mock_llm)
    _events, first_booked = await _drain(session)
    assert first_booked == "completed"
    # The second turn's check fails, and both tries at its hand-back are filtered.
    mock_llm.generate_object_code = AsyncMock(side_effect=RuntimeError("provider hiccup"))
    events, booked = await _drain(session)
    shown = "".join(e.text for e in events if isinstance(e, StreamTextDelta))
    assert "- Old step" not in shown
    assert shown == (
        "The model provider's content filter stopped my summary of this request. "
        "This is what I did before I stopped:\n\n"
        "- New step\n\n"
        'Reply "keep going" to continue from where I stopped.'
    )
    assert booked == "content_filter"
    assert not plans


async def test_round_cap_filtered_twice_reports_the_limit_and_the_steps(workspace):
    descriptions = iter([
        "Load the sales table", "Join the regions", "Sum revenue by region",
        "Chart the totals", "Save the chart",
    ])
    mock_llm = make_mock_llm()

    def _plan(**kw):
        if "tools" not in kw:
            return _filtered_stream()
        return _Iter([StreamComplete(response=_described_call(1, next(descriptions, "Extra step")))])

    mock_llm.plan_stream = MagicMock(side_effect=_plan)
    mock_llm.generate_object_code = AsyncMock(
        return_value=_VerifierVerdict(status="COMPLETE", reason="ok")
    )
    session = _plain(workspace, mock_llm)
    session._max_tool_rounds = 2
    # The second step reports a failure and the fourth raises: both count as failed.
    session.tool_registry.dispatch_tool = AsyncMock(side_effect=[
        ToolOutcome(content="revenue total = 48213", ok=True),
        ToolOutcome(content="join failed", ok=False),
        ToolOutcome(content="revenue total = 48213", ok=True),
        RuntimeError("renderer crashed"),
        ToolOutcome(content="saved", ok=True),
    ])
    events, booked = await _drain(session)
    shown = "".join(e.text for e in events if isinstance(e, StreamTextDelta))
    assert shown == (
        "This task reached its limit of 2 steps per request."
        "\n\nThe model provider's content filter stopped my summary of this request. "
        "This is what I did before I stopped:\n\n"
        "- Load the sales table\n"
        "- Join the regions (failed)\n"
        "- Sum revenue by region\n"
        "- Chart the totals (failed)\n"
        "- Save the chart\n\n"
        'Reply "keep going" to continue from where I stopped.'
    )
    # It quotes no tool output, so the figure a step returned stays out.
    assert "48213" not in shown
    assert session.history[-1] == {"role": "assistant", "content": shown}
    assert _REFUSAL not in str(session.history)
    assert _handback_calls(session) == 2
    assert booked == "content_filter"
    assert events[-1] == StreamTurnEnded(ended_by="content_filter")


def test_report_lists_only_the_latest_steps():
    steps = [_TurnStep(description=f"Step {n}", errored=False) for n in range(1, 13)]
    report = _handback_report(reached=None, steps=steps)
    assert "These are the last things I did before I stopped:" in report
    assert "- Step 1\n" not in report and "- Step 2\n" not in report
    assert "- Step 3\n" in report and "- Step 12\n" in report


@pytest.mark.parametrize(
    "tool_input,expected",
    [
        ({"action": "exec", "one_line_description": "Load the\n  sales table"}, "Load the sales table"),
        ({"action": "view", "name": "main"}, "scratchpad view"),
        ({"action": "install", "one_line_description": "  "}, "scratchpad install"),
    ],
)
def test_step_without_a_description_names_its_action(tool_input, expected):
    call = ToolCall(id="tc_1", name="scratchpad", input=tool_input)
    assert _step_description(call) == expected


def test_step_of_a_tool_without_an_action_names_the_tool():
    call = ToolCall(id="tc_1", name="connect_new_datasource", input={"engine": "postgres"})
    assert _step_description(call) == "connect_new_datasource"


async def test_filtered_main_reply_names_the_filter(workspace):
    # Main replies stream as they arrive; the turn's reason still says the filter fired.
    mock_llm = make_mock_llm()
    mock_llm.plan_stream = MagicMock(side_effect=lambda **kw: _filtered_stream())
    session = _plain(workspace, mock_llm)
    events, booked = await _drain(session)
    assert booked == "content_filter"
    assert events[-1] == StreamTurnEnded(ended_by="content_filter")
