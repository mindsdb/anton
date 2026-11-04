"""A turn that runs past its time limit asks before it stops, then hands back.

Hosts that set no limit (desktop) never see it. The cloud pod sets it from the
request. "Keep going" adds one more window, not four. Every other answer ends
the turn with the same summary as the other limits, recorded as
``time_limit``.

Drives the real ``turn_stream`` with the spend-ceiling suite's scripted LLM.
The session's clock is swapped for one that moves 100 s per model call, so a
180 s limit is reached at the second round's check.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from anton.core.interaction import elicit as elicit_module
from anton.core.llm.provider import StreamAskUser, StreamTurnEnded
from anton.core.session import _HANDBACK_NOT_USER_CLAUSE, _VerifierVerdict
from anton.core.settings import CoreSettings

from tests.test_limit_continue import KEEP_GOING, STOP_HERE, _run, _ScriptedElicitor
from tests.test_spend_ceiling import (
    _history_text,
    _session,
    _text,
    _tool_call,
    workspace,  # noqa: F401 — fixture
)

LIMIT_S = 180
STEP_S = 100


@pytest.fixture
def clock(monkeypatch):
    """The session's clock, which `_timed_session` moves on per model call."""
    now = [0.0]
    monkeypatch.setattr("anton.core.session.time", SimpleNamespace(monotonic=lambda: now[0]))
    return now


def _timed_session(workspace, responses, clock):
    # A large step and token budget, so the time limit is what decides.
    session = _session(workspace, responses=responses, ceiling=10**9, per_call=1_000)
    session._max_tool_rounds = 500
    session._max_turn_seconds = LIMIT_S
    # No verification unless a test asks for it: these runs are about the clock.
    session._verify_min_tool_rounds = 10**6
    plan_stream = session._llm.plan_stream

    def _timed_plan_stream(**kwargs):
        clock[0] += STEP_S
        return plan_stream(**kwargs)

    session._llm.plan_stream = _timed_plan_stream
    return session


def test_there_is_no_time_limit_unless_a_host_sets_one():
    assert CoreSettings().max_turn_seconds == 0


async def test_a_session_without_a_limit_never_asks(workspace, clock):
    session = _timed_session(workspace, [_tool_call(i) for i in range(1, 8)] + [_text("all done")], clock)
    session._max_turn_seconds = 0
    elicitor = _ScriptedElicitor()
    session.elicitor = elicitor
    _, kwargs = await _run(session)
    assert kwargs["ended_by"] == "completed"
    assert elicitor.requests == []


async def test_stop_here_at_the_time_limit_hands_back(workspace, clock):
    session = _timed_session(workspace, [_tool_call(i) for i in range(1, 8)], clock)
    elicitor = _ScriptedElicitor(STOP_HERE)
    session.elicitor = elicitor
    events, kwargs = await _run(session)
    assert kwargs["ended_by"] == "time_limit"
    assert events[-1] == StreamTurnEnded(ended_by="time_limit")
    history = _history_text(session)
    assert "This task reached its time limit of 3 minutes per request." in history
    assert _HANDBACK_NOT_USER_CLAUSE in history
    request = elicitor.requests[0]
    assert request.prompt == "This task reached its time limit of 3 minutes per request. Keep going?"
    assert request.options[0].detail == "Allow up to 3 more minutes"


async def test_no_answer_at_the_time_limit_hands_back(workspace, clock):
    session = _timed_session(workspace, [_tool_call(i) for i in range(1, 8)], clock)
    session.elicitor = None
    _, kwargs = await _run(session)
    assert kwargs["ended_by"] == "time_limit"


async def test_keep_going_adds_one_window_and_asks_again_at_its_end(workspace, clock):
    """Reached at 200 s (limit 180). One "Keep going" moves the limit to 360,
    reached at 400 s, where the second answer stops the turn. A 4x grant would
    have moved it to 900 and finished the work without a second question."""
    session = _timed_session(workspace, [_tool_call(i) for i in range(1, 12)] + [_text("all done")], clock)
    elicitor = _ScriptedElicitor(KEEP_GOING, STOP_HERE)
    session.elicitor = elicitor
    events, kwargs = await _run(session)
    assert kwargs["ended_by"] == "time_limit"
    assert kwargs["limit_continues"] == "1"
    assert len(elicitor.requests) == 2
    assert elicitor.requests[1].prompt.startswith(
        "This task reached its time limit of 6 minutes per request."
    )
    assert sum(isinstance(ev, StreamAskUser) for ev in events) == 2


async def test_keep_going_lets_the_work_finish(workspace, clock):
    session = _timed_session(workspace, [_tool_call(1), _tool_call(2), _text("all done")], clock)
    session.elicitor = _ScriptedElicitor(KEEP_GOING)
    _, kwargs = await _run(session)
    assert kwargs["ended_by"] == "completed"
    assert kwargs["limit_continues"] == "1"


async def test_the_time_question_is_asked_after_the_question_budget_is_spent(
    workspace, clock, monkeypatch,
):
    """Counted questions are refused once the model used its budget; the time
    limit still asks, or a long turn would end without the user getting a say."""
    monkeypatch.setattr(elicit_module, "MAX_QUESTIONS_PER_TURN", 0)
    session = _timed_session(workspace, [_tool_call(1), _tool_call(2), _text("all done")], clock)
    session.elicitor = _ScriptedElicitor(KEEP_GOING)
    _, kwargs = await _run(session)
    assert kwargs["ended_by"] == "completed"
    assert kwargs["limit_continues"] == "1"


async def test_the_step_limit_still_reports_itself_first(workspace, clock):
    """Both limits reached in the same round (the fifth: one step, three of
    grace, and 500 s against 450): the step limit, checked first, is the one
    recorded."""
    session = _timed_session(workspace, [_tool_call(i) for i in range(1, 8)], clock)
    session._max_tool_rounds = 1
    session._max_turn_seconds = 450
    session.elicitor = _ScriptedElicitor(STOP_HERE)
    _, kwargs = await _run(session)
    assert kwargs["ended_by"] == "round_cap"


async def test_a_continuation_past_the_limit_asks_at_the_continuation_gate(workspace, clock):
    """The verifier's INCOMPLETE starts a continuation that answers in text, so
    the per-round check never runs; the gate asks instead."""
    session = _timed_session(
        workspace, [_tool_call(1), _text("partial"), _text("more"), _text("more")], clock,
    )
    session._verify_min_tool_rounds = 1
    session._llm.generate_object_code = AsyncMock(
        return_value=_VerifierVerdict(status="INCOMPLETE", reason="not done")
    )
    elicitor = _ScriptedElicitor(STOP_HERE)
    session.elicitor = elicitor
    _, kwargs = await _run(session)
    assert kwargs["ended_by"] == "time_limit"
    assert elicitor.requests[0].prompt.startswith("This task reached its time limit")


async def test_no_verdict_is_asked_after_the_time_limit_stopped_the_turn(workspace, clock):
    """An INCOMPLETE verdict would start a continuation straight past the
    user's "Stop here". Counts verifier calls rather than raising from one: the
    verdict site turns an exception into a hand-back and would hide a regression."""
    session = _timed_session(workspace, [_tool_call(i) for i in range(1, 8)], clock)
    session._verify_min_tool_rounds = 1
    verdict = AsyncMock(return_value=_VerifierVerdict(status="INCOMPLETE", reason="not done"))
    session._llm.generate_object_code = verdict
    session.elicitor = _ScriptedElicitor(STOP_HERE)
    _, kwargs = await _run(session)
    assert kwargs["ended_by"] == "time_limit"
    assert not [c for c in verdict.call_args_list if c.args and c.args[0] is _VerifierVerdict]


async def test_after_three_more_windows_the_limit_hands_back_without_asking(workspace, clock):
    """The total is four windows; past it the turn ends with a summary rather
    than asking again, so the hosting side's ceiling never cuts it off."""
    session = _timed_session(workspace, [_tool_call(i) for i in range(1, 30)], clock)
    elicitor = _ScriptedElicitor(KEEP_GOING, KEEP_GOING, KEEP_GOING, KEEP_GOING)
    session.elicitor = elicitor
    _, kwargs = await _run(session)
    assert kwargs["ended_by"] == "time_limit"
    assert kwargs["limit_continues"] == "3"
    assert len(elicitor.requests) == 3
