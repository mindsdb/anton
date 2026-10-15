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

from anton.core.llm.provider import StreamComplete, StreamTextDelta, StreamTurnEnded
from anton.core.session import ChatSession, ChatSessionConfig, _VerifierVerdict


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
