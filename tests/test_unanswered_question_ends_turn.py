"""An ask_user question that times out or is skipped ends the turn.

Before this, only the tool result told the model to stop ("Do not act on an
assumed answer. End your turn now"). The model's next response could still call
any tool, and anton ran it: an agent asked whether to go ahead, got no answer,
and made the change anyway.

These drive the real `turn_stream` with a scripted model, a pad that runs each
cell's code in this process, and an elicitor that answers at once.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anton.core.backends.base import Cell
from anton.core.interaction.elicit import AskAnswer, AskRequest
from anton.core.llm.provider import (
    LLMResponse,
    StreamComplete,
    StreamEvent,
    StreamTaskProgress,
    StreamTextDelta,
    StreamTurnEnded,
    ToolCall,
    Usage,
)
from anton.core.session import ChatSession, ChatSessionConfig, _VerifierVerdict
from anton.core.settings import CoreSettings
from tests.conftest import make_mock_llm

QUESTION = "Create approved.txt in this project?"

APPROVE = AskAnswer(status="answered", values=("approve",))
TIMEOUT = AskAnswer(status="timeout")
SKIP = AskAnswer(status="cancelled")

# Long enough for a slow CI box; a turn that never ends fails here instead of hanging.
DEADLINE_S = 10


class _Pad:
    """A scratchpad runtime with no subprocess. Runs each cell's code in this
    process, so a cell that writes a file really writes it, and records it."""

    def __init__(self) -> None:
        self.ran: list[str] = []

    async def install_packages(self, packages: list[str]) -> str:
        return "All packages already installed."

    async def execute_streaming(self, code: str, **kwargs: object):
        self.ran.append(code)
        exec(code, {})
        yield Cell(code=code, stdout="", stderr="", error=None)

    async def cancel(self) -> None:
        pass


class _AnswersAtOnce:
    """Answers every question with `answer` as soon as it is asked."""

    supported_kinds = ("choice",)
    answer_hint = "hint"
    timeout_s = 300

    def __init__(self, answer: AskAnswer) -> None:
        self.answer = answer
        self.asked = 0

    async def begin(self, question_id: str, request: AskRequest) -> None:
        pass

    async def ask(self, question_id: str, request: AskRequest) -> AskAnswer:
        self.asked += 1
        return self.answer

    async def end(self, question_id: str) -> None:
        pass


class _Model:
    """A scripted model. Each response sends the next round's tool calls, an
    empty round answers in text, and once the rounds run out it answers
    "Done.". `calls` counts the responses the turn asked it for."""

    def __init__(self, *rounds: list[ToolCall]) -> None:
        self.rounds = list(rounds)
        self.calls = 0

    def plan_stream(self, **kwargs: object):
        self.calls += 1
        calls = self.rounds.pop(0) if self.rounds else None
        if calls:
            response = LLMResponse(
                content="Working.", tool_calls=calls,
                usage=Usage(input_tokens=10, output_tokens=20), stop_reason="tool_use",
            )
        elif calls == []:
            response = LLMResponse(
                content="Here is the answer.", tool_calls=[],
                usage=Usage(input_tokens=10, output_tokens=20), stop_reason="end_turn",
            )
        else:
            response = LLMResponse(
                content="Done.", tool_calls=[],
                usage=Usage(input_tokens=10, output_tokens=20), stop_reason="end_turn",
            )

        async def gen():
            yield StreamComplete(response=response)

        return gen()


def _ask(call_id: str = "tc_ask") -> ToolCall:
    return ToolCall(id=call_id, name="ask_user", input={
        "question": QUESTION,
        "options": [{"value": "approve", "label": "Approve"}, {"value": "cancel", "label": "Cancel"}],
    })


def _cell(code: str, call_id: str) -> ToolCall:
    return ToolCall(id=call_id, name="scratchpad", input={
        "action": "exec", "name": "main", "code": code, "one_line_description": "step",
    })


def _write(project: Path, call_id: str = "tc_write") -> ToolCall:
    return _cell(f"open({str(project / 'approved.txt')!r}, 'w').write('yes')\n", call_id)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return tmp_path


def _session(
    project: Path, model: _Model, answer: AskAnswer, *, verdict: str = "COMPLETE",
) -> tuple[ChatSession, _Pad]:
    """A session with the write gate on. A new file gets no card, so the gate
    doesn't stand between the question and the write: only the stop does.

    It carries a `session_id`: a turn with none and no counted model call books
    no `turn_completed` event.
    """
    llm = make_mock_llm()
    llm.plan_stream = model.plan_stream
    llm.generate_object_code = AsyncMock(
        return_value=_VerifierVerdict(status=verdict, reason="scripted")
    )
    workspace = MagicMock(base=project)
    workspace.artifacts_dir = project / "artifacts"
    session = ChatSession(ChatSessionConfig(
        llm_client=llm,
        workspace=workspace,
        settings=CoreSettings(write_approval=True),
        elicitor=_AnswersAtOnce(answer),
        session_id="conv-t",
    ))
    pad = _Pad()
    session._scratchpads.get_or_create = AsyncMock(return_value=pad)
    return session, pad


@dataclass
class _TurnSeen:
    """What one turn streamed, and the `ended_by` its `turn_completed` event booked."""

    events: list[StreamEvent] = field(default_factory=list)
    ended_by: str = ""


async def _turn(session: ChatSession, message: str = "go") -> _TurnSeen:
    seen = _TurnSeen()

    async def drive() -> None:
        async for event in session.turn_stream(message):
            seen.events.append(event)

    with patch("anton.analytics.send_event") as send:
        await asyncio.wait_for(drive(), DEADLINE_S)
    seen.ended_by = next(
        c.kwargs["ended_by"] for c in send.call_args_list if c.args[1] == "turn_completed"
    )
    return seen


async def _run(session: ChatSession) -> _TurnSeen:
    """Drive one turn, then close the session."""
    try:
        return await _turn(session)
    finally:
        await session.close()


def _text(events: list[StreamEvent]) -> str:
    return "".join(e.text for e in events if isinstance(e, StreamTextDelta))


def _tool_results(session: ChatSession) -> dict[str, str]:
    """Each tool_result in history, by the id of the call it answers."""
    # `ChatSession._history` holds raw message dicts; there is no message model to read instead.
    return {
        block["tool_use_id"]: str(block["content"])
        for message in session._history if isinstance(message.get("content"), list)
        for block in message["content"] if block.get("type") == "tool_result"
    }


def _tool_uses(session: ChatSession) -> list[str]:
    return [
        block["id"]
        for message in session._history if isinstance(message.get("content"), list)
        for block in message["content"] if block.get("type") == "tool_use"
    ]


unanswered = pytest.mark.parametrize("answer", [TIMEOUT, SKIP], ids=["timeout", "skip"])


@unanswered
async def test_a_call_after_an_unanswered_question_never_runs(project, answer):
    model = _Model([_ask()], [_write(project)])
    session, pad = _session(project, model, answer)
    await _run(session)
    assert pad.ran == []
    assert not (project / "approved.txt").exists()
    assert model.calls == 1, "the model is not asked for another response after the question"


@unanswered
async def test_a_call_in_the_same_response_as_the_question_never_runs(project, answer):
    model = _Model([_ask(), _write(project)])
    session, pad = _session(project, model, answer)
    await _run(session)
    assert pad.ran == []
    assert not (project / "approved.txt").exists()
    assert model.calls == 1
    results = _tool_results(session)
    assert set(_tool_uses(session)) == set(results), "every call in history keeps its result"
    assert results["tc_write"].startswith("Not run")


async def test_an_answered_question_keeps_the_turn_going(project):
    write = _write(project)
    model = _Model([_ask()], [write])
    session, pad = _session(project, model, APPROVE)
    seen = await _run(session)
    assert pad.ran == [write.input["code"]]
    assert (project / "approved.txt").read_text() == "yes"
    assert seen.ended_by == "completed"
    assert not any(isinstance(e, StreamTurnEnded) for e in seen.events)


@pytest.mark.parametrize("answer,says", [
    (TIMEOUT, "haven't had an answer"),
    (SKIP, "You skipped my question"),
], ids=["timeout", "skip"])
async def test_the_turn_ends_on_a_line_that_names_the_open_question(project, answer, says):
    session, _ = _session(project, _Model([_ask()]), answer)
    seen = await _run(session)
    closing = session._history[-1]
    assert closing["role"] == "assistant"
    assert QUESTION in closing["content"] and says in closing["content"]
    assert _text(seen.events).endswith(closing["content"]), "the user reads the line history keeps"
    assert seen.ended_by == "question_unanswered"
    assert seen.events[-1] == StreamTurnEnded(ended_by="question_unanswered")


async def test_the_next_turn_runs_the_change_once_the_question_is_answered(project):
    """Cowork keeps a timed-out card answerable after its turn and sends a late
    answer as the next message, so the stop holds for its own turn only."""
    write = _write(project)
    model = _Model([_ask()])
    session, pad = _session(project, model, TIMEOUT)
    try:
        await _turn(session)
        model.rounds = [[write]]
        await _turn(session, "Approve")
    finally:
        await session.close()
    assert pad.ran == [write.input["code"]]
    assert (project / "approved.txt").read_text() == "yes"


async def test_a_completion_check_that_says_incomplete_starts_no_new_round(project):
    """Two tool rounds bring the completion check in, and an INCOMPLETE
    verdict re-enters the tool loop with tools offered. After an unanswered
    question the turn must end before that check, or the next round runs."""
    look = _cell("total = 1 + 1\n", "tc_look")
    model = _Model([look], [_ask()], [_write(project)])
    session, pad = _session(project, model, TIMEOUT, verdict="INCOMPLETE")
    await _run(session)
    assert pad.ran == [look.input["code"]]
    assert not (project / "approved.txt").exists()
    assert model.calls == 2


async def test_a_question_unanswered_after_a_continuation_leaves_the_earlier_answer_standing(project):
    """A continuation tells the host its reply replaces the answer the user
    read. The closing line isn't that reply, so it is announced as a hand-back
    first, as `_stream_handback_diagnosis` does."""
    model = _Model([_cell("a = 1\n", "tc_a")], [_cell("b = 2\n", "tc_b")], [], [_ask()])
    session, _ = _session(project, model, TIMEOUT, verdict="INCOMPLETE")
    events = (await _run(session)).events
    phases = [e.phase for e in events if isinstance(e, StreamTaskProgress)]
    assert "continuation" in phases
    assert phases[phases.index("continuation"):].count("handback") == 1
    handback = next(
        i for i, e in enumerate(events)
        if isinstance(e, StreamTaskProgress) and e.phase == "handback"
    )
    assert QUESTION in _text(events[handback:])
