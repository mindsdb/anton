"""A turn that reaches its spend ceiling or round cap asks before it stops.

"Keep going" raises the limit inside the same turn; any other outcome hands
back exactly as before. Ending the turn instead made the user's reply a new
turn with a fresh limit, so work bigger than one limit stopped again and again
while the user kept typing "keep going".

Drives the real ``turn_stream`` with the spend-ceiling suite's scripted LLM,
plus an elicitor that answers from a script.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from anton.core.interaction.elicit import AskAnswer
from anton.core.llm.provider import StreamAskUser, StreamTaskProgress
from anton.core.session import (
    _HANDBACK_NOT_USER_CLAUSE,
    _LIMIT_CONTINUE_MULTIPLIER,
    _FinishAuthorization,
    _VerifierVerdict,
)

from tests.test_spend_ceiling import (
    CEILING,
    PER_CALL,
    _history_text,
    _session,
    _text,
    _tool_call,
    workspace,  # noqa: F401 — fixture
)

KEEP_GOING = AskAnswer(status="answered", values=("continue",))
STOP_HERE = AskAnswer(status="answered", values=("stop",))


class _ScriptedElicitor:
    supported_kinds = ("choice",)
    answer_hint = "hint"
    timeout_s = 300

    def __init__(self, *answers: AskAnswer) -> None:
        self.answers = list(answers)
        self.requests = []

    async def begin(self, question_id, request):
        pass

    async def ask(self, question_id, request):
        self.requests.append(request)
        return self.answers.pop(0) if self.answers else STOP_HERE

    async def end(self, question_id):
        pass


async def _run(session, prompt="do the thing"):
    with patch("anton.analytics.send_event") as send:
        events = [ev async for ev in session.turn_stream(prompt)]
    return events, send.call_args.kwargs


async def test_keep_going_at_the_ceiling_finishes_in_the_same_turn(workspace):
    """The run that stops at the plain ceiling finishes after one answer."""
    session = _session(workspace, responses=[_tool_call(i) for i in range(1, 14)] + [_text("all done")])
    session.elicitor = _ScriptedElicitor(KEEP_GOING)
    events, kwargs = await _run(session)
    assert kwargs["ended_by"] == "completed"
    assert kwargs["limit_continues"] == "1"
    assert int(kwargs["tokens_total"]) > CEILING
    assert any(isinstance(ev, StreamAskUser) for ev in events), "the host sees the question"
    assert "Do NOT retry automatically" not in _history_text(session)


async def test_stop_here_hands_back_as_before(workspace):
    session = _session(workspace, responses=[_tool_call(i) for i in range(1, 14)])
    session.elicitor = _ScriptedElicitor(STOP_HERE)
    _, kwargs = await _run(session)
    assert kwargs["ended_by"] == "spend_ceiling"
    assert kwargs["limit_continues"] == "0"
    assert _HANDBACK_NOT_USER_CLAUSE in _history_text(session)


async def test_unanswered_question_hands_back(workspace):
    session = _session(workspace, responses=[_tool_call(i) for i in range(1, 14)])
    session.elicitor = _ScriptedElicitor(AskAnswer(status="timeout"))
    _, kwargs = await _run(session)
    assert kwargs["ended_by"] == "spend_ceiling"


async def test_one_answer_buys_a_bounded_window(workspace):
    """A run that never finishes asks again at the raised ceiling, and stops
    there when the user says so."""
    session = _session(workspace, responses=[_tool_call(i) for i in range(1, 80)])
    session._max_tool_rounds = 100  # so the ceiling, not the round cap, decides
    elicitor = _ScriptedElicitor(KEEP_GOING, STOP_HERE)
    session.elicitor = elicitor
    _, kwargs = await _run(session)
    assert kwargs["ended_by"] == "spend_ceiling"
    assert kwargs["limit_continues"] == "1"
    assert len(elicitor.requests) == 2
    total = int(kwargs["tokens_total"])
    assert CEILING < total <= (1 + _LIMIT_CONTINUE_MULTIPLIER) * CEILING + 2 * PER_CALL


async def test_the_question_states_what_continuing_costs(workspace):
    session = _session(workspace, responses=[_tool_call(i) for i in range(1, 14)])
    elicitor = _ScriptedElicitor(STOP_HERE)
    session.elicitor = elicitor
    await _run(session)
    request = elicitor.requests[0]
    assert f"reached its limit of {CEILING:,} tokens per request" in request.prompt
    assert "Max tokens per task" in request.prompt
    assert [o.value for o in request.options] == ["continue", "stop"]
    assert f"{_LIMIT_CONTINUE_MULTIPLIER * CEILING:,}" in request.options[0].detail
    assert not request.allow_custom


async def test_keep_going_at_the_round_cap_raises_it(workspace):
    work = [_tool_call(i) for i in range(1, 9)] + [_text("all done")]
    session = _session(workspace, responses=work, per_call=1_000)
    session._max_tool_rounds = 3
    session.elicitor = _ScriptedElicitor(KEEP_GOING)
    _, kwargs = await _run(session)
    assert kwargs["ended_by"] == "completed"
    assert kwargs["limit_continues"] == "1"


async def test_stop_at_the_round_cap_hands_back(workspace):
    session = _session(workspace, responses=[_tool_call(i) for i in range(1, 9)], per_call=1_000)
    session._max_tool_rounds = 3
    session.elicitor = _ScriptedElicitor(STOP_HERE)
    _, kwargs = await _run(session)
    assert kwargs["ended_by"] == "round_cap"
    assert _HANDBACK_NOT_USER_CLAUSE in _history_text(session)


def _structured_calls(session, *, authorized: bool) -> AsyncMock:
    """Answer the finish-authorization check, and let any verifier verdict
    through as COMPLETE."""

    async def _answer(schema, **_kwargs):
        if schema is _FinishAuthorization:
            return _FinishAuthorization(authorized=authorized)
        return _VerifierVerdict(status="COMPLETE", reason="done")

    calls = AsyncMock(side_effect=_answer)
    session._llm.generate_object_code = calls
    return calls


def _authorization_checks(calls: AsyncMock) -> int:
    return sum(1 for c in calls.call_args_list if c.args and c.args[0] is _FinishAuthorization)


async def test_the_user_s_own_keep_going_passes_the_limit_without_asking(workspace):
    """The incident: the user wrote "keep going, don't stop until you are
    finished", and the turn stopped at the limit again."""
    session = _session(workspace, responses=[_tool_call(i) for i in range(1, 14)] + [_text("all done")])
    elicitor = _ScriptedElicitor(STOP_HERE)
    session.elicitor = elicitor
    _structured_calls(session, authorized=True)
    events, kwargs = await _run(session, "Keep going, don't stop until you are finished")
    assert kwargs["ended_by"] == "completed"
    assert kwargs["limit_message_continues"] == "1"
    assert kwargs["limit_continues"] == "0"
    assert not elicitor.requests, "the user already answered"
    notes = [ev.message for ev in events if isinstance(ev, StreamTaskProgress)]
    assert any(
        f"reached its limit of {CEILING:,} tokens per request" in n and "as you asked" in n
        for n in notes
    ), "the user is told which limit their message carried it past"


async def test_the_user_s_message_carries_each_limit_once(workspace):
    """A run that never finishes is asked at the raised limit, so one message
    cannot authorize unbounded spend."""
    session = _session(workspace, responses=[_tool_call(i) for i in range(1, 80)])
    session._max_tool_rounds = 100
    elicitor = _ScriptedElicitor(STOP_HERE)
    session.elicitor = elicitor
    calls = _structured_calls(session, authorized=True)
    _, kwargs = await _run(session, "Keep going, don't stop")
    assert kwargs["ended_by"] == "spend_ceiling"
    assert kwargs["limit_message_continues"] == "1"
    assert len(elicitor.requests) == 1
    raised = (1 + _LIMIT_CONTINUE_MULTIPLIER) * CEILING
    assert f"reached its limit of {raised:,} tokens per request" in elicitor.requests[0].prompt
    assert _authorization_checks(calls) == 1, "checked once per turn, not per limit"


async def test_an_ordinary_request_still_asks(workspace):
    session = _session(workspace, responses=[_tool_call(i) for i in range(1, 14)])
    elicitor = _ScriptedElicitor(STOP_HERE)
    session.elicitor = elicitor
    _structured_calls(session, authorized=False)
    _, kwargs = await _run(session, "Convert the RFP to match the strategy")
    assert kwargs["ended_by"] == "spend_ceiling"
    assert kwargs["limit_message_continues"] == "0"
    assert len(elicitor.requests) == 1


async def test_a_failed_check_falls_back_to_asking(workspace):
    session = _session(workspace, responses=[_tool_call(i) for i in range(1, 14)])
    elicitor = _ScriptedElicitor(STOP_HERE)
    session.elicitor = elicitor
    session._llm.generate_object_code = AsyncMock(side_effect=RuntimeError("model down"))
    _, kwargs = await _run(session, "keep going")
    assert kwargs["ended_by"] == "spend_ceiling"
    assert len(elicitor.requests) == 1


async def test_the_hand_back_names_the_limit_and_how_to_continue(workspace):
    session = _session(workspace, responses=[_tool_call(i) for i in range(1, 14)])
    session.elicitor = _ScriptedElicitor(STOP_HERE)
    await _run(session)
    text = _history_text(session)
    assert f"reached its limit of {CEILING:,} tokens per request" in text
    assert 'replying \\"keep going\\" continues' in text or 'replying "keep going" continues' in text
