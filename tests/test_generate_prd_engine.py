"""engine.run_gathering_loop: phase 1's bounded ReAct loop. Mirrors
generate_artifact/engine.py's `_run_loop` shape, with ask_user routed
through sub_tools.dispatch_ask_user instead of handle_ask_user."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from anton.core.llm.provider import LLMResponse, ToolCall, Usage
from anton.core.tools.generate_prd import engine
from anton.core.tools.generate_prd.state import PrdState


def _response(content="", tool_calls=None) -> LLMResponse:
    return LLMResponse(content=content, tool_calls=tool_calls or [], usage=Usage(input_tokens=1, output_tokens=1))


def _tc(name, input, id="tc1") -> ToolCall:
    return ToolCall(id=id, name=name, input=input)


def _state(session, **over) -> PrdState:
    base = dict(
        session=session,
        slug="s",
        artifact_path=Path("/tmp/s"),
        artifact_type="html-app",
        user_request="build a clock",
        agent_understanding="an analog clock",
        known_data="",
        user_preferences="",
    )
    base.update(over)
    return PrdState(**base)


def _session_with_plan_sequence(*responses) -> SimpleNamespace:
    """One shared iterator behind both `plan` and `code`.

    `run_gathering_loop` calls `plan` on round 0 and `code` on every round
    after — two independent `AsyncMock(side_effect=responses)` would each
    restart their own iterator at `responses[0]`, so round 1 would silently
    replay round 0's response instead of advancing to `responses[1]`. A
    single shared iterator behind both callables is what makes `responses`
    actually describe "round 0, round 1, round 2, ..." in order.
    """
    it = iter(responses)

    async def _next(**kwargs):
        return next(it)

    llm = SimpleNamespace(plan=AsyncMock(side_effect=_next), code=AsyncMock(side_effect=_next))
    return SimpleNamespace(_llm=llm, question_count=0, elicitor=None, emit=AsyncMock())


async def test_finish_gathering_sets_artifact_type_and_notes():
    session = _session_with_plan_sequence(
        _response(tool_calls=[_tc("finish_gathering", {"summary": "ready", "artifact_type": "fullstack-stateless-app", "notes": "use CoinGecko"})]),
    )
    state = _state(session)
    await engine.run_gathering_loop(state)
    assert state.final_artifact_type == "fullstack-stateless-app"
    assert state.gathering_notes == "use CoinGecko"


async def test_no_tool_calls_leaves_final_artifact_type_empty():
    """A model that stops with plain text (no finish_gathering) is the
    best-effort case the orchestrator falls back on — see Task 6."""
    session = _session_with_plan_sequence(_response(content="I think we're done."))
    state = _state(session)
    await engine.run_gathering_loop(state)
    assert state.final_artifact_type == ""
    assert state.gathering_notes == "I think we're done."
    # Recorded in `messages`, not just `gathering_notes` — phase 2 reads
    # only `state.messages`, so this is what actually carries the model's
    # best-effort summary forward into draft_brief/write_prd.
    assert state.messages[-1] == {"role": "assistant", "content": "I think we're done."}


async def test_finish_gathering_falls_back_to_the_registered_type_when_invented(monkeypatch):
    """The schema's `enum` is a hint, not an enforced constraint — a model
    can still emit a type outside ARTIFACT_TYPES. Left unvalidated here,
    that string would only fail much later, inside write_prd's
    `ArtifactStore.update(type=...)` (a ValueError that crashes the whole
    generate_prd call) — see prd-design.md's live-testing feedback."""
    session = _session_with_plan_sequence(
        _response(tool_calls=[_tc("finish_gathering", {"summary": "ready", "artifact_type": "interactive-dashboard"})]),
    )
    state = _state(session)  # artifact_type="html-app" — see _state's default
    await engine.run_gathering_loop(state)
    assert state.final_artifact_type == "html-app"


async def test_finish_gathering_keeps_a_valid_type_that_differs_from_the_registered_one():
    session = _session_with_plan_sequence(
        _response(tool_calls=[_tc("finish_gathering", {"summary": "ready", "artifact_type": "fullstack-stateless-app"})]),
    )
    state = _state(session)
    await engine.run_gathering_loop(state)
    assert state.final_artifact_type == "fullstack-stateless-app"


async def test_scratchpad_call_is_dispatched_to_the_real_handler(monkeypatch):
    async def fake_handle_scratchpad(session, inp):
        assert inp == {"action": "view", "name": "s"}
        return "cell 1: ..."

    monkeypatch.setattr(
        "anton.core.tools.tool_handlers.handle_scratchpad", fake_handle_scratchpad
    )
    session = _session_with_plan_sequence(
        _response(tool_calls=[_tc("scratchpad", {"action": "view", "name": "s"})]),
        _response(tool_calls=[_tc("finish_gathering", {"summary": "ok", "artifact_type": "html-app"})]),
    )
    state = _state(session)
    await engine.run_gathering_loop(state)
    assert state.final_artifact_type == "html-app"
    # The scratchpad result must have reached the model as a tool_result.
    result_blocks = state.messages[2]["content"]
    assert any(b.get("content") == "cell 1: ..." for b in result_blocks)


async def test_ask_user_is_dispatched_via_elicit_not_handle_ask_user(monkeypatch):
    from anton.core.interaction.elicit import AskAnswer

    async def fake_elicit(session, question_id, request):
        return AskAnswer(status="answered", values=("dark",))

    monkeypatch.setattr("anton.core.interaction.elicit.elicit", fake_elicit)
    session = _session_with_plan_sequence(
        _response(tool_calls=[_tc("ask_user", {"question": "Theme?", "options": [{"value": "dark"}, {"value": "light"}]})]),
        _response(tool_calls=[_tc("finish_gathering", {"summary": "ok", "artifact_type": "html-app"})]),
    )
    state = _state(session)
    await engine.run_gathering_loop(state)
    assert state.final_artifact_type == "html-app"
    assert "Theme?" in state.qa_log_markdown()
    assert "dark" in state.qa_log_markdown()


async def test_ask_user_omitted_from_tools_when_budget_is_zero():
    """When gathering_question_budget is 0, the ask_user schema must not be
    offered at all — offering a tool guaranteed to answer `limit` just burns
    a round telling the model that (see prd-design.md, review iteration 3)."""
    from anton.core.interaction.elicit import MAX_QUESTIONS_PER_TURN

    session = _session_with_plan_sequence(
        _response(tool_calls=[_tc("finish_gathering", {"summary": "ok", "artifact_type": "html-app"})]),
    )
    session.question_count = MAX_QUESTIONS_PER_TURN  # budget exhausted before phase 1 even starts
    state = _state(session)
    await engine.run_gathering_loop(state)
    tools_seen = session._llm.plan.call_args.kwargs["tools"]
    assert "ask_user" not in {t["name"] for t in tools_seen}


async def test_round_budget_exhausted_without_finish_gathering(monkeypatch):
    responses = [
        _response(tool_calls=[_tc("scratchpad", {"action": "view", "name": "s"}, id=f"tc{i}")])
        for i in range(engine.MAX_ROUNDS)
    ]
    monkeypatch.setattr(
        "anton.core.tools.tool_handlers.handle_scratchpad",
        AsyncMock(return_value="(empty)"),
    )
    session = _session_with_plan_sequence(*responses)
    state = _state(session)
    await engine.run_gathering_loop(state)
    assert state.final_artifact_type == ""


async def test_re_entry_appends_a_continue_message_instead_of_resetting_history():
    """A second call (from orchestrator's back_to_gathering branch) must
    keep phase 2's brief/confirm exchange in `messages`, not wipe it."""
    session = _session_with_plan_sequence(
        _response(tool_calls=[_tc("finish_gathering", {"summary": "ok", "artifact_type": "html-app"})]),
    )
    state = _state(session)
    state.messages = [{"role": "user", "content": "## User request\n..."}, {"role": "assistant", "content": "brief text"}]
    await engine.run_gathering_loop(state)
    assert state.messages[0]["content"] == "## User request\n..."
    assert state.messages[1]["content"] == "brief text"
    assert "Continue gathering" in state.messages[2]["content"]


async def test_each_round_restarts_the_spinner_before_the_llm_call():
    """Live-testing feedback (ENG-969): `elicit()` stops the host spinner
    for `ask_user` and never restarts it, so a round that follows one must
    signal `reasoning_start` itself — otherwise the gap between the user's
    answer and the model's next reply renders as a silent pause."""
    from anton.core.llm.provider import StreamTaskProgress

    session = _session_with_plan_sequence(
        _response(tool_calls=[_tc("finish_gathering", {"summary": "ok", "artifact_type": "html-app"})]),
    )
    state = _state(session)
    await engine.run_gathering_loop(state)
    assert session.emit.await_count >= 1
    phases = {call.args[0].phase for call in session.emit.await_args_list}
    assert phases == {"reasoning_start"}
    assert all(
        isinstance(call.args[0], StreamTaskProgress) for call in session.emit.await_args_list
    )


async def test_finish_gathering_logs_an_llm_call_and_a_done_node():
    trace = MagicMock()
    session = _session_with_plan_sequence(
        _response(tool_calls=[_tc("finish_gathering", {"summary": "ok", "artifact_type": "html-app"})]),
    )
    state = _state(session, trace_log=trace)
    await engine.run_gathering_loop(state)
    assert trace.llm_call.call_args.kwargs["node"] == "gathering"
    trace.node.assert_called_once_with("gathering", "done", detail="finish_gathering: type=html-app")


async def test_round_budget_exhausted_logs_a_fail_node(monkeypatch):
    responses = [
        _response(tool_calls=[_tc("scratchpad", {"action": "view", "name": "s"}, id=f"tc{i}")])
        for i in range(engine.MAX_ROUNDS)
    ]
    monkeypatch.setattr(
        "anton.core.tools.tool_handlers.handle_scratchpad",
        AsyncMock(return_value="(empty)"),
    )
    trace = MagicMock()
    session = _session_with_plan_sequence(*responses)
    state = _state(session, trace_log=trace)
    await engine.run_gathering_loop(state)
    trace.node.assert_any_call("gathering", "fail", detail="MAX_ROUNDS exhausted without finish_gathering")


async def test_ask_user_dispatch_logs_a_node_with_the_answer(monkeypatch):
    from anton.core.interaction.elicit import AskAnswer

    async def fake_elicit(session, question_id, request):
        return AskAnswer(status="answered", values=("dark",))

    monkeypatch.setattr("anton.core.interaction.elicit.elicit", fake_elicit)
    trace = MagicMock()
    session = _session_with_plan_sequence(
        _response(tool_calls=[_tc("ask_user", {"question": "Theme?", "options": [{"value": "dark"}, {"value": "light"}]})]),
        _response(tool_calls=[_tc("finish_gathering", {"summary": "ok", "artifact_type": "html-app"})]),
    )
    state = _state(session, trace_log=trace)
    await engine.run_gathering_loop(state)
    trace.node.assert_any_call("ask_user", "answered", detail="Theme? -> dark")


async def test_scratchpad_dispatch_logs_input_and_output(monkeypatch):
    async def fake_handle_scratchpad(session, inp):
        return "cell 1: ..."

    monkeypatch.setattr(
        "anton.core.tools.tool_handlers.handle_scratchpad", fake_handle_scratchpad
    )
    trace = MagicMock()
    session = _session_with_plan_sequence(
        _response(tool_calls=[_tc("scratchpad", {"action": "view", "name": "s"})]),
        _response(tool_calls=[_tc("finish_gathering", {"summary": "ok", "artifact_type": "html-app"})]),
    )
    state = _state(session, trace_log=trace)
    await engine.run_gathering_loop(state)
    trace.scratchpad.assert_called_once_with(
        node="scratchpad", input={"action": "view", "name": "s"}, output="cell 1: ..."
    )


async def test_a_tool_call_with_a_parse_error_is_asked_to_retry_not_dispatched():
    """Mirrors generate_artifact/engine.py's `_run_loop`: a streamed tool call
    that failed to parse as JSON must not reach dispatch with `input={}` —
    that produces a confusing "missing required field" trail instead of
    telling the model to just re-emit the call."""
    bad_tc = ToolCall(id="tc1", name="finish_gathering", input={}, parse_error="unterminated string")
    session = _session_with_plan_sequence(
        _response(tool_calls=[bad_tc]),
        _response(tool_calls=[_tc("finish_gathering", {"summary": "ok", "artifact_type": "html-app"})]),
    )
    state = _state(session)
    await engine.run_gathering_loop(state)
    assert state.final_artifact_type == "html-app"  # recovered on the retry
    first_result_blocks = state.messages[2]["content"]
    assert "malformed tool input" in first_result_blocks[0]["content"]
