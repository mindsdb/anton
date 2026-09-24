"""engine.run_gathering_loop: phase 1's bounded ReAct loop. Mirrors
generate_artifact/engine.py's `_run_loop` shape, with ask_user routed
through sub_tools.dispatch_ask_user instead of handle_ask_user."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from anton.core.llm.provider import LLMResponse, ToolCall, Usage
from anton.core.tools.generate_artifact.discovery import engine
from anton.core.tools.generate_artifact.discovery.state import PrdState


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


async def test_finish_gathering_renders_the_structured_fields_into_notes():
    """The typed fields replace free-form `notes`; what draft_brief reads
    off the history is the rendered markdown, and the two lists the brief
    presents differently are kept on the state as lists."""
    session = _session_with_plan_sequence(
        _response(tool_calls=[_tc("finish_gathering", {
            "summary": "Dashboard over the orders table.",
            "artifact_type": "fullstack-stateless-app",
            "data_sources": ["orders table"],
            "data_findings": [{
                "source": "orders table",
                "verified_by": "scratchpad `g`, cell 2",
                "shape": "id int, amount numeric",
                "sample": "1 | 149.90",
            }],
            "constraints": ["reads an external DB only"],
            "assumptions": ["last 30 days by default"],
            "open_points": ["count cancelled orders?"],
        })]),
    )
    state = _state(session)
    await engine.run_gathering_loop(state)
    notes = state.gathering_notes
    assert notes.startswith("Dashboard over the orders table.")
    assert "### Data findings" in notes and "scratchpad `g`, cell 2" in notes
    assert "### Constraints" in notes and "reads an external DB only" in notes
    assert "### Assumptions" in notes and "### Open points" in notes
    assert state.assumptions == ["last 30 days by default"]
    assert state.open_points == ["count cancelled orders?"]


async def test_finish_gathering_tolerates_fields_of_the_wrong_shape():
    """The schema is a hint; the model's JSON is not trusted to match it.
    A string where a list was asked for is still an answer and is kept;
    `null` and blank items are dropped; a non-list `data_findings` is
    ignored."""
    session = _session_with_plan_sequence(
        _response(tool_calls=[_tc("finish_gathering", {
            "summary": "ready",
            "artifact_type": "html-app",
            "assumptions": "a single string instead of a list",
            "open_points": [None, "", 42],
            "data_findings": "not a list",
        })]),
    )
    state = _state(session)
    await engine.run_gathering_loop(state)
    assert state.assumptions == ["a single string instead of a list"]
    assert state.open_points == ["42"]
    assert state.gathering_notes == (
        "ready\n\n### Assumptions\n- a single string instead of a list"
        "\n\n### Open points\n- 42"
    )


async def test_finish_gathering_keeps_list_fields_sent_as_strings():
    """Seen live: `constraints`, `assumptions` and `open_points` all
    arrived as strings. The old parser returned empty
    lists for all three, `gathering_notes` shrank to the summary, and
    `discovery.json` held no assumption and no open point — the hot path
    survived only because `draft_brief` reads the tool call off the history.
    `open_points` also began with the model's own tool-call markup."""
    session = _session_with_plan_sequence(
        _response(tool_calls=[_tc("finish_gathering", {
            "summary": "Offline game in one HTML file; no external data.",
            "artifact_type": "html-app",
            "constraints": "Arrows + space, active only on your turn\n- on-screen buttons for mobile (request)",
            "assumptions": "Points per cleared cell of the matching letter",
            "open_points": "\n<parameter name=\"open_points\">What \"your symbol\" means in single-player: the badge alternates (default: yes)",
        })]),
    )
    state = _state(session)
    await engine.run_gathering_loop(state)
    assert state.assumptions == ["Points per cleared cell of the matching letter"]
    assert state.open_points == [
        'What "your symbol" means in single-player: the badge alternates (default: yes)'
    ]
    notes = state.gathering_notes
    assert "### Constraints\n- Arrows + space, active only on your turn\n- on-screen buttons for mobile (request)" in notes
    assert "### Assumptions" in notes and "### Open points" in notes
    assert "<parameter" not in notes


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


async def test_finish_gathering_falls_back_when_the_type_has_no_generator():
    """`document` is in ARTIFACT_TYPES, so it used to be accepted here;
    `settle_artifact_type("document")` then read as fullstack
    (`!= "html-app"`), the pipeline built a backend for a document, and the
    handler refused every repeat call for the slug."""
    session = _session_with_plan_sequence(
        _response(tool_calls=[_tc("finish_gathering", {"summary": "ready", "artifact_type": "document"})]),
    )
    state = _state(session)
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


async def test_ask_user_stays_in_the_array_and_is_refused_by_the_gate():
    """It used to be dropped from the schema list when the budget hit zero.

    That is no longer available: the array is the cached prefix for every
    call in phases A-D, and editing it mid-run costs a full cache miss on the
    largest context in the pipeline. The guarantee moves into the gate — the
    tool is offered but the call is refused without reaching `elicit`, and
    the refusal says why.
    """
    from anton.core.interaction.elicit import MAX_QUESTIONS_PER_TURN

    session = _session_with_plan_sequence(
        _response(tool_calls=[_tc("ask_user", {"question": "which theme?", "options": []})]),
        _response(tool_calls=[_tc("finish_gathering", {"summary": "ok", "artifact_type": "html-app"})]),
    )
    session.question_count = MAX_QUESTIONS_PER_TURN  # budget exhausted before gathering starts
    state = _state(session)
    await engine.run_gathering_loop(state)

    tools_seen = session._llm.plan.call_args.kwargs["tools"]
    assert "ask_user" in {t["name"] for t in tools_seen}

    refusals = [
        block["content"]
        for message in state.messages
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if block.get("type") == "tool_result"
    ]
    assert any("budget" in r.lower() for r in refusals)


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
    """`elicit()` stops the host spinner
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


# ── Which declared sources count as verified ────────────────────────────────

async def test_a_fetched_web_page_verifies_the_source_it_came_from(monkeypatch):
    """A source that IS a web page is verified by having been fetched.

    Requiring a scratchpad cell instead sends every web-sourced request —
    "turn this article into a dashboard" being the common one — through the
    emergency data loop on every single run, to re-download a page the
    gathering phase already read.
    """
    import anton.core.tools.web_tools as web_tools

    monkeypatch.setattr(
        web_tools, "handle_web_fetch_fallback", AsyncMock(return_value="page body")
    )
    session = _session_with_plan_sequence(
        _response(tool_calls=[_tc("web_fetch", {"url": "https://example.com/a"})]),
        _response(tool_calls=[_tc("finish_gathering", {
            "artifact_type": "html-app",
            "notes": "read it",
            "data_sources": ["the article"],
        })]),
    )
    state = _state(session)

    await engine.run_gathering_loop(state)

    assert state.declared_sources == ["the article"]
    assert state.unverified_sources == []


async def test_a_web_tool_outcome_is_unwrapped_to_its_text(monkeypatch):
    """Both web fallbacks return a `ToolOutcome`. Passed through as-is, the
    model's tool_result and the `web_notes` excerpt carried
    `ToolOutcome(content='...', ok=True)` — the dataclass repr instead of the
    page."""
    import anton.core.tools.web_tools as web_tools
    from anton.core.tools.registry import ToolOutcome

    monkeypatch.setattr(
        web_tools, "handle_web_fetch_fallback",
        AsyncMock(return_value=ToolOutcome(content="page body", ok=True)),
    )
    session = _session_with_plan_sequence(
        _response(tool_calls=[_tc("web_fetch", {"url": "https://example.com/a"})]),
        _response(tool_calls=[_tc("finish_gathering", {"artifact_type": "html-app", "summary": "read"})]),
    )
    state = _state(session)

    await engine.run_gathering_loop(state)

    assert state.web_calls[0]["excerpt"] == "page body"
    results = [b for m in state.messages if isinstance(m["content"], list)
               for b in m["content"] if b.get("type") == "tool_result"]
    assert results[0]["content"] == "page body"


async def test_a_source_nothing_was_run_against_stays_unverified():
    """The condition the emergency data loop exists for: the model declared
    a source it never touched."""
    session = _session_with_plan_sequence(
        _response(tool_calls=[_tc("finish_gathering", {
            "artifact_type": "html-app",
            "notes": "assumed",
            "data_sources": ["orders table"],
        })]),
    )
    state = _state(session)

    await engine.run_gathering_loop(state)

    assert state.unverified_sources == ["orders table"]


async def test_only_cells_that_ran_are_recorded_for_the_data_notes(monkeypatch):
    """A live run recorded an exec the single-scratchpad guard had
    refused: both generators then received its code with the
    refusal text as its "Output". A refused or failed cell is not working
    data-access code and stays out of `scratchpad_execs`."""
    from anton.core.tools.registry import ToolOutcome

    outcomes = iter([
        ToolOutcome(content="You already have an active scratchpad ('rentals_app') …",
                    ok=True, reason="new_scratchpad_challenged"),
        ToolOutcome(content="Traceback … UndefinedTable", ok=False, reason="cell_error"),
        ToolOutcome(content="[output] 504 rows", ok=True),
        "plain string result",
    ])

    async def fake_handle_scratchpad(session, inp):
        return next(outcomes)

    monkeypatch.setattr(
        "anton.core.tools.tool_handlers.handle_scratchpad", fake_handle_scratchpad
    )
    execs = [
        _tc("scratchpad", {"action": "exec", "name": "hr", "code": "print(1)"}, id="a"),
        _tc("scratchpad", {"action": "exec", "name": "rentals_app", "code": "bad()"}, id="b"),
        _tc("scratchpad", {"action": "exec", "name": "rentals_app", "code": "print(2)"}, id="c"),
        _tc("scratchpad", {"action": "exec", "name": "rentals_app", "code": "print(3)"}, id="d"),
    ]
    session = _session_with_plan_sequence(
        _response(tool_calls=execs),
        _response(tool_calls=[_tc("finish_gathering", {"summary": "ok", "artifact_type": "html-app"})]),
    )
    state = _state(session)
    await engine.run_gathering_loop(state)
    assert [(x["code"], x["output"]) for x in state.scratchpad_execs] == [
        ("print(2)", "[output] 504 rows"),
        ("print(3)", "plain string result"),
    ]
    # The model still sees every result, refusal included — only the notes skip it.
    result_blocks = state.messages[2]["content"]
    assert len(result_blocks) == 4
