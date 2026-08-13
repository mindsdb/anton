"""Main-loop output-budget truncation recovery (ENG-1042).

A completion can burn its entire ``max_tokens`` on internal reasoning (or
narration) and return nothing — no text, no tool call, no error. The session
has had a recovery for this since early on, but it gated on
``stop_reason in ("max_tokens", "length")`` and the MindsHub gateway then
reported ``finish_reason: "stop"`` at the cap (ENG-1082), so the recovery was
dead code for every hosted user: 38 fully-silent generations across 12 users in
one week. ENG-1082 was fixed 2026-08-03; the token-count gate below is kept
because it cannot regress upstream, and the e2e stub still emulates the old
dishonest ``"stop"`` deliberately.

What must hold now:

1. The gate fires on token-count evidence (`looks_truncated`), so the
   gateway's dishonest ``"stop"`` no longer disables it.
2. The retry is never an unchanged re-issue — the three prod failures in the
   ticket ARE an unchanged retry loop dying identically. The retry must raise
   the output budget and inject a corrective nudge.
3. The nudge matches the variant: "continue where you left off" is meaningless
   when nothing visible was produced.
4. If the retry also dies silently, the user sees an explicit failure notice —
   a turn must never end silently.
5. Completions that finish inside the budget are untouched.
6. A tool call the cap opened up does not count as "finished": the
   repair pass makes such a call parseable, so without this the round looked
   complete and its handler ran on arguments the model never emitted.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.conftest import make_mock_llm

from anton.core.llm.provider import (
    LLMResponse,
    StreamComplete,
    StreamTextDelta,
    ToolCall,
    Usage,
)
from anton.core.session import (
    _TRUNCATED_CONTINUE_NUDGE,
    _TRUNCATED_LEAD,
    _TRUNCATED_SILENT_NUDGE,
    _TRUNCATED_TOOL_CALL_NUDGE,
    _TRUNCATED_TOOL_CALL_TAIL,
    _TRUNCATION_FAILURE_NOTICE,
    ChatSession,
    ChatSessionConfig,
    _VerifierVerdict,
)

BUDGET = 8192


@pytest.fixture()
def workspace():
    # Keep scratchpad venvs inside the repo workspace (pytest runs sandboxed
    # and can't write to the real home directory).
    base = Path(__file__).resolve().parents[1] / ".pytest-workspace"
    base.mkdir(parents=True, exist_ok=True)
    return MagicMock(base=base)


def _response(
    content: str = "",
    output_tokens: int = 20,
    stop_reason: str | None = "end_turn",
    tool_calls: list[ToolCall] | None = None,
) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        usage=Usage(input_tokens=10, output_tokens=output_tokens),
        stop_reason=stop_reason,
    )


def _silent_at_cap(output_tokens: int = BUDGET) -> LLMResponse:
    """The prod signature: whole budget spent, nothing visible, and the
    gateway calls it a normal stop (ENG-1082)."""
    return _response(content="", output_tokens=output_tokens, stop_reason="stop")


class _FakeAsyncIter:
    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


class _ScriptedPlanStream:
    """`plan_stream` fake that plays back one response per call and records
    every call's kwargs, so tests can assert what the retry actually sent."""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        # Snapshot messages — the session mutates the same history list
        # between calls, so a live reference would alias every entry.
        recorded = dict(kwargs)
        recorded["messages"] = [dict(m) for m in kwargs.get("messages", [])]
        self.calls.append(recorded)
        return _FakeAsyncIter([StreamComplete(response=self._responses.pop(0))])


def _make_session(responses: list[LLMResponse], workspace) -> tuple[ChatSession, _ScriptedPlanStream]:
    mock_llm = make_mock_llm()
    mock_llm.max_tokens = BUDGET
    script = _ScriptedPlanStream(responses)
    mock_llm.plan_stream = script
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))
    return session, script


async def _run_turn(session: ChatSession, prompt: str = "build the forecast workbook"):
    events = []
    try:
        async for event in session.turn_stream(prompt):
            events.append(event)
    finally:
        await session.close()
    return events


# --------------------------------------------------------------------------
# 1. The gate fires on the gateway dialect (the dead-code revival).
# --------------------------------------------------------------------------


async def test_silent_burn_with_finish_reason_stop_is_recovered(workspace):
    """output_tokens == budget with stop_reason "stop" — exactly what the
    gateway reports (ENG-1082) and exactly what the old stop_reason gate
    could never see. One retry must happen and its answer must reach the
    user."""
    session, script = _make_session(
        [_silent_at_cap(), _response("Here is the forecast plan.", output_tokens=40)],
        workspace,
    )

    events = await _run_turn(session)

    assert len(script.calls) == 2, "the truncated call must be retried once"
    final = [e for e in events if isinstance(e, StreamComplete)]
    assert final and final[-1].response.content == "Here is the forecast plan."


async def test_honest_length_stop_reason_still_fires(workspace):
    """Providers that report truncation honestly (OpenAI "length") keep
    working even below the cap — `looks_truncated` honours both signals."""
    session, script = _make_session(
        [
            _response(content="", output_tokens=100, stop_reason="length"),
            _response("Recovered.", output_tokens=10),
        ],
        workspace,
    )

    await _run_turn(session)

    assert len(script.calls) == 2


# --------------------------------------------------------------------------
# 2. The retry is never an unchanged re-issue (ticket Done-when).
# --------------------------------------------------------------------------


async def test_retry_is_not_an_unchanged_reissue(workspace):
    """The three prod failures in ENG-1042 are an unchanged retry loop dying
    identically 14 minutes apart. The retry must differ from the failed call
    in BOTH the budget and the messages."""
    session, script = _make_session(
        [_silent_at_cap(), _response("Done.", output_tokens=10)],
        workspace,
    )

    await _run_turn(session)

    first, retry = script.calls
    assert retry.get("max_tokens") == BUDGET * 2, (
        "the retry must raise the output budget, not re-issue the same call"
    )
    assert retry["messages"] != first["messages"], (
        "the retry must inject a corrective nudge into the conversation"
    )


async def test_retry_budget_scales_with_the_client_budget(workspace):
    """The doubled budget is relative to the budget the call actually ran
    with (the configurable client default — ENG-1042 Fix 4), not a magic
    constant."""
    mock_llm = make_mock_llm()
    mock_llm.max_tokens = 1000
    script = _ScriptedPlanStream(
        [
            _response(content="", output_tokens=1000, stop_reason="stop"),
            _response("ok", output_tokens=5),
        ]
    )
    mock_llm.plan_stream = script
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))

    await _run_turn(session)

    assert script.calls[1].get("max_tokens") == 2000


# --------------------------------------------------------------------------
# 3. The nudge matches the variant.
# --------------------------------------------------------------------------


def _user_texts(messages: list[dict]) -> list[str]:
    return [str(m.get("content")) for m in messages if m.get("role") == "user"]


async def test_silent_burn_gets_the_answer_now_nudge(workspace):
    """Nothing visible was produced — 'continue where you left off' is
    meaningless. The model must be told to answer immediately instead."""
    session, script = _make_session(
        [_silent_at_cap(), _response("Done.", output_tokens=10)],
        workspace,
    )

    await _run_turn(session)

    retry_user_texts = "\n".join(_user_texts(script.calls[1]["messages"]))
    assert _TRUNCATED_SILENT_NUDGE in retry_user_texts
    assert _TRUNCATED_CONTINUE_NUDGE not in retry_user_texts


async def test_partial_text_gets_the_continue_nudge_and_keeps_the_partial(workspace):
    """Truncated-text variant: the partial answer must be preserved in
    history (so the continuation actually continues it) and the nudge is the
    'continue where you left off' one."""
    partial = "## Forecast method per account\n1. Revenue: run-rate…"
    session, script = _make_session(
        [
            _response(content=partial, output_tokens=BUDGET, stop_reason="stop"),
            _response("…2. COGS: seasonal average. Workbook built.", output_tokens=30),
        ],
        workspace,
    )

    await _run_turn(session)

    retry_messages = script.calls[1]["messages"]
    retry_user_texts = "\n".join(_user_texts(retry_messages))
    assert _TRUNCATED_CONTINUE_NUDGE in retry_user_texts
    assert _TRUNCATED_SILENT_NUDGE not in retry_user_texts
    assert any(
        m.get("role") == "assistant" and partial in str(m.get("content"))
        for m in retry_messages
    ), "the truncated partial answer must be in history for the continuation"


# --------------------------------------------------------------------------
# 4. A double failure is surfaced, never silent.
# --------------------------------------------------------------------------


async def test_double_silent_burn_surfaces_a_visible_failure(workspace):
    """The retry also burns its (doubled) budget with nothing visible. The
    user must see an explicit notice — the one outcome ENG-1042 forbids is
    the turn ending silently."""
    session, script = _make_session(
        [_silent_at_cap(), _silent_at_cap(output_tokens=BUDGET * 2)],
        workspace,
    )

    events = await _run_turn(session)

    assert len(script.calls) == 2, "one retry, then stop — no unchanged loop"
    notice_deltas = [
        e for e in events
        if isinstance(e, StreamTextDelta) and _TRUNCATION_FAILURE_NOTICE in e.text
    ]
    assert notice_deltas, "the user must be told the turn died"
    # The notice must precede the final StreamComplete so it lands inside
    # the message rather than after the renderer closes it.
    notice_idx = events.index(notice_deltas[0])
    complete_idx = max(
        i for i, e in enumerate(events) if isinstance(e, StreamComplete)
    )
    assert notice_idx < complete_idx


async def test_double_failure_is_recorded_in_history(workspace):
    """The failure notice must also land in history so the next turn's model
    knows the previous turn died rather than silently vanished."""
    session, script = _make_session(
        [_silent_at_cap(), _silent_at_cap(output_tokens=BUDGET * 2)],
        workspace,
    )

    await _run_turn(session)

    assert any(
        m.get("role") == "assistant"
        and _TRUNCATION_FAILURE_NOTICE in str(m.get("content"))
        for m in session.history
    )


# --------------------------------------------------------------------------
# 5. Healthy completions are untouched.
# --------------------------------------------------------------------------


async def test_completion_inside_the_budget_is_not_retried(workspace):
    session, script = _make_session(
        [_response("All done.", output_tokens=500)], workspace
    )

    await _run_turn(session)

    assert len(script.calls) == 1


async def test_at_cap_with_a_tool_call_is_not_intercepted(workspace):
    """A response that hit the cap but still delivered an *intact* tool call
    proceeds into the tool loop — the cut landed after the call, so there is
    nothing to recover. (The damaged-call case is below.)"""
    tool_call = ToolCall(
        id="tc_1",
        name="scratchpad",
        input={"action": "exec", "name": "main", "code": "print(1)"},
    )
    session, script = _make_session(
        [
            _response(
                content="",
                output_tokens=BUDGET,
                stop_reason="stop",
                tool_calls=[tool_call],
            ),
            _response("Tool output looks right — done.", output_tokens=30),
        ],
        workspace,
    )
    # The tool round trips the completion verifier — give it a clean verdict
    # so it doesn't spend the scripted responses on continuations.
    session._llm.generate_object_code = AsyncMock(
        return_value=_VerifierVerdict(status="COMPLETE", reason="done")
    )

    await _run_turn(session)

    # Call 2 is the post-tool follow-up, not a truncation retry: no raised
    # budget, no nudge.
    assert script.calls[1].get("max_tokens") is None
    assert _TRUNCATED_SILENT_NUDGE not in "\n".join(
        _user_texts(script.calls[1]["messages"])
    )


# --------------------------------------------------------------------------
# 6. A tool call cut open at the cap is not a usable tool call.
# --------------------------------------------------------------------------


async def test_repaired_tool_call_at_the_cap_is_retried_not_dispatched(workspace):
    """The shape measured in prod: the budget ran out *inside* the arguments.

    ``safe_parse_tool_input``'s repair pass closes the open brace, so the call
    arrives parseable with ``parse_error`` unset — the round therefore looked
    complete, and the handler ran on arguments the model never finished (in the
    trace, a ``scratchpad exec`` whose ``code`` was simply gone). It has to take
    the truncation retry instead.
    """
    cut_call = ToolCall(
        id="tc_cut",
        name="scratchpad",
        # What the repair pass returns for
        # '{"action": "exec", "name": "main", "estimated_execution_time_seconds": 15'
        input={"action": "exec", "name": "main", "estimated_execution_time_seconds": 15},
        repaired=True,
    )
    session, script = _make_session(
        [
            _response(content="", output_tokens=BUDGET, stop_reason="stop", tool_calls=[cut_call]),
            _response("Re-issued the call in smaller parts. Done.", output_tokens=30),
        ],
        workspace,
    )

    await _run_turn(session)

    assert len(script.calls) == 2, "a cut-open tool call must be retried, not dispatched"
    retry = script.calls[1]
    assert retry.get("max_tokens") == BUDGET * 2
    retry_user_texts = "\n".join(_user_texts(retry["messages"]))
    assert _TRUNCATED_TOOL_CALL_NUDGE in retry_user_texts, (
        "the nudge must name the real failure — a cut tool call, not silence"
    )


async def test_unparseable_tool_call_at_the_cap_is_retried_with_more_budget(workspace):
    """Same round, worse damage: the body could not be salvaged at all.

    The dispatcher's ``parse_error`` short-circuit asks the model to re-emit
    over the tool protocol, but at the same budget that just ran out — so when
    the cap is the cause, the raised-budget retry is the recovery that can
    actually succeed.
    """
    broken_call = ToolCall(
        id="tc_broken",
        name="scratchpad",
        input={},
        parse_error="Unterminated string starting at: line 1 column 42",
    )
    session, script = _make_session(
        [
            _response(content="", output_tokens=BUDGET, stop_reason="length", tool_calls=[broken_call]),
            _response("Recovered.", output_tokens=20),
        ],
        workspace,
    )

    await _run_turn(session)

    assert len(script.calls) == 2
    assert script.calls[1].get("max_tokens") == BUDGET * 2


async def test_cut_open_tool_call_in_a_continuation_round_is_also_retried(workspace):
    """The same rule has to hold on the *second* gate, inside the tool loop.

    That is where the shape is most likely: history is at its longest by then,
    so the output budget is what runs out first. The gate is duplicated in the
    session (pre-loop and in-loop), and fixing only the first one leaves every
    round after the first tool call dispatching half a call.
    """
    intact = ToolCall(id="tc_1", name="scratchpad", input={"action": "exec", "name": "main", "code": "print(1)"})
    cut = ToolCall(id="tc_2", name="scratchpad", input={"action": "exec", "name": "main"}, repaired=True)
    session, script = _make_session(
        [
            _response(content="", output_tokens=40, stop_reason="tool_use", tool_calls=[intact]),
            _response(content="", output_tokens=BUDGET, stop_reason="stop", tool_calls=[cut]),
            _response("Split the cell up — done.", output_tokens=30),
        ],
        workspace,
    )
    session._llm.generate_object_code = AsyncMock(
        return_value=_VerifierVerdict(status="COMPLETE", reason="done")
    )

    await _run_turn(session)

    assert len(script.calls) == 3, "the continuation round's cut call must be retried"
    assert script.calls[2].get("max_tokens") == BUDGET * 2


async def test_a_repaired_call_below_the_cap_is_never_dispatched(workspace):
    """The dispatcher backstop: no handler ever runs on repaired arguments.

    Here the round finished well inside its budget, so neither truncation gate
    fires — the repair had some other cause (a dropped connection, a model
    glitch). The arguments are still a body the model never finished, so the
    call has to come back as an is_error tool_result asking for a re-emit
    rather than reaching `prepare_scratchpad_exec` with no `code`. This is also
    what protects the one path the gates can't see: a truncation retry that
    itself came back cut.
    """
    cut = ToolCall(
        id="tc_net",
        name="scratchpad",
        input={"action": "exec", "name": "main"},
        repaired=True,
    )
    session, script = _make_session(
        [
            _response(content="", output_tokens=120, stop_reason="tool_use", tool_calls=[cut]),
            _response("Re-issued it with the code inline. Done.", output_tokens=30),
        ],
        workspace,
    )
    session._llm.generate_object_code = AsyncMock(
        return_value=_VerifierVerdict(status="COMPLETE", reason="done")
    )

    await _run_turn(session)

    # No truncation retry (the round was nowhere near the cap) — the follow-up
    # round carries the error tool_result instead.
    assert script.calls[1].get("max_tokens") is None
    results = [
        block
        for message in script.calls[1]["messages"]
        for block in (message.get("content") if isinstance(message.get("content"), list) else [])
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert results, "the cut call must produce a tool_result, not a handler run"
    assert results[0]["is_error"] is True
    assert "arrived incomplete" in results[0]["content"]


async def test_a_round_that_lost_text_and_a_tool_call_gets_both_instructions(workspace):
    """Both halves can go at once, and both have to be asked for.

    Paragraphs were already streamed to the user and appended to history, and
    then the tool call was cut open. Telling the model only to re-issue the call
    leaves it free to rewrite the answer from the top, which puts two copies of
    it in history; telling it only to continue leaves the call unmade.
    """
    partial = "## Forecast method per account\n1. Revenue: run-rate…"
    cut = ToolCall(id="tc_cut", name="scratchpad", input={"action": "exec"}, repaired=True)
    session, script = _make_session(
        [
            _response(content=partial, output_tokens=BUDGET, stop_reason="stop", tool_calls=[cut]),
            _response("…2. COGS: seasonal average.", output_tokens=30),
        ],
        workspace,
    )

    await _run_turn(session)

    retry_user_texts = "\n".join(_user_texts(script.calls[1]["messages"]))
    assert _TRUNCATED_CONTINUE_NUDGE in retry_user_texts, "the text has to be continued"
    assert _TRUNCATED_TOOL_CALL_TAIL in retry_user_texts, "the call has to be re-issued"
    assert retry_user_texts.count(_TRUNCATED_LEAD) == 1, (
        "one message, not two nudges stapled together"
    )
    assert any(
        m.get("role") == "assistant" and partial in str(m.get("content"))
        for m in script.calls[1]["messages"]
    ), "the partial answer must stay in history for the continuation"


async def test_the_non_streaming_turn_also_refuses_a_cut_open_call(workspace):
    """`turn()` is the sibling caller, and it dispatches the same tool calls.

    It has no truncation retry of its own, so the refusal *is* the recovery
    here: the handler must not run, and the model must get an is_error
    tool_result it can answer by re-emitting the call. Without this the loop
    reaches `dispatch_tool` with arguments the model never finished, on a path
    the streaming gates never see.
    """
    cut = ToolCall(
        id="tc_cut",
        name="scratchpad",
        input={"action": "exec", "name": "main"},
        repaired=True,
    )
    mock_llm = make_mock_llm()
    mock_llm.max_tokens = BUDGET
    mock_llm.plan = AsyncMock(side_effect=[
        _response(content="", output_tokens=BUDGET, stop_reason="tool_use", tool_calls=[cut]),
        _response("Re-issued it in smaller parts. Done.", output_tokens=30),
    ])
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))
    session.tool_registry.dispatch_tool = AsyncMock()

    try:
        await session.turn("build the forecast workbook")
    finally:
        await session.close()

    session.tool_registry.dispatch_tool.assert_not_called()
    results = [
        block
        for message in session.history
        for block in (message.get("content") if isinstance(message.get("content"), list) else [])
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert results, "the cut call must be answered, not executed"
    assert results[0]["is_error"] is True
    assert "arrived incomplete" in results[0]["content"]


async def test_one_damaged_call_among_intact_ones_still_retries(workspace):
    """Dispatching the intact half of a cut-open round runs part of what the
    model was still in the middle of asking for."""
    intact = ToolCall(id="tc_ok", name="memorize", input={"content": "prefers weekly reports"})
    cut = ToolCall(id="tc_cut", name="scratchpad", input={"action": "exec"}, repaired=True)
    session, script = _make_session(
        [
            _response(content="", output_tokens=BUDGET, stop_reason="stop", tool_calls=[intact, cut]),
            _response("Done.", output_tokens=15),
        ],
        workspace,
    )

    await _run_turn(session)

    assert len(script.calls) == 2
    assert script.calls[1].get("max_tokens") == BUDGET * 2
