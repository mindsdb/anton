"""Main-loop output-budget truncation recovery (ENG-1042).

A completion can burn its entire ``max_tokens`` on internal reasoning (or
narration) and return nothing — no text, no tool call, no error. The session
has had a recovery for this since early on, but it gated on
``stop_reason in ("max_tokens", "length")`` and the MindsHub gateway reports
``finish_reason: "stop"`` at the cap (ENG-1082), so the recovery was dead code
for every hosted user: 38 fully-silent generations across 12 users in one week.

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
    _TRUNCATED_SILENT_NUDGE,
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
        for m in session._history
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
    """A response that hit the cap but still delivered a usable tool call
    proceeds into the tool loop — the recovery only owns the no-tool-call
    case. (Damaged tool-call JSON is the structured-output path's job,
    ENG-1081.)"""
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
