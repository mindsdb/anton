"""Per-turn spend ceiling (ENG-1286).

Drives the real ``turn_stream``/``turn`` rather than poking the predicate,
because the ceiling's whole job is wiring: which loop it is checked in, what it
does to the verifier, and whether the message the user read reaches history.

The fake LLM calls ``usage_listener`` exactly as ``LLMClient`` does — the
listener is the narrow waist ENG-1288 installs, so a stub that skips it would
leave ``total_tokens`` at 0 and every one of these tests would pass by never
reaching the gate.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import make_mock_llm

from anton.core.llm.provider import (
    LLMResponse,
    StreamComplete,
    ToolCall,
    Usage,
)
from anton.core.session import ChatSession, ChatSessionConfig, _SPEND_CEILING_RESERVE

CEILING = 1_000_000
# Per call: enough that two calls clear (CEILING - reserve) and one does not,
# so the tests can place the trip at a known round.
PER_CALL = 500_000


@pytest.fixture()
def workspace():
    base = Path(__file__).resolve().parents[1] / ".pytest-workspace"
    base.mkdir(parents=True, exist_ok=True)
    return MagicMock(base=base)


def _usage(n: int = PER_CALL) -> Usage:
    # Spread across the four components on purpose: the ceiling counts RAW
    # tokens, so a cache-heavy call must count the same as a fresh one. A gate
    # that read only `input_tokens` would pass every other test in this file.
    q = n // 4
    return Usage(
        input_tokens=q, output_tokens=q,
        cache_read_tokens=q, cache_creation_tokens=n - 3 * q,
    )


def _tool_call(i: int = 1) -> LLMResponse:
    return LLMResponse(
        content="working",
        tool_calls=[ToolCall(id=f"tc_{i}", name="scratchpad",
                             input={"action": "view", "name": "main"})],
        usage=_usage(), stop_reason="tool_use",
    )


def _text(text: str = "done") -> LLMResponse:
    return LLMResponse(content=text, tool_calls=[], usage=_usage(),
                       stop_reason="end_turn")


def _session(workspace, *, ceiling: int = CEILING, responses=None,
             per_call: int = PER_CALL) -> ChatSession:
    """Session whose LLM emits `responses` in order and reports usage per call.

    Falls back to a text reply once the script runs out, so a test that fails to
    trip the gate terminates instead of looping.
    """
    mock_llm = make_mock_llm()
    script = list(responses or [])

    def _plan_stream(**kwargs):
        async def _gen():
            resp = script.pop(0) if script else _text()
            # Exactly what LLMClient does on every completion (ENG-1288).
            if mock_llm.usage_listener is not None:
                mock_llm.usage_listener("planning", "test-model", _usage(per_call))
            yield StreamComplete(response=resp)
        return _gen()

    async def _plan(**kwargs):
        resp = script.pop(0) if script else _text()
        if mock_llm.usage_listener is not None:
            mock_llm.usage_listener("planning", "test-model", _usage(per_call))
        return resp

    mock_llm.usage_listener = None
    mock_llm.plan_stream = _plan_stream
    mock_llm.plan = _plan
    settings = MagicMock()
    settings.max_tool_rounds = 25
    settings.max_continuations = 3
    settings.verify_min_tool_rounds = 1
    settings.max_turn_tokens = ceiling
    settings.max_consecutive_errors = 5
    settings.resilience_nudge_at = 2
    settings.context_pressure_threshold = 0.7
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))
    # Override post-construction: ChatSession reads a real CoreSettings in
    # __init__, and only these two knobs matter here.
    session._max_turn_tokens = ceiling
    from anton.core.tools.registry import ToolOutcome
    session.tool_registry.dispatch_tool = AsyncMock(
        return_value=ToolOutcome(content="stubbed tool result", ok=True)
    )
    return session


def _history_text(session) -> str:
    out = []
    for m in session._history:
        c = m.get("content")
        if isinstance(c, str):
            out.append(c)
    return "\n".join(out)


async def _run(session, prompt="do the thing"):
    async for _ in session.turn_stream(prompt):
        pass


async def test_ceiling_stops_the_turn_and_marks_the_exit(workspace):
    """The turn stops on the ceiling and books `spend_ceiling`."""
    session = _session(workspace, responses=[_tool_call(i) for i in range(1, 12)])
    with patch("anton.analytics.send_event") as send:
        await _run(session)
    assert send.call_args.kwargs["ended_by"] == "spend_ceiling"


async def test_total_is_bounded_by_the_ceiling(workspace):
    """The turn's spend lands at or under the ceiling.

    The reserve is what makes this hold: the hand-back diagnosis is itself an
    LLM call, so gating AT the ceiling would overshoot it by that call's cost.
    """
    session = _session(workspace, responses=[_tool_call(i) for i in range(1, 20)])
    with patch("anton.analytics.send_event") as send:
        await _run(session)
    # Analytics properties go over the wire as strings.
    assert int(send.call_args.kwargs["tokens_total"]) <= CEILING


async def test_trips_inside_one_tool_loop_with_no_continuation(workspace):
    """The runaway shape is consecutive tool calls in ONE loop.

    A continuation-gate-only check never sees it — the measured tail is 13-26
    consecutive scratchpad calls that reach the round cap having triggered no
    continuation at all. This is the regression guard for that placement.
    """
    session = _session(workspace, responses=[_tool_call(i) for i in range(1, 12)])
    with patch("anton.analytics.send_event") as send:
        await _run(session)
    kwargs = send.call_args.kwargs
    assert kwargs["ended_by"] == "spend_ceiling"
    assert int(kwargs["continuations"]) == 0, "must trip without needing a continuation"
    assert int(kwargs["rounds"]) < 25, "must trip before the round cap, not because of it"


async def test_handback_asks_the_user_and_is_persisted_as_streamed(workspace):
    """ENG-1155 property: history holds what the user actually read.

    Also asserts the message asks rather than merely announcing — a ceiling the
    user cannot get past is worse than no ceiling.
    """
    session = _session(workspace, responses=[_tool_call(i) for i in range(1, 12)])
    with patch("anton.analytics.send_event"):
        await _run(session)
    text = _history_text(session)
    assert "Do NOT retry automatically" in text
    assert "ask if they'd like you to continue" in text
    # The streamed diagnosis, not the pre-stop reply, is the tail of history.
    assert session._history[-1]["role"] == "assistant"


async def test_verification_is_skipped_after_a_ceiling_trip(workspace):
    """No verdict call once the ceiling has stopped the turn.

    Verification would spend more of the budget we just declared exhausted, and
    an INCOMPLETE verdict would force a continuation straight past the ceiling.
    """
    session = _session(workspace, responses=[_tool_call(i) for i in range(1, 12)])
    session._llm.generate_object_code = AsyncMock(
        side_effect=AssertionError("verifier must not run after a ceiling trip")
    )
    with patch("anton.analytics.send_event"):
        await _run(session)


async def test_turn_under_the_ceiling_is_untouched(workspace):
    """No behaviour change for a turn that never reaches the gate."""
    session = _session(workspace, responses=[_tool_call(1), _text("all done")],
                       per_call=1_000)
    with patch("anton.analytics.send_event") as send:
        await _run(session)
    kwargs = send.call_args.kwargs
    assert kwargs["ended_by"] == "completed"
    text = _history_text(session)
    assert "Do NOT retry automatically" not in text


async def test_ceiling_of_zero_disables_the_gate(workspace):
    """0 means off — a host on older settings keeps pre-ENG-1286 behaviour.

    Uses per-call usage far above any plausible ceiling, so a gate that ignored
    the 0 would certainly trip.
    """
    session = _session(workspace, ceiling=0,
                       responses=[_tool_call(i) for i in range(1, 6)] + [_text()],
                       per_call=5_000_000)
    with patch("anton.analytics.send_event") as send:
        await _run(session)
    assert send.call_args.kwargs["ended_by"] != "spend_ceiling"


async def test_reserve_is_held_back_for_the_handback(workspace):
    """The gate fires below the ceiling by the reserve, not at it."""
    from anton.core.turn_cost import TurnCost
    session = _session(workspace)
    session._turn_cost = TurnCost()  # no calls yet -> reserve is the floor
    session._turn_cost.input_tokens = CEILING - _SPEND_CEILING_RESERVE - 1
    assert not session._spend_ceiling_reached()
    session._turn_cost.input_tokens = CEILING - _SPEND_CEILING_RESERVE
    assert session._spend_ceiling_reached()


async def test_reserve_scales_with_the_turn_s_own_call_size(workspace):
    """A big-context turn reserves more, because its remaining calls cost more.

    Without this the bound is a promise the ceiling cannot keep: two calls land
    after the last passing check, and at 190k+ of context per call they exceed
    a flat 200k reserve on their own.
    """
    from anton.core.turn_cost import TurnCost
    session = _session(workspace)
    session._turn_cost = TurnCost()
    session._turn_cost.peak_context_tokens = 190_000
    assert session._spend_ceiling_gate() == CEILING - 380_000
    # A small-context turn keeps the floor rather than shrinking below it.
    session._turn_cost.peak_context_tokens = 1_000
    assert session._spend_ceiling_gate() == CEILING - _SPEND_CEILING_RESERVE


async def test_cache_reads_count_toward_the_ceiling(workspace):
    """Raw tokens, not cost-weighted — cache reads draw the user's allowance
    at full 1:1 weight, so a cache-heavy turn must trip like any other.
    """
    from anton.core.turn_cost import TurnCost
    session = _session(workspace)
    session._turn_cost = TurnCost()
    session._turn_cost.cache_read_tokens = CEILING - _SPEND_CEILING_RESERVE
    assert session._spend_ceiling_reached(), (
        "a turn made almost entirely of cache reads still exhausts the "
        "user's included-token allowance"
    )
