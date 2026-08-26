"""Round-cap grace (ENG-1893).

Exercises `max_tool_rounds`'s "always ask" hand-back in isolation: `per_call`
usage is kept tiny relative to the default 1.25M `max_turn_tokens` so the
spend ceiling never interferes, and `max_tool_rounds` is set directly on the
session (not through settings) so these stay cheap to script.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import make_mock_llm

from anton.core.llm.provider import LLMResponse, StreamComplete, ToolCall, Usage
from anton.core.session import ChatSession, ChatSessionConfig, _ROUND_CAP_GRACE_ROUNDS


@pytest.fixture()
def workspace():
    base = Path(__file__).resolve().parents[1] / ".pytest-workspace"
    base.mkdir(parents=True, exist_ok=True)
    return MagicMock(base=base)


def _usage(n: int = 1_000) -> Usage:
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
    return LLMResponse(content=text, tool_calls=[], usage=_usage(), stop_reason="end_turn")


def _session(workspace, *, max_tool_rounds: int, responses=None) -> ChatSession:
    """Session whose LLM emits `responses` in order, falling back to a text
    reply once the script runs out — a test that fails to trip the cap
    terminates instead of looping.
    """
    mock_llm = make_mock_llm()
    script = list(responses or [])

    def _plan_stream(**kwargs):
        async def _gen():
            resp = script.pop(0) if script else _text()
            if mock_llm.usage_listener is not None:
                mock_llm.usage_listener("planning", "test-model", _usage())
            yield StreamComplete(response=resp)
        return _gen()

    async def _plan(**kwargs):
        resp = script.pop(0) if script else _text()
        if mock_llm.usage_listener is not None:
            mock_llm.usage_listener("planning", "test-model", _usage())
        return resp

    mock_llm.usage_listener = None
    mock_llm.plan_stream = _plan_stream
    mock_llm.plan = _plan
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))
    session._max_tool_rounds = max_tool_rounds
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


async def test_round_cap_grants_one_time_grace_then_stops(workspace):
    """The first breach is silent; running through the whole grace window
    still asks, same as today past that point."""
    max_rounds = 5
    session = _session(
        workspace, max_tool_rounds=max_rounds,
        responses=[_tool_call(i) for i in range(1, max_rounds + _ROUND_CAP_GRACE_ROUNDS + 3)],
    )
    with patch("anton.analytics.send_event") as send:
        await _run(session)
    assert send.call_args.kwargs["ended_by"] == "round_cap"
    assert session._round_cap_grace_used
    assert int(send.call_args.kwargs["rounds"]) == max_rounds + _ROUND_CAP_GRACE_ROUNDS


async def test_round_cap_grace_also_applies_to_the_non_streaming_turn(workspace):
    """`turn()` mirrors turn_stream's grace, like it already mirrors the
    plain round cap and spend ceiling.

    Public API with no in-tree caller, but its cost books are wired, so an
    unmirrored grace here would under-report every host that uses it.
    """
    max_rounds = 5
    session = _session(
        workspace, max_tool_rounds=max_rounds,
        responses=[_tool_call(i) for i in range(1, max_rounds + _ROUND_CAP_GRACE_ROUNDS + 3)],
    )
    with patch("anton.analytics.send_event") as send:
        await session.turn("do the thing")
    assert send.call_args.kwargs["ended_by"] == "round_cap"
    assert session._round_cap_grace_used
    assert int(send.call_args.kwargs["rounds"]) == max_rounds + _ROUND_CAP_GRACE_ROUNDS


async def test_round_cap_grace_lets_a_near_finish_wrap_up_without_asking(workspace):
    """A task that actually finishes inside the grace window never asks at all."""
    max_rounds = 5
    session = _session(
        workspace, max_tool_rounds=max_rounds,
        responses=[_tool_call(i) for i in range(1, max_rounds + _ROUND_CAP_GRACE_ROUNDS + 1)]
        + [_text("done")],
    )
    with patch("anton.analytics.send_event") as send:
        await _run(session)
    assert send.call_args.kwargs["ended_by"] != "round_cap"
    text = _history_text(session)
    assert "ask if they'd like you to continue" not in text


async def test_round_under_the_cap_is_untouched(workspace):
    """No behaviour change for a turn that never reaches the cap."""
    session = _session(workspace, max_tool_rounds=5,
                       responses=[_tool_call(1), _text("done")])
    with patch("anton.analytics.send_event") as send:
        await _run(session)
    assert send.call_args.kwargs["ended_by"] == "completed"
    assert not session._round_cap_grace_used


async def test_round_cap_grace_does_not_multiply_across_continuations(workspace):
    """A one-shot grace must not become the ambient cap for every continuation.

    `_round_cap_grace_used` used to gate the effective cap directly, so once
    ANY continuation exceeded the base cap it stayed extended to M+3 for the
    rest of the turn — every later continuation got the +3 window for free.
    Here every continuation's
    scripted work (6 tool calls) exceeds M=5, so before the fix none of them
    ever actually hit the cap and the turn ran all 3 continuations to
    completion (24 rounds, ended_by=handback_budget).

    With the grant gated on a per-continuation `_round_cap_grace_active`
    flag, only the FIRST continuation to breach gets the wider cap; the next
    continuation that also breaches ends the turn outright at the base cap,
    same as it would with no grace at all. That bounds the turn at
    2M + grace, not `max_continuations * (M + grace)`.
    """
    from anton.core.session import _ROUND_CAP_GRACE_ROUNDS, _VerifierVerdict

    max_rounds = 5
    per_continuation = max_rounds + 1  # exceeds M, comfortably inside M + grace
    session = _session(
        workspace, max_tool_rounds=max_rounds,
        responses=(
            [_tool_call(i) for i in range(1, per_continuation + 1)] + [_text("partial")]
        )
        * 3,
    )
    verdict = _VerifierVerdict(status="INCOMPLETE", reason="not done", close_to_done=False)
    session._llm.generate_object_code = AsyncMock(return_value=verdict)
    with patch("anton.analytics.send_event") as send:
        await _run(session)
    kwargs = send.call_args.kwargs
    assert kwargs["ended_by"] == "round_cap"
    assert int(kwargs["continuations"]) <= 1, (
        "the grace must not let a second continuation also breach for free"
    )
    assert int(kwargs["rounds"]) <= 2 * max_rounds + _ROUND_CAP_GRACE_ROUNDS
    assert session._round_cap_grace_used
