"""The round cap (`max_tool_rounds`) in isolation.

`per_call` usage is kept tiny relative to the default `max_turn_tokens` so the
spend ceiling never interferes, and `max_tool_rounds` is set directly on the
session (not through settings) so these stay cheap to script.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import make_mock_llm

from anton.core.llm.provider import LLMResponse, StreamComplete, ToolCall, Usage
from anton.core.session import (
    ChatSession,
    ChatSessionConfig,
    _VerifierVerdict,
)


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


async def _run(session, prompt="do the thing"):
    async for _ in session.turn_stream(prompt):
        pass


async def test_round_under_the_cap_is_untouched(workspace):
    """No behaviour change for a turn that never reaches the cap."""
    session = _session(workspace, max_tool_rounds=5,
                       responses=[_tool_call(1), _text("done")])
    verdict = _VerifierVerdict(status="COMPLETE", reason="done", close_to_done=False)
    session._llm.generate_object_code = AsyncMock(return_value=verdict)
    with patch("anton.analytics.send_event") as send:
        await _run(session)
    assert send.call_args.kwargs["ended_by"] == "completed"


async def test_a_zero_cap_denies_the_first_round(workspace):
    """`max_tool_rounds=0` means "no tool rounds"."""
    session = _session(workspace, max_tool_rounds=0, responses=[_tool_call(1)])
    with patch("anton.analytics.send_event") as send:
        await _run(session)
    kwargs = send.call_args.kwargs
    assert kwargs["ended_by"] == "round_cap"
    assert kwargs["rounds"] == "0"
    assert kwargs["grace_granted"] == ""


async def test_the_ceiling_grace_is_reported(workspace):
    """The spend ceiling keeps its one-time grace, and `grace_granted` names it.

    Needs a real per-call token cost to trip the ceiling, which `_session`'s
    `usage_listener` stub ignores (always reports the fixed default `_usage()`
    regardless of the scripted response) — so this builds its own session
    rather than growing that shared helper's surface for one test.
    """
    per_call = 50_000
    mock_llm = make_mock_llm()
    script = [
        LLMResponse(
            content="working",
            tool_calls=[ToolCall(id=f"tc_{i}", name="scratchpad",
                                 input={"action": "view", "name": "main"})],
            usage=_usage(per_call), stop_reason="tool_use",
        ) for i in range(1, 16)
    ]

    def _plan_stream(**kwargs):
        async def _gen():
            resp = script.pop(0) if script else _text()
            if mock_llm.usage_listener is not None:
                mock_llm.usage_listener("planning", "test-model", _usage(per_call))
            yield StreamComplete(response=resp)
        return _gen()

    mock_llm.usage_listener = None
    mock_llm.plan_stream = _plan_stream
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))
    session._max_tool_rounds = 100
    session._max_turn_tokens = 300_000
    from anton.core.tools.registry import ToolOutcome
    session.tool_registry.dispatch_tool = AsyncMock(
        return_value=ToolOutcome(content="stubbed tool result", ok=True)
    )
    with patch("anton.analytics.send_event") as send:
        await _run(session)
    kwargs = send.call_args.kwargs
    assert kwargs["grace_granted"] == "ceiling"
    assert session._spend_ceiling_grace_used
