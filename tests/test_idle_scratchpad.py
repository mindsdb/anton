"""Idle-scratchpad loop breaker: a model that keeps announcing a tool call in
scratchpad cells instead of making it loses the scratchpad for one round."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.conftest import make_mock_llm

from anton.core.idle_scratchpad import IDLE_SCRATCHPAD_NOTICE, is_idle_scratchpad_call
from anton.core.llm.provider import LLMResponse, StreamComplete, ToolCall, Usage
from anton.core.session import ChatSession, ChatSessionConfig
from anton.core.tools.registry import ToolOutcome


@pytest.mark.parametrize("code", [
    "", "   ", "# fetch with the Notion tool next", "print('')", "print('Ready to query HubSpot.')",
    "pass", "print('a')\nprint('b', end='')", "'just a note'",
])
def test_code_that_does_no_work_is_idle(code):
    assert is_idle_scratchpad_call("scratchpad", {"action": "exec", "code": code}) is True


@pytest.mark.parametrize("code", [
    "import os\nprint(os.environ.get('X'))", "x = 1", "print(f'{len(rows)} rows')", "print(rows)",
    "this is not python (",
])
def test_code_that_does_work_is_not_idle(code):
    assert is_idle_scratchpad_call("scratchpad", {"action": "exec", "code": code}) is False


def test_removing_a_pad_is_idle_and_other_tools_never_are():
    assert is_idle_scratchpad_call("scratchpad", {"action": "remove", "name": "general"}) is True
    assert is_idle_scratchpad_call("scratchpad", {"action": "view", "name": "general"}) is False
    assert is_idle_scratchpad_call("hubspot__search_crm_objects", {"action": "exec", "code": ""}) is False


@pytest.fixture()
def workspace():
    base = Path(__file__).resolve().parents[1] / ".pytest-workspace"
    base.mkdir(parents=True, exist_ok=True)
    return MagicMock(base=base)


def _usage() -> Usage:
    return Usage(input_tokens=250, output_tokens=250, cache_read_tokens=250, cache_creation_tokens=250)


def _call(i: int, name: str = "scratchpad", code: str = "print('Ready.')") -> LLMResponse:
    tool_input = {"action": "exec", "name": "general", "code": code} if name == "scratchpad" else {}
    return LLMResponse(
        content="", tool_calls=[ToolCall(id=f"tc_{i}", name=name, input=tool_input)],
        usage=_usage(), stop_reason="tool_use",
    )


def _session(workspace, responses):
    """Session whose LLM plays `responses`, then answers in text; records the
    tool names offered on every model call."""
    mock_llm = make_mock_llm()
    script = list(responses)
    offered: list[set[str]] = []

    def _plan_stream(**kwargs):
        offered.append({t["name"] for t in kwargs.get("tools") or []})

        async def _gen():
            resp = script.pop(0) if script else LLMResponse(
                content="done", tool_calls=[], usage=_usage(), stop_reason="end_turn")
            yield StreamComplete(response=resp)
        return _gen()

    async def _plan(**kwargs):
        return LLMResponse(content="done", tool_calls=[], usage=_usage(), stop_reason="end_turn")

    mock_llm.usage_listener = None
    mock_llm.plan_stream = _plan_stream
    mock_llm.plan = _plan
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))
    session.tool_registry.dispatch_tool = AsyncMock(return_value=ToolOutcome(content="ok", ok=True))

    async def _dispatch_stream(*args, **kwargs):
        yield ToolOutcome(content="ok", ok=True)

    session.tool_registry.dispatch_tool_stream = _dispatch_stream
    return session, offered


async def _run(session):
    async for _ in session.turn_stream("summarize my hubspot data"):
        pass


def _notices(session) -> int:
    return sum(
        1 for m in session._history if isinstance(m.get("content"), list)
        for block in m["content"] if block.get("type") == "text" and block.get("text") == IDLE_SCRATCHPAD_NOTICE
    )


async def test_two_idle_rounds_withhold_the_scratchpad_for_the_next_round(workspace):
    session, offered = _session(workspace, [_call(1), _call(2)])
    await _run(session)

    assert "scratchpad" in offered[0] and "scratchpad" in offered[1]
    assert "scratchpad" not in offered[2]
    assert _notices(session) == 1


async def test_the_scratchpad_comes_back_after_a_real_tool_call(workspace):
    session, offered = _session(workspace, [_call(1), _call(2), _call(3, name="publish_or_preview")])
    await _run(session)

    assert "scratchpad" not in offered[2]
    assert "scratchpad" in offered[3]


async def test_real_scratchpad_work_is_never_interrupted(workspace):
    work = "import json\nprint(json.dumps({'rows': 3}))"
    session, offered = _session(workspace, [_call(i, code=work) for i in range(1, 5)])
    await _run(session)

    assert all("scratchpad" in names for names in offered)
    assert _notices(session) == 0
