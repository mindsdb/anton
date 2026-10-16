"""A failed tool call's stored tool_result carries `is_error: true`.

The flag is the Messages API's own marker for a failed tool call. The model
reads it, and so does cowork-server's figure matcher, which reads the turn's
history rows and leaves an is_error result out of the steps a figure can come
from. The matcher reads only numbers, so without the flag it takes 500 from an
MCP tool's error body ("limit must be less than or equal to 200; received
500") as a figure source, even though the words say the call failed.

The flag means "this call produced no usable output", so it is set only on a
handler's explicit failure verdict (`ToolOutcome.ok=False` or a raise):

- an unclassified result (`ok=None`) stays unflagged, since its failure
  verdict is a text guess;
- a scratchpad result stays unflagged even when its cell raised, because the
  cell may have printed real data before the error.

The results anton writes itself for a call that never ran carry the flag too,
and the compaction summary keeps the failure marker in its text.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from anton.cloud_turn.history_rows import split_turn_into_rows
from anton.core.llm.provider import LLMResponse, ToolCall, Usage
from anton.core.llm.serialize import serialize_content
from anton.core.mcp.client import McpSession
from anton.core.mcp.registry import discover_tool_defs
from anton.core.session import ChatSession, ChatSessionConfig
from anton.core.tools.registry import ToolOutcome
from anton.memory.history_store import repair_replayed_tool_ids
from tests.conftest import make_mock_llm

MCP_ERROR = "limit must be less than or equal to 200; received 500"
BREAKER_MARK = "times in a row. Stop retrying this approach."


@pytest.fixture()
def workspace():
    base = Path(__file__).resolve().parents[1] / ".pytest-workspace"
    base.mkdir(parents=True, exist_ok=True)
    return MagicMock(base=base)


def _usage() -> Usage:
    return Usage(input_tokens=500, output_tokens=500)


def _scripted_plan(*calls: ToolCall):
    """One tool round per call, then a text answer on every later call."""
    queue = list(calls)

    async def plan(**_kw) -> LLMResponse:
        if queue:
            return LLMResponse(
                content="working", tool_calls=[queue.pop(0)], usage=_usage(), stop_reason="tool_use",
            )
        return LLMResponse(content="done", tool_calls=[], usage=_usage(), stop_reason="end_turn")

    return plan


def _session(workspace, *, calls: list[ToolCall]) -> ChatSession:
    llm = make_mock_llm()
    llm.plan = _scripted_plan(*calls)
    session = ChatSession(ChatSessionConfig(llm_client=llm, workspace=workspace, session_id="conv-err"))
    session._max_turn_tokens = 0
    return session


async def _run(session: ChatSession) -> None:
    async for _ in session.turn_stream("go"):
        pass


def _results(session: ChatSession) -> dict[str, dict]:
    """Every tool_result in history, by tool_use_id."""
    return {
        block["tool_use_id"]: block
        for msg in session.history
        if msg.get("role") == "user" and isinstance(msg.get("content"), list)
        for block in msg["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    }


def _crm_server() -> MCPServer:
    server = MCPServer(name="crm")

    @server.tool()
    def search_deals(limit: int) -> str:
        if limit > 200:
            raise ToolError(MCP_ERROR)
        return '{"results": [{"amount": "48250.00"}]}'

    return server


async def test_a_failed_mcp_calls_stored_result_is_marked_is_error(workspace):
    session = _session(workspace, calls=[
        ToolCall(id="toolu_bad", name="hubspot__search_deals", input={"limit": 500}),
        ToolCall(id="toolu_ok", name="hubspot__search_deals", input={"limit": 10}),
    ])
    async with McpSession(server=_crm_server()) as mcp_session:
        for tool_def in await discover_tool_defs(mcp_session, engine="hubspot", access_mode="write"):
            session.tool_registry.register_tool(tool_def)
        await _run(session)

    results = _results(session)
    assert MCP_ERROR in str(results["toolu_bad"]["content"])
    assert results["toolu_bad"].get("is_error") is True
    assert "48250.00" in str(results["toolu_ok"]["content"])
    assert "is_error" not in results["toolu_ok"]
    # The rows cowork-server persists and reads keep the flag.
    rows = split_turn_into_rows(session.history)
    row_results = {
        b["tool_use_id"]: b for row in rows if row["role"] == "user" for b in row["content"]
    }
    assert row_results["toolu_bad"].get("is_error") is True
    assert "is_error" not in row_results["toolu_ok"]


async def test_the_circuit_breaker_text_rides_inside_the_flagged_result(workspace):
    calls = [ToolCall(id=f"toolu_{i}", name="hubspot__search_deals", input={}) for i in range(5)]
    session = _session(workspace, calls=calls)
    session.tool_registry.dispatch_tool = AsyncMock(side_effect=[
        ToolOutcome(content=MCP_ERROR, ok=False, reason="mcp_tool_error") for _ in calls
    ])
    await _run(session)

    results = _results(session)
    last = results["toolu_4"]
    assert BREAKER_MARK in last["content"]
    assert last.get("is_error") is True
    assert all(results[c.id].get("is_error") is True for c in calls)


async def test_a_raising_tool_is_marked_is_error(workspace):
    session = _session(workspace, calls=[ToolCall(id="toolu_r", name="hubspot__search_deals", input={})])
    session.tool_registry.dispatch_tool = AsyncMock(side_effect=RuntimeError("connection reset"))
    await _run(session)

    result = _results(session)["toolu_r"]
    assert "connection reset" in result["content"]
    assert result.get("is_error") is True


async def test_a_failed_multimodal_result_is_marked_is_error(workspace):
    session = _session(workspace, calls=[ToolCall(id="toolu_m", name="hubspot__search_deals", input={})])
    session.tool_registry.dispatch_tool = AsyncMock(return_value=ToolOutcome(
        content=[{"type": "text", "text": MCP_ERROR}], ok=False, reason="mcp_tool_error",
    ))
    await _run(session)

    result = _results(session)["toolu_m"]
    assert result["content"] == [{"type": "text", "text": MCP_ERROR}]
    assert result.get("is_error") is True


async def test_a_failed_scratchpad_cell_stays_unflagged(workspace):
    # The cell printed a real figure before it raised; marking the result
    # is_error would hide that output from every consumer.
    session = _session(workspace, calls=[
        ToolCall(id="toolu_s", name="scratchpad", input={
            "action": "exec", "name": "m", "code": "print('Total revenue: 20437.08'); {}['region']",
        }),
    ])
    session.tool_registry.dispatch_tool = AsyncMock(return_value=ToolOutcome(
        content="[output]\nTotal revenue: 20437.08\n[error]\nKeyError: 'region'",
        ok=False, reason="KeyError: 'region'",
    ))
    await _run(session)

    result = _results(session)["toolu_s"]
    assert "20437.08" in result["content"]
    assert "is_error" not in result


async def test_an_unclassified_result_stays_unflagged(workspace):
    # ok=None: the handler gave no verdict. The legacy text fallback counts
    # "failed" toward the error streak, but that guess must not hide the output.
    session = _session(workspace, calls=[ToolCall(id="toolu_u", name="recall", input={})])
    session.tool_registry.dispatch_tool = AsyncMock(return_value=ToolOutcome(
        content="0 records failed validation; 42 passed", ok=None,
    ))
    await _run(session)

    result = _results(session)["toolu_u"]
    assert "42 passed" in result["content"]
    assert "is_error" not in result


def test_the_debug_log_serializer_keeps_the_flag():
    failed = {"type": "tool_result", "tool_use_id": "toolu_1", "content": MCP_ERROR, "is_error": True}
    ok = {"type": "tool_result", "tool_use_id": "toolu_2", "content": "48250.00"}
    assert serialize_content([failed, ok]) == [failed, ok]


def test_a_call_cancelled_mid_turn_gets_a_flagged_result(workspace):
    # repair_history answers a call that never ran; nothing in it is output.
    session = ChatSession(ChatSessionConfig(
        llm_client=make_mock_llm(), workspace=workspace, initial_history=[
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_c", "name": "hubspot__search_deals", "input": {}},
            ]},
        ],
    ))
    session.repair_history()

    result = _results(session)["toolu_c"]
    assert result["content"] == "Cancelled by user."
    assert result.get("is_error") is True


def test_a_replayed_call_with_no_recorded_result_gets_a_flagged_fill():
    # Two calls lost their ids; the reply holds one result, so the second
    # call gets a result anton writes itself.
    history = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "", "name": "hubspot__search_deals", "input": {"limit": 1}},
            {"type": "tool_use", "id": "", "name": "hubspot__search_deals", "input": {"limit": 2}},
        ]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "", "content": "48250.00"}]},
    ]
    fixed, _ = repair_replayed_tool_ids(history)

    fill = fixed[2]["content"][1]
    assert fill["content"] == "[no result was recorded for this call]"
    assert fill.get("is_error") is True
    assert "is_error" not in fixed[2]["content"][0]


async def test_the_compaction_summary_input_marks_a_failed_result(workspace):
    # Once a failed call folds into the summary, the summarizer's text is
    # all that is left of it, so the text must say the call failed.
    failed_pair = [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_f", "name": "hubspot__search_deals", "input": {}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_f", "content": MCP_ERROR, "is_error": True},
        ]},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_g", "name": "hubspot__search_deals", "input": {}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_g", "content": "48250.00"},
        ]},
    ]
    history = [{"role": "user", "content": "go " + "x" * 50}, *failed_pair]
    history += [
        {"role": "assistant" if i % 2 == 0 else "user", "content": "y" * 50} for i in range(9)
    ]
    llm = make_mock_llm()
    llm.summarize = AsyncMock(return_value=LLMResponse(content="## Goal\nx", usage=_usage()))
    session = ChatSession(ChatSessionConfig(llm_client=llm, workspace=workspace, initial_history=history))
    assert await session._summarize_history() is True

    sent = llm.summarize.await_args.kwargs["messages"][0]["content"]
    assert f"[tool_result error]: {MCP_ERROR}" in sent
    assert "[tool_result]: 48250.00" in sent
