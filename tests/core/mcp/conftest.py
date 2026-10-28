from __future__ import annotations

import pytest
from mcp.server.mcpserver import MCPServer
from mcp_types import CallToolResult, TextContent

GENIE_REASONING = "GENIE-REASONING-TEXT " * 600
GENIE_FINAL_ANSWER = "Revenue was 42 last quarter."
GENIE_RESULT_ROWS = 250
GENIE_POLLS_IN_PROGRESS = 2


@pytest.fixture()
def genie_final_answer() -> str:
    """The final answer the Genie stub server reports for a completed ask."""
    return GENIE_FINAL_ANSWER


@pytest.fixture()
def genie_result_rows() -> int:
    """The total row count the Genie stub server reports for a query result."""
    return GENIE_RESULT_ROWS


@pytest.fixture()
def stub_mcp_server() -> MCPServer:
    """An in-process MCP server exposing a few stub tools — the SDK's own
    test pattern (no sockets, no threads, no subprocess). See
    anton/core/mcp/__init__.py for why this is enough to validate the client
    without a real remote MCP server."""
    server = MCPServer(name="stub")

    @server.tool()
    def add(a: int, b: int) -> str:
        return str(a + b)

    @server.tool()
    def boom() -> str:
        raise ValueError("kaboom")

    return server


def _genie_result(text: str, structured: dict) -> CallToolResult:
    """A tool result carrying both Genie's long text and its structured fields."""
    return CallToolResult(content=[TextContent(type="text", text=text)], structured_content=structured)


@pytest.fixture()
def genie_stub_server() -> MCPServer:
    """An in-process server shaped like Databricks Genie One.

    `genie_ask` returns at once with `in_progress`; `genie_poll_response`
    reports `in_progress` GENIE_POLLS_IN_PROGRESS times, then completes with
    a long reasoning text beside the structured fields. `view_ask` is not on
    the live server and stays here only to prove it is never exposed.
    """
    server = MCPServer(name="genie-stub")
    polls = {"count": 0}

    @server.tool()
    def genie_ask(question: str, conversation_id: str | None = None) -> CallToolResult:
        ids = {"conversation_id": conversation_id or "conv-1", "response_id": "resp-1"}
        return _genie_result("Genie started working.", {**ids, "status": "in_progress"})

    @server.tool()
    def genie_poll_response(conversation_id: str, response_id: str) -> CallToolResult:
        polls["count"] += 1
        ids = {"conversation_id": conversation_id, "response_id": response_id}
        if polls["count"] <= GENIE_POLLS_IN_PROGRESS:
            return _genie_result(GENIE_REASONING, {**ids, "status": "in_progress", "progress_steps": ["thinking"]})
        return _genie_result(
            GENIE_REASONING,
            {
                **ids,
                "status": "completed",
                "deep_link": "https://dbc-1.cloud.databricks.com/genie/x",
                "final_answer": GENIE_FINAL_ANSWER,
                "progress_steps": ["thinking", GENIE_REASONING],
                "narration_instruction": "Your reply MUST include the link.",
                "query_items": [{"item_id": "pub_1_output", "sql": "SELECT 1"}],
            },
        )

    @server.tool()
    def genie_get_query_result(conversation_id: str, response_id: str, item_id: str) -> CallToolResult:
        rows = [[str(index), "x"] for index in range(GENIE_RESULT_ROWS)]
        return _genie_result(
            "table text",
            {
                "conversation_id": conversation_id,
                "response_id": response_id,
                "item_id": item_id,
                "columns": [{"name": "n", "type_text": "INT"}, {"name": "label", "type_text": "STRING"}],
                "rows": rows,
                "total_row_count": GENIE_RESULT_ROWS,
                "truncated": False,
                "statement_state": "SUCCEEDED",
                "ready": True,
            },
        )

    @server.tool()
    def genie_cancel_response(conversation_id: str) -> str:
        return "cancelled"

    @server.tool()
    def view_ask(question: str) -> str:
        return "not on the live server"

    return server
