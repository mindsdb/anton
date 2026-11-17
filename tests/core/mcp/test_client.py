from __future__ import annotations

import asyncio

import mcp
import mcp_types
import pytest

from anton.core.mcp.client import McpSession
from anton.core.mcp.errors import McpPermanentError, McpTransientError


async def test_list_tools_returns_the_stub_servers_tools(stub_mcp_server):
    async with McpSession(server=stub_mcp_server) as session:
        tools = await session.list_tools()
    assert {t.name for t in tools} == {"add", "boom"}


async def test_call_tool_success_flattens_text_content(stub_mcp_server):
    async with McpSession(server=stub_mcp_server) as session:
        content, is_error = await session.call_tool("add", {"a": 1, "b": 2})
    assert content == "3"
    assert is_error is False


async def test_call_tool_error_surfaces_as_is_error_not_an_exception(stub_mcp_server):
    """Per the MCP spec, a tool-originated failure comes back INSIDE the
    result (is_error=True), not as a protocol-level exception — the caller
    (registry.py) decides what that means, this layer just reports it."""
    async with McpSession(server=stub_mcp_server) as session:
        content, is_error = await session.call_tool("boom", {})
    assert is_error is True
    assert isinstance(content, str)


async def test_session_used_outside_context_manager_raises(stub_mcp_server):
    session = McpSession(server=stub_mcp_server)
    with pytest.raises(RuntimeError):
        await session.list_tools()


def test_requires_exactly_one_of_url_or_server(stub_mcp_server):
    with pytest.raises(ValueError):
        McpSession()
    with pytest.raises(ValueError):
        McpSession(url="https://example.test", server=stub_mcp_server)


async def test_call_tool_retries_once_on_transient_failure(stub_mcp_server, monkeypatch):
    """A transport/protocol-level failure (not a tool-reported is_error) gets
    exactly one retry against a FRESH session (a dead transport's task group
    can't serve a second call — see McpSession._reconnect) before raising.
    Patched at the class level, not the instance: reconnecting builds a new
    `mcp.Client` object, so an instance-level patch would only ever see the
    first attempt."""
    calls = {"n": 0}
    real_call_tool = mcp.Client.call_tool

    async def flaky(self, name, arguments):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("transport hiccup")
        return await real_call_tool(self, name, arguments)

    monkeypatch.setattr(mcp.Client, "call_tool", flaky)
    async with McpSession(server=stub_mcp_server) as session:
        content, is_error = await session.call_tool("add", {"a": 2, "b": 2})

    assert calls["n"] == 2
    assert content == "4"
    assert is_error is False


async def test_call_tool_raises_transient_error_after_exhausting_the_retry(stub_mcp_server, monkeypatch):
    async def always_fails(self, name, arguments):
        raise ConnectionError("still down")

    monkeypatch.setattr(mcp.Client, "call_tool", always_fails)
    async with McpSession(server=stub_mcp_server) as session:
        with pytest.raises(McpTransientError):
            await session.call_tool("add", {"a": 1, "b": 1})


async def test_call_tool_never_retries_a_permanent_failure(stub_mcp_server, monkeypatch):
    async with McpSession(server=stub_mcp_server) as session:
        calls = {"n": 0}

        async def rejected(name, arguments):
            calls["n"] += 1
            raise McpPermanentError("token rejected")

        monkeypatch.setattr(session._client, "call_tool", rejected)

        with pytest.raises(McpPermanentError):
            await session.call_tool("add", {"a": 1, "b": 1})

    assert calls["n"] == 1


async def test_call_tool_reconnects_on_a_non_exception_transport_failure(stub_mcp_server, monkeypatch):
    """Killing a server mid-session can surface as asyncio.CancelledError or
    an (Base)ExceptionGroup rather than a plain Exception, depending on which
    background task of the transport noticed first (verified by hand,
    ENG-1816) — `except Exception` alone would miss it. This still gets one
    reconnect-and-retry, same as a plain exception."""
    calls = {"n": 0}
    real_call_tool = mcp.Client.call_tool

    async def flaky(self, name, arguments):
        calls["n"] += 1
        if calls["n"] == 1:
            raise asyncio.CancelledError("simulated transport-task-group teardown")
        return await real_call_tool(self, name, arguments)

    monkeypatch.setattr(mcp.Client, "call_tool", flaky)
    async with McpSession(server=stub_mcp_server) as session:
        content, is_error = await session.call_tool("add", {"a": 5, "b": 5})

    assert calls["n"] == 2
    assert content == "10"
    assert is_error is False


async def test_call_tool_reraises_the_original_error_when_a_non_exception_failure_persists(
    stub_mcp_server, monkeypatch
):
    """If reconnecting doesn't help, the ORIGINAL exception propagates as-is
    — never disguised as McpTransientError, since that could be real task
    cancellation (e.g. the whole turn being cancelled) rather than a
    business-logic failure a ToolOutcome should swallow."""

    async def always_cancelled(self, name, arguments):
        raise asyncio.CancelledError("still gone")

    monkeypatch.setattr(mcp.Client, "call_tool", always_cancelled)
    async with McpSession(server=stub_mcp_server) as session:
        with pytest.raises(asyncio.CancelledError):
            await session.call_tool("add", {"a": 1, "b": 1})


async def test_call_tool_never_retries_a_real_task_cancellation(stub_mcp_server, monkeypatch):
    """A CancelledError delivered because THIS task's own cancellation was
    requested (Task.cancelling() > 0) must propagate immediately, with no
    reconnect/retry attempted — retrying would silently swallow real
    cancellation, exactly what call_tool's docstring promises never happens.
    Distinguishes this from a CancelledError merely leaking up from the
    transport's own internal task group (cancelling() == 0 there — see
    test_call_tool_reconnects_on_a_non_exception_transport_failure, which
    still retries because nothing called .cancel() on the test's own task)."""
    calls = {"n": 0}

    async def always_cancelled(self, name, arguments):
        calls["n"] += 1
        raise asyncio.CancelledError()

    class _FakeCancelledTask:
        def cancelling(self) -> int:
            return 1

    monkeypatch.setattr(mcp.Client, "call_tool", always_cancelled)
    monkeypatch.setattr(asyncio, "current_task", lambda: _FakeCancelledTask())

    async with McpSession(server=stub_mcp_server) as session:
        with pytest.raises(asyncio.CancelledError):
            await session.call_tool("add", {"a": 1, "b": 1})

    assert calls["n"] == 1  # no retry attempted


async def test_a_guarded_session_retry_logs_no_part_of_its_url_path(caplog):
    secret = "SECRET-PATH-7f3a"
    error = RuntimeError(f"Client error for url 'https://mcp.example.com/{secret}/mcp'")

    class _FailingClient:
        async def call_tool(self, name, arguments):
            raise error

    session = McpSession(url=f"https://mcp.example.com/{secret}/mcp", guarded=True)
    session._client = _FailingClient()

    async def failed_reconnect():
        raise error

    session._reconnect = failed_reconnect
    caplog.set_level("DEBUG")
    with pytest.raises(McpTransientError):
        await session.call_tool("add", {"a": 1})
    assert "reconnect failed" in caplog.text
    assert secret not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def _paged_server(page_count: int, *, endless: bool = False):
    """A stub `mcp.Client.list_tools` that serves one tool per page."""
    cursors_seen: list[str | None] = []

    async def list_tools(self, *, cursor=None, **kwargs):
        cursors_seen.append(cursor)
        index = 0 if cursor is None else int(cursor)
        last = not endless and index == page_count - 1
        return mcp_types.ListToolsResult(
            tools=[mcp_types.Tool(name=f"tool_{index}", input_schema={"type": "object"})],
            next_cursor=None if last else str(index + 1),
        )

    return list_tools, cursors_seen


async def test_list_tools_follows_the_page_cursor_to_the_last_page(stub_mcp_server, monkeypatch):
    list_tools, cursors_seen = _paged_server(3)
    monkeypatch.setattr(mcp.Client, "list_tools", list_tools)
    async with McpSession(server=stub_mcp_server) as session:
        tools = await session.list_tools()
    assert [tool.name for tool in tools] == ["tool_0", "tool_1", "tool_2"]
    assert cursors_seen == [None, "1", "2"]


async def test_list_tools_stops_a_server_that_never_ends_its_pages(stub_mcp_server, monkeypatch):
    list_tools, cursors_seen = _paged_server(0, endless=True)
    monkeypatch.setattr(mcp.Client, "list_tools", list_tools)
    async with McpSession(server=stub_mcp_server) as session:
        tools = await session.list_tools()
    assert len(tools) == len(cursors_seen) == 20
