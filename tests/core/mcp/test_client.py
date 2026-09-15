from __future__ import annotations

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
    exactly one retry with a fresh call before raising."""
    async with McpSession(server=stub_mcp_server) as session:
        real_call_tool = session._client.call_tool
        calls = {"n": 0}

        async def flaky(name, arguments):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("transport hiccup")
            return await real_call_tool(name, arguments)

        monkeypatch.setattr(session._client, "call_tool", flaky)
        content, is_error = await session.call_tool("add", {"a": 2, "b": 2})

    assert calls["n"] == 2
    assert content == "4"
    assert is_error is False


async def test_call_tool_raises_transient_error_after_exhausting_the_retry(stub_mcp_server, monkeypatch):
    async with McpSession(server=stub_mcp_server) as session:

        async def always_fails(name, arguments):
            raise ConnectionError("still down")

        monkeypatch.setattr(session._client, "call_tool", always_fails)

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
