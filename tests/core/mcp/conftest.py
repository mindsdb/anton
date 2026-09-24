from __future__ import annotations

import pytest
from mcp.server.mcpserver import MCPServer


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
