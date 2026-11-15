"""A 401/403 from a remote MCP server, through the real SDK client: the SDK
turns it into a JSON-RPC error with no status, so McpSession records it off
the wire."""
from __future__ import annotations

import json

import httpx2
import mcp
import pytest

from anton.core.mcp.client import McpSession
from anton.core.mcp.errors import McpPermanentError, McpTokenRejected

_URL = "https://mcp.example.test/mcp"


def _fake_server(*, open_status: int = 200, call_status: int = 200):
    def handle(request: httpx2.Request) -> httpx2.Response:
        if request.method != "POST":
            return httpx2.Response(405)
        message = json.loads(request.content)
        method = message.get("method")
        if method == "initialize":
            if open_status != 200:
                return httpx2.Response(open_status)
            return httpx2.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {
                        "protocolVersion": message["params"]["protocolVersion"],
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "fake", "version": "1"},
                    },
                },
            )
        if "id" not in message:
            return httpx2.Response(202)
        if method == "tools/list":
            tool = {"name": "anything", "inputSchema": {"type": "object"}}
            return httpx2.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": {"tools": [tool]}})
        if call_status != 200:
            return httpx2.Response(call_status)
        return httpx2.Response(
            200, json={"jsonrpc": "2.0", "id": message["id"], "result": {"content": [{"type": "text", "text": "ok"}]}}
        )

    return handle


@pytest.fixture()
def fake_server(monkeypatch):
    real_client = httpx2.AsyncClient

    def install(**statuses):
        handler = _fake_server(**statuses)
        monkeypatch.setattr(
            httpx2, "AsyncClient", lambda **kw: real_client(transport=httpx2.MockTransport(handler), **kw)
        )

    return install


@pytest.mark.parametrize("status", [401, 403])
async def test_a_rejected_token_on_open_raises_token_rejected(fake_server, status):
    fake_server(open_status=status)
    with pytest.raises(McpTokenRejected) as caught:
        async with McpSession(url=_URL, access_token="tok"):
            pass
    assert caught.value.status_code == status


@pytest.mark.parametrize("status", [401, 403])
async def test_a_rejected_token_on_a_call_raises_token_rejected_without_retrying(fake_server, status):
    fake_server(call_status=status)
    async with McpSession(url=_URL, access_token="tok") as session:
        with pytest.raises(McpTokenRejected) as caught:
            await session.call_tool("anything", {})
    assert caught.value.status_code == status
    assert isinstance(caught.value, McpPermanentError)


async def test_a_server_error_on_open_is_not_a_rejected_token(fake_server):
    fake_server(open_status=500)
    with pytest.raises(Exception) as caught:
        async with McpSession(url=_URL, access_token="tok"):
            pass
    assert not isinstance(caught.value, McpTokenRejected)


async def test_a_server_error_on_a_call_is_not_a_rejected_token(fake_server):
    fake_server(call_status=500)
    async with McpSession(url=_URL, access_token="tok") as session:
        with pytest.raises(Exception) as caught:
            await session.call_tool("anything", {})
    assert not isinstance(caught.value, McpPermanentError)


async def test_a_working_server_answers_the_call(fake_server):
    fake_server()
    async with McpSession(url=_URL, access_token="tok") as session:
        assert await session.call_tool("anything", {}) == ("ok", False)
