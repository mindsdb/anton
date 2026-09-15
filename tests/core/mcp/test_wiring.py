from __future__ import annotations

import pytest

from anton.core.mcp import wiring
from anton.core.mcp.client import McpSession


class FakeDataVault:
    """Minimal `DataVault`-shaped stub — only the methods wiring.py reads."""

    def __init__(self, records: dict[tuple[str, str], dict]) -> None:
        self._records = records

    def read_record(self, engine: str, name: str):
        record = self._records.get((engine, name))
        if record is None:
            return None
        return {"fields": record}

    def load(self, engine: str, name: str):
        record = self._records.get((engine, name))
        return dict(record) if record else None

    def list_connections(self):
        return [{"engine": e, "name": n, "created_at": ""} for (e, n) in self._records]


@pytest.fixture()
def patch_mcp_session_to_stub(monkeypatch, stub_mcp_server):
    """Every `McpSession(url=..., access_token=...)` call in wiring.py opens
    the in-process stub server instead of a real remote connection."""

    def fake_session_factory(*, url=None, access_token=None, **kwargs):
        return McpSession(server=stub_mcp_server)

    monkeypatch.setattr(wiring, "McpSession", fake_session_factory)


async def test_no_mcp_connections_is_a_safe_no_op(patch_mcp_session_to_stub):
    vault = FakeDataVault({("linear", "me"): {"_method": "oauth", "access_token": "tok"}})
    tool_defs, sessions = await wiring.discover_mcp_tools_async(
        vault, [{"engine": "linear", "name": "me"}]
    )
    assert tool_defs == []
    assert sessions == []


async def test_mcp_connection_yields_namespaced_tools_and_an_open_session(patch_mcp_session_to_stub):
    vault = FakeDataVault(
        {("hubspot", "acme"): {"_method": "mcp", "_access_mode": "write", "access_token": "tok"}}
    )
    tool_defs, sessions = await wiring.discover_mcp_tools_async(
        vault, [{"engine": "hubspot", "name": "acme"}]
    )
    assert {d.name for d in tool_defs} == {"hubspot__add", "hubspot__boom"}
    assert len(sessions) == 1

    await wiring.close_mcp_sessions(sessions)


async def test_missing_access_token_skips_the_connection_without_failing_the_turn(patch_mcp_session_to_stub):
    vault = FakeDataVault({("hubspot", "acme"): {"_method": "mcp", "_access_mode": "write"}})
    tool_defs, sessions = await wiring.discover_mcp_tools_async(
        vault, [{"engine": "hubspot", "name": "acme"}]
    )
    assert tool_defs == []
    assert sessions == []


async def test_unknown_engine_skips_the_connection_without_failing_the_turn(patch_mcp_session_to_stub):
    vault = FakeDataVault(
        {("some-future-engine", "acme"): {"_method": "mcp", "access_token": "tok"}}
    )
    tool_defs, sessions = await wiring.discover_mcp_tools_async(
        vault, [{"engine": "some-future-engine", "name": "acme"}]
    )
    assert tool_defs == []
    assert sessions == []


async def test_access_mode_defaults_to_read_when_unset(patch_mcp_session_to_stub):
    """No connect-time UI feeds `_access_mode` (see ENG-1816's "Decisions
    made in review") — every new MCP connection is hardcoded to read."""
    vault = FakeDataVault({("hubspot", "acme"): {"_method": "mcp", "access_token": "tok"}})
    tool_defs, sessions = await wiring.discover_mcp_tools_async(
        vault, [{"engine": "hubspot", "name": "acme"}]
    )
    # `add`/`boom` aren't in HubSpot's read/write table, so under the
    # fail-closed default read mode sees neither.
    assert tool_defs == []
    await wiring.close_mcp_sessions(sessions)


async def test_call_mcp_tool_opens_calls_and_closes_without_registering_anything(
    monkeypatch, stub_mcp_server
):
    monkeypatch.setattr(
        wiring, "McpSession", lambda *, url=None, access_token=None, **kw: McpSession(server=stub_mcp_server)
    )
    result = await wiring.call_mcp_tool("hubspot", "tok", "add", a=5, b=6)
    assert result == "11"


async def test_call_mcp_tool_raises_on_a_tool_level_error(monkeypatch, stub_mcp_server):
    monkeypatch.setattr(
        wiring, "McpSession", lambda *, url=None, access_token=None, **kw: McpSession(server=stub_mcp_server)
    )
    with pytest.raises(RuntimeError):
        await wiring.call_mcp_tool("hubspot", "tok", "boom")
