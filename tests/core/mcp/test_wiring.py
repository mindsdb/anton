from __future__ import annotations

import asyncio
import contextvars

import httpx2
import pytest

from anton.core.datasources.data_vault import LocalDataVault
from anton.core.mcp import servers, wiring
from anton.core.mcp.client import McpSession
from anton.core.mcp.databricks import GenieCredentialsRefused, genie_mcp_url
from anton.utils.datasources import begin_ds_turn_scope, scrub_credentials

HOST = "dbc-1.cloud.databricks.com"
SECRET = "sentinel-secret-1"
TOKEN = "sentinel-token"
GENIE_FIELDS = {
    "_method": "genie-mcp",
    "workspace_url": f"https://{HOST}/",
    "client_id": "cid",
    "_client_secret": SECRET,
}


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


async def test_notion_write_mode_registers_every_tool(patch_mcp_session_to_stub):
    vault = FakeDataVault({("notion", "acme"): {"_method": "mcp", "_access_mode": "write", "access_token": "tok"}})
    tool_defs, sessions = await wiring.discover_mcp_tools_async(
        vault, [{"engine": "notion", "name": "acme"}]
    )
    assert {d.name for d in tool_defs} == {"notion__add", "notion__boom"}
    await wiring.close_mcp_sessions(sessions)


async def test_two_connections_for_the_same_engine_get_disambiguated_names(patch_mcp_session_to_stub):
    """The vault explicitly supports more than one connection per engine
    (data_vault.py: `name` "disambiguates when an org has more than one
    connection for this engine"). Without disambiguation both connections'
    tools would collide under `{engine}__{tool_name}` and
    ToolRegistry.register_tool would silently drop the second connection's
    tools as duplicates."""
    vault = FakeDataVault(
        {
            ("hubspot", "acme"): {"_method": "mcp", "_access_mode": "write", "access_token": "tok-1"},
            ("hubspot", "other-org"): {"_method": "mcp", "_access_mode": "write", "access_token": "tok-2"},
        }
    )
    tool_defs, sessions = await wiring.discover_mcp_tools_async(
        vault,
        [{"engine": "hubspot", "name": "acme"}, {"engine": "hubspot", "name": "other-org"}],
    )

    assert {d.name for d in tool_defs} == {
        "hubspot__acme__add", "hubspot__acme__boom",
        "hubspot__other-org__add", "hubspot__other-org__boom",
    }
    assert len(sessions) == 2
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


class RecordingLogger:
    """Stands in for a module logger and keeps every formatted line."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def _record(self, message: str, *args, **kwargs) -> None:
        self.lines.append(message % args if args else message)

    debug = info = warning = error = exception = _record


@pytest.fixture()
def genie_vault(tmp_path) -> LocalDataVault:
    """A real file vault holding one Genie connection."""
    vault = LocalDataVault(tmp_path)
    vault.save("databricks", "genie-1", dict(GENIE_FIELDS))
    return vault


@pytest.fixture()
def minted(monkeypatch) -> list[tuple[str, str, str]]:
    """Patch `mint_token` to record its arguments and hand back the sentinel token."""
    calls: list[tuple[str, str, str]] = []

    async def fake_mint(host, client_id, client_secret, *, transport=None):
        calls.append((host, client_id, client_secret))
        return TOKEN

    monkeypatch.setattr(servers, "mint_token", fake_mint)
    return calls


@pytest.fixture()
def opened_sessions(monkeypatch, genie_stub_server) -> list[dict]:
    """Point every `McpSession` wiring opens at the Genie stub, recording url and token."""
    opened: list[dict] = []

    def fake_session_factory(*, url=None, access_token=None, **kwargs):
        opened.append({"url": url, "access_token": access_token})
        return McpSession(server=genie_stub_server)

    monkeypatch.setattr(wiring, "McpSession", fake_session_factory)
    return opened


async def test_a_genie_connection_is_discovered_with_a_minted_token(genie_vault, minted, opened_sessions):
    tool_defs, sessions = await wiring.discover_mcp_tools_async(
        genie_vault, [{"engine": "databricks", "name": "genie-1"}]
    )
    try:
        assert {d.name for d in tool_defs} == {
            "databricks__genie_ask",
            "databricks__genie_poll_response",
            "databricks__genie_get_query_result",
            "databricks__genie_cancel_response",
        }
        assert minted == [(HOST, "cid", SECRET)]
        assert opened_sessions == [{"url": genie_mcp_url(HOST), "access_token": TOKEN}]
    finally:
        await wiring.close_mcp_sessions(sessions)


def test_the_secret_is_not_in_the_genie_connection_environment(tmp_path):
    vault = LocalDataVault(tmp_path)
    vault.save("databricks", "genie-1", dict(GENIE_FIELDS))
    env = vault.env_for("databricks", "genie-1")
    assert env["DS_DATABRICKS_GENIE_1__WORKSPACE_URL"] == f"https://{HOST}/"
    assert env["DS_DATABRICKS_GENIE_1__CLIENT_ID"] == "cid"
    assert not any(SECRET in value for value in env.values())


async def test_refused_credentials_skip_the_connection_and_log_no_secret(
    genie_vault, opened_sessions, monkeypatch
):
    async def refused(host, client_id, client_secret, *, transport=None):
        raise GenieCredentialsRefused("Databricks refused the service principal's client ID or secret")

    monkeypatch.setattr(servers, "mint_token", refused)
    recorder = RecordingLogger()
    monkeypatch.setattr(wiring, "logger", recorder)

    tool_defs, sessions = await wiring.discover_mcp_tools_async(
        genie_vault, [{"engine": "databricks", "name": "genie-1"}]
    )

    assert (tool_defs, sessions) == ([], [])
    assert opened_sessions == []
    assert any("databricks" in line and "genie-1" in line and "GenieCredentialsRefused" in line for line in recorder.lines)
    assert not any(SECRET in line for line in recorder.lines)


async def test_a_bad_stored_host_is_skipped_without_a_request(tmp_path, minted, opened_sessions, monkeypatch):
    vault = LocalDataVault(tmp_path)
    vault.save("databricks", "genie-1", {**GENIE_FIELDS, "workspace_url": "https://evil.example.com"})
    recorder = RecordingLogger()
    monkeypatch.setattr(wiring, "logger", recorder)

    tool_defs, sessions = await wiring.discover_mcp_tools_async(
        vault, [{"engine": "databricks", "name": "genie-1"}]
    )

    assert (tool_defs, sessions) == ([], [])
    assert minted == []
    assert opened_sessions == []
    assert any("InvalidWorkspaceHost" in line for line in recorder.lines)
    assert not any("evil" in line for line in recorder.lines)


async def test_a_rejected_token_at_connect_skips_the_connection(genie_vault, minted, monkeypatch):
    request = httpx2.Request("POST", genie_mcp_url(HOST))
    response = httpx2.Response(401, request=request, text=f"rejected {TOKEN}")
    rejection = ExceptionGroup(
        "unhandled errors in a TaskGroup",
        [httpx2.HTTPStatusError(f"401 for {TOKEN}", request=request, response=response)],
    )

    class RejectedSession:
        """A session whose connect fails the way a 401 at `initialize` does."""

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            raise rejection

    monkeypatch.setattr(wiring, "McpSession", RejectedSession)
    recorder = RecordingLogger()
    monkeypatch.setattr(wiring, "logger", recorder)

    tool_defs, sessions = await wiring.discover_mcp_tools_async(
        genie_vault, [{"engine": "databricks", "name": "genie-1"}]
    )

    assert (tool_defs, sessions) == ([], [])
    assert any("databricks" in line and "genie-1" in line and "ExceptionGroup" in line for line in recorder.lines)
    assert not any(TOKEN in line or SECRET in line for line in recorder.lines)


async def test_hubspot_and_notion_connections_are_unchanged(monkeypatch, stub_mcp_server):
    assert await servers.resolve_mcp_endpoint("hubspot", "mcp", {"access_token": "tok-h"}) == (
        "https://mcp.hubspot.com",
        "tok-h",
    )
    assert await servers.resolve_mcp_endpoint("notion", "mcp", {"access_token": "tok-n"}) == (
        "https://mcp.notion.com/mcp",
        "tok-n",
    )
    assert await servers.resolve_mcp_endpoint("hubspot", "mcp", {}) is None
    assert await servers.resolve_mcp_endpoint("hubspot", "genie-mcp", dict(GENIE_FIELDS)) is None

    opened: list[dict] = []

    def fake_session_factory(*, url=None, access_token=None, **kwargs):
        opened.append({"url": url, "access_token": access_token})
        return McpSession(server=stub_mcp_server)

    monkeypatch.setattr(wiring, "McpSession", fake_session_factory)
    vault = FakeDataVault(
        {
            ("hubspot", "acme"): {"_method": "mcp", "_access_mode": "write", "access_token": "tok-h"},
            ("notion", "acme"): {"_method": "mcp", "_access_mode": "write", "access_token": "tok-n"},
        }
    )
    tool_defs, sessions = await wiring.discover_mcp_tools_async(
        vault, [{"engine": "hubspot", "name": "acme"}, {"engine": "notion", "name": "acme"}]
    )
    try:
        assert {d.name for d in tool_defs} == {"hubspot__add", "hubspot__boom", "notion__add", "notion__boom"}
        assert sorted(opened, key=lambda item: item["url"]) == [
            {"url": "https://mcp.hubspot.com", "access_token": "tok-h"},
            {"url": "https://mcp.notion.com/mcp", "access_token": "tok-n"},
        ]
        outcome = await next(d for d in tool_defs if d.name == "hubspot__add").handler(None, {"a": 2, "b": 3})
        assert (outcome.content, outcome.ok, outcome.reason) == ("5", True, "")
    finally:
        await wiring.close_mcp_sessions(sessions)


async def test_the_secret_and_token_are_scrubbed_in_the_turn(minted):
    async def turn() -> str:
        begin_ds_turn_scope()
        endpoint = await servers.resolve_mcp_endpoint("databricks", "genie-mcp", dict(GENIE_FIELDS))
        assert endpoint == (genie_mcp_url(HOST), TOKEN)
        return scrub_credentials(f"secret={SECRET} token={TOKEN}")

    scrubbed = await asyncio.create_task(turn(), context=contextvars.Context())

    assert SECRET not in scrubbed
    assert TOKEN not in scrubbed
    assert scrubbed == "secret=[DS_DATABRICKS__CLIENT_SECRET] token=[DS_DATABRICKS__ACCESS_TOKEN]"
