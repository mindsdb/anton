from __future__ import annotations

import re
import socket

import pytest
from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations

from anton.core.mcp import wiring
from anton.core.mcp.client import McpSession
from anton.core.mcp.errors import McpEgressBlocked, McpHostUnresolved
from anton.core.mcp.names import parse_namespaced_tool_name


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


def _custom_record(**overrides) -> dict:
    record = {
        "_method": "mcp",
        "_server_url": "https://mcp.example.com/mcp",
        "_allowed_tools": '["add"]',
        "_label": "Acme Tracker",
        "access_token": "tok",
    }
    record.update(overrides)
    return {key: value for key, value in record.items() if value is not None}


@pytest.fixture()
def recorded_sessions(monkeypatch, stub_mcp_server):
    """Record every `McpSession` wiring builds; guarded ones are real
    sessions, so the address check runs, and the rest open the stub."""
    built: list[dict] = []

    def factory(*, url=None, access_token=None, guarded=False, **kwargs):
        built.append({"url": url, "access_token": access_token, "guarded": guarded})
        if guarded and url is not None and "10.0.0.5" in url:
            return McpSession(url=url, access_token=access_token, guarded=True)
        return McpSession(server=stub_mcp_server)

    monkeypatch.setattr(wiring, "McpSession", factory)
    return built


async def test_a_custom_connection_uses_its_own_url_through_a_guarded_session(recorded_sessions):
    vault = FakeDataVault({("custom_mcp", "acme-tracker"): _custom_record()})
    tool_defs, sessions = await wiring.discover_mcp_tools_async(
        vault, [{"engine": "custom_mcp", "name": "acme-tracker"}]
    )
    assert recorded_sessions == [{"url": "https://mcp.example.com/mcp", "access_token": "tok", "guarded": True}]
    assert [d.name for d in tool_defs] == ["custom_mcp__acme-tracker__add"]
    await wiring.close_mcp_sessions(sessions)


async def test_a_tool_the_server_adds_later_stays_off(recorded_sessions):
    vault = FakeDataVault({("custom_mcp", "acme-tracker"): _custom_record(_allowed_tools='["add", "renamed"]')})
    tool_defs, sessions = await wiring.discover_mcp_tools_async(
        vault, [{"engine": "custom_mcp", "name": "acme-tracker"}]
    )
    assert [d.name for d in tool_defs] == ["custom_mcp__acme-tracker__add"]
    await wiring.close_mcp_sessions(sessions)


@pytest.mark.parametrize("allowed", [None, "", "not json", '{"add": true}', '["add", 1]', "add"])
async def test_an_unreadable_tool_list_skips_the_connection_before_any_request(recorded_sessions, allowed):
    vault = FakeDataVault({("custom_mcp", "acme-tracker"): _custom_record(_allowed_tools=allowed)})
    tool_defs, sessions = await wiring.discover_mcp_tools_async(
        vault, [{"engine": "custom_mcp", "name": "acme-tracker"}]
    )
    assert (tool_defs, sessions, recorded_sessions) == ([], [], [])


@pytest.mark.parametrize(
    "server_url",
    [None, "", "http://mcp.example.com/mcp", "https://mcp.example.com/mcp?key=secret", "https://u:p@mcp.example.com/"],
)
async def test_an_unusable_server_url_skips_the_connection_before_any_request(recorded_sessions, server_url):
    vault = FakeDataVault({("custom_mcp", "acme-tracker"): _custom_record(_server_url=server_url)})
    tool_defs, sessions = await wiring.discover_mcp_tools_async(
        vault, [{"engine": "custom_mcp", "name": "acme-tracker"}]
    )
    assert (tool_defs, sessions, recorded_sessions) == ([], [], [])


async def test_a_built_in_engine_ignores_a_server_url_on_its_record(recorded_sessions):
    vault = FakeDataVault(
        {
            ("hubspot", "acme"): {
                "_method": "mcp",
                "_access_mode": "write",
                "_server_url": "https://attacker.example.com/mcp",
                "access_token": "tok",
            }
        }
    )
    tool_defs, sessions = await wiring.discover_mcp_tools_async(vault, [{"engine": "hubspot", "name": "acme"}])
    assert recorded_sessions == [{"url": "https://mcp.hubspot.com", "access_token": "tok", "guarded": False}]
    await wiring.close_mcp_sessions(sessions)


async def test_a_custom_server_at_a_private_address_is_skipped_and_the_turn_keeps_its_other_tools(recorded_sessions):
    vault = FakeDataVault(
        {
            ("custom_mcp", "inside"): _custom_record(_server_url="https://10.0.0.5/mcp"),
            ("hubspot", "acme"): {"_method": "mcp", "_access_mode": "write", "access_token": "tok"},
        }
    )
    tool_defs, sessions = await wiring.discover_mcp_tools_async(
        vault, [{"engine": "custom_mcp", "name": "inside"}, {"engine": "hubspot", "name": "acme"}]
    )
    assert {d.name for d in tool_defs} == {"hubspot__add", "hubspot__boom"}
    assert len(sessions) == 1
    await wiring.close_mcp_sessions(sessions)


async def test_a_no_login_custom_server_connects_without_a_token(recorded_sessions):
    vault = FakeDataVault({("custom_mcp", "open-data"): _custom_record(access_token=None)})
    tool_defs, sessions = await wiring.discover_mcp_tools_async(
        vault, [{"engine": "custom_mcp", "name": "open-data"}]
    )
    assert recorded_sessions[0]["access_token"] is None
    assert [d.name for d in tool_defs] == ["custom_mcp__open-data__add"]
    await wiring.close_mcp_sessions(sessions)


async def test_a_custom_connection_switched_off_registers_nothing(recorded_sessions):
    vault = FakeDataVault({("custom_mcp", "acme-tracker"): _custom_record(_access_mode="none")})
    tool_defs, sessions = await wiring.discover_mcp_tools_async(
        vault, [{"engine": "custom_mcp", "name": "acme-tracker"}]
    )
    assert tool_defs == []
    await wiring.close_mcp_sessions(sessions)


async def test_call_mcp_tool_does_not_serve_custom_servers():
    with pytest.raises(ValueError):
        await wiring.call_mcp_tool("custom_mcp", "tok", "add")


async def test_listing_a_custom_servers_tools_reports_the_read_only_label(monkeypatch):
    server = MCPServer(name="labelled")

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    def search_issues(query: str) -> str:
        return query

    @server.tool(annotations=ToolAnnotations(read_only_hint=False))
    def create_issue(title: str) -> str:
        return title

    @server.tool()
    def sync() -> str:
        return "ok"

    @server.tool(annotations=ToolAnnotations(destructive_hint=False))
    def archive(issue_id: str) -> str:
        return issue_id

    @server.tool(name="files.read")
    def files_read() -> str:
        return "x"

    built: list[dict] = []

    def factory(*, url=None, access_token=None, guarded=False, **kwargs):
        built.append({"url": url, "access_token": access_token, "guarded": guarded})
        return McpSession(server=server)

    monkeypatch.setattr(wiring, "McpSession", factory)
    _answer_lookups_with(monkeypatch, "93.184.215.14")
    tools = await wiring.list_custom_server_tools(
        "https://mcp.example.com/mcp", "tok", connection_name="acme-tracker"
    )
    assert built == [{"url": "https://mcp.example.com/mcp", "access_token": "tok", "guarded": True}]
    assert {(tool.name, tool.read_only) for tool in tools} == {
        ("search_issues", True),
        ("create_issue", False),
        ("sync", False),
        ("archive", False),
    }


def _answer_lookups_with(monkeypatch, *addresses: str) -> None:
    """Make wiring's host lookup answer with `addresses`, or fail when none are given."""

    async def resolver(host: str, port: int):
        if not addresses:
            raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port)) for address in addresses]

    real_resolve_public = wiring.resolve_public
    monkeypatch.setattr(
        wiring, "resolve_public", lambda host, port: real_resolve_public(host, port, resolver=resolver)
    )


@pytest.mark.parametrize("url", ["https://10.0.0.5/mcp", "https://169.254.169.254/mcp", "http://mcp.example.com/mcp"])
async def test_listing_tools_at_a_refused_address_raises_before_connecting(url):
    with pytest.raises(McpEgressBlocked):
        await wiring.list_custom_server_tools(url, "tok", connection_name="acme-tracker")


async def test_listing_tools_at_a_name_that_does_not_resolve_says_so_before_connecting(recorded_sessions, monkeypatch):
    _answer_lookups_with(monkeypatch)
    with pytest.raises(McpHostUnresolved):
        await wiring.list_custom_server_tools("https://nowhere.example.com/mcp", "tok", connection_name="x")
    assert recorded_sessions == []


async def test_listing_tools_at_a_name_resolving_privately_is_refused_before_connecting(recorded_sessions, monkeypatch):
    _answer_lookups_with(monkeypatch, "10.0.0.5")
    with pytest.raises(McpEgressBlocked):
        await wiring.list_custom_server_tools("https://inside.example.com/mcp", "tok", connection_name="x")
    assert recorded_sessions == []


SECRET_PATH = "SECRET-PATH-7f3a"
SECRET_ERROR = f"Client error '404 Not Found' for url 'https://mcp.example.com/{SECRET_PATH}/mcp'"


class _FailsToOpen(McpSession):
    async def _connect(self) -> None:
        raise RuntimeError(SECRET_ERROR)


class _FailsToList(McpSession):
    async def _connect(self) -> None:
        return None

    async def list_tools(self):
        raise RuntimeError(SECRET_ERROR)

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _FailsToClose(McpSession):
    async def __aexit__(self, exc_type, exc, tb) -> None:
        raise RuntimeError(SECRET_ERROR)


@pytest.mark.parametrize("session_class", [_FailsToOpen, _FailsToList])
async def test_a_custom_server_failure_logs_no_part_of_its_url_path(monkeypatch, caplog, session_class):
    monkeypatch.setattr(
        wiring,
        "McpSession",
        lambda *, url=None, access_token=None, guarded=False, **kw: session_class(
            url=url, access_token=access_token, guarded=guarded
        ),
    )
    vault = FakeDataVault(
        {("custom_mcp", "acme-tracker"): _custom_record(_server_url=f"https://mcp.example.com/{SECRET_PATH}/mcp")}
    )
    caplog.set_level("DEBUG")
    tool_defs, sessions = await wiring.discover_mcp_tools_async(
        vault, [{"engine": "custom_mcp", "name": "acme-tracker"}]
    )
    assert (tool_defs, sessions) == ([], [])
    assert "RuntimeError" in caplog.text
    assert SECRET_PATH not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


async def test_a_built_in_server_failure_still_logs_the_full_error(monkeypatch, caplog):
    monkeypatch.setattr(wiring, "McpSession", lambda *, url=None, access_token=None, **kw: _FailsToOpen(url=url))
    vault = FakeDataVault({("hubspot", "acme"): {"_method": "mcp", "access_token": "tok"}})
    caplog.set_level("DEBUG")
    await wiring.discover_mcp_tools_async(vault, [{"engine": "hubspot", "name": "acme"}])
    assert SECRET_PATH in caplog.text


async def test_closing_a_custom_session_logs_no_part_of_its_url_path(caplog):
    caplog.set_level("DEBUG")
    await wiring.close_mcp_sessions([_FailsToClose(url="https://mcp.example.com/mcp", guarded=True)])
    assert "RuntimeError" in caplog.text
    assert SECRET_PATH not in caplog.text


async def test_two_hubspot_portals_with_long_names_both_stay_callable(monkeypatch):
    """HubSpot connection names are `<email>-<portal name>`, uncapped."""
    server = MCPServer(name="hubspot-stub")

    @server.tool(name="search_crm_objects")
    def search(query: str) -> str:
        return f"found {query}"

    @server.tool(name="discover_hubspot_schema")
    def schema() -> str:
        return "schema"

    monkeypatch.setattr(wiring, "McpSession", lambda **kwargs: McpSession(server=server))
    first = "jo@example.com-Acme Corporation Sales Portal EU"
    second = "jo@example.com-Acme Corporation Sales Portal US"
    record = {"_method": "mcp", "_access_mode": "read", "access_token": "tok"}
    vault = FakeDataVault({("hubspot", first): dict(record), ("hubspot", second): dict(record)})
    tool_defs, sessions = await wiring.discover_mcp_tools_async(
        vault, [{"engine": "hubspot", "name": first}, {"engine": "hubspot", "name": second}]
    )
    names = [d.name for d in tool_defs]
    assert len(names) == len(set(names)) == 4
    assert all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) for name in names)
    search_def = next(d for d in tool_defs if d.name.endswith("__search_crm_objects"))
    outcome = await search_def.handler(None, {"query": "deals"})
    assert (outcome.ok, outcome.content) == (True, "found deals")
    await wiring.close_mcp_sessions(sessions)


async def test_two_notion_workspaces_with_long_names_both_stay_callable(monkeypatch):
    """Two connections whose 51-character names would push tool names past 64."""
    server = MCPServer(name="notion-stub")

    @server.tool(name="notion-search")
    def search(query: str) -> str:
        return f"found {query}"

    @server.tool(name="notion-fetch")
    def fetch(id: str) -> str:
        return f"page {id}"

    monkeypatch.setattr(wiring, "McpSession", lambda **kwargs: McpSession(server=server))
    first = "jo-example-com-" + "1b2c3d4e-0000-4000-8000-000000000001"
    second = "jo-example-com-" + "1b2c3d4e-0000-4000-8000-000000000002"
    assert len(first) == len(second) == 51
    record = {"_method": "mcp", "_access_mode": "read", "access_token": "tok"}
    vault = FakeDataVault({("notion", first): dict(record), ("notion", second): dict(record)})
    tool_defs, sessions = await wiring.discover_mcp_tools_async(
        vault, [{"engine": "notion", "name": first}, {"engine": "notion", "name": second}]
    )
    names = [d.name for d in tool_defs]
    assert len(names) == len(set(names)) == 4
    assert all(len(name) <= 64 for name in names)
    for tool_def in tool_defs:
        parts = parse_namespaced_tool_name(name=tool_def.name)
        assert parts is not None and parts.engine == "notion"
    fetch_def = next(d for d in tool_defs if d.name.endswith("__notion-fetch"))
    outcome = await fetch_def.handler(None, {"id": "abc"})
    assert (outcome.ok, outcome.content) == (True, "page abc")
    await wiring.close_mcp_sessions(sessions)
