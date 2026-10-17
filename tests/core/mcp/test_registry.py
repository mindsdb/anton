from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from anton.core.mcp.access import TOOL_ACCESS
from anton.core.mcp.client import McpSession
from anton.core.mcp.names import ToolNameParts, namespaced_tool_name, parse_namespaced_tool_name
from anton.core.mcp.registry import discover_tool_defs
from anton.core.tools.registry import ToolOutcome


def test_namespaced_tool_name():
    assert namespaced_tool_name("hubspot", "search_crm_objects") == "hubspot__search_crm_objects"


def test_namespaced_tool_name_disambiguates_by_connection_when_given():
    """Two connections for the same engine in one turn (the vault explicitly
    supports this) must not produce identical tool names, or the second
    connection's tools silently collide with the first's under
    ToolRegistry's skip-duplicate-by-name behavior."""
    assert (
        namespaced_tool_name("hubspot", "search_crm_objects", connection_name="acme")
        == "hubspot__acme__search_crm_objects"
    )
    assert (
        namespaced_tool_name("hubspot", "search_crm_objects", connection_name="acme")
        != namespaced_tool_name("hubspot", "search_crm_objects", connection_name="other-org")
    )


def test_parse_namespaced_tool_name_splits_engine_and_tool():
    assert parse_namespaced_tool_name(name="hubspot__search_crm_objects") == ToolNameParts(
        engine="hubspot", connection_name=None, tool_name="search_crm_objects"
    )


def test_parse_namespaced_tool_name_reads_the_connection_between_engine_and_tool():
    """A connection name may itself contain "__", so the engine is the text
    before the first "__", the tool is the text after the last, and
    everything between is the connection."""
    assert parse_namespaced_tool_name(name="hubspot__acme__search_crm_objects") == ToolNameParts(
        engine="hubspot", connection_name="acme", tool_name="search_crm_objects"
    )
    assert parse_namespaced_tool_name(name="hubspot__acme__eu__search_crm_objects") == ToolNameParts(
        engine="hubspot", connection_name="acme__eu", tool_name="search_crm_objects"
    )


def test_parse_namespaced_tool_name_round_trips_namespaced_tool_name():
    """A connection's name is free text, so it may start or end with "_" or
    hold "__" anywhere. Every classified HubSpot tool still parses back to
    the engine, connection and tool that built its name."""
    for connection_name in (None, "acme", "acme__eu", "crm_", "a____b", "crm__", "__crm"):
        for tool_name in TOOL_ACCESS["hubspot"]:
            name = namespaced_tool_name("hubspot", tool_name, connection_name=connection_name)
            assert parse_namespaced_tool_name(name=name) == ToolNameParts(
                engine="hubspot", connection_name=connection_name, tool_name=tool_name
            ), name


def test_parse_namespaced_tool_name_rejects_a_name_that_is_not_namespaced():
    for name in (
        "web_fetch",
        "read_text_file",
        "__search_crm_objects",
        "hubspot__",
        "hubspot____get_crm_objects",
        "",
    ):
        assert parse_namespaced_tool_name(name=name) is None, name


async def test_discover_tool_defs_namespaces_and_filters_by_access_mode(stub_mcp_server):
    """`add`/`boom` aren't in HubSpot's table, so under the fail-closed
    default they're write-tier — read mode should see neither."""
    async with McpSession(server=stub_mcp_server) as session:
        read_defs = await discover_tool_defs(session, engine="hubspot", access_mode="read")
        write_defs = await discover_tool_defs(session, engine="hubspot", access_mode="write")

    assert read_defs == []
    assert {d.name for d in write_defs} == {"hubspot__add", "hubspot__boom"}


async def test_discover_tool_defs_none_mode_skips_listing_entirely(stub_mcp_server, monkeypatch):
    async with McpSession(server=stub_mcp_server) as session:
        called = {"list_tools": False}

        async def fail_if_called():
            called["list_tools"] = True
            raise AssertionError("list_tools should not be called for access_mode='none'")

        monkeypatch.setattr(session, "list_tools", fail_if_called)
        defs = await discover_tool_defs(session, engine="hubspot", access_mode="none")

    assert defs == []
    assert called["list_tools"] is False


async def test_registered_tool_def_input_schema_matches_the_discovered_tool(stub_mcp_server):
    async with McpSession(server=stub_mcp_server) as session:
        defs = await discover_tool_defs(session, engine="hubspot", access_mode="write")

    add_def = next(d for d in defs if d.name == "hubspot__add")
    assert add_def.input_schema["properties"].keys() == {"a", "b"}


async def test_handler_calls_back_into_the_open_session_and_returns_tool_outcome(stub_mcp_server):
    async with McpSession(server=stub_mcp_server) as session:
        defs = await discover_tool_defs(session, engine="hubspot", access_mode="write")
        add_def = next(d for d in defs if d.name == "hubspot__add")
        boom_def = next(d for d in defs if d.name == "hubspot__boom")

        outcome = await add_def.handler(None, {"a": 3, "b": 4})
        assert isinstance(outcome, ToolOutcome)
        assert outcome.content == "7"
        assert outcome.ok is True

        error_outcome = await boom_def.handler(None, {})
        assert isinstance(error_outcome, ToolOutcome)
        assert error_outcome.ok is False
        assert error_outcome.reason == "mcp_tool_error"


async def test_handler_reports_permanent_failure_as_reconnect_message(stub_mcp_server, monkeypatch):
    async with McpSession(server=stub_mcp_server) as session:
        defs = await discover_tool_defs(session, engine="hubspot", access_mode="write")
        add_def = next(d for d in defs if d.name == "hubspot__add")

        from anton.core.mcp.errors import McpPermanentError

        async def rejected(name, arguments):
            raise McpPermanentError("token rejected")

        monkeypatch.setattr(session, "call_tool", rejected)
        outcome = await add_def.handler(None, {"a": 1, "b": 1})

    assert outcome.ok is False
    assert outcome.reason == "mcp_permanent_error"
    assert "reconnect" in outcome.content.lower() or "hubspot" in outcome.content.lower()


async def test_discover_tool_defs_threads_connection_name_into_namespacing(stub_mcp_server):
    async with McpSession(server=stub_mcp_server) as session:
        defs = await discover_tool_defs(
            session, engine="hubspot", access_mode="write", connection_name="acme"
        )
    assert {d.name for d in defs} == {"hubspot__acme__add", "hubspot__acme__boom"}
