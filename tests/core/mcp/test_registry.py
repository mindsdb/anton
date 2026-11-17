from __future__ import annotations

import re

import pytest
from mcp.server.mcpserver import MCPServer

from anton.core.mcp.access import TOOL_ACCESS
from anton.core.mcp.client import McpSession
from anton.core.mcp.errors import McpEgressBlocked, McpPermanentError
from anton.core.mcp.names import (
    ToolNameParts,
    namespaced_tool_name,
    parse_namespaced_tool_name,
    registrable_tool_name,
)
from anton.core.mcp.registry import discover_tool_defs, without_empty_optionals
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


async def test_handler_leaves_out_empty_optional_arguments_of_a_read_tool(monkeypatch):
    """Some models send cursor="" instead of omitting it; Notion rejects that."""
    monkeypatch.setitem(TOOL_ACCESS, "hubspot", {"list_pages": "read"})
    server = MCPServer(name="list-stub")

    @server.tool()
    def list_pages(query: str, cursor: str | None = None) -> str:
        return f"query={query!r} cursor={cursor!r}"

    async with McpSession(server=server) as session:
        defs = await discover_tool_defs(session, engine="hubspot", access_mode="write")
        handler = next(d for d in defs if d.name == "hubspot__list_pages").handler

        outcome = await handler(None, {"query": "", "cursor": ""})
        assert outcome.content == "query='' cursor=None"

        outcome = await handler(None, {"query": "x", "cursor": "abc"})
        assert outcome.content == "query='x' cursor='abc'"


async def test_handler_passes_empty_arguments_of_a_write_tool_through(monkeypatch):
    """On a write tool "" can mean "clear this field", so it must reach the server."""
    monkeypatch.setitem(TOOL_ACCESS, "hubspot", {})
    server = MCPServer(name="update-stub")

    @server.tool()
    def update_contact(contact_id: str, phone: str | None = None) -> str:
        return f"contact_id={contact_id!r} phone={phone!r}"

    async with McpSession(server=server) as session:
        defs = await discover_tool_defs(session, engine="hubspot", access_mode="write")
        handler = next(d for d in defs if d.name == "hubspot__update_contact").handler

        outcome = await handler(None, {"contact_id": "42", "phone": ""})
        assert outcome.content == "contact_id='42' phone=''"


def test_empty_optionals_are_dropped_inside_nested_objects():
    """HubSpot's discover_hubspot_schema takes its cursor inside `query`."""
    schema = {
        "type": "object",
        "required": ["query"],
        "properties": {"query": {
            "type": "object",
            "required": ["_queryType"],
            "properties": {"_queryType": {"type": "string"}, "cursor": {"type": "string"}, "limit": {"type": "integer"}},
        }},
    }
    args = {"query": {"_queryType": "GET_OBJECT_TYPES", "cursor": "", "limit": 100, "typeNameFilter": []}}
    assert without_empty_optionals(args, schema, schema) == {
        "query": {"_queryType": "GET_OBJECT_TYPES", "limit": 100, "typeNameFilter": []},
    }


def test_a_required_field_is_kept_even_when_empty_at_any_depth():
    schema = {
        "type": "object",
        "$defs": {"Filter": {"type": "object", "required": ["value"], "properties": {"value": {"type": "string"}}}},
        "properties": {"filters": {"type": "array", "items": {"$ref": "#/$defs/Filter"}}},
        "required": [],
    }
    args = {"filters": [{"value": "", "note": None}], "cursor": None}
    assert without_empty_optionals(args, schema, schema) == {"filters": [{"value": ""}]}


def test_a_field_required_by_any_branch_is_kept():
    schema = {"anyOf": [
        {"type": "object", "required": ["q"], "properties": {"q": {"type": "string"}}},
        {"type": "object", "properties": {"id": {"type": "string"}}},
    ]}
    assert without_empty_optionals({"q": "", "id": ""}, schema, schema) == {"q": ""}


def test_a_name_that_fits_keeps_its_connection_name():
    assert namespaced_tool_name("hubspot", "add", connection_name="other-org") == "hubspot__other-org__add"
    assert namespaced_tool_name("hubspot", "add", connection_name="my_org") == "hubspot__my_org__add"


def test_a_51_character_connection_name_gives_short_unique_names_that_parse_back():
    notion_tools = ["notion-search", "notion-fetch", "notion-get-comments", "notion-query-data-sources"]
    connections = ["jo-example-com-" + f"1b2c3d4e-0000-4000-8000-00000000000{n}" for n in (1, 2)]
    names = [
        registrable_tool_name("notion", tool, connection_name=connection)
        for connection in connections
        for tool in notion_tools
    ]
    assert all(name is not None and len(name) <= 64 for name in names)
    assert len(set(names)) == len(names)
    for name, tool in zip(names, notion_tools * 2):
        parts = parse_namespaced_tool_name(name=name)
        assert parts is not None and (parts.engine, parts.tool_name) == ("notion", tool)


def test_a_connection_name_the_model_api_refuses_is_shortened_when_registering():
    name = registrable_tool_name("custom_mcp", "search", connection_name="Acme Tracker")
    assert name is not None and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name)
    assert name == registrable_tool_name("custom_mcp", "search", connection_name="Acme Tracker")


def test_namespaced_tool_name_shortens_on_length_only():
    """cowork-server builds labels from these names; a short name keeps its
    connection text, and the label code redacts it there."""
    assert namespaced_tool_name("hubspot", "search", connection_name="token=abc") == "hubspot__token=abc__search"


@pytest.mark.parametrize("tool_name", ["files.read", "github/search", "get user", "tool:list", "über_tool", "a__b", "_private", ""])
def test_a_tool_name_without_a_valid_registered_name_is_refused(tool_name):
    assert registrable_tool_name("custom_mcp", tool_name, connection_name="my-team-search-server") is None


def test_a_tool_name_still_too_long_after_shortening_is_refused():
    assert registrable_tool_name("custom_mcp", "x" * 60, connection_name="my-team-search-server") is None


async def test_unusable_tool_names_are_left_out_and_their_siblings_register():
    server = MCPServer(name="odd-names")
    for odd in ["files.read", "github/search", "get user", "tool:list", "über_tool", "a__b", "_private"]:
        server.tool(name=odd)(lambda: "x")
    server.tool(name="search_issues")(lambda: "ok")

    async with McpSession(server=server) as session:
        defs = await discover_tool_defs(
            session,
            engine="custom_mcp",
            access_mode="read",
            connection_name="my-team-search-server",
            allowed_tools=frozenset({tool.name for tool in await session.list_tools()}),
        )
    assert [d.name for d in defs] == ["custom_mcp__my-team-search-server__search_issues"]
    assert all(parse_namespaced_tool_name(name=d.name) is not None for d in defs)


async def test_allowed_tools_replace_the_access_table(stub_mcp_server):
    async with McpSession(server=stub_mcp_server) as session:
        defs = await discover_tool_defs(
            session, engine="custom_mcp", access_mode="read", allowed_tools=frozenset({"add"})
        )
    assert [d.name for d in defs] == ["custom_mcp__add"]


async def test_an_empty_allowed_list_registers_nothing(stub_mcp_server):
    async with McpSession(server=stub_mcp_server) as session:
        defs = await discover_tool_defs(session, engine="custom_mcp", access_mode="write", allowed_tools=frozenset())
    assert defs == []


async def test_allowed_tools_still_register_nothing_when_the_connection_is_off(stub_mcp_server):
    async with McpSession(server=stub_mcp_server) as session:
        defs = await discover_tool_defs(
            session, engine="custom_mcp", access_mode="none", allowed_tools=frozenset({"add"})
        )
    assert defs == []


async def test_an_allowed_custom_tool_gets_its_arguments_unchanged(monkeypatch):
    """Holds even when the access table marks a tool of that name read."""
    monkeypatch.setitem(TOOL_ACCESS, "custom_mcp", {"list_pages": "read"})
    server = MCPServer(name="custom-stub")

    @server.tool()
    def list_pages(query: str, cursor: str | None = None) -> str:
        return f"query={query!r} cursor={cursor!r}"

    async with McpSession(server=server) as session:
        defs = await discover_tool_defs(
            session, engine="custom_mcp", access_mode="read", allowed_tools=frozenset({"list_pages"})
        )
        outcome = await defs[0].handler(None, {"query": "", "cursor": ""})
    assert outcome.content == "query='' cursor=''"


async def test_a_refused_address_has_its_own_message_naming_the_server(stub_mcp_server, monkeypatch):
    async with McpSession(server=stub_mcp_server) as session:
        defs = await discover_tool_defs(
            session,
            engine="custom_mcp",
            access_mode="read",
            allowed_tools=frozenset({"add"}),
            display_name="Acme Tracker",
        )

        async def refused(name, arguments):
            raise McpEgressBlocked("not public")

        monkeypatch.setattr(session, "call_tool", refused)
        outcome = await defs[0].handler(None, {"a": 1, "b": 1})
    assert outcome.ok is False
    assert outcome.reason == "mcp_egress_blocked"
    assert "Acme Tracker" in outcome.content
    assert "reconnect" not in outcome.content


async def test_a_rejected_token_message_names_the_server_by_its_label(stub_mcp_server, monkeypatch):
    async with McpSession(server=stub_mcp_server) as session:
        defs = await discover_tool_defs(
            session,
            engine="custom_mcp",
            access_mode="read",
            allowed_tools=frozenset({"add"}),
            display_name="Acme Tracker",
        )

        async def rejected(name, arguments):
            raise McpPermanentError("token rejected")

        monkeypatch.setattr(session, "call_tool", rejected)
        outcome = await defs[0].handler(None, {"a": 1, "b": 1})
    assert outcome.reason == "mcp_permanent_error"
    assert "The Acme Tracker connection needs to be reconnected" in outcome.content
    assert "custom_mcp" not in outcome.content
