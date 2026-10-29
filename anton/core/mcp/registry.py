"""Converts a discovered MCP tool list into native `ToolDef`s.

Two things happen here, in order, before the LLM ever sees a tool name:

1. **Namespacing** — every tool registers as ``{engine}__{tool_name}``, never
   its bare name. Cheap to do from the start; avoids a rename later if a
   core tool or a second MCP connector (Linear, PostHog) ever collides on a
   bare name — a rename after the fact could affect any conversation whose
   history already references the old one. When a turn has more than one
   connection for the *same* engine — the vault explicitly supports this
   (`data_vault.py`: a connection's `name` "disambiguates when an org has
   more than one connection for this engine") — the caller (`wiring.py`)
   threads that connection's own `name` in too, so the two connections'
   tools don't collide under `ToolRegistry.register_tool`'s
   skip-duplicate-by-name behavior and silently shadow one another.
   `names.parse_namespaced_tool_name` splits a registered name back into
   engine, connection and tool for callers that only see the name (for
   example a stored tool row). Both name helpers live in `names`, so such a
   caller does not import the MCP SDK.
2. **Access filtering** — `access.allowed_for_mode` decides which tools
   survive at all for this connection's mode. This *is* the governance
   mechanism for MCP tool access (see access.py's module docstring); nothing
   downstream re-checks it.

Async handlers need no bridging: every existing `ToolDef.handler` in this
codebase already is `async def`, and `ToolRegistry.dispatch_tool_stream`
awaits the result directly — a new MCP handler fits the same contract with
nothing new to design.
"""

from __future__ import annotations

from typing import Any

from anton.core.mcp.access import AccessMode, allowed_for_mode
from anton.core.mcp.client import McpSession
from anton.core.mcp.errors import McpPermanentError, McpTransientError
from anton.core.mcp.names import namespaced_tool_name
from anton.core.tools.registry import ToolOutcome
from anton.core.tools.tool_defs import ToolDef


async def discover_tool_defs(
    mcp_session: McpSession,
    *,
    engine: str,
    access_mode: AccessMode,
    connection_name: str | None = None,
) -> list[ToolDef]:
    """List `mcp_session`'s tools, apply the access filter, and wrap each
    surviving one as a namespaced `ToolDef` whose handler calls back into
    this same open session.

    `access_mode="none"` returns an empty list without even listing tools
    remotely — nothing to register, no reason to pay the round-trip.

    `connection_name` disambiguates the registered tool names when this
    engine has more than one connection in the same turn — see
    `namespaced_tool_name`.
    """
    if access_mode == "none":
        return []
    tools = await mcp_session.list_tools()
    return [
        ToolDef(
            name=namespaced_tool_name(engine, tool.name, connection_name=connection_name),
            description=tool.description or "",
            input_schema=tool.input_schema,
            handler=_make_handler(
                mcp_session,
                engine=engine,
                bare_tool_name=tool.name,
                input_schema=tool.input_schema or {},
                drop_empty_optionals=allowed_for_mode(engine, tool.name, "read"),
            ),
        )
        for tool in tools
        if allowed_for_mode(engine, tool.name, access_mode)
    ]


_MAX_SCHEMA_DEPTH = 20


def _schema_branches(schema: Any, root: dict, depth: int) -> list[dict]:
    """`schema` plus every anyOf/oneOf/allOf branch, with local `$ref`s
    resolved: everywhere a field's `required`/`properties` can come from."""
    if not isinstance(schema, dict) or depth > _MAX_SCHEMA_DEPTH:
        return []
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/"):
        node: Any = root
        for part in ref[2:].split("/"):
            node = node.get(part) if isinstance(node, dict) else None
        return _schema_branches(node, root, depth + 1)
    branches = [schema]
    for key in ("anyOf", "oneOf", "allOf"):
        for branch in schema.get(key) or ():
            branches.extend(_schema_branches(branch, root, depth + 1))
    return branches


def without_empty_optionals(value: Any, schema: Any, root: dict, depth: int = 0) -> Any:
    """`value` with every "" or null field left out, at any depth, unless
    some branch of its schema marks that field required.

    Some models fill optional fields with "" or null instead of omitting them,
    and servers validate those as real values: Notion and HubSpot both reject
    cursor="" as an invalid cursor, HubSpot's one level down inside `query`.
    Applied to read tools only: on a write tool "" or null can be a deliberate
    "clear this field", which must reach the server.
    """
    if depth > _MAX_SCHEMA_DEPTH:
        return value
    branches = _schema_branches(schema, root, depth)
    if isinstance(value, dict):
        required = {name for branch in branches for name in branch.get("required") or ()}
        cleaned = {}
        for key, item in value.items():
            if key not in required and (item is None or item == ""):
                continue
            item_schemas = [
                branch["properties"][key]
                for branch in branches
                if isinstance(branch.get("properties"), dict) and key in branch["properties"]
            ]
            cleaned[key] = without_empty_optionals(item, {"anyOf": item_schemas}, root, depth + 1)
        return cleaned
    if isinstance(value, list):
        item_schemas = [branch["items"] for branch in branches if isinstance(branch.get("items"), dict)]
        return [without_empty_optionals(item, {"anyOf": item_schemas}, root, depth + 1) for item in value]
    return value


def _make_handler(
    mcp_session: McpSession, *, engine: str, bare_tool_name: str, input_schema: dict, drop_empty_optionals: bool
):
    async def handle_mcp_call(session, tc_input: dict) -> ToolOutcome:  # noqa: ARG001 - session unused, contract requires it
        arguments = tc_input or {}
        if drop_empty_optionals:
            arguments = without_empty_optionals(arguments, input_schema, input_schema)
        try:
            content, is_error = await mcp_session.call_tool(bare_tool_name, arguments)
        except McpPermanentError:
            return ToolOutcome(
                content=(
                    f"The {engine} connection needs to be reconnected before "
                    f"this tool can run again."
                ),
                ok=False,
                reason="mcp_permanent_error",
            )
        except McpTransientError:
            # McpSession.call_tool() already retried once internally before
            # raising this — a calm message, not an alarming one: this is a
            # rate limit or transport hiccup, never implying the user broke
            # something (see ENG-1816 "Decisions made in review").
            return ToolOutcome(
                content=f"{engine} is temporarily unavailable. Try again shortly.",
                ok=False,
                reason="mcp_transient_error",
            )
        return ToolOutcome(content=content, ok=not is_error, reason="mcp_tool_error" if is_error else "")

    return handle_mcp_call
