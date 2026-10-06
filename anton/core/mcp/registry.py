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

from anton.core.mcp.access import AccessMode, allowed_for_mode
from anton.core.mcp.client import McpSession
from anton.core.mcp.errors import McpPermanentError, McpTransientError
from anton.core.tools.registry import ToolOutcome
from anton.core.tools.tool_defs import ToolDef


def namespaced_tool_name(engine: str, tool_name: str, *, connection_name: str | None = None) -> str:
    """`{engine}__{tool_name}` in the common single-connection-per-engine
    case; `{engine}__{connection_name}__{tool_name}` when `connection_name`
    is given (multiple connections for this engine in the same turn — see
    module docstring). Left out by default so the common case's tool names
    stay exactly what every existing test/doc example already shows."""
    if connection_name is None:
        return f"{engine}__{tool_name}"
    return f"{engine}__{connection_name}__{tool_name}"


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
            handler=_make_handler(mcp_session, engine=engine, bare_tool_name=tool.name),
        )
        for tool in tools
        if allowed_for_mode(engine, tool.name, access_mode)
    ]


def _make_handler(mcp_session: McpSession, *, engine: str, bare_tool_name: str):
    async def handle_mcp_call(session, tc_input: dict) -> ToolOutcome:  # noqa: ARG001 - session unused, contract requires it
        try:
            content, is_error = await mcp_session.call_tool(bare_tool_name, tc_input)
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
