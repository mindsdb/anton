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

Databricks Genie gets its own handlers (`_GENIE_CALLS`): `genie_ask` only
starts an answer, so the ask and poll handlers wait for it with
`databricks.wait_for_genie_answer` and return the short structured fields,
never Genie's long reasoning text; query results are cut to
`GENIE_RESULT_ROW_CAP` rows. Every other tool returns the server's content as is.

Async handlers need no bridging: every existing `ToolDef.handler` in this
codebase already is `async def`, and `ToolRegistry.dispatch_tool_stream`
awaits the result directly — a new MCP handler fits the same contract with
nothing new to design.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from anton.core.mcp.access import AccessMode, allowed_for_mode
from anton.core.mcp.client import McpSession
from anton.core.mcp.databricks import GENIE_IN_PROGRESS, GenieAnswerFailed, wait_for_genie_answer
from anton.core.mcp.errors import McpPermanentError, McpTransientError
from anton.core.mcp.names import namespaced_tool_name
from anton.core.tools.registry import ToolOutcome
from anton.core.tools.tool_defs import ToolDef
from anton.utils.datasources import hold_secret_value


async def discover_tool_defs(
    mcp_session: McpSession,
    *,
    engine: str,
    access_mode: AccessMode,
    connection_name: str | None = None,
    held_secrets: tuple[tuple[str, str], ...] = (),
) -> list[ToolDef]:
    """List `mcp_session`'s tools, apply the access filter, and wrap each
    surviving one as a namespaced `ToolDef` whose handler calls back into
    this same open session.

    `access_mode="none"` returns an empty list without even listing tools
    remotely — nothing to register, no reason to pay the round-trip.

    `connection_name` disambiguates the registered tool names when this
    engine has more than one connection in the same turn — see
    `namespaced_tool_name`.

    `held_secrets` are `(label, value)` pairs each handler holds for the
    running turn's `scrub_credentials` before it calls the server.
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
                poll_tool_name=namespaced_tool_name(engine, "genie_poll_response", connection_name=connection_name),
                held_secrets=held_secrets,
            ),
        )
        for tool in tools
        if allowed_for_mode(engine, tool.name, access_mode)
    ]


_MAX_SCHEMA_DEPTH = 20

#: Rows a Genie query result may hand the model; the largest seen live was 12.
GENIE_RESULT_ROW_CAP = 200
# Genie's raw reply text is long reasoning addressed to the agent, so a reply without
# structured fields becomes one of these fixed messages, never that text.
_GENIE_ANSWER_FAILED = "Genie did not report the answer's status."
_GENIE_RESULT_FAILED = "Genie did not return the query result."
_GENIE_ANSWER_FIELDS = ("status", "conversation_id", "response_id", "final_answer", "query_items", "deep_link")
_GENIE_RESULT_FIELDS = (
    "conversation_id",
    "response_id",
    "item_id",
    "columns",
    "rows",
    "total_row_count",
    "truncated",
    "statement_state",
    "ready",
)


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


def _genie_answer_outcome(answer: dict[str, Any], poll_tool_name: str) -> ToolOutcome:
    """Shape a Genie poll's structured fields into the model's short JSON reply.

    A finished answer keeps only `_GENIE_ANSWER_FIELDS`, with each query item
    reduced to its `item_id`. One still in progress keeps only its ids and a
    note naming `poll_tool_name`. No side effects.
    """
    if answer.get("status") == GENIE_IN_PROGRESS:
        reply = {
            "status": GENIE_IN_PROGRESS,
            "conversation_id": answer.get("conversation_id"),
            "response_id": answer.get("response_id"),
            "note": f"Genie is still working; call {poll_tool_name} with these ids.",
        }
    else:
        reply = {key: answer[key] for key in _GENIE_ANSWER_FIELDS if key in answer}
        if isinstance(reply.get("query_items"), list):
            reply["query_items"] = [
                {"item_id": item.get("item_id")} for item in reply["query_items"] if isinstance(item, dict)
            ]
    return ToolOutcome(content=json.dumps(reply), ok=True, reason="")


async def _call_genie_answer(
    mcp_session: McpSession, bare_tool_name: str, arguments: dict, poll_tool_name: str
) -> ToolOutcome:
    """Run `genie_ask` or `genie_poll_response` and wait while Genie is still working.

    Returns the short JSON reply from `_genie_answer_outcome`. An error, a
    reply with no structured status, an in-progress reply without both ids,
    or a failed wait is a tool error with a fixed message, never the
    server's text. MCP errors propagate.
    """
    structured, _, is_error = await mcp_session.call_tool_structured(bare_tool_name, arguments)
    failed = ToolOutcome(content=_GENIE_ANSWER_FAILED, ok=False, reason="mcp_tool_error")
    if is_error or not isinstance(structured, dict) or "status" not in structured:
        return failed
    if structured["status"] == GENIE_IN_PROGRESS:
        conversation_id, response_id = structured.get("conversation_id"), structured.get("response_id")
        if not conversation_id or not response_id:
            return failed
        try:
            structured = await wait_for_genie_answer(mcp_session, conversation_id, response_id)
        except GenieAnswerFailed:
            return failed
    return _genie_answer_outcome(structured, poll_tool_name)


async def _call_genie_query_result(
    mcp_session: McpSession, bare_tool_name: str, arguments: dict, poll_tool_name: str
) -> ToolOutcome:
    """Run `genie_get_query_result` and keep at most `GENIE_RESULT_ROW_CAP` rows.

    Returns the result's structured fields as JSON, with a note when rows
    were cut. An error or a reply with no structured result is a tool error
    with a fixed message, never the server's text. `poll_tool_name` is
    unused; it keeps the Genie call signature. MCP errors propagate.
    """
    structured, _, is_error = await mcp_session.call_tool_structured(bare_tool_name, arguments)
    if is_error or not isinstance(structured, dict):
        return ToolOutcome(content=_GENIE_RESULT_FAILED, ok=False, reason="mcp_tool_error")
    result = {key: structured[key] for key in _GENIE_RESULT_FIELDS if key in structured}
    rows = result.get("rows")
    if isinstance(rows, list) and len(rows) > GENIE_RESULT_ROW_CAP:
        total = result.get("total_row_count", len(rows))
        result["rows"] = rows[:GENIE_RESULT_ROW_CAP]
        result["note"] = f"Showing the first {GENIE_RESULT_ROW_CAP} of {total} rows."
    return ToolOutcome(content=json.dumps(result), ok=True, reason="")


_GenieCall = Callable[[McpSession, str, dict, str], Awaitable[ToolOutcome]]
_GENIE_CALLS: dict[str, _GenieCall] = {
    "genie_ask": _call_genie_answer,
    "genie_poll_response": _call_genie_answer,
    "genie_get_query_result": _call_genie_query_result,
}


def _make_handler(
    mcp_session: McpSession,
    *,
    engine: str,
    bare_tool_name: str,
    input_schema: dict,
    drop_empty_optionals: bool,
    poll_tool_name: str,
    held_secrets: tuple[tuple[str, str], ...],
):
    """Build the `ToolDef` handler that calls `bare_tool_name` on `mcp_session`.

    A Databricks Genie tool in `_GENIE_CALLS` goes through its own call;
    every other tool returns the server's content. Either way a permanent MCP
    error becomes the reconnect message and a transient one the retry
    message. `poll_tool_name` is the registered name the Genie note points at.
    Each call first holds `held_secrets` for the running turn's scrub.
    """
    genie_call = _GENIE_CALLS.get(bare_tool_name) if engine == "databricks" else None

    async def handle_mcp_call(session, tc_input: dict) -> ToolOutcome:  # noqa: ARG001 - session unused, contract requires it
        # Held here, inside the turn: discovery runs before the turn opens its scrub scope.
        for label, value in held_secrets:
            hold_secret_value(label, value)
        arguments = tc_input or {}
        if drop_empty_optionals:
            arguments = without_empty_optionals(arguments, input_schema, input_schema)
        try:
            if genie_call is not None:
                return await genie_call(mcp_session, bare_tool_name, arguments, poll_tool_name)
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
