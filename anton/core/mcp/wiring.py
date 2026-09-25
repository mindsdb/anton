"""Entry points other modules call to actually use this package.

**Discovery happens before the `ChatSession` exists, not after** — this is
load-bearing, not a style choice. Two things forced it during implementation:

1. `ToolDef` needs its `input_schema` populated before `_build_tools()` hands
   the list to the model, so the MCP `tools/list` call has to happen at the
   start of tool-schema assembly. "One session per turn" therefore means:
   open + discover once, near the top of the turn, before the first
   `_build_tools()` call — not deferred until the model decides to call an
   MCP tool.
2. `ChatSession._build_tools()` populates the registry exactly once per
   session (`if not self.tool_registry: ...`) — registering tools directly
   onto an already-constructed session's `tool_registry` *before* that first
   call would make the registry non-empty and make it skip building the core
   tools (scratchpad, ask_user, ...) entirely. And on `cowork-server`'s cloud
   path specifically, `_build_tools()` also re-enforces a fixed
   `tool_allowlist` on *every* call, not just the first — a name registered
   after the fact and left out of that allowlist would survive exactly one
   round then get silently unregistered on the turn's next tool-list build.

Both problems disappear if discovery happens first and its `ToolDef`s are
folded into `ChatSessionConfig.tools` (the same mechanism every
host-registered tool already uses) — and, on the cloud path, into that
turn's `tool_allowlist` too — before `ChatSession(config)` is even called.
See `anton/cloud_turn/session.py: build_cloud_chat_session` for the cloud
caller; the desktop harness (`cowork-server`, Stage 2) is the other.

Three entry points:
  - `discover_mcp_tools_async` / `discover_mcp_tools` (sync wrapper) — the
    full path: for every `method == "mcp"` connection, open a session,
    discover + filter + namespace its tools. Returns the `ToolDef`s to fold
    into `ChatSessionConfig.tools` plus the open `McpSession`s a caller must
    keep alive for the turn and close at its end (there is no `ChatSession`
    yet at this point to hang them on).
  - `call_mcp_tool` — a lighter helper with no ToolDef/registry/filter
    involvement, for a caller that just needs one result (e.g.
    cowork-server's identity-bridge endpoint calling `get_user_details`/
    `get_organization_details` for a connector Electron can't speak MCP to
    directly).
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from typing import TYPE_CHECKING, Any

from anton.core.mcp.access import AccessMode
from anton.core.mcp.client import McpSession
from anton.core.mcp.registry import discover_tool_defs
from anton.core.mcp.servers import mcp_server_url

if TYPE_CHECKING:
    from anton.core.datasources.data_vault import DataVault
    from anton.core.tools.tool_defs import ToolDef

logger = logging.getLogger(__name__)


def _method_of(vault: "DataVault", engine: str, name: str) -> str | None:
    """The `_method` bookkeeping field a connection's record carries — the
    one thing that distinguishes an MCP connection from any other, without a
    vault schema change (same `_`-prefixed convention as `_label`/
    `_connector_id`/`_picked_files`, confirmed excluded from injected env
    vars)."""
    record = vault.read_record(engine, name)
    if record is None:
        return None
    return (record.get("fields") or {}).get("_method")


def _access_mode_of(vault: "DataVault", engine: str, name: str) -> AccessMode:
    """`_access_mode` defaults to "read" — every new MCP connection is
    hardcoded to that at connect time (no connect-time UI feeds it; see
    ENG-1816's "Decisions made in review"). An unrecognized stored value
    also falls back to "read" rather than failing the whole turn over it."""
    record = vault.read_record(engine, name) or {}
    mode = (record.get("fields") or {}).get("_access_mode", "read")
    return mode if mode in ("read", "write", "none") else "read"


async def discover_mcp_tools_async(
    vault: "DataVault", connections: list[dict[str, str]]
) -> tuple[list["ToolDef"], list[McpSession]]:
    """Discover native tools for every `method == "mcp"` connection in
    `connections`. Returns `(tool_defs, open_sessions)` — fold `tool_defs`
    into `ChatSessionConfig.tools` (and, on the cloud path, into
    `tool_allowlist`) and keep `open_sessions` alive for the turn, closing
    each at turn end.

    A no-op — returns `([], [])` — for a vault/turn with no MCP connections,
    so this is always safe to call unconditionally. A single connection
    failing to connect or list tools is logged and skipped; it never fails
    the whole turn over one bad connector.
    """
    # Vault reads (_method_of/_access_mode_of, below) block on a synchronous
    # HTTP call for the cloud vault (TurnKeyDataVault._fetch) — a pre-existing
    # limitation shared by every vault caller today (e.g.
    # restore_namespaced_env), not something introduced or fixable here.
    # This first pass over `connections` is cheap and sequential regardless;
    # only the per-connection MCP round-trips below are worth parallelizing.
    mcp_connections: list[tuple[str, str]] = []
    for connection in connections:
        engine, name = connection.get("engine"), connection.get("name")
        if not engine or not name:
            continue
        if _method_of(vault, engine, name) == "mcp":
            mcp_connections.append((engine, name))
    if not mcp_connections:
        return [], []

    # A turn with two connections for the same engine (the vault explicitly
    # supports this — see registry.py's module docstring) needs its tool
    # names disambiguated by connection, or the second connection's tools
    # silently collide with the first's under ToolRegistry's
    # skip-duplicate-by-name behavior. Left out for the common single-
    # connection case so its tool names stay exactly what they've always been.
    engine_counts = Counter(engine for engine, _ in mcp_connections)

    results = await asyncio.gather(
        *(
            _discover_one(vault, engine, name, disambiguate=engine_counts[engine] > 1)
            for engine, name in mcp_connections
        )
    )

    tool_defs: list["ToolDef"] = []
    sessions: list[McpSession] = []
    for defs, session in results:
        tool_defs.extend(defs)
        if session is not None:
            sessions.append(session)
    return tool_defs, sessions


async def _discover_one(
    vault: "DataVault", engine: str, name: str, *, disambiguate: bool
) -> tuple[list["ToolDef"], McpSession | None]:
    """One connection's open-session-then-discover step, isolated so
    `discover_mcp_tools_async` can run every connection concurrently via
    `asyncio.gather` — a bad connector never fails the whole turn over it,
    it just contributes `([], None)`."""
    url = mcp_server_url(engine)
    if url is None:
        logger.warning("discover_mcp_tools: no known MCP server URL for engine %s", engine)
        return [], None
    access_token = (vault.load(engine, name) or {}).get("access_token")
    if not access_token:
        logger.warning("discover_mcp_tools: no access token for %s/%s", engine, name)
        return [], None

    mcp_session = McpSession(url=url, access_token=access_token)
    try:
        await mcp_session.__aenter__()
    except Exception:
        logger.exception("discover_mcp_tools: failed to open MCP session for %s/%s", engine, name)
        return [], None
    try:
        access_mode = _access_mode_of(vault, engine, name)
        tool_defs = await discover_tool_defs(
            mcp_session,
            engine=engine,
            access_mode=access_mode,
            connection_name=name if disambiguate else None,
        )
    except Exception:
        logger.exception("discover_mcp_tools: tool discovery failed for %s/%s", engine, name)
        await mcp_session.__aexit__(None, None, None)
        return [], None
    return tool_defs, mcp_session


def discover_mcp_tools(
    vault: "DataVault", connections: list[dict[str, str]]
) -> tuple[list["ToolDef"], list[McpSession]]:
    """Sync wrapper for a caller with no running event loop of its own —
    `build_cloud_chat_session`, which already runs inside
    `run_in_executor` (see `cloud_turn/__main__.py`). Never call this from
    inside a running event loop; call `discover_mcp_tools_async` directly
    there instead (cowork-server's desktop harness, already `async def`)."""
    return asyncio.run(discover_mcp_tools_async(vault, connections))


async def close_mcp_sessions(sessions: list[McpSession]) -> None:
    """Close every session `discover_mcp_tools[_async]` opened. Best-effort:
    one session failing to close cleanly must not stop the others, or mask
    whatever the turn was already tearing down for."""
    for session in sessions:
        try:
            await session.__aexit__(None, None, None)
        except Exception:
            logger.exception("close_mcp_sessions: error closing an MCP session")


async def call_mcp_tool(engine: str, access_token: str, tool_name: str, **kwargs: Any) -> str | list[dict]:
    """Open a session, call one tool, close it. No ToolDef/registry/filter
    involvement — for a caller that just needs a single result (e.g.
    cowork-server's identity-bridge endpoint calling `get_user_details`/
    `get_organization_details`, since Electron can't speak MCP itself)."""
    url = mcp_server_url(engine)
    if url is None:
        raise ValueError(f"no known MCP server URL for engine {engine!r}")
    async with McpSession(url=url, access_token=access_token) as mcp_session:
        content, is_error = await mcp_session.call_tool(tool_name, kwargs)
        if is_error:
            raise RuntimeError(f"{engine} MCP tool {tool_name!r} returned an error: {content!r}")
        return content
