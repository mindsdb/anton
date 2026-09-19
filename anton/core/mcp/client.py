"""One open MCP session, for the life of a turn.

Session lifetime is resolved by scope, not by adding infrastructure: a
session only needs to live for one turn's tool-calling sequence, so it's
opened near the top of the turn (see ``wiring.py``) and closed at turn end
— no session broker, no Redis, on either desktop or cloud, since both
already rebuild their whole ``ChatSession`` fresh every turn regardless.

Built on the official ``modelcontextprotocol`` Python SDK (v2 — ``mcp>=2,<3``
in pyproject.toml; v2 renamed ``FastMCP`` to ``MCPServer`` and replaced the
old in-memory test helper). Pass either a remote server (``url`` +
``access_token``, mandatory PKCE/S256 OAuth per HubSpot's own
``.well-known/oauth-authorization-server`` metadata — no Dynamic Client
Registration needed, it's a fixed client_id/secret model like every other
connector) or an in-process ``MCPServer``/``Server`` instance directly (the
SDK's own test pattern — no sockets, no threads, no subprocess).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx2
import mcp
from mcp_types import Tool as McpTool

from anton.core.mcp.errors import McpPermanentError, classify_mcp_error

logger = logging.getLogger(__name__)

#: Recommended, not a hard gate (see ENG-1816 open questions): cloud turns
#: already pay a synchronous, blocking cost per connector (the turn-key
#: OAuth token fetch — anton/cloud_turn/__main__.py's own comment on that)
#: before this even lands. The initialize()+tools/list() round-trip stacks
#: directly on top. Log anything slower for a second look; re-measure again
#: once a real server (not a local reference/stub) is in the loop.
_SLOW_ROUND_TRIP_MS = 500.0


class McpSession:
    """An open session against one MCP server.

    Use as an async context manager::

        async with McpSession(url=url, access_token=token) as mcp_session:
            tools = await mcp_session.list_tools()
            content, is_error = await mcp_session.call_tool(name, args)

    Exactly one of `url` or `server` must be given. `server` is for tests —
    an in-process `MCPServer`/`Server` instance connects directly, with no
    network involved at all.
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        access_token: str | None = None,
        server: Any | None = None,
        read_timeout_seconds: float = 30.0,
    ) -> None:
        if (url is None) == (server is None):
            raise ValueError("McpSession needs exactly one of `url` or `server`")
        self._url = url
        self._access_token = access_token
        self._server = server
        self._read_timeout_seconds = read_timeout_seconds
        self._http_client: httpx2.AsyncClient | None = None
        self._client: mcp.Client | None = None

    async def __aenter__(self) -> "McpSession":
        await self._connect()
        return self

    async def _connect(self) -> None:
        if self._server is not None:
            target: Any = self._server
        else:
            headers = {"Authorization": f"Bearer {self._access_token}"} if self._access_token else {}
            self._http_client = httpx2.AsyncClient(headers=headers)
            from mcp.client.streamable_http import streamable_http_client

            target = streamable_http_client(self._url, http_client=self._http_client)

        client = mcp.Client(
            target, raise_exceptions=True, read_timeout_seconds=self._read_timeout_seconds
        )
        started = time.monotonic()
        try:
            await client.__aenter__()
        except Exception:
            if self._http_client is not None:
                await self._http_client.aclose()
            raise
        self._log_round_trip("initialize()", started)
        self._client = client

    async def _reconnect(self) -> None:
        """Close the dead client (best-effort) and open a fresh one against
        the same target/token. Required, not optional, for `call_tool`'s
        retry: once one task in a session's transport task group has failed,
        the whole group — and the `Client` built on it — is unusable, so
        simply calling `call_tool()` again on the same object cannot
        succeed."""
        if self._client is not None:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception:
                logger.exception("McpSession: error closing the dead client before reconnecting")
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:
                logger.exception("McpSession: error closing the dead http client before reconnecting")
        self._client = None
        self._http_client = None
        await self._connect()

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Tear down the session. A cleanup failure here must never replace
        or mask an exception already propagating out of the `async with`
        block — the underlying SDK's in-process transport wraps teardown
        errors in an `ExceptionGroup` when exiting mid-exception, which would
        otherwise turn a clean, catchable error (e.g. `call_mcp_tool`'s
        `RuntimeError` on a tool-level failure) into an opaque group a caller
        can't match on. Swallow (log-only) a cleanup error when one is
        already in flight; let it propagate normally otherwise."""
        try:
            if self._client is not None:
                await self._client.__aexit__(exc_type, exc, tb)
        except Exception:
            if exc_type is not None:
                logger.exception("McpSession: error closing session while another exception was propagating")
            else:
                raise
        finally:
            if self._http_client is not None:
                await self._http_client.aclose()
            self._client = None

    async def list_tools(self) -> list[McpTool]:
        """List every tool this server advertises, unfiltered."""
        client = self._require_client()
        started = time.monotonic()
        result = await client.list_tools()
        self._log_round_trip("tools/list()", started)
        return list(result.tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> tuple[str | list[dict], bool]:
        """Call one tool. Returns (content, is_error).

        `is_error=True` means the *tool itself* reported a failure (per the
        MCP spec, that's how a tool-originated error is meant to surface —
        inside a normal result, not as a protocol-level exception) — the
        caller decides what that means for its own retry/resilience
        bookkeeping (see registry.py's handler wrapper).

        A transport/protocol-level failure gets one retry against a FRESH
        session (same stored token) before a `McpPermanentError`/
        `McpTransientError` reaches the caller — mirrors the OAuth refresh
        path's own permanent-vs-transient split (see errors.py). A fresh
        session, not just a fresh call on the same one, is required: once
        one task in the transport's task group has failed, the whole group —
        and the `Client` built on it — is unusable. A permanent failure
        (rejected/expired token) is never retried.

        Verified by hand (ENG-1816): killing the server mid-session surfaces
        here as anything from a plain `httpx2.ConnectError` to an
        `asyncio.CancelledError`/`ExceptionGroup`, depending on which of the
        transport's own background tasks (the streamable-HTTP transport
        holds a long-lived background reader) noticed the failure first —
        both shapes were reproduced across otherwise-identical runs. The
        non-`Exception` shapes are still retried once the same way, but if
        the retry also fails, the ORIGINAL exception is re-raised as-is
        rather than being disguised as `McpTransientError` — real task
        cancellation (e.g. the whole turn being cancelled) must never be
        silently swallowed into a business-logic failure.

        That last guarantee only covers the case where retrying doesn't
        help. A `CancelledError` can also mean the *caller's own* task was
        actually cancelled (a real timeout or shutdown) rather than a
        `CancelledError` merely leaking up from a sibling task inside the
        SDK's own transport task group dying for an unrelated reason — and
        those two cases must never be handled the same way: retrying after
        real cancellation would silently absorb it, so the retry appears to
        just "run a bit longer" instead of stopping, exactly the failure
        mode this method's own docstring promises never happens.
        `asyncio.Task.cancelling()` (3.11+) is the sanctioned way to tell
        them apart — it counts actual `cancel()` calls against the current
        task, 0 for a leaked child-task `CancelledError`, >=1 for a real
        one — verified empirically both ways before relying on it. A real
        cancellation is re-raised immediately, no retry attempted.
        """
        last_error: BaseException | None = None
        for attempt in (1, 2):
            try:
                result = await self._require_client().call_tool(name, arguments)
                return _flatten_content(result.content), result.is_error
            except (KeyboardInterrupt, SystemExit, GeneratorExit):
                raise
            except BaseException as exc:  # noqa: BLE001 - reclassified/reraised below
                if isinstance(exc, Exception):
                    classified = classify_mcp_error(exc)
                    if isinstance(classified, McpPermanentError):
                        raise classified from exc
                    last_error = classified
                else:
                    current_task = asyncio.current_task()
                    if current_task is not None and current_task.cancelling() > 0:
                        raise
                    last_error = exc
                if attempt == 2:
                    break
                logger.info("MCP tool %s: transport failure, reconnecting and retrying once: %s", name, last_error)
                try:
                    await self._reconnect()
                except Exception:
                    logger.exception("MCP tool %s: reconnect failed after a transport failure", name)
                    break
        assert last_error is not None
        raise last_error

    def _require_client(self) -> mcp.Client:
        if self._client is None:
            raise RuntimeError("McpSession used outside its `async with` block")
        return self._client

    def _log_round_trip(self, label: str, started: float) -> None:
        elapsed_ms = (time.monotonic() - started) * 1000
        log = logger.warning if elapsed_ms > _SLOW_ROUND_TRIP_MS else logger.info
        log("MCP %s took %.0fms (url=%s)", label, elapsed_ms, self._url or "<in-process>")


def _flatten_content(blocks: list[Any]) -> str | list[dict]:
    """Reduce a CallToolResult's content blocks to anton's ToolDef handler
    contract (str | list[dict]): plain text collapses to one string (the
    common case); anything with a non-text block is left as a list of dicts
    so no content is silently dropped."""
    if all(getattr(block, "type", None) == "text" for block in blocks):
        return "\n".join(getattr(block, "text", "") for block in blocks)
    return [block.model_dump(by_alias=True) for block in blocks]
