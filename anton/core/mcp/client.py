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
import contextlib
import logging
import time
from typing import Any

import httpx2
import mcp
from mcp_types import CallToolResult
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

# The SDK's own timeouts: httpx's 5 s default is too short, and a server may hold the stream open.
_HTTP_TIMEOUT_SECONDS = 30.0
_HTTP_READ_TIMEOUT_SECONDS = 300.0


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
        """Store the target; nothing connects until `__aenter__`.

        Raises `ValueError` unless exactly one of `url` or `server` is given.
        """
        if (url is None) == (server is None):
            raise ValueError("McpSession needs exactly one of `url` or `server`")
        self._url = url
        self._access_token = access_token
        self._server = server
        self._read_timeout_seconds = read_timeout_seconds
        self._http_client: httpx2.AsyncClient | None = None
        self._client: mcp.Client | None = None
        self._owner: asyncio.Task[BaseException | None] | None = None
        self._release: asyncio.Event | None = None
        self._reconnect_lock = asyncio.Lock()
        self._entered = False
        self._closed = False

    async def __aenter__(self) -> "McpSession":
        await self._connect()
        self._entered = True
        return self

    async def _connect(self) -> None:
        """Open a fresh client in its own owner task and wait until it is ready.

        Builds the HTTP client (url target) or uses the in-process server,
        then starts `_hold_client` in a new task and waits for its handshake.
        Sets `_client`, `_owner` and `_release` on success. Raises whatever
        the handshake raised; the owner has closed the HTTP client by then.
        """
        http_client: httpx2.AsyncClient | None = None
        if self._server is not None:
            target: Any = self._server
        else:
            headers = {"Authorization": f"Bearer {self._access_token}"} if self._access_token else {}
            http_client = httpx2.AsyncClient(
                headers=headers,
                timeout=httpx2.Timeout(
                    _HTTP_TIMEOUT_SECONDS,
                    read=_HTTP_READ_TIMEOUT_SECONDS,
                ),
            )
            self._http_client = http_client
            from mcp.client.streamable_http import streamable_http_client

            target = streamable_http_client(self._url, http_client=http_client)

        client = mcp.Client(
            target, raise_exceptions=True, read_timeout_seconds=self._read_timeout_seconds
        )
        opened: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        release = asyncio.Event()
        started = time.monotonic()
        owner = asyncio.create_task(
            self._hold_client(client, http_client, opened, release), name="mcp-session-owner"
        )
        try:
            await opened
        except asyncio.CancelledError:
            owner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await owner
            raise
        except BaseException:
            await owner
            raise
        self._log_round_trip("initialize()", started)
        self._client = client
        self._owner = owner
        self._release = release

    async def _hold_client(
        self,
        client: mcp.Client,
        http_client: httpx2.AsyncClient | None,
        opened: asyncio.Future[None],
        release: asyncio.Event,
    ) -> BaseException | None:
        """Own one `mcp.Client` for its whole life, inside this one task.

        Invariant: the SDK's transport and session open anyio task groups and
        cancel scopes, which anyio binds to the task that entered them. They
        must be exited in that same task, and when one of their background
        tasks fails anyio cancels that task until the scopes exit. So no
        caller's task may enter or exit a client: discovery opens it in a
        short-lived task, tools call it from child tasks, the turn closes it
        from another. This task is the only one that does either.

        Resolves `opened` once the handshake is done (or with its failure), then
        waits for `release`, or for anyio to cancel it because the transport
        died, and exits the client. Always closes `http_client`. Returns the
        error raised while exiting after `release` was set, so the closer can
        report it; a transport death is logged here instead.
        """
        try:
            async with client:
                if not opened.done():
                    opened.set_result(None)
                await release.wait()
        except Exception as exc:
            if not opened.done():
                opened.set_exception(exc)
            elif release.is_set():
                return exc
            else:
                logger.warning("McpSession: MCP transport closed after %s", _describe(exc))
        except BaseException as exc:
            if opened.done():
                raise
            # Handed to the waiting caller, which re-raises it; it must never wait forever.
            if isinstance(exc, asyncio.CancelledError):
                opened.cancel()
            else:
                opened.set_exception(exc)
        finally:
            if http_client is not None:
                await http_client.aclose()
        return None

    async def _close_client(self) -> BaseException | None:
        """Ask the current owner task to exit its client and wait for it.

        Clears `_client`, `_owner`, `_release` and `_http_client`. Returns the
        error the client raised while exiting, or None. Safe from any task.
        The teardown is shielded, so a cancelled closer re-raises at once while
        the owner still finishes the SDK's session DELETE and the HTTP close.
        An owner its event loop already cancelled (opened under one
        `asyncio.run`, closed under another) is logged and skipped.
        """
        owner, release = self._owner, self._release
        self._client = None
        self._owner = None
        self._release = None
        self._http_client = None
        if owner is None or release is None:
            return None
        release.set()
        if owner.cancelled():
            logger.warning("McpSession: the session's owner task was already cancelled by its event loop")
            return None
        return await asyncio.shield(owner)

    async def _reconnect(self, failed: mcp.Client | None) -> None:
        """Replace `failed` with a fresh client against the same target/token.

        Required, not optional, for `call_tool`'s retry: once one task in a
        session's transport task group has failed, the whole group, and the
        `Client` built on it, is unusable. A no-op when another call already
        replaced `failed`; `failed=None` means no client was live, and a fresh
        one is opened unless another call already did. Raises what `_connect`
        raises; a close error from the dead client is logged, never raised.
        """
        async with self._reconnect_lock:
            if self._client is not failed:
                return
            close_error = await self._close_client()
            if close_error is not None:
                logger.warning("McpSession: closing the dead client failed with %s", _describe(close_error))
            await self._connect()

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Tear down the session; safe from any task.

        The owner task exits the client, so this never crosses task scopes. A
        cleanup error is raised when nothing else is propagating, and only
        logged when an exception is already leaving the `async with` block,
        so it never replaces a clean, catchable error with an opaque one.
        """
        self._closed = True
        close_error = await self._close_client()
        if close_error is None:
            return
        if exc_type is not None:
            logger.warning(
                "McpSession: closing the session failed with %s while another exception was propagating",
                _describe(close_error),
            )
            return
        raise close_error

    async def list_tools(self) -> list[McpTool]:
        """List every tool this server advertises, unfiltered."""
        self._check_open()
        client = await self._acquire_client()
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
        result = await self._call_with_retry(name, arguments)
        return _flatten_content(result.content), result.is_error

    async def call_tool_structured(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, str | list[dict], bool]:
        """Call one tool and keep its structured result too.

        Returns `(structured_content, content, is_error)`: the tool's MCP
        `structured_content` (None when it sent none) beside what `call_tool`
        returns. Retries, classifies and re-raises exactly as `call_tool`
        does, since both go through `_call_with_retry`.
        """
        result = await self._call_with_retry(name, arguments)
        return result.structured_content, _flatten_content(result.content), result.is_error

    async def _call_with_retry(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        """Run one `tools/call` with the retry and cancellation rules `call_tool` documents.

        Returns the raw `CallToolResult`. Raises `McpPermanentError` at once,
        `McpTransientError` or the original non-`Exception` failure after one
        retry against a fresh session. May reconnect the session as a side effect.
        """
        last_error: BaseException | None = None
        self._check_open()
        for attempt in (1, 2):
            client: mcp.Client | None = None
            try:
                client = await self._acquire_client()
                return await client.call_tool(name, arguments)
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
                    await self._reconnect(client)
                except Exception:
                    logger.exception("MCP tool %s: reconnect failed after a transport failure", name)
                    break
        assert last_error is not None
        raise last_error

    def _check_open(self) -> None:
        """Raise `RuntimeError` unless the session was entered and not yet closed."""
        if not self._entered or self._closed:
            raise RuntimeError("McpSession used outside its `async with` block")

    async def _acquire_client(self) -> mcp.Client:
        """Return the live client, opening one when a failed reconnect left none.

        A call that starts while another call reconnects waits on the lock and
        gets the new client. Raises what `_connect` raises, which the caller
        classifies like any transport failure; `RuntimeError` once closed.
        """
        client = self._client
        if client is not None:
            return client
        async with self._reconnect_lock:
            self._check_open()
            if self._client is None:
                await self._connect()
            assert self._client is not None
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


def _describe(exc: BaseException) -> str:
    """Name `exc` by class only, with the classes inside an exception group.

    Class names only: a transport error's message can carry a server URL or
    a token-bearing request, and these lines are logged.
    """
    if isinstance(exc, BaseExceptionGroup):
        return f"{type(exc).__name__}({', '.join(_describe(inner) for inner in exc.exceptions)})"
    return type(exc).__name__
