from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx2
import mcp
import pytest

from anton.core.mcp.client import McpSession
from anton.core.mcp.errors import McpPermanentError, McpTransientError
from anton.core.mcp.wiring import close_mcp_sessions


async def test_list_tools_returns_the_stub_servers_tools(stub_mcp_server):
    async with McpSession(server=stub_mcp_server) as session:
        tools = await session.list_tools()
    assert {t.name for t in tools} == {"add", "boom"}


async def test_call_tool_success_flattens_text_content(stub_mcp_server):
    async with McpSession(server=stub_mcp_server) as session:
        content, is_error = await session.call_tool("add", {"a": 1, "b": 2})
    assert content == "3"
    assert is_error is False


async def test_call_tool_error_surfaces_as_is_error_not_an_exception(stub_mcp_server):
    """Per the MCP spec, a tool-originated failure comes back INSIDE the
    result (is_error=True), not as a protocol-level exception — the caller
    (registry.py) decides what that means, this layer just reports it."""
    async with McpSession(server=stub_mcp_server) as session:
        content, is_error = await session.call_tool("boom", {})
    assert is_error is True
    assert isinstance(content, str)


async def test_session_used_outside_context_manager_raises(stub_mcp_server):
    session = McpSession(server=stub_mcp_server)
    with pytest.raises(RuntimeError):
        await session.list_tools()


def test_requires_exactly_one_of_url_or_server(stub_mcp_server):
    with pytest.raises(ValueError):
        McpSession()
    with pytest.raises(ValueError):
        McpSession(url="https://example.test", server=stub_mcp_server)


async def test_call_tool_retries_once_on_transient_failure(stub_mcp_server, monkeypatch):
    """A transport/protocol-level failure (not a tool-reported is_error) gets
    exactly one retry against a FRESH session (a dead transport's task group
    can't serve a second call — see McpSession._reconnect) before raising.
    Patched at the class level, not the instance: reconnecting builds a new
    `mcp.Client` object, so an instance-level patch would only ever see the
    first attempt."""
    calls = {"n": 0}
    real_call_tool = mcp.Client.call_tool

    async def flaky(self, name, arguments):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("transport hiccup")
        return await real_call_tool(self, name, arguments)

    monkeypatch.setattr(mcp.Client, "call_tool", flaky)
    async with McpSession(server=stub_mcp_server) as session:
        content, is_error = await session.call_tool("add", {"a": 2, "b": 2})

    assert calls["n"] == 2
    assert content == "4"
    assert is_error is False


async def test_call_tool_raises_transient_error_after_exhausting_the_retry(stub_mcp_server, monkeypatch):
    async def always_fails(self, name, arguments):
        raise ConnectionError("still down")

    monkeypatch.setattr(mcp.Client, "call_tool", always_fails)
    async with McpSession(server=stub_mcp_server) as session:
        with pytest.raises(McpTransientError):
            await session.call_tool("add", {"a": 1, "b": 1})


async def test_call_tool_never_retries_a_permanent_failure(stub_mcp_server, monkeypatch):
    async with McpSession(server=stub_mcp_server) as session:
        calls = {"n": 0}

        async def rejected(name, arguments):
            calls["n"] += 1
            raise McpPermanentError("token rejected")

        monkeypatch.setattr(session._client, "call_tool", rejected)

        with pytest.raises(McpPermanentError):
            await session.call_tool("add", {"a": 1, "b": 1})

    assert calls["n"] == 1


async def test_call_tool_reconnects_on_a_non_exception_transport_failure(stub_mcp_server, monkeypatch):
    """Killing a server mid-session can surface as asyncio.CancelledError or
    an (Base)ExceptionGroup rather than a plain Exception, depending on which
    background task of the transport noticed first (verified by hand,
    ENG-1816) — `except Exception` alone would miss it. This still gets one
    reconnect-and-retry, same as a plain exception."""
    calls = {"n": 0}
    real_call_tool = mcp.Client.call_tool

    async def flaky(self, name, arguments):
        calls["n"] += 1
        if calls["n"] == 1:
            raise asyncio.CancelledError("simulated transport-task-group teardown")
        return await real_call_tool(self, name, arguments)

    monkeypatch.setattr(mcp.Client, "call_tool", flaky)
    async with McpSession(server=stub_mcp_server) as session:
        content, is_error = await session.call_tool("add", {"a": 5, "b": 5})

    assert calls["n"] == 2
    assert content == "10"
    assert is_error is False


async def test_call_tool_reraises_the_original_error_when_a_non_exception_failure_persists(
    stub_mcp_server, monkeypatch
):
    """If reconnecting doesn't help, the ORIGINAL exception propagates as-is
    — never disguised as McpTransientError, since that could be real task
    cancellation (e.g. the whole turn being cancelled) rather than a
    business-logic failure a ToolOutcome should swallow."""

    async def always_cancelled(self, name, arguments):
        raise asyncio.CancelledError("still gone")

    monkeypatch.setattr(mcp.Client, "call_tool", always_cancelled)
    async with McpSession(server=stub_mcp_server) as session:
        with pytest.raises(asyncio.CancelledError):
            await session.call_tool("add", {"a": 1, "b": 1})


async def test_call_tool_never_retries_a_real_task_cancellation(stub_mcp_server, monkeypatch):
    """A CancelledError delivered because THIS task's own cancellation was
    requested (Task.cancelling() > 0) must propagate immediately, with no
    reconnect/retry attempted — retrying would silently swallow real
    cancellation, exactly what call_tool's docstring promises never happens.
    Distinguishes this from a CancelledError merely leaking up from the
    transport's own internal task group (cancelling() == 0 there — see
    test_call_tool_reconnects_on_a_non_exception_transport_failure, which
    still retries because nothing called .cancel() on the test's own task)."""
    calls = {"n": 0}

    async def always_cancelled(self, name, arguments):
        calls["n"] += 1
        raise asyncio.CancelledError()

    class _FakeCancelledTask:
        def cancelling(self) -> int:
            return 1

    monkeypatch.setattr(mcp.Client, "call_tool", always_cancelled)
    monkeypatch.setattr(asyncio, "current_task", lambda: _FakeCancelledTask())

    async with McpSession(server=stub_mcp_server) as session:
        with pytest.raises(asyncio.CancelledError):
            await session.call_tool("add", {"a": 1, "b": 1})

    assert calls["n"] == 1  # no retry attempted


async def test_url_session_http_client_uses_the_sdk_timeouts_and_keeps_its_auth_header(monkeypatch):
    """httpx's 5 s default timed out Genie calls under load; the session must
    carry the MCP SDK's timeouts and still send the bearer header."""

    def refuse_transport(url, *, http_client):
        raise ConnectionError("no network in tests")

    monkeypatch.setattr("mcp.client.streamable_http.streamable_http_client", refuse_transport)
    session = McpSession(url="https://example.test/mcp", access_token="tok-123")

    with pytest.raises(ConnectionError):
        await session._connect()

    http_client = session._http_client
    assert http_client is not None
    assert http_client.timeout.connect == 30.0
    assert http_client.timeout.write == 30.0
    assert http_client.timeout.pool == 30.0
    assert http_client.timeout.read == 300.0
    assert http_client.headers["Authorization"] == "Bearer tok-123"


_HANDSHAKE_METHODS = frozenset({"initialize", "server/discover"})


class _DroppingTransport(httpx2.AsyncBaseTransport):
    """Routes requests to an in-process ASGI app, failing chosen `tools/call` POSTs.

    `drop` holds the 1-based numbers of the `tools/call` requests that raise
    `httpx2.ReadTimeout` before reaching the app, the shape a dropped
    connection took live; `refuse_handshake` does the same for handshake
    requests with `httpx2.ConnectError`. `tool_calls` and `handshakes` count
    every request of that kind seen.
    """

    def __init__(self, app: Any, drop: set[int], refuse_handshake: set[int] | None = None) -> None:
        self._inner = httpx2.ASGITransport(app=app)
        self._drop = drop
        self._refuse_handshake = refuse_handshake or set()
        self.tool_calls = 0
        self.handshakes = 0

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        """Forward `request` to the app, or fail a dropped call or a refused handshake."""
        method = json.loads(request.content or b"{}").get("method") if request.method == "POST" else None
        if method == "tools/call":
            self.tool_calls += 1
            if self.tool_calls in self._drop:
                raise httpx2.ReadTimeout("simulated dropped connection", request=request)
        if method in _HANDSHAKE_METHODS:
            self.handshakes += 1
            if self.handshakes in self._refuse_handshake:
                raise httpx2.ConnectError("simulated refused reconnect", request=request)
        return await self._inner.handle_async_request(request)


@pytest.fixture()
def dropping_url_server(stub_mcp_server, monkeypatch):
    """Serve the stub over streamable HTTP in-process and route every session to it.

    Returns an async context manager taking the `tools/call` numbers to drop
    and yielding the transport, so a test can count the calls that reached
    the server. Entered in the test body: the server's task group must be
    exited in the task that entered it, and an async fixture's teardown runs
    in another task.
    """
    app = stub_mcp_server.streamable_http_app(json_response=True, stateless_http=True)
    real_async_client = httpx2.AsyncClient

    @contextlib.asynccontextmanager
    async def serve(
        drop: set[int], refuse_handshake: set[int] | None = None
    ) -> AsyncIterator[_DroppingTransport]:
        """Run the server and route McpSession's HTTP client through a failing transport."""
        transport = _DroppingTransport(app, drop, refuse_handshake)

        class _RoutedClient(real_async_client):
            """An `httpx2.AsyncClient` that always sends through `transport`."""

            def __init__(self, **kwargs: Any) -> None:
                """Build the client as McpSession asks, with the test transport forced in."""
                super().__init__(transport=transport, **kwargs)

        monkeypatch.setattr(httpx2, "AsyncClient", _RoutedClient)
        async with stub_mcp_server.session_manager.run():
            yield transport

    return serve


_STUB_URL = "http://127.0.0.1:8000/mcp"


async def _call_in_child_task(session: McpSession, a: int, b: int) -> tuple[str | list[dict], bool]:
    """Call `add` from a fresh task, as a turn's tool handler does, bounded so a hang fails."""
    return await asyncio.wait_for(asyncio.create_task(session.call_tool("add", {"a": a, "b": b})), timeout=5)


async def test_a_dropped_call_from_a_child_task_retries_and_the_opener_closes_cleanly(dropping_url_server):
    """The session is opened in this task and a tool handler calls it from a
    child task. A dropped first request is retried once and answered, a later
    call works, and the opener closes the session without an error or a hang."""
    async with dropping_url_server({1}) as transport:
        session = McpSession(url=_STUB_URL)
        await asyncio.wait_for(session.__aenter__(), timeout=5)

        assert await _call_in_child_task(session, 1, 2) == ("3", False)
        assert await _call_in_child_task(session, 2, 3) == ("5", False)
        await asyncio.wait_for(session.__aexit__(None, None, None), timeout=5)

    assert transport.tool_calls == 3
    assert asyncio.current_task().cancelling() == 0


async def test_a_session_opened_in_a_finished_task_survives_a_drop_and_closes_cleanly(dropping_url_server):
    """Discovery opens each session inside an `asyncio.gather` task that has
    finished before any tool runs, and the turn closes it from another task.
    Neither the retry nor that close may depend on the opening task."""
    async with dropping_url_server({1}):
        session = McpSession(url=_STUB_URL)
        await asyncio.wait_for(asyncio.gather(session.__aenter__()), timeout=5)

        assert await _call_in_child_task(session, 1, 2) == ("3", False)
        assert await _call_in_child_task(session, 4, 4) == ("8", False)
        await asyncio.wait_for(session.__aexit__(None, None, None), timeout=5)


async def test_a_call_dropped_twice_raises_a_transient_error_not_a_cancellation(dropping_url_server):
    """When the retry is dropped too, the caller gets `McpTransientError`. A
    `CancelledError` here would end the turn as if the person pressed Stop."""
    async with dropping_url_server({1, 2}) as transport:
        session = McpSession(url=_STUB_URL)
        await asyncio.wait_for(asyncio.gather(session.__aenter__()), timeout=5)

        with pytest.raises(McpTransientError):
            await _call_in_child_task(session, 1, 2)
        assert await _call_in_child_task(session, 2, 2) == ("4", False)
        await asyncio.wait_for(session.__aexit__(None, None, None), timeout=5)

    assert transport.tool_calls == 3


async def test_a_session_recovers_on_the_next_call_after_a_failed_reconnect(dropping_url_server):
    """The first call is dropped and its reconnect is refused, so it ends in
    `McpTransientError`. The session must not stay dead for the turn: the
    next call opens a fresh client and is answered."""
    async with dropping_url_server({1}, refuse_handshake={2}) as transport:
        session = McpSession(url=_STUB_URL)
        await asyncio.wait_for(asyncio.gather(session.__aenter__()), timeout=5)

        with pytest.raises(McpTransientError):
            await _call_in_child_task(session, 1, 2)
        assert await _call_in_child_task(session, 3, 4) == ("7", False)
        await asyncio.wait_for(session.__aexit__(None, None, None), timeout=5)

    assert transport.handshakes == 3


async def test_parallel_calls_hit_by_one_drop_share_a_single_reconnect(dropping_url_server):
    """Two tool calls run at once and the first request drops. Both are
    answered, and the session reconnects once, not once per caller."""
    async with dropping_url_server({1}) as transport:
        session = McpSession(url=_STUB_URL)
        await asyncio.wait_for(asyncio.gather(session.__aenter__()), timeout=5)

        first = asyncio.create_task(session.call_tool("add", {"a": 1, "b": 1}))
        second = asyncio.create_task(session.call_tool("add", {"a": 2, "b": 2}))
        results = await asyncio.wait_for(asyncio.gather(first, second), timeout=5)
        await asyncio.wait_for(session.__aexit__(None, None, None), timeout=5)

    assert results == [("2", False), ("4", False)]
    assert transport.handshakes == 2


def test_a_session_opened_under_one_event_loop_closes_cleanly_under_another(stub_mcp_server):
    """The cloud turn opens sessions under one `asyncio.run` and closes them
    under another. The first loop's shutdown has cancelled the owner task by
    then, and closing must return normally rather than raise `CancelledError`."""
    session = McpSession(server=stub_mcp_server)

    async def open_and_call() -> tuple[str | list[dict], bool]:
        """Open the session and make one call, bounded."""
        await asyncio.wait_for(session.__aenter__(), timeout=5)
        return await asyncio.wait_for(session.call_tool("add", {"a": 1, "b": 2}), timeout=5)

    async def close() -> None:
        """Close the session from a fresh loop, bounded."""
        await asyncio.wait_for(close_mcp_sessions([session]), timeout=5)

    assert asyncio.run(open_and_call()) == ("3", False)
    asyncio.run(close())
    assert session._client is None


async def _close_while_connecting(session: McpSession, monkeypatch, reopen) -> list[Any]:
    """Run `reopen()` with `_connect` held open, close the session meanwhile, then let it finish.

    Returns the owner tasks the held `_connect` started, so a test can check
    nothing was left running. `reopen()` must raise `RuntimeError`.
    """
    real_connect = session._connect
    entered, gate = asyncio.Event(), asyncio.Event()
    owners: list[Any] = []

    async def held_connect() -> None:
        """Wait for the test's go, then open the client as usual and record its owner."""
        entered.set()
        await gate.wait()
        await real_connect()
        owners.append(session._owner)

    monkeypatch.setattr(session, "_connect", held_connect)
    task = asyncio.create_task(reopen())
    await asyncio.wait_for(entered.wait(), timeout=5)
    await asyncio.wait_for(session.__aexit__(None, None, None), timeout=5)
    gate.set()
    with pytest.raises(RuntimeError):
        await asyncio.wait_for(task, timeout=5)
    return owners


async def test_a_close_during_a_reconnect_releases_the_fresh_client(stub_mcp_server, monkeypatch):
    """The turn closes the session while a call is reconnecting it. The client
    that reconnect opens afterwards must be closed too, not left running."""
    session = McpSession(server=stub_mcp_server)
    await asyncio.wait_for(session.__aenter__(), timeout=5)

    owners = await _close_while_connecting(session, monkeypatch, lambda: session._reconnect(session._client))

    assert len(owners) == 1 and owners[0].done()
    assert session._client is None and session._owner is None


async def test_a_close_while_a_call_reopens_a_missing_client_releases_it(stub_mcp_server, monkeypatch):
    """After a failed reconnect left no client, a call opens one. A close that
    lands while it opens must still leave nothing running."""
    session = McpSession(server=stub_mcp_server)
    await asyncio.wait_for(session.__aenter__(), timeout=5)
    await asyncio.wait_for(session._close_client(), timeout=5)

    owners = await _close_while_connecting(session, monkeypatch, session._acquire_client)

    assert len(owners) == 1 and owners[0].done()
    assert session._client is None and session._owner is None
