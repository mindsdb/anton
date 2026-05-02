"""Runtime orchestration — wakes per-session agent runtimes.

A :class:`RuntimeOrchestrator` is the dispatch system's hand-off point to
the actual agent runtime. The router calls :meth:`wake` after appending a
new ``messages_in`` row; the orchestrator ensures *something* is running
that will pick it up.

Two implementations ship:

  - :class:`InProcessRuntimeOrchestrator` — a pluggable in-process loop.
    Useful for tests, the CLI, and embedded deployments. Takes a
    user-supplied async callable that produces a reply; runs once per
    session in a background task.
  - :class:`LocalScratchpadOrchestrator` — bridges to Anton's existing
    ``LocalScratchpadRuntime`` (anton/core/backends/local.py). Threaded
    later — kept as a stub here so the import surface is stable.

Both share the same :class:`RuntimeOrchestrator` Protocol from
:mod:`anton.core.dispatch.router`. Cloud deployments can supply a third
implementation that talks to scratchpad_service over HTTP.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from anton.core.dispatch.entities import AgentGroup, Session
from anton.core.dispatch.router import RuntimeOrchestrator
from anton.core.dispatch.session_store import SessionStoreProtocol


# Agent callable contract. Given (agent_group, session, content_dict),
# return a string reply or ``None`` to suppress.
AgentCallable = Callable[[AgentGroup, Session, Any], Awaitable[str | None]]


class InProcessRuntimeOrchestrator(RuntimeOrchestrator):
    """Background-task orchestrator for in-process agents.

    Each session gets one long-running task that polls its store for
    undelivered ``messages_in`` rows, calls ``agent_fn``, and appends the
    reply to ``messages_out``. The router's :meth:`deliver` side then
    pushes those replies out via the originating adapter.

    Attributes:
        agent_fn: User-supplied async callable that produces replies.
        store_opener: Async callable returning a session store; the
            orchestrator uses this so it doesn't depend on the repository
            directly. The router wires this up at construction time.
        poll_interval_s: Sleep between empty polls. Loops are otherwise
            event-driven via :meth:`wake`'s wakeup signal.
    """

    def __init__(
        self,
        agent_fn: AgentCallable,
        store_opener: Callable[[Session], Awaitable[SessionStoreProtocol]],
        *,
        poll_interval_s: float = 0.5,
    ) -> None:
        self.agent_fn = agent_fn
        self.store_opener = store_opener
        self.poll_interval_s = poll_interval_s
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._wakeups: dict[str, asyncio.Event] = {}
        self._stops: dict[str, asyncio.Event] = {}

    async def wake(self, session: Session, agent_group: AgentGroup) -> None:
        """Ensure a runtime task exists for this session and notify it."""
        wakeup = self._wakeups.get(session.id)
        if wakeup is None:
            wakeup = asyncio.Event()
            stop = asyncio.Event()
            self._wakeups[session.id] = wakeup
            self._stops[session.id] = stop
            self._tasks[session.id] = asyncio.create_task(
                self._run(session, agent_group, wakeup, stop),
                name=f"dispatch-runtime-{session.id}",
            )
        wakeup.set()

    async def stop(self, session: Session) -> None:
        """Signal the per-session task to stop and await its exit."""
        stop = self._stops.pop(session.id, None)
        wakeup = self._wakeups.pop(session.id, None)
        task = self._tasks.pop(session.id, None)
        if stop is not None:
            stop.set()
        if wakeup is not None:
            wakeup.set()
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                task.cancel()

    async def stop_all(self) -> None:
        """Stop every running session task. Used at dispatcher shutdown."""
        for sid in list(self._tasks):
            stop = self._stops.get(sid)
            wakeup = self._wakeups.get(sid)
            if stop is not None:
                stop.set()
            if wakeup is not None:
                wakeup.set()
        tasks = list(self._tasks.values())
        self._tasks.clear()
        self._wakeups.clear()
        self._stops.clear()
        for t in tasks:
            try:
                await asyncio.wait_for(t, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                t.cancel()

    # -----------------------------------------------------------------
    # Internal loop
    # -----------------------------------------------------------------

    async def _run(
        self,
        session: Session,
        agent_group: AgentGroup,
        wakeup: asyncio.Event,
        stop: asyncio.Event,
    ) -> None:
        """Per-session loop: drain messages_in → invoke agent → append messages_out."""
        store = await self.store_opener(session)
        try:
            while not stop.is_set():
                # Drain pending inbound rows.
                rows = store.read_undelivered("in")
                for row in rows:
                    if stop.is_set():
                        break
                    try:
                        reply = await self.agent_fn(agent_group, session, row.content)
                    except Exception as e:
                        reply = f"[agent error] {e!r}"
                    finally:
                        store.mark_delivered(row.rowid, "in")
                    if reply:
                        store.append("out", "reply", {"text": reply})

                if stop.is_set():
                    break

                # Wait for next wakeup or the poll fallback.
                wakeup.clear()
                try:
                    await asyncio.wait_for(wakeup.wait(), timeout=self.poll_interval_s)
                except asyncio.TimeoutError:
                    pass
        finally:
            try:
                store.close()
            except Exception:
                pass


class LocalScratchpadOrchestrator(RuntimeOrchestrator):
    """Bridges dispatch sessions to Anton's ``LocalScratchpadRuntime``.

    Each session maps to one ``LocalScratchpadRuntime`` instance with
    vault credentials injected via ``extra_env``. The session's SQLite
    store is mounted into the runtime so the agent can read/write
    messages directly.

    Stubbed for now — the wiring lives in
    :mod:`anton.core.backends.local` and will be filled in once the
    in-process flow has been validated end-to-end via the CLI adapter.
    """

    async def wake(self, session: Session, agent_group: AgentGroup) -> None:
        raise NotImplementedError(
            "LocalScratchpadOrchestrator.wake is not yet implemented; "
            "use InProcessRuntimeOrchestrator for now."
        )

    async def stop(self, session: Session) -> None:
        raise NotImplementedError
