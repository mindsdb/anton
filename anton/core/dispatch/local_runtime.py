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
  - :class:`LocalScratchpadOrchestrator` — bridges to Anton's
    :func:`anton.core.runtime.build_chat_session`. Each dispatch session
    maps to one cached :class:`ChatSession` so conversation memory and
    vault credentials persist across inbound messages.

Both share the same :class:`RuntimeOrchestrator` Protocol from
:mod:`anton.core.dispatch.router`. Cloud deployments can supply a third
implementation that talks to scratchpad_service over HTTP.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from anton.core.dispatch.entities import AgentGroup, Session
from anton.core.dispatch.router import RuntimeOrchestrator
from anton.core.dispatch.session_store import SessionStoreProtocol

logger = logging.getLogger(__name__)


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


def _extract_text(content: Any) -> str:
    """Pull a user-visible text string out of a stored ``messages_in`` row.

    The router stores inbound messages as a dict::

        {"id": ..., "content": <Any>, "sender_id": ..., "sender_name": ...,
         "thread_id": ...}

    where ``content`` is whatever the channel adapter handed in. Most
    channels send a string; some send a dict (e.g. Chat-SDK style).
    Anything non-string falls back to ``repr`` so the agent at least sees
    *something* it can reason about instead of crashing on type mismatch.
    """
    payload = content
    if isinstance(payload, dict):
        inner = payload.get("content")
        if inner is not None:
            payload = inner
    if isinstance(payload, str):
        return payload
    return repr(payload)


class LocalScratchpadOrchestrator(RuntimeOrchestrator):
    """Bridges dispatch sessions to Anton's :class:`ChatSession`.

    One :class:`ChatSession` is built per dispatch session and cached for
    the lifetime of the runtime task — conversation memory, vault env
    injection, and history all persist across inbound messages.

    Each inbound row drains through ``ChatSession.turn_stream(text)``;
    text deltas are concatenated into a single user-visible reply written
    to ``messages_out`` (kind ``"reply"``). Tool-result / progress events
    are forwarded as separate rows (kind ``"activity"``) so dashboards
    that care about agent internals can render them — channel adapters
    typically ignore non-``"reply"`` rows.

    Parameters
    ----------
    store_opener
        Async callable returning the session's SQLite store. Wired by the
        router at construction time.
    extra_tools
        Tools added on top of the Anton defaults (e.g. publish, datasource
        connect). Defaults to none.
    system_prompt_suffix
        Free-form text appended to the system prompt — hosts use this to
        describe their UI affordances.
    poll_interval_s
        Sleep between empty polls. Otherwise event-driven via :meth:`wake`.
    emit_activity_rows
        Write tool-result and progress events to ``messages_out`` as
        ``"activity"`` rows. Set False to keep the outbox text-only.
    """

    def __init__(
        self,
        store_opener: Callable[[Session], Awaitable[SessionStoreProtocol]],
        *,
        extra_tools: list[Any] | None = None,
        system_prompt_suffix: str | None = None,
        poll_interval_s: float = 0.5,
        emit_activity_rows: bool = True,
    ) -> None:
        self.store_opener = store_opener
        self.extra_tools = extra_tools
        self.system_prompt_suffix = system_prompt_suffix
        self.poll_interval_s = poll_interval_s
        self.emit_activity_rows = emit_activity_rows
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._wakeups: dict[str, asyncio.Event] = {}
        self._stops: dict[str, asyncio.Event] = {}
        # One ChatSession per dispatch session.id, lazily built on first inbound.
        self._chat_sessions: dict[str, Any] = {}

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
                name=f"dispatch-scratchpad-{session.id}",
            )
        wakeup.set()

    async def stop(self, session: Session) -> None:
        """Signal the per-session task to stop and await its exit."""
        stop = self._stops.pop(session.id, None)
        wakeup = self._wakeups.pop(session.id, None)
        task = self._tasks.pop(session.id, None)
        chat = self._chat_sessions.pop(session.id, None)
        if stop is not None:
            stop.set()
        if wakeup is not None:
            wakeup.set()
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                task.cancel()
        # Close the ChatSession only after the task has stopped using it —
        # close() tears down scratchpad subprocesses, which would otherwise
        # leak (orphaned processes) once the cache entry is dropped.
        if chat is not None:
            await self._close_chat_session(chat)

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
        chats = list(self._chat_sessions.values())
        self._tasks.clear()
        self._wakeups.clear()
        self._stops.clear()
        self._chat_sessions.clear()
        for t in tasks:
            try:
                await asyncio.wait_for(t, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                t.cancel()
        # Tear down every cached ChatSession after its task has stopped, so
        # scratchpad subprocesses are terminated rather than orphaned.
        for chat in chats:
            await self._close_chat_session(chat)

    @staticmethod
    async def _close_chat_session(chat: Any) -> None:
        """Close one cached ChatSession; best-effort, never raises.

        ChatSession.close() releases scratchpad subprocesses and other
        runtime resources. Shutdown must not abort because one session
        failed to close cleanly, so failures are logged and swallowed.
        """
        close = getattr(chat, "close", None)
        if close is None:
            return
        try:
            await close()
        except Exception:
            logger.exception("failed to close dispatch ChatSession")

    # -----------------------------------------------------------------
    # Internal: build / cache the Anton ChatSession
    # -----------------------------------------------------------------

    async def _get_chat_session(
        self,
        session: Session,
        agent_group: AgentGroup,
    ) -> Any:
        cached = self._chat_sessions.get(session.id)
        if cached is not None:
            return cached

        # Lazy import keeps the dispatch module importable on hosts where
        # Anton's heavy deps aren't installed (e.g. router-only test rigs).
        from anton.core.runtime import build_chat_session

        chat = await build_chat_session(
            session_id=session.id,
            workspace_path=str(agent_group.workspace) if agent_group.workspace else None,
            extra_tools=self.extra_tools,
            system_prompt_suffix=self.system_prompt_suffix,
        )
        self._chat_sessions[session.id] = chat
        return chat

    # -----------------------------------------------------------------
    # Internal: per-session loop
    # -----------------------------------------------------------------

    async def _run(
        self,
        session: Session,
        agent_group: AgentGroup,
        wakeup: asyncio.Event,
        stop: asyncio.Event,
    ) -> None:
        """Per-session loop: drain messages_in → run turn_stream → append messages_out."""
        store = await self.store_opener(session)
        try:
            while not stop.is_set():
                rows = store.read_undelivered("in")
                for row in rows:
                    if stop.is_set():
                        break
                    text = _extract_text(row.content)
                    try:
                        reply = await self._run_turn(session, agent_group, store, text)
                    except Exception as exc:
                        logger.exception(
                            "ChatSession turn failed for session %s", session.id
                        )
                        reply = self._safe_error_message(exc)
                    finally:
                        store.mark_delivered(row.rowid, "in")
                    if reply:
                        store.append("out", "reply", {"text": reply})

                if stop.is_set():
                    break

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

    async def _run_turn(
        self,
        session: Session,
        agent_group: AgentGroup,
        store: SessionStoreProtocol,
        text: str,
    ) -> str:
        """Run one ChatSession turn, collect deltas, optionally emit activity rows."""
        from anton.core.llm.provider import (
            StreamComplete,
            StreamContextCompacted,
            StreamTaskProgress,
            StreamTextDelta,
            StreamToolResult,
        )

        chat = await self._get_chat_session(session, agent_group)
        parts: list[str] = []

        async for event in chat.turn_stream(text):
            if isinstance(event, StreamTextDelta):
                parts.append(event.text)
            elif isinstance(event, StreamTaskProgress):
                if self.emit_activity_rows:
                    store.append("out", "activity", {
                        "kind": "progress",
                        "phase": event.phase,
                        "message": event.message,
                    })
            elif isinstance(event, StreamToolResult):
                if self.emit_activity_rows:
                    store.append("out", "activity", {
                        "kind": "tool_result",
                        "name": event.name,
                        "action": event.action,
                        "content": event.content,
                    })
            elif isinstance(event, StreamContextCompacted):
                if self.emit_activity_rows:
                    store.append("out", "activity", {
                        "kind": "context",
                        "message": event.message,
                    })
            elif isinstance(event, StreamComplete):
                # End-of-turn marker — nothing to write; the loop falls through
                # and the consolidated reply gets appended by `_run`.
                pass

        return "".join(parts).strip()

    @staticmethod
    def _safe_error_message(exc: Exception) -> str:
        """Render an exception as a user-facing error with API keys redacted."""
        try:
            from anton.core.runtime import safe_redact_error
            return f"[agent error] {safe_redact_error(exc)}"
        except Exception:
            return f"[agent error] {exc!r}"
