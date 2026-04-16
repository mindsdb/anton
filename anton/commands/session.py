"""Slash-command handlers for /resume."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console

from anton.config.settings import AntonSettings
from anton.utils.prompt import prompt_or_cancel
from anton.chat_session import rebuild_session

if TYPE_CHECKING:
    from anton.chat import ChatSession
    from anton.core.memory.episodes import EpisodicMemory
    from anton.memory.history_store import HistoryStore
    from anton.core.memory.cortex import Cortex
    from anton.workspace import Workspace


async def handle_resume(
    console: Console,
    settings: AntonSettings,
    state: dict,
    self_awareness,
    cortex: "Cortex | None",
    workspace: "Workspace | None",
    session: "ChatSession",
    episodic: "EpisodicMemory | None" = None,
    history_store: "HistoryStore | None" = None,
) -> "tuple[ChatSession, str | None]":
    """Show session picker and resume a previous chat session.

    Returns (new_session, resumed_session_id) or (original_session, None).
    """
    from rich.table import Table

    if history_store is None:
        console.print("[anton.warning]History store not available.[/]")
        console.print()
        return session, None

    sessions = history_store.list_sessions(limit=10)
    if not sessions:
        console.print()
        console.print("[anton.warning]No previous sessions to resume.[/]")
        console.print()
        return session, None

    console.print()
    console.print("[anton.cyan]Recent sessions:[/]")
    console.print()

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="bold", width=3)
    table.add_column("Date", style="anton.cyan")
    table.add_column("Turns", justify="right")
    table.add_column("Preview")

    for i, s in enumerate(sessions, 1):
        table.add_row(str(i), s["date"], str(s["turns"]), s["preview"])

    console.print(table)
    console.print()

    choices = [str(i) for i in range(1, len(sessions) + 1)] + ["q"]
    choice = await prompt_or_cancel(
        "(anton) Select session (or q to cancel)", choices=choices, default="q"
    )
    if choice is None or choice == "q":
        console.print()
        return session, None

    idx = int(choice) - 1
    selected = sessions[idx]
    sid = selected["session_id"]

    history = history_store.load(sid)
    if history is None:
        console.print("[anton.error]Failed to load session history.[/]")
        console.print()
        return session, None

    # Resume episodic memory for this session
    if episodic is not None and episodic.enabled:
        episodic.resume_session(sid)

    # Close old scratchpads
    if session._scratchpads.list_pads():
        await session._scratchpads.close_all()

    # Build new session with restored history
    new_session = rebuild_session(
        settings=settings,
        state=state,
        self_awareness=self_awareness,
        cortex=cortex,
        workspace=workspace,
        console=console,
        episodic=episodic,
        history_store=history_store,
        session_id=sid,
    )
    new_session._history = list(history)
    new_session._turn_count = sum(1 for m in history if m.get("role") == "user")

    console.print()
    console.print(
        f"[anton.success]Resumed session from {selected['date']} ({selected['turns']} turns)[/]"
    )
    console.print()

    return new_session, sid
