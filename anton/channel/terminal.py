from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from anton.channel.base import Channel
from anton.events.types import (
    AntonEvent,
    PromptUser,
    StatusUpdate,
    TaskComplete,
    TaskFailed,
)


_PHASE_LABELS = {
    "planning": "[bold blue]Planning[/]",
    "skill_discovery": "[bold cyan]Discovering skills[/]",
    "skill_building": "[bold magenta]Building skill[/]",
    "executing": "[bold yellow]Executing[/]",
    "complete": "[bold green]Complete[/]",
    "failed": "[bold red]Failed[/]",
}


class TerminalChannel(Channel):
    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._status_ctx = None

    async def emit(self, event: AntonEvent) -> None:
        if isinstance(event, StatusUpdate):
            label = _PHASE_LABELS.get(event.phase.value, event.phase.value)
            eta = f" (ETA ~{int(event.eta_seconds)}s)" if event.eta_seconds else ""
            msg = f"{label}  {event.message}{eta}"
            # Stop any existing spinner before starting a new one
            if self._status_ctx is not None:
                self._status_ctx.stop()
                self._status_ctx = None
            self._status_ctx = self.console.status(msg, spinner="dots")
            self._status_ctx.start()

        elif isinstance(event, TaskComplete):
            self._stop_spinner()
            self.console.print(
                Panel(event.summary, title="Done", border_style="green")
            )

        elif isinstance(event, TaskFailed):
            self._stop_spinner()
            self.console.print(
                Panel(event.error_summary, title="Failed", border_style="red")
            )

        elif isinstance(event, PromptUser):
            self._stop_spinner()
            # prompt handled via prompt() method; this is for display only
            self.console.print(f"[bold]?[/] {event.question}")

    async def prompt(self, question: str) -> str:
        self._stop_spinner()
        return await asyncio.to_thread(Prompt.ask, question)

    async def close(self) -> None:
        self._stop_spinner()

    def _stop_spinner(self) -> None:
        if self._status_ctx is not None:
            self._status_ctx.stop()
            self._status_ctx = None


_THEMED_PHASE_LABELS = {
    "planning": "[phase.planning]Planning[/]",
    "skill_discovery": "[phase.skill_discovery]Discovering skills[/]",
    "skill_building": "[phase.skill_building]Building skill[/]",
    "executing": "[phase.executing]Executing[/]",
    "complete": "[phase.complete]Complete[/]",
    "failed": "[phase.failed]Failed[/]",
}


class CLIChannel(Channel):
    def __init__(self) -> None:
        from anton.channel.theme import build_rich_theme, detect_color_mode, get_palette

        mode = detect_color_mode()
        self._palette = get_palette(mode)
        self.console = Console(theme=build_rich_theme(mode))
        self._status_ctx = None

    async def emit(self, event: AntonEvent) -> None:
        if isinstance(event, StatusUpdate):
            label = _THEMED_PHASE_LABELS.get(event.phase.value, event.phase.value)
            eta = f" (ETA ~{int(event.eta_seconds)}s)" if event.eta_seconds else ""
            msg = f"{label}  {event.message}{eta}"
            if self._status_ctx is not None:
                self._status_ctx.stop()
                self._status_ctx = None
            self._status_ctx = self.console.status(msg, spinner="dots")
            self._status_ctx.start()

        elif isinstance(event, TaskComplete):
            self._stop_spinner()
            self.console.print(
                Panel(event.summary, title="Done", border_style="anton.success")
            )

        elif isinstance(event, TaskFailed):
            self._stop_spinner()
            self.console.print(
                Panel(event.error_summary, title="Failed", border_style="anton.error")
            )

        elif isinstance(event, PromptUser):
            self._stop_spinner()
            self.console.print(f"[bold]?[/] {event.question}")

    async def prompt(self, question: str) -> str:
        self._stop_spinner()
        return await asyncio.to_thread(Prompt.ask, question)

    async def close(self) -> None:
        self._stop_spinner()

    def _stop_spinner(self) -> None:
        if self._status_ctx is not None:
            self._status_ctx.stop()
            self._status_ctx = None
