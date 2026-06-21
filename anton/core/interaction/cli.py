"""Terminal implementation of :class:`SelectionElicitor` for standalone CLI runs.

Renders a numbered picker to the console and reads the user's choice. The agent
loop already pauses the escape watcher around interactive tools, so this only
has to print the options and await a number.
"""

from __future__ import annotations

from anton.core.interaction.selection import SelectionRequest

__all__ = ["CLISelectionElicitor"]


class CLISelectionElicitor:
    """Picker that prompts on the terminal (standalone ``anton`` chat)."""

    def __init__(self, console) -> None:
        self._console = console

    async def elicit(self, request: SelectionRequest) -> str | None:
        from anton.utils.prompt import prompt_or_cancel

        options = request.options
        if not options:
            return None

        self._console.print(f"\n[bold]{request.prompt}[/]")
        for index, option in enumerate(options, start=1):
            icon = "📁" if option.kind == "folder" else "📄"
            detail = f"  [dim]{option.detail}[/]" if option.detail else ""
            self._console.print(f"  [bold]{index}[/]. {icon} {option.label}{detail}")

        choice = await prompt_or_cancel(
            "Select a number (Esc to cancel)",
            choices=[str(i) for i in range(1, len(options) + 1)],
        )
        if choice is None:
            return None
        try:
            selected = int(choice) - 1
        except ValueError:
            return None
        return options[selected].value if 0 <= selected < len(options) else None
