from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.text import Text

if TYPE_CHECKING:
    from rich.console import Console


@dataclass
class _ToolActivity:
    tool_id: str
    name: str
    json_parts: list[str] = field(default_factory=list)
    description: str = ""
    current_progress: str = ""
    step_count: int = 0


_TOOL_LABELS: dict[str, str] = {
    "scratchpad": "Scratchpad",
    "update_context": "Context",
}

_MAX_DESC = 60


def _tool_display_text(name: str, input_json: str) -> str:
    """Map tool name + raw JSON input to a human-readable description."""
    label = _TOOL_LABELS.get(name, name)
    try:
        data = json.loads(input_json)
    except (json.JSONDecodeError, TypeError):
        return label

    desc = ""
    if name == "scratchpad":
        desc = data.get("one_line_description") or data.get("action", "")
    elif name == "update_context":
        updates = data.get("updates", [])
        desc = f"{len(updates)} file(s)"
    if desc:
        if len(desc) > _MAX_DESC:
            desc = desc[:_MAX_DESC - 1] + "\u2026"
        return f"{label}({desc})"
    return label

THINKING_MESSAGES = [
    "Consulting the sacred docs...",
    "Rebasing my neurons...",
    "Spinning up inference hamsters...",
    "Parsing the vibes...",
    "Asking the rubber duck...",
    "Aligning my attention heads...",
    "Searching the latent space...",
    "Unrolling the loops...",
    "Compiling thoughts...",
    "Warming up the transformer...",
    "Descending the gradient...",
    "Sampling from the posterior...",
    "Tokenizing reality...",
    "Running a forward pass...",
    "Traversing the context window...",
    "Optimizing the objective...",
    "Softmaxing the options...",
    "Backpropagating insights...",
    "Loading weights...",
    "Crunching embeddings...",
]

WORKING_FOOTER_MESSAGES = [
    "working through your request",
    "piecing together a solution",
    "reasoning through the problem",
    "exploring the best approach",
    "connecting the dots for you",
    "building your answer step by step",
    "untangling the problem for you",
    "chewing on this one carefully",
    "cooking up a solid answer",
    "wiring together a solution",
]

TOOL_MESSAGES = [
    "Rolling up sleeves...",
    "Firing up the agent...",
    "Handing off to the crew...",
    "Dispatching the task...",
    "Engaging autopilot...",
    "Letting the tools cook...",
]

PHASE_LABELS = {
    "memory_recall": "Memory",
    "planning": "Planning",
    "executing": "Executing",
    "complete": "Complete",
    "failed": "Failed",
    "scratchpad": "Scratchpad",
}


class StreamDisplay:
    """Manages a Rich Live display for streaming LLM responses."""

    def __init__(self, console: Console, toolbar: dict | None = None) -> None:
        self._console = console
        self._live: object | None = None
        self._initial_text = ""
        self._buffer = ""
        self._started = False
        self._toolbar = toolbar
        self._activities: list[_ToolActivity] = []
        self._thinking_msg: str = ""
        self._in_tool_phase = False
        self._answer_started = False
        self._footer_msg: str = ""

    def _set_status(self, text: str) -> None:
        if self._toolbar is not None:
            self._toolbar["status"] = text

    def start(self) -> None:
        msg = random.choice(THINKING_MESSAGES)  # noqa: S311
        self._thinking_msg = msg
        self._set_status(msg)
        spinner = Spinner("dots", text=Text(f" {msg}", style="anton.muted"))
        self._live = Live(
            spinner,
            console=self._console,
            refresh_per_second=12,
            transient=True,
        )
        self._live.start()
        self._initial_text = ""
        self._buffer = ""
        self._started = False
        self._activities = []
        self._in_tool_phase = False
        self._answer_started = False
        self._footer_msg = random.choice(WORKING_FOOTER_MESSAGES)  # noqa: S311

    def append_text(self, delta: str) -> None:
        if self._live is None:
            return
        if self._in_tool_phase:
            self._buffer += delta
            self._answer_started = True
        else:
            self._initial_text += delta
        self._started = True
        self._refresh_live()

    def show_tool_result(self, content: str) -> None:
        """Display a tool result (e.g. scratchpad dump) directly to the user."""
        if self._live is None:
            return
        if self._buffer:
            self._buffer += "\n\n"
        self._buffer += content
        self._started = True
        self._refresh_live()

    def show_tool_execution(self, task: str) -> None:
        """Backward-compatible wrapper — delegates to on_tool_use_start."""
        self.on_tool_use_start(f"_compat_{id(task)}", task)

    def on_tool_use_start(self, tool_id: str, name: str) -> None:
        """Track a new tool use and update the live display."""
        if self._live is None:
            return
        self._in_tool_phase = True
        activity = _ToolActivity(tool_id=tool_id, name=name)
        self._activities.append(activity)
        self._refresh_live()

    def on_tool_use_delta(self, tool_id: str, json_delta: str) -> None:
        """Accumulate JSON input deltas for a tool use."""
        for act in self._activities:
            if act.tool_id == tool_id:
                act.json_parts.append(json_delta)
                return

    def on_tool_use_end(self, tool_id: str) -> None:
        """Finalize a tool use: parse accumulated JSON and set description."""
        for act in self._activities:
            if act.tool_id == tool_id:
                raw = "".join(act.json_parts)
                act.description = _tool_display_text(act.name, raw)
                self._refresh_live()
                return

    def update_progress(self, phase: str, message: str, eta: float | None = None) -> None:
        """Update the Live display with agent progress (phase + message + optional ETA)."""
        if self._live is None:
            return

        # For scratchpad streaming, show progress on the activity line itself
        if phase == "scratchpad" and self._activities:
            for act in reversed(self._activities):
                if act.name == "scratchpad":
                    act.current_progress = message
                    break
            self._refresh_live()
            return

        label = PHASE_LABELS.get(phase, phase)
        eta_str = f"  ~{int(eta)}s" if eta else ""
        status = f"{label}  {message}{eta_str}"
        self._set_status(status)
        self._refresh_live()

    def finish(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

        # Eagerly finalize any activities that never got on_tool_use_end
        for act in self._activities:
            if not act.description and act.json_parts:
                raw = "".join(act.json_parts)
                act.description = _tool_display_text(act.name, raw)

        if self._activities:
            # Print initial text as muted "inner speech" (thinking before acting)
            if self._initial_text:
                self._console.print(Text(self._initial_text.rstrip(), style="anton.muted"))
            # Print finalized activity tree
            self._console.print(self._build_activity_tree(final=True))
            # Print answer
            if self._buffer:
                self._console.print(Text("anton> ", style="anton.cyan"), end="")
                self._console.print(Markdown(self._buffer))
        else:
            # No tools — print response normally
            all_text = self._initial_text + self._buffer
            if all_text:
                self._console.print(Text("anton> ", style="anton.cyan"), end="")
                self._console.print(Markdown(all_text))

        self._console.print()

    def abort(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    # --- Private helpers ---

    def _build_activity_tree(self, final: bool = False) -> Text:
        """Render the activity tree as styled Text."""
        lines = Text()
        for i, act in enumerate(self._activities):
            label = act.description or _TOOL_LABELS.get(act.name, act.name)
            if i == 0:
                lines.append("\u23bf ")
            else:
                lines.append("  ")
            lines.append(label, style="bold")
            if act.current_progress:
                lines.append(f" \u2190 {act.current_progress}", style="anton.muted")
            lines.append("\n")
        return lines

    def _refresh_live(self) -> None:
        """Recompose the live display: anton> initial, spinner, tree, answer."""
        if self._live is None:
            return

        parts: list = []

        if self._activities:
            # Show initial text as muted "inner speech" at top
            if self._initial_text:
                parts.append(Text(self._initial_text.rstrip(), style="anton.muted"))

            # Spinner while tools are running (before answer streams)
            if not self._answer_started:
                spinner = Spinner("dots", text=Text(f" {self._thinking_msg}", style="anton.muted"))
                parts.append(spinner)

            # Activity tree
            parts.append(self._build_activity_tree())

            # Answer text below tree
            if self._buffer:
                parts.append(Text())
                parts.append(Markdown(self._buffer))
        elif self._initial_text:
            # Pure text streaming, no tools yet
            parts.append(Markdown(self._initial_text))
        else:
            # Nothing yet — just spinner
            spinner = Spinner("dots", text=Text(f" {self._thinking_msg}", style="anton.muted"))
            parts.append(spinner)

        # Working footer — visible while streaming
        footer = Text(f"\n\u23f5\u23f5 Esc to stop \u2014 {self._footer_msg}", style="#ff69b4")
        parts.append(footer)

        self._live.update(Group(*parts) if len(parts) > 1 else parts[0])
