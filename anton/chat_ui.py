from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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
    eta_str: str = ""


_TOOL_LABELS: dict[str, str] = {
    "scratchpad": "Scratchpad",
    "memorize": "Memory",
    "recall": "Recall",
}

_MAX_DESC = 60

_REFRESH_FPS = 6


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
    elif name == "memorize":
        entries = data.get("entries", [])
        desc = f"{len(entries)} entry/entries"
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

ANALYZING_MESSAGES = [
    "Analyzing results...",
    "Reading the output...",
    "Digesting the results...",
    "Making sense of the output...",
    "Processing results...",
    "Reviewing the output...",
]

CANCEL_MESSAGES = [
    "Ok, dropping everything\u2026",
    "Alright, pulling the plug\u2026",
    "Stopping the presses\u2026",
    "Hitting the brakes\u2026",
    "Winding down\u2026",
    "Wrapping it up\u2026",
    "Ok, letting go of this one\u2026",
    "Understood, shutting it down\u2026",
    "Copy that, standing down\u2026",
    "Roger, aborting mission\u2026",
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
    """Manages streaming LLM output with permanent prints and a tiny Live spinner.

    Content is printed permanently (scrollable) as it arrives.
    Live is used ONLY for a small spinner+footer at the bottom (1-2 lines),
    so transient cleanup always works regardless of terminal emulator.
    """

    def __init__(self, console: Console, toolbar: dict | None = None) -> None:
        self._console = console
        self._live: Live | None = None
        self._toolbar = toolbar
        self._activities: list[_ToolActivity] = []
        self._thinking_msg: str = ""
        self._footer_msg: str = ""
        self._cancel_msg: str = ""
        self._buffer = ""  # answer text accumulated during streaming
        self._in_tool_phase = False
        self._last_was_tool = False
        self._initial_text = ""
        self._initial_printed = False
        self._active = False

    def _set_status(self, text: str) -> None:
        if self._toolbar is not None:
            self._toolbar["status"] = text

    # --- Tiny Live: just spinner + footer (1-2 lines) ---

    def _start_spinner(self, text: str | None = None) -> None:
        """Start or restart the tiny spinner Live."""
        self._stop_spinner()
        self._live = Live(
            self._build_spinner_display(),
            console=self._console,
            refresh_per_second=_REFRESH_FPS,
            transient=True,
        )
        self._live.start()

    def _stop_spinner(self) -> None:
        """Stop the spinner (transient=True clears it — always safe, it's tiny)."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _update_spinner(self) -> None:
        """Update the spinner display."""
        if self._live is not None:
            self._live.update(self._build_spinner_display())

    def _build_spinner_display(self) -> Spinner:
        """Build the spinner with optional footer text."""
        if self._cancel_msg:
            footer = f"  \u23f5\u23f5 {self._cancel_msg}"
        else:
            footer = f"  \u23f5\u23f5 Esc to stop \u2014 {self._footer_msg}"
        spinner = Spinner(
            "dots",
            text=Text(f" {self._thinking_msg}  ", style="anton.muted")
            + Text(footer, style="#ff69b4"),
        )
        return spinner

    # --- Public API ---

    def start(self) -> None:
        self._thinking_msg = random.choice(THINKING_MESSAGES)  # noqa: S311
        self._footer_msg = random.choice(WORKING_FOOTER_MESSAGES)  # noqa: S311
        self._set_status(self._thinking_msg)
        self._activities = []
        self._buffer = ""
        self._initial_text = ""
        self._initial_printed = False
        self._in_tool_phase = False
        self._last_was_tool = False
        self._cancel_msg = ""
        self._active = True
        self._start_spinner()

    def append_text(self, delta: str) -> None:
        if not self._active:
            return
        if self._in_tool_phase:
            self._buffer += delta
            self._last_was_tool = False
        else:
            self._initial_text += delta

    def show_tool_result(self, content: str) -> None:
        """Print a tool result permanently (immediately scrollable)."""
        if not self._active:
            return
        self._stop_spinner()
        self._console.print(Markdown(content))
        self._last_was_tool = True
        self._start_spinner()

    def show_tool_execution(self, task: str) -> None:
        """Backward-compatible wrapper — delegates to on_tool_use_start."""
        self.on_tool_use_start(f"_compat_{id(task)}", task)

    def on_tool_use_start(self, tool_id: str, name: str) -> None:
        """Track a new tool use and print the activity line."""
        if not self._active:
            return
        self._in_tool_phase = True
        self._last_was_tool = True
        activity = _ToolActivity(tool_id=tool_id, name=name)
        self._activities.append(activity)
        # Activity line will be printed when we know the description (on_tool_use_end)

    def on_tool_use_delta(self, tool_id: str, json_delta: str) -> None:
        """Accumulate JSON input deltas for a tool use."""
        for act in self._activities:
            if act.tool_id == tool_id:
                act.json_parts.append(json_delta)
                return

    def on_tool_use_end(self, tool_id: str) -> None:
        """Finalize a tool use and print its activity line permanently."""
        for act in self._activities:
            if act.tool_id == tool_id:
                raw = "".join(act.json_parts)
                act.description = _tool_display_text(act.name, raw)
                # Scratchpad lines are printed by scratchpad_start (which has the ETA)
                if act.name != "scratchpad":
                    self._stop_spinner()
                    self._print_activity_line(act)
                    self._start_spinner()
                return

    def update_progress(self, phase: str, message: str, eta: float | None = None) -> None:
        """Update progress — just changes the spinner text."""
        if not self._active:
            return

        if phase == "analyzing":
            self._thinking_msg = random.choice(ANALYZING_MESSAGES)  # noqa: S311
            self._update_spinner()
            return

        if phase == "scratchpad_start" and self._activities:
            for act in reversed(self._activities):
                if act.name == "scratchpad":
                    act.description = _tool_display_text(act.name, "".join(act.json_parts)) or f"Scratchpad({message})"
                    if eta:
                        act.eta_str = f"~{int(eta)}s"
                    # Print/update the activity line
                    self._stop_spinner()
                    self._print_activity_line(act)
                    self._start_spinner()
                    break
            return

        if phase == "scratchpad" and self._activities:
            for act in reversed(self._activities):
                if act.name == "scratchpad":
                    act.current_progress = message
                    break
            self._thinking_msg = message
            self._update_spinner()
            return

        label = PHASE_LABELS.get(phase, phase)
        eta_str = f"  ~{int(eta)}s" if eta else ""
        self._thinking_msg = f"{label}  {message}{eta_str}"
        self._set_status(self._thinking_msg)
        self._update_spinner()

    def finish(self) -> None:
        """Stop spinner and print the final answer."""
        self._stop_spinner()

        # Print initial text as muted "inner speech" (if not already printed)
        if self._initial_text and not self._initial_printed:
            if self._activities:
                self._console.print(Text(self._initial_text.rstrip(), style="anton.muted"))
            # If no activities, initial text IS the answer — handled below

        # Print answer
        if self._activities:
            if self._buffer:
                self._console.print(Text("anton> ", style="anton.cyan"), end="")
                self._console.print(Markdown(self._buffer))
        else:
            all_text = self._initial_text + self._buffer
            if all_text:
                self._console.print(Text("anton> ", style="anton.cyan"), end="")
                self._console.print(Markdown(all_text))

        self._active = False
        self._console.print()

    def abort(self) -> None:
        self._stop_spinner()
        self._active = False

    def show_context_compacted(self, message: str) -> None:
        """Show a notification that context was compacted."""
        if not self._active:
            return
        self._stop_spinner()
        self._console.print(Text(f"> {message}", style="anton.muted"))
        self._start_spinner()

    def show_cancelling(self) -> None:
        """Update the footer to acknowledge that cancellation is in progress."""
        self._cancel_msg = random.choice(CANCEL_MESSAGES)  # noqa: S311
        self._update_spinner()

    # --- Private helpers ---

    def _print_activity_line(self, act: _ToolActivity) -> None:
        """Print a single activity line permanently."""
        line = Text()
        label = act.description or _TOOL_LABELS.get(act.name, act.name)
        prefix = "\u23bf " if act is self._activities[0] else "  "
        line.append(prefix)
        line.append(label, style="bold")
        if act.eta_str:
            line.append(f" {act.eta_str}", style="anton.muted")
        self._console.print(line)
