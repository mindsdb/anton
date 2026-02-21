from __future__ import annotations

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
    "skill_discovery": "Skills",
    "skill_building": "Building",
    "executing": "Executing",
    "complete": "Complete",
    "failed": "Failed",
}

_TOOL_LABELS: dict[str, str] = {
    "execute_task": "Running task",
    "scratchpad": "Running scratchpad",
    "minds": "Querying data",
    "update_context": "Updating context",
    "request_secret": "Requesting secret",
}


@dataclass
class _Step:
    label: str
    done: bool = False


class StreamDisplay:
    """Manages a Rich Live display for streaming LLM responses."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self._live: object | None = None
        self._buffer = ""
        self._started = False
        self._steps: list[_Step] = []

    def _render(self) -> Group:
        """Build the composite Live display from steps + text buffer."""
        parts: list[Text | Spinner | Markdown] = []
        for step in self._steps:
            if step.done:
                parts.append(Text(f"  ✓ {step.label}", style="anton.muted"))
            else:
                parts.append(
                    Spinner("dots", text=Text(f" {step.label}", style="anton.muted"))
                )
        if self._buffer:
            parts.append(Text(""))  # blank line separator
            parts.append(Markdown(self._buffer))
        return Group(*parts)

    def start(self) -> None:
        msg = random.choice(THINKING_MESSAGES)  # noqa: S311
        self._steps = [_Step(label=msg)]
        self._buffer = ""
        self._started = False
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=12,
            transient=True,
        )
        self._live.start()

    def append_text(self, delta: str) -> None:
        if self._live is None:
            return
        # On first text delta, mark all steps done
        if not self._started:
            for step in self._steps:
                step.done = True
            self._started = True
        self._buffer += delta
        self._live.update(self._render())

    def show_tool_result(self, content: str) -> None:
        """Display a tool result (e.g. scratchpad dump) directly to the user."""
        if self._live is None:
            return
        # Mark all steps done
        for step in self._steps:
            step.done = True
        if self._buffer:
            self._buffer += "\n\n"
        self._buffer += content
        self._started = True
        self._live.update(self._render())

    def show_tool_execution(self, name: str, description: str = "") -> None:
        if self._live is None:
            return
        base_label = _TOOL_LABELS.get(name, name)

        if description:
            # Truncate description to 20 words
            words = description.split()
            if len(words) > 20:
                description = " ".join(words[:20]) + "..."
            full_label = f"{base_label} — {description}"
            # Update the current active step's label (set by the earlier no-desc event)
            for step in self._steps:
                if not step.done:
                    step.label = full_label
                    self._live.update(self._render())
                    return
            # No active step to update — add new one
            self._steps.append(_Step(label=full_label))
        else:
            # No description — mark previous active step done, add new
            for step in self._steps:
                if not step.done:
                    step.done = True
            self._steps.append(_Step(label=base_label))

        self._live.update(self._render())

    def update_progress(self, phase: str, message: str, eta: float | None = None) -> None:
        """Update the Live display with agent progress (phase + message + optional ETA)."""
        if self._live is None:
            return
        label = PHASE_LABELS.get(phase, phase)
        eta_str = f"  ~{int(eta)}s" if eta else ""
        full_label = f"{label}  {message}{eta_str}"
        # Truncate to 20 words
        words = full_label.split()
        if len(words) > 20:
            full_label = " ".join(words[:20]) + "..."
        # Update current active step's label
        for step in self._steps:
            if not step.done:
                step.label = full_label
                break
        self._live.update(self._render())

    def finish(self, input_tokens: int, output_tokens: int, elapsed: float, ttft: float | None) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

        # Print finalized step list if there was more than one step
        if len(self._steps) > 1:
            for step in self._steps:
                self._console.print(Text(f"  ✓ {step.label}", style="anton.muted"))
            self._console.print()

        # Print final rendered response
        if self._buffer:
            self._console.print(Text("anton> ", style="anton.cyan"), end="")
            self._console.print(Markdown(self._buffer))

        # Stats line
        parts = [f"{input_tokens} in / {output_tokens} out", f"{elapsed:.1f}s"]
        if ttft is not None:
            parts.append(f"TTFT {int(ttft * 1000)}ms")
        stats = "  ".join(parts)
        self._console.print(Text(stats, style="anton.muted"))
        self._console.print()
        self._console.print()

    def abort(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
