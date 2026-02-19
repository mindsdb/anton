from __future__ import annotations

import random
from typing import TYPE_CHECKING

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


class StreamDisplay:
    """Manages a Rich Live display for streaming LLM responses."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self._live: object | None = None
        self._buffer = ""
        self._started = False

    def start(self) -> None:
        msg = random.choice(THINKING_MESSAGES)  # noqa: S311
        spinner = Spinner("dots", text=Text(f" {msg}", style="anton.muted"))
        self._live = Live(
            spinner,
            console=self._console,
            refresh_per_second=12,
            transient=True,
        )
        self._live.start()
        self._buffer = ""
        self._started = False

    def append_text(self, delta: str) -> None:
        if self._live is None:
            return
        self._buffer += delta
        self._started = True
        self._live.update(Markdown(self._buffer))

    def show_tool_execution(self, task: str) -> None:
        if self._live is None:
            return
        msg = random.choice(TOOL_MESSAGES)  # noqa: S311
        spinner = Spinner("dots", text=Text(f" {msg}", style="anton.muted"))
        self._live.update(spinner)

    def update_progress(self, phase: str, message: str, eta: float | None = None) -> None:
        """Update the Live display with agent progress (phase + message + optional ETA)."""
        if self._live is None:
            return
        label = PHASE_LABELS.get(phase, phase)
        eta_str = f"  ~{int(eta)}s" if eta else ""
        spinner = Spinner(
            "dots",
            text=Text(f" {label}  {message}{eta_str}", style="anton.muted"),
        )
        self._live.update(spinner)

    def finish(self, input_tokens: int, output_tokens: int, elapsed: float, ttft: float | None) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

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

    def abort(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
