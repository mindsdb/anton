from __future__ import annotations

import asyncio
import json
import os
import random
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

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
    printed: bool = False  # whether the activity line has been printed
    done: bool = False  # whether execution is complete


_TOOL_LABELS: dict[str, str] = {
    "scratchpad": "Scratchpad",
    "memorize": "Memory",
    "recall": "Recall",
}

_MAX_DESC = 60

_REFRESH_FPS = 6

# Max chars of orchestrator text to show in the spinner
_MAX_THOUGHT_LEN = 80


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
            desc = desc[: _MAX_DESC - 1] + "\u2026"
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
        self._buffer = ""  # answer text accumulated during streaming
        self._in_tool_phase = False
        self._last_was_tool = False
        self._initial_text = ""
        self._initial_printed = False
        self._active = False
        # 3-line footer state
        self._line1_fun: str = ""  # Line 1: Esc to cancel — fun message
        self._line2_status: str = ""  # Line 2: ⠸ what's happening now
        self._line3_peek: str = ""  # Line 3: ↳ live peek at output
        self._cancel_msg: str = ""

    def _set_status(self, text: str) -> None:
        if self._toolbar is not None:
            self._toolbar["status"] = text

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

    def _build_spinner_display(self) -> object:
        """Build the 3-line footer display.

        Line 1: ⏵⏵ Esc to cancel — fun message
        Line 2: ⠸ status / what's happening now
        Line 3:   ↳ live peek at streaming output
        """
        from rich.console import Group

        # Line 1: spinner + status
        spinner = Spinner(
            "dots", text=Text(f" {self._line2_status}", style="anton.muted")
        )

        # Line 2: peek (only if there's something to peek at)
        parts: list = [spinner]
        if self._line3_peek:
            line3 = Text()
            line3.append("  \u21b3 ", style="anton.muted")
            line3.append(self._line3_peek, style="dim")
            parts.append(line3)

        # Line 3: control + personality (at the bottom)
        cancel_line = Text()
        if self._cancel_msg:
            cancel_line.append(f"\u23f5\u23f5 {self._cancel_msg}", style="#ff69b4")
        else:
            cancel_line.append(
                f"\u23f5\u23f5 Esc to cancel \u2014 {self._line1_fun}", style="#ff69b4"
            )
        parts.append(cancel_line)

        return Group(*parts)

    def start(self) -> None:
        self._line1_fun = random.choice(THINKING_MESSAGES)  # noqa: S311
        self._line2_status = random.choice(WORKING_FOOTER_MESSAGES)  # noqa: S311
        self._line3_peek = ""
        self._set_status(self._line1_fun)
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
            self._line3_peek = self._extract_peek(self._buffer)
            self._update_spinner()
        else:
            self._initial_text += delta
            self._line3_peek = self._extract_peek(self._initial_text)
            self._update_spinner()

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
        """Track a new tool use."""
        if not self._active:
            return
        self._in_tool_phase = True
        self._last_was_tool = True
        activity = _ToolActivity(tool_id=tool_id, name=name)
        self._activities.append(activity)

    def on_tool_use_delta(self, tool_id: str, json_delta: str) -> None:
        """Accumulate JSON input deltas for a tool use."""
        for act in self._activities:
            if act.tool_id == tool_id:
                act.json_parts.append(json_delta)
                return

    def on_tool_use_end(self, tool_id: str) -> None:
        """Finalize a tool use description. Print non-scratchpad tools immediately.

        Scratchpad lines are deferred to scratchpad_start which has the ETA.
        """
        for act in self._activities:
            if act.tool_id == tool_id:
                raw = "".join(act.json_parts)
                act.description = _tool_display_text(act.name, raw)
                if act.name != "scratchpad":
                    self._stop_spinner()
                    self._print_activity_line(act)
                    act.printed = True
                    self._start_spinner()
                return

    def update_progress(
        self, phase: str, message: str, eta: float | None = None
    ) -> None:
        """Update progress — manages spinner and activity lines."""
        if not self._active:
            return

        if phase == "analyzing":
            self._line1_fun = random.choice(ANALYZING_MESSAGES)  # noqa: S311
            self._line2_status = "Composing response..."
            self._line3_peek = ""
            self._update_spinner()
            return

        if phase == "scratchpad_start":
            # Print the scratchpad activity line NOW (before execution)
            for act in reversed(self._activities):
                if act.name == "scratchpad" and not act.printed:
                    if eta:
                        act.eta_str = f"~{int(eta)}s"
                    self._stop_spinner()
                    self._print_activity_line(act)
                    act.printed = True
                    self._line2_status = act.description
                    self._line3_peek = ""
                    self._start_spinner()
                    break
            return

        if phase == "scratchpad_done":
            # Mark the scratchpad line as complete with actual elapsed time
            for act in reversed(self._activities):
                if act.name == "scratchpad" and act.printed and not act.done:
                    elapsed = eta if eta else 0  # eta_seconds carries elapsed time
                    act.done = True
                    self._stop_spinner()
                    self._print_done_line(act, elapsed)
                    self._line1_fun = random.choice(THINKING_MESSAGES)  # noqa: S311
                    self._line2_status = random.choice(WORKING_FOOTER_MESSAGES)  # noqa: S311
                    self._line3_peek = ""
                    self._start_spinner()
                    break
            return

        if phase == "scratchpad" and self._activities:
            for act in reversed(self._activities):
                if act.name == "scratchpad":
                    act.current_progress = message
                    break
            self._line3_peek = message
            self._update_spinner()
            return

        if phase in ("connect_datasource", "interactive"):
            # Interactive tool — stop spinner so user can see and type
            self._stop_spinner()
            return

        label = PHASE_LABELS.get(phase, phase)
        eta_str = f"  ~{int(eta)}s" if eta else ""
        self._line2_status = f"{label}  {message}{eta_str}"
        self._set_status(self._line2_status)
        self._update_spinner()

    def finish(self) -> None:
        """Stop spinner and print the final answer."""
        self._stop_spinner()

        # Print initial text as muted "inner speech" (if not already printed)
        if self._initial_text and not self._initial_printed:
            if self._activities:
                self._console.print(
                    Text(self._initial_text.rstrip(), style="anton.muted")
                )

        # Print answer
        if self._activities:
            if self._buffer:
                self._console.print(Text("anton> ", style="anton.prompt"), end="")
                self._console.print(Markdown(self._buffer))
        else:
            all_text = self._initial_text + self._buffer
            if all_text:
                self._console.print(Text("anton> ", style="anton.prompt"), end="")
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
        """Update line 1 to acknowledge that cancellation is in progress."""
        self._cancel_msg = random.choice(CANCEL_MESSAGES)  # noqa: S311
        self._line2_status = "Stopping..."
        self._line3_peek = ""
        self._update_spinner()

    def _extract_peek(self, text: str) -> str:
        """Extract the last meaningful line from streaming text for the peek line."""
        lines = text.rstrip().splitlines()
        if not lines:
            return ""
        last = lines[-1].strip()
        if not last:
            return ""
        # Strip markdown formatting
        for ch in ("#", "*", "-", ">", "`"):
            last = last.lstrip(ch).strip()
        if len(last) > _MAX_THOUGHT_LEN:
            last = last[: _MAX_THOUGHT_LEN - 1] + "\u2026"
        return last

    def _print_activity_line(self, act: _ToolActivity) -> None:
        """Print a single activity line permanently (before execution)."""
        line = Text()
        label = act.description or _TOOL_LABELS.get(act.name, act.name)
        prefix = "\u23bf " if act is self._activities[0] else "  "
        line.append(prefix)
        line.append(label, style="bold")
        if act.eta_str:
            line.append(f" {act.eta_str}", style="anton.muted")
        self._console.print(line)

    def _print_done_line(self, act: _ToolActivity, elapsed: float) -> None:
        """Print a completion marker for a finished activity."""
        line = Text()
        line.append("  \u2714 ", style="green")
        elapsed_str = f"{elapsed:.1f}s" if elapsed >= 1 else f"{int(elapsed * 1000)}ms"
        line.append(elapsed_str, style="anton.muted")
        self._console.print(line)


class EscapeWatcher:
    """Detect Escape keypress during streaming via cbreak terminal mode."""

    def __init__(self, on_cancel: Callable[[], None] | None = None) -> None:
        self.cancelled = asyncio.Event()
        self._on_cancel = on_cancel
        self._task: asyncio.Task | None = None
        self._old_settings: list | None = None
        self._stop = False
        self._paused = False

    async def __aenter__(self) -> EscapeWatcher:
        if sys.platform != "win32" and sys.stdin.isatty():
            self._task = asyncio.create_task(self._watch())
        return self

    def pause(self) -> None:
        """Temporarily restore normal terminal mode for interactive prompts."""
        if self._paused or self._old_settings is None:
            return
        import termios
        fd = sys.stdin.fileno()
        termios.tcsetattr(fd, termios.TCSADRAIN, self._old_settings)
        self._paused = True

    def resume(self) -> None:
        """Re-enter cbreak mode after interactive prompts."""
        if not self._paused:
            return
        import tty
        fd = sys.stdin.fileno()
        tty.setcbreak(fd)
        self._paused = False

    async def __aexit__(self, *exc: object) -> None:
        self._stop = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._drain_stdin()

    @staticmethod
    def _drain_stdin() -> None:
        if sys.platform == "win32":
            return
        import fcntl

        fd = sys.stdin.fileno()
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        try:
            while True:
                try:
                    if not os.read(fd, 1024):
                        break
                except BlockingIOError:
                    break
        finally:
            fcntl.fcntl(fd, fcntl.F_SETFL, flags)

    async def _watch(self) -> None:
        if sys.platform == "win32":
            return
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            loop = asyncio.get_running_loop()
            while not self._stop:
                if self._paused:
                    await asyncio.sleep(0.1)
                    continue
                ready = await loop.run_in_executor(
                    None, lambda: select.select([fd], [], [], 0.1)[0]
                )
                if not ready:
                    continue
                ch = os.read(fd, 1)
                if ch == b"\x1b":
                    followup = await loop.run_in_executor(
                        None, lambda: select.select([fd], [], [], 0.05)[0]
                    )
                    if followup:
                        os.read(fd, 32)
                        continue
                    if self._on_cancel is not None:
                        self._on_cancel()
                    self.cancelled.set()
                    return
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, self._old_settings)


class ClosingSpinner:
    """Animated spinner shown while scratchpad processes are being killed."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self._live: object | None = None

    def start(self) -> None:
        spinner = Spinner(
            "dots", text=Text(" Closing scratchpad processes…", style="anton.muted")
        )
        self._live = Live(
            spinner, console=self._console, refresh_per_second=6, transient=True
        )
        self._live.start()

    def stop(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
