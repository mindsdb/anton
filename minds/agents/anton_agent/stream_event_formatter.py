from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field

from minds.agents.anton_agent.anton.llm.provider import (
    StreamComplete,
    StreamContextCompacted,
    StreamTaskProgress,
    StreamTextDelta,
    StreamToolResult,
    StreamToolUseDelta,
    StreamToolUseEnd,
    StreamToolUseStart,
)
from minds.schemas.chat import Role


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

ANALYZING_MESSAGES = [
    "Analyzing results...",
    "Reading the output...",
    "Digesting the results...",
    "Making sense of the output...",
    "Processing results...",
    "Reviewing the output...",
]

PHASE_LABELS = {
    "memory_recall": "Memory",
    "planning": "Planning",
    "executing": "Executing",
    "complete": "Complete",
    "failed": "Failed",
    "scratchpad": "Scratchpad",
}


class AntonStreamEventFormatter:
    """
    Convert Anton streaming events into (role, content) chunks suitable for MessageStreamer.

    - Answer deltas are emitted as role "assistant".
    - All other updates are emitted as role "system".
    """

    def __init__(
        self,
        *,
        tool_result_max_chars: int | None = None,
        progress_throttle_seconds: float = 0.25,
    ) -> None:
        self._tool_result_max_chars = tool_result_max_chars
        self._progress_throttle_seconds = progress_throttle_seconds

        # We don't render a spinner in the API stream, but we still emit the same
        # random status texts as thought chunks (per StreamDisplay semantics).
        self._started = False
        self._thinking_msg = random.choice(THINKING_MESSAGES)  # noqa: S311

        self._tool_activities: dict[str, _ToolActivity] = {}
        self._tool_order: list[str] = []
        self._last_progress_by_phase: dict[str, str] = {}
        self._last_progress_ts: float = 0.0

    def on_event(self, event) -> list[tuple[str, str]]:
        chunks: list[tuple[str, str]] = []

        # Emit a single initial "thinking" text once.
        if not self._started:
            self._started = True
            chunks.append((Role.system, self._thinking_msg))

        if isinstance(event, StreamTextDelta):
            chunks.append((Role.assistant, event.text))
            return chunks

        if isinstance(event, StreamToolUseStart):
            self._tool_activities[event.id] = _ToolActivity(tool_id=event.id, name=event.name)
            self._tool_order.append(event.id)

            if event.name == "scratchpad":
                chunks.append((Role.thought_scratchpad_start, ""))
            elif event.name == "memorize":
                chunks.append((Role.thought_memorize_start, ""))
            elif event.name == "recall":
                chunks.append((Role.thought_recall_start, ""))
            else:
                chunks.append((Role.system, ""))
            return chunks

        if isinstance(event, StreamToolUseDelta):
            act = self._tool_activities.get(event.id)
            if act is not None:
                act.json_parts.append(event.json_delta)
            return chunks

        if isinstance(event, StreamToolUseEnd):
            act = self._tool_activities.get(event.id)
            if act is None:
                return chunks
            raw = "".join(act.json_parts)
            desc = _tool_display_text(act.name, raw)
            act.description = desc

            if act.name == "scratchpad":
                chunks.append((Role.thought_scratchpad_end, raw))
            elif act.name == "memorize":
                chunks.append((Role.thought_memorize_end, raw))
            elif act.name == "recall":
                chunks.append((Role.thought_recall_end, raw))
            else:
                chunks.append((Role.system, raw))
            return chunks

        if isinstance(event, StreamTaskProgress):
            chunks.extend(self._format_progress(event))
            return chunks

        if isinstance(event, StreamToolResult):
            content = self._truncate(event.content, self._tool_result_max_chars)
            # StreamToolResult are only emitted for scratchpad tools.
            chunks.append((Role.thought_scratchpad_result, content))
            return chunks

        if isinstance(event, StreamContextCompacted):
            # StreamDisplay renders this as Markdown in the answer region.
            chunks.append((Role.thought_context_compacted, event.message))
            return chunks

        if isinstance(event, StreamComplete):
            return chunks

        return chunks

    def _format_progress(self, event: StreamTaskProgress) -> list[tuple[str, str]]:
        # StreamDisplay.update_progress("analyzing") changes the spinner copy.
        # We don't show a spinner, but we still emit the same "analyzing" text.
        if event.phase == "analyzing":
            return [(Role.system, random.choice(ANALYZING_MESSAGES))]  # noqa: S311

        # Avoid spamming the client with rapid progress updates.
        now = time.monotonic()
        if (now - self._last_progress_ts) < self._progress_throttle_seconds:
            last = self._last_progress_by_phase.get(event.phase)
            if last == event.message:
                return []

        self._last_progress_ts = now
        self._last_progress_by_phase[event.phase] = event.message

        # Match StreamDisplay's status formatting (label + double-space + message + optional ETA).
        eta_str = f"  ~{int(event.eta_seconds)}s" if event.eta_seconds else ""
        label = PHASE_LABELS.get(event.phase, event.phase)
        text = f"{label}  {event.message}{eta_str}"

        if event.phase in {"planning", "memory_recall"} or event.phase in {"scratchpad", "scratchpad_start"}:
            role = Role.system
        else:
            role = Role.system

        # For scratchpad streaming, StreamDisplay shows progress on the tool line.
        # Keep it prefix-free for the API stream.
        if event.phase == "scratchpad" and self._tool_order:
            # Use the most recent tool label/description we have.
            last_id = self._tool_order[-1]
            act = self._tool_activities.get(last_id)
            if act is not None:
                base = act.description or _TOOL_LABELS.get(act.name, act.name)
                return [(Role.system, f"{base}: {event.message}")]

        return [(role, text)]

    @staticmethod
    def _truncate(text: str, max_chars: int | None) -> str:
        if max_chars is None:
            return text
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1] + "\u2026"
