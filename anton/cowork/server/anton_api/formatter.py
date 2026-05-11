"""SSE event formatter — turns ChatSession.turn_stream() events into
OpenAI Responses API SSE strings.

Emits typed events:
    response.created            (with conversation_id)
    response.in_progress        (thought/tool activity, carries thought_role)
    response.output_text.delta  (assistant text deltas)
    response.completed          (final response object)
    response.failed             (error)
"""

from __future__ import annotations

import json
import time
import uuid
from typing import AsyncIterator, Callable, Optional

from .models import (
    ResponseObject,
    ResponseOutput,
    ResponseOutputContent,
    ResponseStatus,
    Role,
)


PHASE_LABELS = {
    "planning": "Planning",
    "analyzing": "Analyzing",
    "executing": "Executing",
    "scratchpad": "Running code",
    "scratchpad_start": "Running code",
    "scratchpad_done": "Code complete",
    "connect_datasource": "Connecting",
    "interactive": "Interactive",
    "context": "Context",
}

PROGRESS_THROTTLE = 0.25  # seconds


async def format_responses_stream(
    event_stream: AsyncIterator,
    model: str = "anton",
    response_id: str | None = None,
    message_id: str | None = None,
    conversation_id: str | None = None,
    event_sink: Optional[Callable[[str, dict], None]] = None,
) -> AsyncIterator[str]:
    """Yield Responses-API SSE strings derived from ChatSession events.

    `event_sink` (optional) is called with `(event_type, payload_dict)` for
    every event before it's serialised to SSE. Used by the responses
    route to capture a per-turn event log to disk so the client can
    rebuild the Thinking block + scratchpad cells when the conversation
    is reopened (without keeping localStorage state).
    """
    from anton.core.llm.provider import (
        StreamComplete,
        StreamContextCompacted,
        StreamTaskProgress,
        StreamTextDelta,
        StreamToolResult,
        StreamToolUseDelta,
        StreamToolUseEnd,
        StreamToolUseStart,
    )

    resp_id = response_id or f"resp-{uuid.uuid4().hex[:12]}"
    msg_id = message_id or f"msg-{uuid.uuid4().hex[:12]}"
    seq = 0
    last_progress = 0.0
    collected_text: list[str] = []

    def _event(event_type: str, data: dict) -> str:
        # Wall-clock millisecond stamp on every event. The renderer
        # uses this (over `Date.now()` at the moment of replay) so
        # historical conversations rebuild correct reasoning /
        # execution durations: synchronous replay through the stream
        # reducer would otherwise see every `now()` collapse to the
        # same JS-tick value, producing 0ms across the board.
        # Persisted into the turns sidecar via `event_sink`, so the
        # field is also there for future replays.
        if "at_ms" not in data:
            data["at_ms"] = int(time.time() * 1000)
        if event_sink is not None:
            try:
                event_sink(event_type, data)
            except Exception:
                # Recording is best-effort — never break the live stream.
                pass
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    tool_json_parts: dict[str, list[str]] = {}
    tool_names: dict[str, str] = {}

    resp = ResponseObject(id=resp_id, model=model, status=ResponseStatus.created)
    seq += 1
    created_data = {
        "type": "response.created",
        "sequence_number": seq,
        "response": resp.model_dump(),
    }
    if conversation_id:
        created_data["conversation_id"] = conversation_id
    yield _event("response.created", created_data)

    async for event in event_stream:
        if isinstance(event, StreamTextDelta):
            collected_text.append(event.text)
            seq += 1
            yield _event("response.output_text.delta", {
                "type": "response.output_text.delta",
                "sequence_number": seq,
                "item_id": msg_id,
                "delta": event.text,
            })

        elif isinstance(event, StreamToolUseStart):
            tool_names[event.id] = event.name
            tool_json_parts[event.id] = []
            if "scratchpad" in event.name:
                role = Role.thought_scratchpad_start.value
            elif "memorize" in event.name:
                role = Role.thought_memorize_start.value
            elif "recall" in event.name:
                role = Role.thought_recall_start.value
            else:
                role = Role.thought_progress.value
            # `tool_use_id` rides along on start/end/result/progress
            # events so the renderer can correlate them. Without it,
            # multi-tool turns (LLM emits start/end for cells A, B, C
            # upfront, then anton dispatches them sequentially) end up
            # patching the wrong step when results arrive — the
            # frontend's "patch the last scratchpad step" heuristic
            # silently misattributes A's output to C.
            seq += 1
            yield _event("response.in_progress", {
                "type": "response.in_progress",
                "sequence_number": seq,
                "thought_role": role,
                "content": event.name,
                "tool_use_id": event.id,
            })

        elif isinstance(event, StreamToolUseDelta):
            if event.id in tool_json_parts:
                tool_json_parts[event.id].append(event.json_delta)

        elif isinstance(event, StreamToolUseEnd):
            name = tool_names.pop(event.id, "")
            parts = tool_json_parts.pop(event.id, [])
            accumulated = "".join(parts)
            if "scratchpad" in name:
                role = Role.thought_scratchpad_end.value
            elif "memorize" in name:
                role = Role.thought_memorize_end.value
            elif "recall" in name:
                role = Role.thought_recall_end.value
            else:
                role = Role.thought_progress.value
            # 64 KB cap — old 2 KB cap routinely chopped scratchpad
            # JSON mid-`code` field, leaving the desktop renderer with
            # an unparseable string and the inspector showing "No code
            # captured for this cell." 64 KB covers every cell we've
            # seen in practice without bloating the SSE stream or the
            # persisted turns log.
            seq += 1
            yield _event("response.in_progress", {
                "type": "response.in_progress",
                "sequence_number": seq,
                "thought_role": role,
                "content": accumulated[:65536],
                "tool_use_id": event.id,
            })

        elif isinstance(event, StreamToolResult):
            seq += 1
            yield _event("response.in_progress", {
                "type": "response.in_progress",
                "sequence_number": seq,
                "thought_role": Role.thought_scratchpad_result.value,
                "content": event.content[:65536],
                "tool_name": getattr(event, "name", "") or "",
                "tool_action": getattr(event, "action", "") or "",
                "tool_use_id": getattr(event, "id", None) or "",
            })

        elif isinstance(event, StreamTaskProgress):
            # scratchpad_start / scratchpad_done phases now carry the
            # source tool_use_id so the renderer correlates them to
            # the right step instead of the last scratchpad step.
            # We DO NOT throttle scratchpad-phase events even when
            # under PROGRESS_THROTTLE — dropping a scratchpad_done
            # would leave the cell stuck in_progress in the UI.
            phase_str = event.phase or ""
            is_scratchpad_phase = phase_str in ("scratchpad_start", "scratchpad_done")
            now = time.time()
            should_emit = is_scratchpad_phase or (now - last_progress >= PROGRESS_THROTTLE)
            if should_emit:
                if not is_scratchpad_phase:
                    last_progress = now
                label = PHASE_LABELS.get(event.phase, event.phase)
                msg = f"{label}: {event.message}" if event.message else label
                seq += 1
                yield _event("response.in_progress", {
                    "type": "response.in_progress",
                    "sequence_number": seq,
                    "thought_role": Role.thought_progress.value,
                    "content": msg,
                    "phase": event.phase,
                    "message": event.message,
                    "eta_seconds": getattr(event, "eta_seconds", None),
                    "tool_use_id": getattr(event, "id", None) or "",
                })

        elif isinstance(event, StreamContextCompacted):
            seq += 1
            yield _event("response.in_progress", {
                "type": "response.in_progress",
                "sequence_number": seq,
                "thought_role": Role.thought_context_compacted.value,
                "content": event.message,
            })

        elif isinstance(event, StreamComplete):
            pass

    full_text = "".join(collected_text)
    resp_completed = ResponseObject(
        id=resp_id,
        model=model,
        status=ResponseStatus.completed,
        output=[ResponseOutput(
            id=msg_id,
            status=ResponseStatus.completed,
            content=[ResponseOutputContent(text=full_text)],
        )],
    )
    seq += 1
    yield _event("response.completed", {
        "type": "response.completed",
        "sequence_number": seq,
        "response": resp_completed.model_dump(),
    })
