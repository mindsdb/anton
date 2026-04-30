"""Stream event formatter — converts ChatSession events to Responses API SSE.

Ported from anton_servicesrepo/scratchpad_service/chat_formatter.py, scoped to
the Responses API (the Chat Completions formatter lives there until we unify).
"""

from __future__ import annotations

import json
import time
import uuid
from typing import AsyncIterator

from anton.server.models import (
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
}

PROGRESS_THROTTLE = 0.25  # seconds


async def format_responses_stream(
    event_stream: AsyncIterator,
    model: str = "anton",
    response_id: str | None = None,
    message_id: str | None = None,
    conversation_id: str | None = None,
) -> AsyncIterator[str]:
    """Convert ChatSession.turn_stream() events to Responses API SSE strings.

    Emits typed events:
        response.created, response.in_progress (thoughts/tool activity),
        response.output_text.delta (assistant text), response.completed.
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
            seq += 1
            yield _event("response.in_progress", {
                "type": "response.in_progress",
                "sequence_number": seq,
                "thought_role": role,
                "content": event.name,
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
            seq += 1
            yield _event("response.in_progress", {
                "type": "response.in_progress",
                "sequence_number": seq,
                "thought_role": role,
                "content": accumulated[:2000],
            })

        elif isinstance(event, StreamToolResult):
            seq += 1
            yield _event("response.in_progress", {
                "type": "response.in_progress",
                "sequence_number": seq,
                "thought_role": Role.thought_scratchpad_result.value,
                "content": event.content[:2000],
            })

        elif isinstance(event, StreamTaskProgress):
            now = time.time()
            if now - last_progress >= PROGRESS_THROTTLE:
                last_progress = now
                label = PHASE_LABELS.get(event.phase, event.phase)
                msg = f"{label}: {event.message}" if event.message else label
                seq += 1
                yield _event("response.in_progress", {
                    "type": "response.in_progress",
                    "sequence_number": seq,
                    "thought_role": Role.thought_progress.value,
                    "content": msg,
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
