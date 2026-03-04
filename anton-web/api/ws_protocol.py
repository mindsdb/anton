"""Translate internal StreamEvent types into the simplified WebSocket JSON protocol."""

from __future__ import annotations

import re
from pathlib import Path

from anton.llm.provider import (
    StreamComplete,
    StreamContextCompacted,
    StreamEvent,
    StreamTaskProgress,
    StreamTextDelta,
    StreamToolResult,
    StreamToolUseDelta,
    StreamToolUseEnd,
    StreamToolUseStart,
)

_FILE_PATTERN = re.compile(r"(?:^|[\s\"'])(/[\w./ -]+\.(?:html|png|jpg|jpeg|svg|csv|xlsx))", re.MULTILINE)


def event_to_dict(event: StreamEvent) -> dict | None:
    """Convert a StreamEvent to a JSON-serializable dict for the frontend.

    Returns None for events that should be suppressed (tool internals).
    """
    if isinstance(event, StreamTextDelta):
        return {"type": "text_delta", "text": event.text}

    if isinstance(event, StreamTaskProgress):
        return {"type": "status", "message": event.message}

    if isinstance(event, StreamToolUseStart):
        return {"type": "status", "message": f"Working ({event.name})..."}

    if isinstance(event, (StreamToolUseEnd, StreamToolUseDelta)):
        return None

    if isinstance(event, StreamToolResult):
        outputs = _detect_outputs(event.content)
        if outputs:
            return {"type": "outputs", "files": outputs}
        return None

    if isinstance(event, StreamComplete):
        usage = event.response.usage
        return {
            "type": "complete",
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            },
        }

    if isinstance(event, StreamContextCompacted):
        return {"type": "context_compacted", "message": event.message}

    return None


def _detect_outputs(content: str) -> list[dict]:
    """Scan tool result text for file paths that look like renderable outputs."""
    outputs: list[dict] = []
    for match in _FILE_PATTERN.finditer(content):
        filepath = match.group(1)
        p = Path(filepath)
        if not p.is_file():
            continue
        ext = p.suffix.lower().lstrip(".")
        if ext in ("html", "htm"):
            kind = "html"
        elif ext in ("png", "jpg", "jpeg", "svg", "gif", "webp"):
            kind = "image"
        else:
            kind = "file"
        url = f"/api/outputs/{p.name}"
        outputs.append({"kind": kind, "path": filepath, "url": url})
    return outputs
