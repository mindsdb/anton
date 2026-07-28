"""Wire contract between the scratchpad-controller and the pod entrypoint.

Matches what the controller sends (`scratchpad_controller.anton_turn.request_line`)
and what cowork-server consumes off the reply stream. Intentionally minimal and
data-only: the entrypoint reads ONE newline-terminated JSON line on stdin.

Events written back on stdout (JSONL) are the three the controller translates:
  {"kind": "delta", "text": "..."}   - streamed assistant text, one per chunk
  {"kind": "turn_completed"}          - terminal success (no payload)
  {"kind": "turn_failed", "error": "..."}  - terminal failure (scrubbed string)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class TurnRequestV1:
    """One turn to run in the pod. Sent as a single JSON line on stdin."""

    protocol_version: int
    conversation_id: str
    input: str
    #: Mount path the controller passes; the pod uses its own trusted mount and
    #: does not act on this value (kept for wire-compatibility). See session.py.
    workspace_path: str | None = None
    #: Optional model override; None uses the settings default.
    model: str | None = None
    #: DB-authoritative ordered history ({"role","content"} dicts). The pod never
    #: loads its own history; cowork-server owns persistence.
    history: list = field(default_factory=list)

    @staticmethod
    def from_json(raw: str) -> "TurnRequestV1":
        d = json.loads(raw)
        return TurnRequestV1(
            protocol_version=int(d["protocol_version"]),
            conversation_id=str(d["conversation_id"]),
            input=str(d["input"]),
            workspace_path=d.get("workspace_path"),
            model=d.get("model"),
            history=d.get("history") or [],
        )
