from __future__ import annotations
import json
from dataclasses import dataclass, asdict


@dataclass
class TurnRequestV1:
    protocol_version: int
    conversation_id: str
    input: str
    workspace_path: str | None = None
    model: str | None = None

    @staticmethod
    def from_json(raw: str) -> "TurnRequestV1":
        d = json.loads(raw)
        return TurnRequestV1(
            protocol_version=int(d["protocol_version"]),
            conversation_id=str(d["conversation_id"]),
            input=str(d["input"]),
            workspace_path=d.get("workspace_path"),
            model=d.get("model"),
        )


@dataclass
class TurnResultV1:
    protocol_version: int
    kind: str            # "turn_completed" | "turn_failed"
    final_text: str | None = None
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))
