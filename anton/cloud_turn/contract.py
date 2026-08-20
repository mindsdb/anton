"""Wire contract between the scratchpad-controller and the pod entrypoint.

Matches what the controller sends (`scratchpad_controller.anton_turn.request_line`)
and what cowork-server consumes off the reply stream. Intentionally minimal and
data-only: the entrypoint reads ONE newline-terminated JSON line on stdin.

Events written back on stdout (JSONL) are the five the controller translates:
  {"kind": "delta", "text": "..."}   - streamed assistant text, one per chunk
  {"kind": "memory", "entries": [...]}  - pre-terminal; cowork persists these
  {"kind": "skill", "entries": [...]}   - pre-terminal; skill drafts the agent
      built this turn, as [{"slug", "files": {name: text}}]. Staged only: cowork
      surfaces a card and the user saves it, so nothing reaches the skill store.
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
    #: Optional per-turn LLM credential set by cowork ({"provider","api_key","base_url"}).
    #: MVP: always MindsHub (provider="minds-cloud"). None falls back to env settings.
    llm: dict | None = None
    #: Optional memory cowork resolved for this tenant: {"global": {slot: text},
    #: "project": {...}}, slot in profile|rules|lessons. Read-only in the pod.
    memory: dict | None = None
    #: Optional skills cowork resolved for this tenant + project:
    #: {slug: {"files": {relpath: text}}}. Staged read-only in the pod; the pod
    #: never writes skills back (agent-built skills are a desktop draft flow).
    skills: dict | None = None
    #: Optional trace-attribution block cowork resolved for this turn:
    #: ``{"surface": "web", "cowork_server_version": ..., "install_channel": ...}``.
    #: Observability only — nothing here may affect what the turn DOES.
    #:
    #: It has to travel because the pod cannot derive any of it: cowork-server is
    #: not installed in this image, and only the deployment knows which surface it
    #: serves. Without it a web turn is indistinguishable from a desktop one
    #: (ENG-1459), and ENG-1279's server version + install channel are absent too.
    #:
    #: Deliberately one open dict rather than N typed fields: every key otherwise
    #: needs declaring in three repos (cowork-server -> scratchpad-controller's
    #: allowlist -> here), so a block keeps the next key to two.
    trace: dict | None = None

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
            llm=d.get("llm"),
            memory=d.get("memory"),
            skills=d.get("skills"),
            # A controller too old to forward it simply yields None, which reads
            # as "no attribution" rather than failing the turn.
            trace=d.get("trace") if isinstance(d.get("trace"), dict) else None,
        )
