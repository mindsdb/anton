"""Wire contract between the scratchpad-controller and the pod entrypoint.

Matches what the controller sends (`scratchpad_controller.anton_turn.request_line`)
and what cowork-server consumes off the reply stream. Intentionally minimal and
data-only: the entrypoint reads ONE newline-terminated JSON line on stdin.

Events written back on stdout (JSONL):
  {"kind": "delta", "text": "..."}   - streamed assistant text, one per chunk
  {"kind": "progress", "phase": "...", "message": "..."}  - phase notice, rate
      limited to one per 250ms except for the phases the entrypoint exempts.
      `phase: "continuation"` is the boundary the completion verifier crosses
      when it forces a continuation: the text after it supersedes the text
      before it, so cowork replaces the answer rather than appending to it.
      Exempt from the rate limit — dropping it reinstates the duplicated
      answer, since the replacement text still arrives either way.
      `phase: "handback"` is its counterpart: the turn is explaining instead
      of delivering that replacement, so the text after it adds to the answer
      rather than replacing it. Also exempt, and for a sharper reason —
      dropping it lets the explanation pass as the replacement, and the answer
      the user already read is lost from the transcript.
  {"kind": "memory", "entries": [...]}  - pre-terminal; cowork persists these
  {"kind": "skill", "entries": [...]}   - pre-terminal; skill drafts the agent
      built this turn, as [{"slug", "files": {name: text}}]. Staged only: cowork
      surfaces a card and the user saves it, so nothing reaches the skill store.
  {"kind": "history", "rows": [...]}  - pre-terminal; this turn's tool
      block-rows as [{"role", "content"}], where content holds only tool_use
      (assistant) or tool_result (user) blocks. cowork persists them as their
      own hidden `messages` rows so the NEXT turn's history replays a valid
      tool_use -> tool_result sequence. Omitted after a mid-turn compaction or
      on failure, in which case that turn replays text-only.
  {"kind": "compaction", "summary": "...", "covered_through": N}  - pre-terminal;
      this turn folded the first N messages of the REQUEST's `history` into
      `summary`. cowork saves both and seeds `summary` + the uncovered tail next
      turn, instead of resending (and re-summarizing) the whole conversation.
      Mutually exclusive with `history` above, which the same compaction
      suppresses. Omitted on failure too: the next turn re-seeds the same
      history and compacts it again.
  {"kind": "turn_completed"}          - terminal success (no payload)
  {"kind": "turn_failed", "error": "..."}  - terminal failure (scrubbed string)

The tool/round step kinds the controller relays unchanged as `turn_step` are
not spelled out above: `tool_start`, `tool_end`, `tool_result`, `compacted`,
`round_end`. A `heartbeat` only resets the controller's stall timer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

TURN_PROTOCOL_VERSION = 1
DATASOURCE_PROTOCOL_VERSION = 1
MAX_DATASOURCE_CONNECTIONS = 100


def _is_version(value: object, expected: int) -> bool:
    """Whether ``value`` is the JSON integer ``expected``, and not merely equal to it.

    `True == 1` and `1.0 == 1`, and `int(1.9)` is 1, so an equality check or a
    coercion reads an envelope that is not this version as if it were, then
    parses it with this version's schema and drops whatever else it carried.
    """
    return not isinstance(value, bool) and isinstance(value, int) and value == expected


@dataclass(frozen=True)
class DatasourceConnectionRefV1:
    connection_id: int
    credential_version: int


@dataclass(frozen=True)
class DatasourceBlockV1:
    protocol_version: int
    connections: tuple[DatasourceConnectionRefV1, ...]


def _parse_datasource_block(value: object) -> DatasourceBlockV1 | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("datasource must be an object")
    if set(value) != {"protocol_version", "connections"}:
        raise ValueError("datasource contains unsupported fields")
    if not _is_version(value.get("protocol_version"), DATASOURCE_PROTOCOL_VERSION):
        raise ValueError("unsupported datasource protocol version")
    connections = value.get("connections")
    if not isinstance(connections, list) or not connections or len(connections) > MAX_DATASOURCE_CONNECTIONS:
        raise ValueError("datasource connections are invalid")
    refs: list[DatasourceConnectionRefV1] = []
    seen: set[int] = set()
    for connection in connections:
        if not isinstance(connection, dict) or set(connection) != {"connection_id", "credential_version"}:
            raise ValueError("datasource connection reference is invalid")
        connection_id = connection.get("connection_id")
        credential_version = connection.get("credential_version")
        if (
            isinstance(connection_id, bool)
            or not isinstance(connection_id, int)
            or connection_id < 1
            or isinstance(credential_version, bool)
            or not isinstance(credential_version, int)
            or credential_version < 1
            or connection_id in seen
        ):
            raise ValueError("datasource connection reference is invalid")
        seen.add(connection_id)
        refs.append(DatasourceConnectionRefV1(connection_id, credential_version))
    return DatasourceBlockV1(DATASOURCE_PROTOCOL_VERSION, tuple(refs))


@dataclass
class TurnRequestV1:
    """One turn to run in the pod. Sent as a single JSON line on stdin."""

    protocol_version: int
    conversation_id: str
    input: str
    correlation_id: str | None = None
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
    #: ``{"surface": "web", "cowork_server_version": ..., "install_channel": ...}``,
    #: plus ``user_id`` / ``organization_id`` (Keycloak UUIDs, ENG-2121) for the
    #: ``turn_completed`` analytics event.
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
    #: Optional turn-key OAuth credential set by cowork-server:
    #: {"turn_key": str, "connections": [{"engine", "name"}, ...]}. Absent or
    #: empty means no connectors for this turn — see cloud_turn/session.py's
    #: build_cloud_chat_session, which is the only place this is read.
    oauth: dict | None = None
    #: Optional verified datasource references. Only IDs and immutable versions
    #: cross the controller/pod boundary; gateway origin and capabilities do not.
    datasource: DatasourceBlockV1 | None = None
    #: Optional ISO 8601 creation time of the conversation, as cowork-server
    #: resolved it. The pod is new every turn and cannot derive this: history
    #: rows carry no timestamps. Without it the session dates the conversation
    #: from the pod's own clock, so the prompt says a three week old
    #: conversation started today and changes at every midnight.
    started_at: str | None = None

    @staticmethod
    def from_json(raw: str) -> "TurnRequestV1":
        d = json.loads(raw)
        protocol_version = d["protocol_version"]
        if not _is_version(protocol_version, TURN_PROTOCOL_VERSION):
            raise ValueError("unsupported cloud turn protocol version")
        return TurnRequestV1(
            protocol_version=protocol_version,
            conversation_id=str(d["conversation_id"]),
            input=str(d["input"]),
            correlation_id=(d.get("correlation_id") if isinstance(d.get("correlation_id"), str) else None),
            workspace_path=d.get("workspace_path"),
            model=d.get("model"),
            history=d.get("history") or [],
            llm=d.get("llm"),
            memory=d.get("memory"),
            skills=d.get("skills"),
            # A controller too old to forward it simply yields None, which reads
            # as "no attribution" rather than failing the turn.
            trace=d.get("trace") if isinstance(d.get("trace"), dict) else None,
            # Same defensive isinstance check as trace, for the same reason.
            oauth=d.get("oauth") if isinstance(d.get("oauth"), dict) else None,
            datasource=_parse_datasource_block(d.get("datasource")),
            # The controller always sends the key and sends None when
            # cowork-server could not resolve it, so the guard does real work.
            started_at=(
                d.get("started_at") if isinstance(d.get("started_at"), str) else None
            ),
        )
