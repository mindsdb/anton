"""Wire contract between the scratchpad-controller and the pod entrypoint.

Matches what the controller sends (`scratchpad_controller.anton_turn.request_line`)
and what cowork-server consumes off the reply stream. Intentionally minimal and
data-only: the entrypoint reads the request as the first newline-terminated JSON
line on stdin. An interactive turn keeps stdin open and reads one answer line
per user answer after it (see below); the controller never closes stdin on the
live-pod path, and its dev subprocess runner closes it only for
non-interactive turns.

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
      `phase: "tool_progress"` is a streaming tool's step line, `id` the
      tool_use id it belongs to; lines with an id are never rate limited (the
      first creates the step a consumer renders, each one is a step the user
      must see; a tool emits a handful per run), `phase: "tool_done"` closes
      that step and carries `ok` and `eta_seconds`.
      `phase: "tool_peek"` — the live tail of what a streaming tool is
      writing, each one replacing the last, an empty `message` clearing it —
      does NOT appear on this wire: `ToolRegistry.dispatch_tool` relays it
      only to a host that declared `ChatSessionConfig.live_tool_peek`, and
      the pod does not. A consumer that meets it anyway (an older pod, a
      host that opted in later) should drop it unless it renders a live tail.
      `phase: "model_wait"` says a model call is open and has sent nothing
      the wire carried: `message` reads "Waiting for the model (2m 40s)", or
      "The model is still writing (2m 40s)" when the call produced output the
      wire does not carry, and `eta_seconds` is how long the call has been
      open; `id` and `ok` are null. The heartbeat task writes it outside the
      rate limit, at most once per 20 s of otherwise quiet wire, and only
      while the session reports a call waiting on the provider. A tool, a
      cell or an open question reports none, so they still go quiet. A
      consumer shows it as a still-working line and does not save it.
  {"kind": "memory", "entries": [...]}  - pre-terminal; cowork persists these
  {"kind": "skill", "entries": [...]}   - pre-terminal; skill drafts the agent
      built this turn, as [{"slug", "files": {name: text}}]. Staged only: cowork
      surfaces a card and the user saves it, so nothing reaches the skill store.
  {"kind": "connector", "entries": [...]}  - pre-terminal; connector templates
      the agent built this turn with `create_connector_draft`, at most 3, each
      as `ConnectorDraft.to_dict()`. Sent only when the request set
      `connector_drafts`. Staged only: cowork shows a card the user saves.
  {"kind": "history", "rows": [...]}  - pre-terminal; this turn's tool
      block-rows as [{"role", "content"}], where content holds only tool_use
      (assistant) or tool_result (user) blocks. For a turn that did not
      compact, cowork persists them as their own hidden `messages` rows so the
      NEXT turn's history replays a valid tool_use -> tool_result sequence.
      Sent after a mid-turn compaction or hard truncation too: the rows come
      from `ChatSession.turn_messages()`, which still holds the messages the
      swap folded out of history. For such a turn, cowork-server since
      mindsdb/cowork-server#712 reads the rows for figure sources and stores
      none, so the next turn replays that turn text-only; older servers store
      them. Omitted when the turn used no tools, and on failure, in which case
      that turn replays text-only.
  {"kind": "compaction", "summary": "...", "covered_through": N}  - pre-terminal;
      this turn folded the first N messages of the REQUEST's `history` into
      `summary`. cowork saves both and seeds `summary` + the uncovered tail next
      turn, instead of resending (and re-summarizing) the whole conversation.
      Sent before `history` above when both apply. `covered_through` counts
      only seeded messages, never this turn's own. The summary text may still
      describe rows this turn also sends in full under `history`. Omitted on
      failure too: the next turn re-seeds the same history and compacts it
      again.
  {"kind": "ask_user", "id", "prompt", "options": [{"value", "label",
      "detail"}], "select", "allow_custom", "timeout_s"}  - interactive turns
      only: a question the turn is now blocked on. Never rate limited.
  {"kind": "ask_user_answered", "id", "status", "values", "text",
      "answer_id"}  - retires that question. `answer_id` names the stdin answer
      that closed it (a Skip too, with status "cancelled"); null on timeout or
      error.
  {"kind": "ask_user_answer_rejected", "id", "answer_id", "reason"}  - a stdin
      answer that did not close its question; reason is invalid_option |
      not_found | already_answered. The question stays open on invalid_option.
  {"kind": "addon", "type": "response.addon", "addon_id", "state", ...}  - what
      happened to a message the user added mid-task (stdin `addon` lines
      below); any turn, never rate limited. `addon_id` is the stdin line's.
      The controller relays it as a `turn_step` (step "addon"), the way it
      relays the ask_user kinds; an older controller drops it.
      {"kind": "addon", "type": "response.addon", "state": "ready"}  - exactly
      this, no addon_id: emitted once at turn start, before any model call,
      and only when the stdin addon reader is active. From then on an
      `addon` line reaches the running turn; before it, one is refused.
      state "added" | "refused" (+ "text", "decided_by"): the verdict on an
      `addon` line. Refused when no turn is running yet or any more (a line
      that lands before the session is built is refused too, not buffered),
      or past the turn's limits: 20 added messages, 4000 characters each,
      64 KB in all. cowork-server then keeps the message as the next task.
      A repeat of a refused `addon_id` is refused again.
      state "read" (+ "text", "decided_by"): a model call that included it
      has completed. One that fails, or a turn cancelled first, leaves it
      unread, so it comes back as "returned".
      state "returned" (+ "text", "decided_by"): the turn ended before the
      model saw it; sent before the terminal event on every path (completed,
      failed, interrupted, ended on an unanswered question), so the host can
      run it as the next task.
      state "withdrawn" | "already_read" | "returned" (+ "request_id"): the
      verdict on an `addon_withdraw` line. "returned" when the message was
      already returned (it is the next task); "already_read" otherwise,
      including an unknown or refused id.
      An `addon` line repeating a known `addon_id` re-reports that message's
      current state rather than adding it twice.
  {"kind": "browser", "session_id": "..."}  - the browser tool opened (or
      reopened) the person's browser session. cowork shows the browser beside
      the chat, with a viewer link it mints with the person's own credential.
      The line carries no link and no key.
  {"kind": "turn_completed", "artifacts_touched": [slug, ...]}  - terminal
      success. `artifacts_touched` (ARTIFACTS_TOUCHED_KEY) is the sorted
      folder names of the artifacts this turn created or updated through its
      artifact tools, or edited by a scratchpad cell (the session's
      `artifacts_touched`); `[]` means the turn touched none. The list is what
      the tools and cells reported, not every write: a process a cell started
      that writes into an artifact after the cell returns goes unlisted.
      cowork-server reads it in `tracked_artifacts_from_reply` and then opens
      only the listed folders at turn end, plus the folders it queued a repair
      in and its `.locks` folder. The key is left out, and the host then
      diffs the whole artifacts folder, when the session keeps no such
      record, when a snapshot around a cell could not read the artifacts
      folder or an entry in it (`ChatSession.artifacts_touched_complete`),
      and when the list would name more than MAX_ARTIFACTS_TOUCHED (1,000)
      folders.
  {"kind": "turn_failed", "error": "..."}  - terminal failure (scrubbed string)
      `error` is "TypeName: message", so a consumer reads the failure's kind
      off the class name. A MindsHub billing stop that knows when its limit
      lifts adds `"reset_at": "<ISO-8601 instant with a UTC offset>"`; every
      other failure omits the key.

The tool/round step kinds the controller relays unchanged as `turn_step` are
not spelled out above: `tool_start`, `tool_end`, `tool_result`, `compacted`,
`round_end`, and `turn_ended` ({"ended_by": "..."}, sent before the terminal
event when a turn handed back early). A `heartbeat` only resets the
controller's stall timer.
A `tool_result` with `action: "message"` is a tool's message to the user,
rendered as an agent message; the pod sends it only when the request set
`tool_messages`.
A request with `jev_decisions` set to false never asks Jev, MindsHub's
decision model, whether the agent finished: the organization opted out.
A request with `ui_locale` (a tag in `anton.core.llm.ui_locale.UI_LOCALE_NAMES`:
en, es, pt-BR, de, fr, it, ja, ko, zh-CN) tells the agent which
language the user's app is set to (ENG-4149); absent or unsupported adds nothing.

Answer lines on stdin (interactive turns only), one JSON object per line,
at most 64 KB:
  {"kind": "answer", "question_id", "answer_id", "values": [...], "text": "...",
   "skipped": false}

Addon lines on stdin (any turn, same 64 KB bound), ENG-4194:
  {"kind": "addon", "addon_id", "text", "decided_by"}  - hand the running turn
      a message (ChatSession.add_user_input); answered by an `addon` event
      with state "added" or "refused".
  {"kind": "addon_withdraw", "addon_id", "request_id"}  - take back one the
      model has not seen (ChatSession.withdraw_user_input); answered by an
      `addon` event with state "withdrawn", "already_read" or "returned".
  The model sees each added message as a separate text block (never inside a
  tool result), wrapped in <user_addition id="..."> with a marker drawn per
  turn; any user_addition tag inside the text is stripped.
A stdin line of any other kind, or a malformed one, is logged and skipped.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict

from anton.core.browser.config import BrowserConfig
from anton.core.tools.skill_format import validate_name

logger = logging.getLogger(__name__)

TURN_PROTOCOL_VERSION = 1
DATASOURCE_PROTOCOL_VERSION = 1
MAX_DATASOURCE_CONNECTIONS = 100
#: The most custom API connections one turn carries; cowork-server holds the
#: same cap.
MAX_CUSTOM_API_CONNECTIONS = 25
#: The connector id rule anton's draft parser applies, and auth's longest
#: connection name: a block outside them did not come from those writers.
_CUSTOM_CONNECTOR_ID = re.compile(r"[a-z][a-z0-9_]{1,40}")
_CUSTOM_NAME_MAX = 128
_REQUIRED_REF_KEYS = frozenset({"connection_id", "credential_version"})
# Names only, for the prompt. cowork-server's queue check allows the same two.
_OPTIONAL_REF_KEYS = frozenset({"label", "engine"})
_ENGINE_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,31}")
#: Auth caps a connection's name at this many characters; the prompt keeps the same bound.
MAX_CONNECTION_LABEL_LEN = 128

# The `turn_completed` key that names the artifact folders the turn touched.
# scratchpad-controller's `scratchpad_controller/anton_turn.py` relays it and
# cowork-server's `cowork/services/task_objects.py` reads it, under the same name.
ARTIFACTS_TOUCHED_KEY = "artifacts_touched"
# The most folder names that list carries. The entrypoint leaves out a longer
# list; scratchpad-controller's `anton_turn.py` relays none and cowork-server's
# `tracked_artifacts_from_reply` (`cowork/services/task_objects.py`) reads one
# as absent, both against the same cap.
MAX_ARTIFACTS_TOUCHED = 1000


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
    #: The name the user gave the connection, already safe for one prompt line.
    label: str | None = None
    #: Auth's connector id for the connection, such as ``postgres``.
    engine: str | None = None


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
        if (
            not isinstance(connection, dict)
            or not _REQUIRED_REF_KEYS <= set(connection) <= _REQUIRED_REF_KEYS | _OPTIONAL_REF_KEYS
        ):
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
        refs.append(
            DatasourceConnectionRefV1(
                connection_id,
                credential_version,
                label=_usable_label(connection.get("label")),
                engine=_usable_engine(connection.get("engine")),
            )
        )
    return DatasourceBlockV1(DATASOURCE_PROTOCOL_VERSION, tuple(refs))


@dataclass(frozen=True)
class CustomApiConnectionRefV1:
    """One custom API connection the turn may call: auth's id and the member's own names, never a value."""

    connection_id: int
    connector_id: str
    name: str
    credential_version: int


@dataclass(frozen=True, repr=False)
class CustomApiBlockV1:
    """The member's custom API connections and the turn key their grants are bound to.

    ``repr=False`` keeps the key out of any traceback that prints the request.
    """

    turn_key: str
    connections: tuple[CustomApiConnectionRefV1, ...]


def _positive_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 1


def _parse_custom_api_block(value: object) -> CustomApiBlockV1 | None:
    """Parse the optional ``custom_api`` block exactly, or refuse the request."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"turn_key", "connections"}:
        raise ValueError("custom_api must carry exactly turn_key and connections")
    turn_key = value["turn_key"]
    if not isinstance(turn_key, str) or not turn_key:
        raise ValueError("custom_api turn key is missing")
    connections = value["connections"]
    if not isinstance(connections, list) or not connections or len(connections) > MAX_CUSTOM_API_CONNECTIONS:
        raise ValueError("custom_api connections are invalid")
    refs: list[CustomApiConnectionRefV1] = []
    seen: set[int] = set()
    for connection in connections:
        if not isinstance(connection, dict) or set(connection) != {
            "connection_id", "connector_id", "name", "credential_version",
        }:
            raise ValueError("custom_api connection reference is invalid")
        connection_id = connection["connection_id"]
        if (
            not _positive_int(connection_id)
            or not _positive_int(connection["credential_version"])
            or not isinstance(connection["connector_id"], str)
            or not _CUSTOM_CONNECTOR_ID.fullmatch(connection["connector_id"])
            or not isinstance(connection["name"], str)
            or not connection["name"]
            or len(connection["name"]) > _CUSTOM_NAME_MAX
            or connection_id in seen
        ):
            raise ValueError("custom_api connection reference is invalid")
        seen.add(connection_id)
        refs.append(CustomApiConnectionRefV1(
            connection_id, connection["connector_id"], connection["name"], connection["credential_version"],
        ))
    return CustomApiBlockV1(turn_key, tuple(refs))


def _usable_label(value: object) -> str | None:
    """The connection's name as one clean prompt line, or ``None`` when nothing usable is left.

    Spaces of any kind become a plain space and other non-printable characters go, so a
    name cannot start a new prompt line. cowork-server sends the name as the user typed it,
    so this is the only place it is cleaned.
    """
    if not isinstance(value, str):
        return None
    spaced = "".join(" " if unicodedata.category(ch) == "Zs" else ch for ch in value)
    cleaned = "".join(ch for ch in spaced if ch.isprintable()).strip()
    cleaned = cleaned[:MAX_CONNECTION_LABEL_LEN].strip()
    return cleaned or None


def _usable_engine(value: object) -> str | None:
    """Auth's connector id if it has the shape of one, else ``None``.

    A name is only a hint for the model, so an odd value is dropped rather
    than failing the turn. ``fullmatch``, so a trailing newline does not pass.
    """
    if isinstance(value, str) and _ENGINE_PATTERN.fullmatch(value):
        return value
    return None


class TurnSkillEntry(BaseModel):
    """One skill in a turn's ``skills`` block, which keys it by its slug.

    cowork-server sends ``{"files": {relpath: text}, "always_on": true}`` and
    leaves ``always_on`` out, or false, for a skill the agent recalls on demand.
    Frozen, so no field of a parsed entry can be reassigned: staging and the
    always-on prompt section read the same one.
    """

    model_config = ConfigDict(frozen=True)

    #: Each file of the skill's folder, by its path inside that folder
    #: (``SKILL.md``, ``references/recipe.md``), to its text.
    files: dict[str, str]
    #: True also puts the skill's SKILL.md body into every turn's prompt.
    always_on: bool = False


def parse_turn_skills(value: object) -> dict[str, TurnSkillEntry] | None:
    """The request's ``skills`` block as entries by slug, or None when it is absent.

    The block is untrusted wire data, so no part of it fails the turn. A block
    that is not an object names no skills. An entry is dropped when its slug is
    not a valid skill name (with a warning) or its value is not an object.
    Inside an entry, ``files`` that is not an object reads as no files, a file
    whose path or text is not a string is dropped, and only a JSON ``true``
    turns ``always_on`` on. A path that escapes the skill's folder is dropped
    where the pod writes the files (``session._stage_skills``).

    None and ``{}`` stay apart: ``{}`` names no skills, and only an absent block
    lets the pod take the rollout fallback described at
    ``session._SKILLS_ROOT_ENV``.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        return {}
    entries: dict[str, TurnSkillEntry] = {}
    for slug, raw in value.items():
        if not _is_skill_name(slug):
            logger.warning("skills: dropping invalid slug %r", slug)
            continue
        if not isinstance(raw, dict):
            continue
        files = raw.get("files")
        entries[slug] = TurnSkillEntry(
            files={
                rel: text
                for rel, text in (files.items() if isinstance(files, dict) else ())
                if isinstance(rel, str) and isinstance(text, str)
            },
            always_on=raw.get("always_on") is True,
        )
    return entries


def _is_skill_name(slug: object) -> bool:
    if not isinstance(slug, str):
        return False
    try:
        validate_name(slug)
    except ValueError:
        return False
    return True


#: The per-task limits a request may set, with the lowest value each accepts.
#: A turn-token or turn-seconds limit of 0 means no limit to anton, so it is
#: refused here: a producer bug must not turn the limit off.
TURN_LIMIT_MINIMUMS = {
    "max_tool_rounds": 1, "max_continuations": 0, "max_turn_tokens": 1, "max_turn_seconds": 1,
}


def _parse_limits(value: object, conversation_id: str) -> dict[str, int]:
    """The usable limits in a request's ``limits`` block.

    Keeps known names with integer values at or above their minimum and drops
    the rest, so one bad value leaves that limit at the pod's default instead of
    failing the turn. Each drop is logged by name and reason, never by value:
    dropping in silence would look exactly like a controller that sent nothing.
    The name and reason go in log fields, not %-args, because the pod's
    formatter keeps only the static template of a call with args.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        logger.warning(
            "cloud turn ignoring a limits block that is not an object",
            extra={
                "event": "turn_limits_ignored",
                "reason": "limits_block_not_an_object",
                "conversation_id": conversation_id,
            },
        )
        return {}
    limits: dict[str, int] = {}
    for name, limit in value.items():
        if name not in TURN_LIMIT_MINIMUMS:
            # An unknown name is not echoed: it comes off the wire.
            reason = "unknown_limit"
        elif not isinstance(limit, int) or isinstance(limit, bool):
            reason = f"{name}_not_an_integer"
        elif limit < TURN_LIMIT_MINIMUMS[name]:
            reason = f"{name}_below_minimum"
        else:
            limits[name] = limit
            continue
        logger.warning(
            "cloud turn ignoring a limit",
            extra={"event": "turn_limit_ignored", "reason": reason, "conversation_id": conversation_id},
        )
    return limits


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
    #: Optional skills cowork resolved for this turn's person and project:
    #: {slug: {"files": {relpath: text}, "always_on": true}}, kept as the JSON
    #: value; the session reads it through parse_turn_skills (TurnSkillEntry).
    #: When present (even ``{}``), it is the complete set of skills the turn may
    #: recall or list. The pod stages them into a fresh folder each turn and
    #: deletes that folder when the turn ends. It never writes skills back: a
    #: skill the agent builds goes out as a draft on the `skill` event above.
    #: ``"always_on": true`` (organization and team skills) also puts the
    #: skill's SKILL.md body into every turn's prompt, and omitted or false
    #: means recall only. A request without the block stages no skills, except
    #: for the rollout fallback described at session._SKILLS_ROOT_ENV.
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
    #: Optional verified datasource references: IDs and immutable versions, each with
    #: an optional ``label`` (the user's name for it) and ``engine`` (auth's connector
    #: id), which only the prompt reads. No capability or password crosses; the
    #: gateway is on the `llm` base URL's host.
    datasource: DatasourceBlockV1 | None = None
    #: Optional custom API connections of the member, by auth id, with the turn
    #: key their grants are bound to (this turn only). No value.
    custom_api: CustomApiBlockV1 | None = None
    #: Optional ISO 8601 creation time of the conversation, as cowork-server
    #: resolved it. The pod is new every turn and cannot derive this: history
    #: rows carry no timestamps. Without it the session dates the conversation
    #: from the pod's own clock, so the prompt says a three week old
    #: conversation started today and changes at every midnight.
    started_at: str | None = None
    #: Whether cowork-server can answer `ask_user` questions mid-turn (web UI
    #: turns). Only then does the pod register `ask_user` and keep reading
    #: answer lines from stdin. Anything but a JSON `true` reads as False, so a
    #: controller that does not know the field leaves every turn non-interactive.
    interactive: bool = False
    #: Whether the client watching this turn renders a tool's message to the
    #: user (a `tool_result` with `action: "message"`) as an agent message —
    #: the cowork UI does, channel bots do not. Without it the pod drops those
    #: messages and the tool hands the content to the agent instead. Anything
    #: but a JSON `true` reads as False.
    tool_messages: bool = False
    #: Whether the turn may ask Jev, MindsHub's decision model, whether the
    #: agent finished. A missing field or a JSON `true` keeps it on, so a
    #: controller that does not forward the field changes nothing; anything
    #: else turns it off, since a malformed opt-out must not read as consent.
    jev_decisions: bool = True
    #: The language the user's app is set to, a tag in
    #: `anton.core.llm.ui_locale.UI_LOCALE_NAMES` (en, es, pt-BR, de, fr, it,
    #: ja, ko, zh-CN), which cowork-server read from the app's
    #: X-Cowork-UI-Locale header (ENG-4149). The
    #: session validates it (`anton.core.llm.ui_locale`); absent, or a
    #: controller that does not forward it, leaves the prompt unchanged.
    ui_locale: str | None = None
    #: Per-engine connector data cowork-server resolved for this turn:
    #: {engine: {"usage_notes": str}}. Rendered only for engines the turn's
    #: vault connects. Absent when no connected engine has notes.
    connectors: dict | None = None
    #: Whether this turn may build connector templates with
    #: `create_connector_draft`, sent back on the `connector` event. Anything
    #: but a JSON `true` reads as False, so the tool stays off by default.
    connector_drafts: bool = False
    #: Optional reasoning effort the user picked for this turn. Applied to the
    #: planning model only; the name is passed to the provider uninterpreted.
    #: Absent, null, blank, or non-string keeps the configured default.
    reasoning_effort: str | None = None
    #: The MindsHub browser router, when cowork-server turned the browser on for
    #: this turn: {"base_url", "profile", "project"?}, read by
    #: BrowserConfig.from_dict. The pod reaches the router with this turn's
    #: `llm` key. A block with a `token`, or with any address but the https
    #: router, is refused. Absent or malformed means no browser tool this turn.
    browser: BrowserConfig | None = None
    #: The tenant's per-task limits cowork-server resolved: any of
    #: ``max_tool_rounds``, ``max_continuations``, ``max_turn_tokens``,
    #: ``max_turn_seconds``. Each one
    #: present replaces the pod's own default for this turn; an absent one (or
    #: an older controller that sends no block) leaves the default.
    limits: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def from_json(raw: str) -> "TurnRequestV1":
        d = json.loads(raw)
        protocol_version = d["protocol_version"]
        if not _is_version(protocol_version, TURN_PROTOCOL_VERSION):
            raise ValueError("unsupported cloud turn protocol version")
        reasoning_effort = d.get("reasoning_effort")
        if not isinstance(reasoning_effort, str):
            reasoning_effort = ""
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
            custom_api=_parse_custom_api_block(d.get("custom_api")),
            # The controller always sends the key and sends None when
            # cowork-server could not resolve it, so the guard does real work.
            started_at=(
                d.get("started_at") if isinstance(d.get("started_at"), str) else None
            ),
            interactive=d.get("interactive") is True,
            tool_messages=d.get("tool_messages") is True,
            jev_decisions=d.get("jev_decisions", True) is True,
            ui_locale=d.get("ui_locale") if isinstance(d.get("ui_locale"), str) else None,
            # Same defensive isinstance check as oauth/trace.
            connectors=(
                d.get("connectors") if isinstance(d.get("connectors"), dict) else None
            ),
            connector_drafts=d.get("connector_drafts") is True,
            reasoning_effort=reasoning_effort.strip() or None,
            browser=BrowserConfig.from_dict(d.get("browser")),
            limits=_parse_limits(d.get("limits"), str(d["conversation_id"])),
        )
