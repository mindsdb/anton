"""Versioned wire contract between the scratchpad-controller and the pod runner.

The controller sends a :class:`TurnRequestV1` as a single JSON document on the
pod's stdin (EOF-terminated). The runner replies with :class:`TurnEventV1`
records as JSONL on stdout — one JSON object per line, nothing else. Both sides
pin ``protocol_version``.

Design rules:
- **Data only.** The request is inert; the runner never interpolates request
  fields into code.
- **Typed, not free-form.** Roles and content-block types are enumerated; an
  unsupported shape is rejected with a structured error, never coerced.
- **Only implemented fields exist here.** Speculative fields (artifact
  manifests, usage, workspace checkpoints, file history) are intentionally
  absent until they are implemented and tested.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Bump when the request/event shape changes incompatibly.
PROTOCOL_VERSION = 1

# ── Validation limits (item 3) — documented defaults, not scattered literals ──
#: Max messages in the request's input history.
MAX_HISTORY_MESSAGES = 1000
#: Max size of the whole request JSON document, in bytes.
MAX_REQUEST_BYTES = 10 * 1024 * 1024
#: Max characters in a single text (or string) content value.
MAX_TEXT_BLOCK_CHARS = 1_000_000


def _check_text_len(value: str) -> str:
    if len(value) > MAX_TEXT_BLOCK_CHARS:
        raise ValueError(
            f"text block exceeds {MAX_TEXT_BLOCK_CHARS} chars (got {len(value)})"
        )
    return value


# ── Content blocks ────────────────────────────────────────────────────────────
# V1 supports exactly these three block types — the surface Anton's cloud tools
# (scratchpad + artifacts) actually produce/consume. Image and any other block
# type are NOT supported in V1 (see module docs / conversion limitation).


class TextBlockV1(BaseModel):
    type: Literal["text"] = "text"
    text: str

    _v = field_validator("text")(_check_text_len)


class ToolUseBlockV1(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlockV1(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    #: A string result, or a list of text blocks. (Cloud tools return strings.)
    content: str | list[TextBlockV1]
    is_error: bool = False

    @field_validator("content")
    @classmethod
    def _content_len(cls, v):
        if isinstance(v, str):
            return _check_text_len(v)
        return v


ContentBlockV1 = Annotated[
    Union[TextBlockV1, ToolUseBlockV1, ToolResultBlockV1],
    Field(discriminator="type"),
]


class MessageV1(BaseModel):
    """One persistable conversation message. Tool results are carried in
    ``user``-role messages (Anthropic convention)."""

    role: Literal["user", "assistant"]
    content: str | list[ContentBlockV1]

    @field_validator("content")
    @classmethod
    def _content_len(cls, v):
        if isinstance(v, str):
            return _check_text_len(v)
        return v


#: Supported protocol versions (single-element in V1). Derived here so the
#: capabilities manifest and schema export share one source of truth.
SUPPORTED_PROTOCOL_VERSIONS = (PROTOCOL_VERSION,)
#: The content-block ``type`` discriminators V1 represents, derived from models.
SUPPORTED_CONTENT_BLOCK_TYPES = tuple(
    m.model_fields["type"].default
    for m in (TextBlockV1, ToolUseBlockV1, ToolResultBlockV1)
)
#: Message roles V1 accepts.
SUPPORTED_MESSAGE_ROLES = ("user", "assistant")


# ── Capabilities ────────────────────────────────────────────────────────────


class CapabilitiesV1(BaseModel):
    """Feature gates for one cloud turn. Everything unsafe in a shared,
    headless pod is OFF by default and enabled only once it has a cloud-safe
    implementation."""

    model_config = ConfigDict(extra="forbid")

    personal_memory: bool = False
    connectors: bool = False
    local_data_vault: bool = False
    interactive_tools: bool = False
    local_file_history: bool = False
    dotenv_loading: bool = False


# ── Request ───────────────────────────────────────────────────────────────────


class TurnRequestV1(BaseModel):
    """One turn to run in the pod. Sent as JSON on stdin."""

    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal[1] = PROTOCOL_VERSION
    #: Stable per logical turn — the idempotency key (a redelivery reuses it).
    run_id: str
    #: Per delivery attempt.
    attempt_id: str
    conversation_id: str
    #: Authoritative identity for keying/attribution; producers may not spoof it.
    organization_id: str | None = None
    user_id: str | None = None
    #: (org, workspace) key input; maps to a cowork project.
    workspace_id: str | None = None
    # No workspace path on the wire — the mount is trusted pod-side config
    # (see ``anton.cloud_turn.session.resolve_trusted_workspace_path``).
    #: The user's message for THIS turn — text or typed content blocks.
    input: str | list[ContentBlockV1]
    #: DB-authoritative ordered history (immutable input). The pod never loads
    #: its own history; cowork-server owns persistence.
    history: list[MessageV1] = Field(default_factory=list)
    #: Optional model override. Honoured only if in the pod's trusted model
    #: allowlist (default: none), else rejected — see cloud_turn.session.
    model: str | None = None
    capabilities: CapabilitiesV1 = Field(default_factory=CapabilitiesV1)
    #: Absolute deadline as a Unix timestamp in milliseconds. The runner turns
    #: this into a remaining-time soft timeout at startup; the controller's pod
    #: timeout is the hard external backstop. None = no inner deadline.
    deadline_unix_ms: int | None = None

    @field_validator("history")
    @classmethod
    def _history_len(cls, v: list[MessageV1]) -> list[MessageV1]:
        if len(v) > MAX_HISTORY_MESSAGES:
            raise ValueError(
                f"history exceeds {MAX_HISTORY_MESSAGES} messages (got {len(v)})"
            )
        return v


# ── Structured errors (item 4) ──────────────────────────────────────────────


class ErrorCodeV1(str, Enum):
    """Stable V1 failure codes. Values are the wire strings."""

    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED_PROTOCOL_VERSION = "unsupported_protocol_version"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    UNSUPPORTED_MODEL = "unsupported_model"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    MODEL_AUTH_FAILURE = "model_auth_failure"
    MODEL_PROVIDER_FAILURE = "model_provider_failure"
    INTERNAL_TURN_FAILURE = "internal_turn_failure"


class TurnErrorV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: ErrorCodeV1
    #: Short, credential-scrubbed. Full tracebacks stay on stderr only.
    message: str
    retryable: bool = False


# ── Events ────────────────────────────────────────────────────────────────────


class _EventBaseV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal[1] = PROTOCOL_VERSION
    # Nullable ONLY for the pre-validation failure path: a malformed/empty
    # request may carry no valid identifier, and we must not invent a
    # valid-looking one. ``None`` means "the request could not be validated and
    # supplied no usable identifier". Every event for a VALID request always
    # carries real string ids.
    run_id: str | None
    attempt_id: str | None
    #: 1-based monotonic position within the run (started=1, terminal=2).
    sequence: int


class TurnStartedV1(_EventBaseV1):
    """Emitted once, before work begins. Not terminal."""

    kind: Literal["turn.started"] = "turn.started"


class TurnCompletedV1(_EventBaseV1):
    """Terminal success."""

    kind: Literal["turn.completed"] = "turn.completed"
    #: The final assistant message's text.
    final_text: str
    #: Messages GENERATED during this turn (assistant text, tool calls, tool
    #: results, final assistant message), in order. Never the input history.
    output_messages: list[MessageV1] = Field(default_factory=list)


class TurnFailedV1(_EventBaseV1):
    """Terminal failure."""

    kind: Literal["turn.failed"] = "turn.failed"
    error: TurnErrorV1


TurnEventV1 = Annotated[
    Union[TurnStartedV1, TurnCompletedV1, TurnFailedV1],
    Field(discriminator="kind"),
]

TERMINAL_KINDS = frozenset({"turn.completed", "turn.failed"})


def event_line(event: TurnStartedV1 | TurnCompletedV1 | TurnFailedV1) -> str:
    """Serialize one event to a single JSONL line (no trailing newline)."""
    return event.model_dump_json()


def parse_request(raw: str) -> TurnRequestV1:
    """Parse + validate a TurnRequestV1 from a JSON string.

    Raises typed :mod:`anton.cloud_turn.errors` exceptions (imported lazily to
    avoid an import cycle): size / JSON / protocol-version / shape failures each
    map to a distinct structured error code.
    """
    import json

    from anton.cloud_turn.errors import (
        InvalidRequestError,
        UnsupportedProtocolVersionError,
    )

    if len(raw.encode("utf-8")) > MAX_REQUEST_BYTES:
        raise InvalidRequestError(
            f"request exceeds {MAX_REQUEST_BYTES} bytes"
        )
    try:
        data = json.loads(raw)
    except Exception as exc:
        raise InvalidRequestError(f"not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise InvalidRequestError("request must be a JSON object")

    version = data.get("protocol_version", PROTOCOL_VERSION)
    if version != PROTOCOL_VERSION:
        raise UnsupportedProtocolVersionError(
            f"unsupported protocol_version {version!r}; this pod speaks {PROTOCOL_VERSION}"
        )

    from pydantic import ValidationError

    try:
        return TurnRequestV1.model_validate(data)
    except ValidationError as exc:
        raise InvalidRequestError(f"invalid TurnRequestV1: {exc}") from exc
