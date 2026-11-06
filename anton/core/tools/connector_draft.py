"""The `create_connector_draft` tool and the one parser for a connector draft.

A cloud turn may build the template for an API connector the user has no
connector for: its id, label, the credential fields the member will fill in,
and usage notes. Never a value. The pod emits the drafts at end of turn, the
host shows each as a card, and the member saves it explicitly. The member then
enters credentials outside the conversation.

`parse_connector_draft` is the allowlist for that shape. The host imports it to
check a draft again before showing or storing it, so both sides agree on what
a draft may hold.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlsplit

from anton.core.tools.registry import ToolOutcome
from anton.core.tools.tool_defs import ToolDef
from anton.utils.datasources import scrub_output

#: Drafts one turn may hold. A second call with the same id replaces the first.
MAX_DRAFTS_PER_TURN = 3
MAX_FIELDS = 12

_CONNECTOR_ID = re.compile(r"[a-z][a-z0-9_]{1,40}")
_FIELD_NAME = re.compile(r"[a-z][a-z0-9_]{0,39}")
#: An unbroken run this long of the characters keys are made of, holding both
#: letters and digits, reads as a key or token. Words and digitless ids pass.
_TOKEN_RUN = re.compile(r"[A-Za-z0-9_-]{20,}")

_HIDDEN_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})

_TOP_TEXT_CAPS = {"label": 80, "description": 300, "category": 40, "usage_notes": 1600}
_FIELD_TEXT_CAPS = {"label": 80, "placeholder": 120, "help": 300}
_HELP_URL_MAX = 500
#: Prose texts, the only ones that may span lines.
_LONG_TEXTS = frozenset({"usage_notes", "help"})

_TOP_KEYS = frozenset({"connector_id", "help_url", "fields", *_TOP_TEXT_CAPS})
_FIELD_KEYS = frozenset({"name", "secret", "required", *_FIELD_TEXT_CAPS})


class ConnectorDraftError(ValueError):
    """A connector draft that cannot be kept. The message names the field and
    never quotes its value, since the value may be a credential."""


@dataclass(frozen=True)
class ConnectorDraftField:
    name: str
    label: str
    secret: bool = True
    required: bool = True
    placeholder: str | None = None
    help: str | None = None


@dataclass(frozen=True)
class ConnectorDraft:
    connector_id: str
    label: str
    fields: tuple[ConnectorDraftField, ...]
    description: str | None = None
    category: str | None = None
    help_url: str | None = None
    usage_notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """The draft as JSON-ready data, with unset optional keys left out."""
        out = {k: v for k, v in asdict(self).items() if k != "fields" and v is not None}
        out["fields"] = [
            {k: v for k, v in asdict(f).items() if v is not None} for f in self.fields
        ]
        return out


def _looks_like_token(text: str) -> bool:
    """Whether ``text`` holds a long letters-and-digits run with both kinds."""
    for match in _TOKEN_RUN.finditer(text):
        run = match.group()
        if any(c.isdigit() for c in run) and any(c.isalpha() for c in run):
            return True
    return False


def _check_text(value: Any, where: str, cap: int, *, long_text: bool) -> str | None:
    """Return ``value`` stripped, or None when absent or empty.

    Raises:
        ConnectorDraftError: when it is not text, too long, holds control
            characters, or looks like it holds a credential value.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConnectorDraftError(f"{where} must be text.")
    text = value.strip()
    if len(text) > cap:
        raise ConnectorDraftError(f"{where} must be at most {cap} characters.")
    # Format characters and line separators are invisible on the card and can
    # split a key so the run rule below misses it.
    allowed = {"\n"} if long_text else set()
    if any(unicodedata.category(c) in _HIDDEN_CATEGORIES and c not in allowed for c in text):
        raise ConnectorDraftError(f"{where} must not contain control or invisible characters.")
    # The scrubber misses common key shapes in prose (GitHub, AWS), so the run
    # rule applies to every text, the notes included.
    if scrub_output(text) != text or _looks_like_token(text):
        raise ConnectorDraftError(
            f"{where} looks like it holds a credential value. Describe the "
            "credential (its name, where to find it) without the value."
        )
    return text or None


def _check_keys(data: dict, allowed: frozenset[str], where: str) -> None:
    """Refuse keys outside ``allowed`` without naming them; a key is model text too."""
    if not set(data) <= allowed:
        raise ConnectorDraftError(
            f"{where} has a key that is not allowed. Allowed keys: {', '.join(sorted(allowed))}."
        )


def _check_bool(value: Any, where: str, default: bool) -> bool:
    """Return ``value`` when it is a JSON boolean, ``default`` when absent."""
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ConnectorDraftError(f"{where} must be true or false.")
    return value


def _check_help_url(value: Any) -> str | None:
    """Return an https URL with a host and no user info, or None when absent."""
    text = _check_text(value, "help_url", _HELP_URL_MAX, long_text=False)
    if text is None:
        return None
    try:
        parts = urlsplit(text)
    except ValueError as e:
        raise ConnectorDraftError("help_url must be an https:// URL.") from e
    if parts.scheme != "https" or not parts.hostname or "@" in parts.netloc or any(c.isspace() for c in text):
        raise ConnectorDraftError("help_url must be an https:// URL.")
    if parts.query:
        raise ConnectorDraftError("help_url must not have a query string.")
    return text


def _parse_field(data: Any, index: int) -> ConnectorDraftField:
    """Parse one entry of ``fields``."""
    where = f"fields[{index}]"
    if not isinstance(data, dict):
        raise ConnectorDraftError(f"{where} must be an object.")
    _check_keys(data, _FIELD_KEYS, where)
    name = data.get("name")
    if not isinstance(name, str) or not _FIELD_NAME.fullmatch(name):
        raise ConnectorDraftError(
            f"{where}.name must be lowercase letters, digits and underscores, "
            "starting with a letter, at most 40 characters."
        )
    texts = {
        key: _check_text(data.get(key), f"{where}.{key}", cap, long_text=key in _LONG_TEXTS)
        for key, cap in _FIELD_TEXT_CAPS.items()
    }
    if texts["label"] is None:
        raise ConnectorDraftError(f"{where}.label is required.")
    return ConnectorDraftField(
        name=name,
        label=texts["label"],
        secret=_check_bool(data.get("secret"), f"{where}.secret", True),
        required=_check_bool(data.get("required"), f"{where}.required", True),
        placeholder=texts["placeholder"],
        help=texts["help"],
    )


def parse_connector_draft(data: Any) -> ConnectorDraft:
    """Build a connector draft from untrusted data through an allowlist.

    Args:
        data: The draft as the model or a client sent it.

    Returns:
        The draft, with texts stripped and empty optional texts dropped.

    Raises:
        ConnectorDraftError: for any key, value or shape outside the
            allowlist. The message names the field, never the value.
    """
    if not isinstance(data, dict):
        raise ConnectorDraftError("The connector draft must be an object.")
    _check_keys(data, _TOP_KEYS, "The connector draft")
    connector_id = data.get("connector_id")
    if not isinstance(connector_id, str) or not _CONNECTOR_ID.fullmatch(connector_id):
        raise ConnectorDraftError(
            "connector_id must be lowercase letters, digits and underscores, "
            "starting with a letter, 2 to 41 characters."
        )
    texts = {
        key: _check_text(data.get(key), key, cap, long_text=key in _LONG_TEXTS)
        for key, cap in _TOP_TEXT_CAPS.items()
    }
    if texts["label"] is None:
        raise ConnectorDraftError("label is required.")
    raw_fields = data.get("fields")
    if not isinstance(raw_fields, list) or not 1 <= len(raw_fields) <= MAX_FIELDS:
        raise ConnectorDraftError(f"fields must be a list of 1 to {MAX_FIELDS} fields.")
    fields = tuple(_parse_field(f, i) for i, f in enumerate(raw_fields))
    if len({f.name for f in fields}) != len(fields):
        raise ConnectorDraftError("Each field name must be unique.")
    return ConnectorDraft(
        connector_id=connector_id,
        label=texts["label"],
        fields=fields,
        description=texts["description"],
        category=texts["category"],
        help_url=_check_help_url(data.get("help_url")),
        usage_notes=texts["usage_notes"],
    )


class ConnectorDraftBox:
    """The drafts one turn built, in call order, for the entrypoint to drain."""

    def __init__(self) -> None:
        self._drafts: dict[str, ConnectorDraft] = {}

    def put(self, draft: ConnectorDraft) -> bool:
        """Keep ``draft``, replacing one with the same id.

        Returns:
            False when the turn already holds the maximum of other drafts.
        """
        if draft.connector_id not in self._drafts and len(self._drafts) >= MAX_DRAFTS_PER_TURN:
            return False
        self._drafts[draft.connector_id] = draft
        return True

    def drain(self) -> list[dict[str, Any]]:
        """Return the drafts as data and forget them, so a second call is empty."""
        drained = [d.to_dict() for d in self._drafts.values()]
        self._drafts.clear()
        return drained


_DESCRIPTION = (
    "Build the template for a connector to an API the user needs and has no "
    "connector for. Pass the connector's id, a label, the credential fields "
    "the user will fill in (for example api_key, or client_id and "
    "client_secret), and usage notes on how to call the API with them. Never "
    "pass a credential value: no key, token, password or secret goes into any "
    "field of this tool, and never ask the user for one. The template is "
    "staged, not saved: the user sees it as a card in the chat and saves it, "
    "then enters their credentials in Connect Apps and Data. Only API "
    "connectors that take keys or tokens are supported; not database drivers "
    "and not sign-in flows that redirect to the provider."
)

_TEXT = {"type": "string"}
_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "connector_id": {
            "type": "string",
            "description": "Lowercase id, letters, digits and underscores, e.g. 'acme_crm'.",
        },
        "label": {"type": "string", "description": "Display name, e.g. 'Acme CRM'."},
        "description": {"type": "string", "description": "One sentence on what it connects to."},
        "category": _TEXT,
        "help_url": {"type": "string", "description": "https link to the provider's API key docs."},
        "usage_notes": {
            "type": "string",
            "description": "How to call the API: base URL, auth header and scheme, key endpoints. No values.",
        },
        "fields": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_FIELDS,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Lowercase field name, e.g. 'api_key'."},
                    "label": _TEXT,
                    "secret": {"type": "boolean", "description": "Mask the input. Defaults to true."},
                    "required": {"type": "boolean", "description": "Defaults to true."},
                    "placeholder": _TEXT,
                    "help": {"type": "string", "description": "Where the user finds this value."},
                },
                "required": ["name", "label"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["connector_id", "label", "fields"],
    "additionalProperties": False,
}


def connector_draft_tool(box: ConnectorDraftBox) -> ToolDef:
    """The `create_connector_draft` tool, keeping its drafts in ``box``."""

    async def handle(session: Any, tc_input: dict) -> ToolOutcome:
        """Parse one draft and keep it in ``box``; refusals name the field only."""
        try:
            draft = parse_connector_draft(tc_input)
        except ConnectorDraftError as e:
            return ToolOutcome(content=json.dumps({"error": str(e)}), ok=False, reason="invalid_draft")
        if not box.put(draft):
            return ToolOutcome(
                content=json.dumps({
                    "error": f"This turn already built {MAX_DRAFTS_PER_TURN} connector templates."
                }),
                ok=False, reason="draft_limit",
            )
        return ToolOutcome(content=json.dumps({
            "connector_id": draft.connector_id,
            "status": "staged",
            "next": (
                "The user sees this template as a card and saves it, then adds "
                "their credentials in Connect Apps and Data. Tell them that."
            ),
        }), ok=True)

    return ToolDef(
        name="create_connector_draft",
        description=_DESCRIPTION,
        input_schema=_INPUT_SCHEMA,
        handler=handle,
    )
