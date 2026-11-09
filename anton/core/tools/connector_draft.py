"""The `create_connector_draft` tool and the one parser for a connector draft.

A cloud turn may build the template for an API connector the user has no
connector for: its id, label, the credential fields the member will fill in,
usage notes, and a check: one GET whose path and headers say how to test the
credentials. Never a value, and never the API's address: the member types
that when they connect, so a template cannot choose where a key is sent. The pod emits the drafts at end of turn, the
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
#: One below the store's limit, which also keeps the address the member types.
MAX_FIELDS = 11
#: Field names the host stores beside the member's values.
RESERVED_FIELD_NAMES = frozenset({"base_url"})
MAX_CHECK_HEADERS = 5

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

_TOP_KEYS = frozenset({"connector_id", "help_url", "fields", "check", *_TOP_TEXT_CAPS})
_FIELD_KEYS = frozenset({"name", "secret", "required", *_FIELD_TEXT_CAPS})
_CHECK_KEYS = frozenset({"path", "headers"})

_CHECK_PATH_MAX = 300
_CHECK_HEADER_VALUE_MAX = 200
_HEADER_NAME = re.compile(r"[A-Za-z][A-Za-z0-9-]{0,40}")
_PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_]{0,39})\}")
#: Headers the request itself owns, or that would change what it does or where it goes.
_REFUSED_HEADERS = frozenset({
    "host", "cookie", "content-length", "transfer-encoding", "connection", "keep-alive", "te",
    "trailer", "upgrade", "expect", "accept-encoding",
    "x-http-method-override", "x-http-method", "x-method-override",
})


class ConnectorDraftError(ValueError):
    """A connector draft that cannot be kept. The message names the field and
    never quotes its value, since the value may be a credential."""


@dataclass(frozen=True)
class ConnectorDraftField:
    """One credential field the member fills in; it describes, never holds, a value."""

    name: str
    label: str
    secret: bool = True
    required: bool = True
    placeholder: str | None = None
    help: str | None = None


@dataclass(frozen=True)
class ConnectorDraftCheck:
    """How to test the credentials: one GET of ``path`` on the address the member types.

    ``headers`` pairs each header name with its value, where ``{field}`` stands
    for that field's value.
    """

    path: str
    headers: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        """The check as JSON-ready data, its headers as a name-to-value object."""
        return {"path": self.path, "headers": dict(self.headers)}


@dataclass(frozen=True)
class ConnectorDraft:
    """A connector template as the allowlist accepted it, ready to show or save."""

    connector_id: str
    label: str
    fields: tuple[ConnectorDraftField, ...]
    description: str | None = None
    category: str | None = None
    help_url: str | None = None
    usage_notes: str | None = None
    check: ConnectorDraftCheck | None = None

    def to_dict(self) -> dict[str, Any]:
        """The draft as JSON-ready data, with unset optional keys left out."""
        out = {
            k: v for k, v in asdict(self).items() if k not in ("fields", "check") and v is not None
        }
        out["fields"] = [
            {k: v for k, v in asdict(f).items() if v is not None} for f in self.fields
        ]
        if self.check is not None:
            out["check"] = self.check.to_dict()
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
        # Reading the port validates it; an out-of-range one raises here.
        parts.port
    except ValueError as e:
        raise ConnectorDraftError("help_url must be an https:// URL.") from e
    if parts.scheme != "https" or not parts.hostname or "@" in parts.netloc or any(c.isspace() for c in text):
        raise ConnectorDraftError("help_url must be an https:// URL.")
    if parts.query:
        raise ConnectorDraftError("help_url must not have a query string.")
    return text


def _is_printable_ascii(text: str, *, space: bool) -> bool:
    """Whether every character of ``text`` is printable ASCII, the space only when allowed."""
    low = 0x20 if space else 0x21
    return all(low <= ord(c) <= 0x7E for c in text)


def _check_path(value: Any) -> str:
    """Return the check's path: an absolute path with an optional query, never a host.

    A path the request URL could read as naming another host (``//``, ``\\``,
    a ``.`` or ``..`` segment) is refused, and so is anything a placeholder
    could reach: a value never goes into the URL. ``%`` is refused too, so an
    encoded value cannot pass the token rule.
    """
    text = _check_text(value, "check.path", _CHECK_PATH_MAX, long_text=False)
    if text is None:
        raise ConnectorDraftError("check.path is required.")
    if (
        not text.startswith("/")
        or "//" in text
        or not _is_printable_ascii(text, space=False)
        or any(c in text for c in "\\#{}%")
    ):
        raise ConnectorDraftError("check.path must be a path such as /v1/me, with no host.")
    segments = text.split("?", 1)[0].split("/")
    if any(segment in (".", "..") for segment in segments):
        raise ConnectorDraftError("check.path must not have . or .. segments.")
    return text


def _check_headers(value: Any, fields: tuple[ConnectorDraftField, ...]) -> tuple[tuple[str, str], ...]:
    """Return the check's headers; a value may place a declared field as ``{name}``.

    At least one header places a secret field, or the check cannot tell a
    working key from a wrong one.
    """
    if not isinstance(value, dict) or not 1 <= len(value) <= MAX_CHECK_HEADERS:
        raise ConnectorDraftError(f"check.headers must be an object of 1 to {MAX_CHECK_HEADERS} headers.")
    by_name = {f.name: f for f in fields}
    seen: set[str] = set()
    headers: list[tuple[str, str]] = []
    places_secret = False
    for name, header_value in value.items():
        # A header name is model text: one that reads as a token is refused before any message names it.
        if not isinstance(name, str) or not _HEADER_NAME.fullmatch(name) or _looks_like_token(name):
            raise ConnectorDraftError("check.headers has a header name that is not allowed.")
        lowered = name.lower()
        if lowered in _REFUSED_HEADERS or lowered.startswith("proxy-") or lowered in seen:
            raise ConnectorDraftError("check.headers has a header name that is not allowed.")
        seen.add(lowered)
        where = f"check.headers.{name}"
        text = _check_text(header_value, where, _CHECK_HEADER_VALUE_MAX, long_text=False)
        if text is None or not _is_printable_ascii(text, space=True):
            raise ConnectorDraftError(f"{where} must be printable text.")
        placed = _PLACEHOLDER.findall(text)
        if "{" in _PLACEHOLDER.sub("", text) or "}" in _PLACEHOLDER.sub("", text):
            raise ConnectorDraftError(f"{where} may use braces only as {{field_name}}.")
        if any(name_ not in by_name for name_ in placed):
            raise ConnectorDraftError(f"{where} names a field the draft does not declare.")
        places_secret = places_secret or any(by_name[name_].secret for name_ in placed)
        headers.append((name, text))
    if not places_secret:
        raise ConnectorDraftError("check.headers must place a secret field, such as {api_key}.")
    return tuple(headers)


def _parse_check(data: Any, fields: tuple[ConnectorDraftField, ...]) -> ConnectorDraftCheck | None:
    """Parse the optional ``check``, or return None when absent."""
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ConnectorDraftError("check must be an object.")
    _check_keys(data, _CHECK_KEYS, "check")
    return ConnectorDraftCheck(path=_check_path(data.get("path")), headers=_check_headers(data.get("headers"), fields))


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
    if name in RESERVED_FIELD_NAMES:
        raise ConnectorDraftError(f"{where}.name is reserved; the member types the API address when connecting.")
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
        check=_parse_check(data.get("check"), fields),
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
    "client_secret), usage notes on how to call the API with them, and a "
    "check: the path of one GET that answers only with valid credentials (for "
    "example /v1/me) and the header that carries them (for example "
    "Authorization: Bearer {api_key}). The user types the API's address "
    "when connecting, so leave it out of the check. Never "
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
            "description": "How to call the API: auth header and scheme, key endpoints. No values.",
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
_INPUT_SCHEMA["properties"]["check"] = {
    "type": "object",
    "description": "How to test the credentials: one GET, on the address the user types.",
    "properties": {
        "path": {"type": "string", "description": "The request path, e.g. '/v1/me'. No host."},
        "headers": {
            "type": "object",
            "description": "Header name to value; {field_name} places a field's value, e.g. 'Bearer {api_key}'.",
            "additionalProperties": {"type": "string"},
            "maxProperties": MAX_CHECK_HEADERS,
        },
    },
    "required": ["path", "headers"],
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
