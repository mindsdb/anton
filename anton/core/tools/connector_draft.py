"""The `create_connector_draft` tool and the one parser for a connector draft.

A cloud turn may build the template for an API connector the user has no
connector for: its id, label, the credential fields the member will fill in,
an auth section saying how those fields go on every request, usage notes,
and a check: the path of one GET that tests the credentials. Never a value,
and never the API's address: the member types that when they connect, so
nothing can send a key anywhere else. Usage notes are prose and may name any
host; nothing fetches or follows them. The pod emits the drafts at end of
turn, the host shows each as a card, and the member saves it explicitly. The
member then enters credentials outside the conversation.

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
#: Field names the host stores beside the member's values or reads as display names.
RESERVED_FIELD_NAMES = frozenset({"base_url", "label", "user_label"})

_CONNECTOR_ID = re.compile(r"[a-z][a-z0-9_]{1,40}")
_FIELD_NAME = re.compile(r"[a-z][a-z0-9_]{0,39}")
#: An unbroken run this long of the characters keys are made of, holding both
#: letters and digits, reads as a key or token. Words and digitless ids pass.
_TOKEN_RUN = re.compile(r"[A-Za-z0-9_-]{20,}")
#: The same rule for an id or field name, where ``_`` separates words.
_ID_RUN = re.compile(r"[a-z0-9]{20,}")

_HIDDEN_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})

_TOP_TEXT_CAPS = {"label": 80, "description": 300, "category": 40, "usage_notes": 1600}
_FIELD_TEXT_CAPS = {"label": 80, "placeholder": 120, "help": 300}
_HELP_URL_MAX = 500
#: Prose texts, the only ones that may span lines.
_LONG_TEXTS = frozenset({"usage_notes", "help"})

_TOP_KEYS = frozenset({"connector_id", "help_url", "fields", "check", "auth", *_TOP_TEXT_CAPS})
_FIELD_KEYS = frozenset({"name", "secret", "required", *_FIELD_TEXT_CAPS})
_CHECK_KEYS = frozenset({"path"})
#: How the member's values go on a request; later types join this list.
_AUTH_KEYS = {
    "header": frozenset({"type", "headers"}),
    "basic": frozenset({"type", "username", "password"}),
}
MAX_AUTH_HEADERS = 5

_CHECK_PATH_MAX = 300
_AUTH_TEMPLATE_MAX = 200
_HEADER_NAME = re.compile(r"[A-Za-z][A-Za-z0-9-]{0,40}")
_PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_]{0,39})\}")
#: Lowercase names a template may not set: transport-owned, read by proxies as
#: client, host, path or method, or passed by the call helper. Others pass.
_REFUSED_HEADERS = frozenset({
    "host", "cookie", "via", "max-forwards", "range", "expect", "accept-encoding",
    "connection", "keep-alive", "te", "trailer", "transfer-encoding", "upgrade",
    "forwarded", "x-real-ip", "x-client-ip", "x-cluster-client-ip", "true-client-ip",
    "cf-connecting-ip", "fastly-client-ip",
    "x-original-url", "x-rewrite-url", "x-original-host", "x-host", "x-http-host-override",
    "x-http-method-override", "x-http-method", "x-method-override",
    "accept", "accept-language", "if-none-match", "if-modified-since",
})
_REFUSED_HEADER_PREFIXES = ("proxy-", "x-forwarded-", "x-envoy-", "content-")


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
    """How to test the credentials: one GET of ``path`` on the address the
    member types, carrying the template's ``auth``."""

    path: str

    def to_dict(self) -> dict[str, Any]:
        """The check as JSON-ready data."""
        return {"path": self.path}


@dataclass(frozen=True)
class ConnectorDraftAuth:
    """How the member's values go on every request, as templates where
    ``{field}`` stands for that field's value.

    ``header`` sets each of ``headers``. ``basic`` sends
    ``Authorization: Basic`` of ``username:password``.
    """

    type: str
    headers: tuple[tuple[str, str], ...] = ()
    username: str | None = None
    password: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """The auth section as JSON-ready data, holding only its type's keys."""
        if self.type == "basic":
            return {"type": "basic", "username": self.username, "password": self.password}
        return {"type": self.type, "headers": dict(self.headers)}


@dataclass(frozen=True)
class ConnectorDraft:
    """A connector template as the allowlist accepted it, ready to show or save."""

    connector_id: str
    label: str
    fields: tuple[ConnectorDraftField, ...]
    auth: ConnectorDraftAuth
    description: str | None = None
    category: str | None = None
    help_url: str | None = None
    usage_notes: str | None = None
    check: ConnectorDraftCheck | None = None

    def to_dict(self) -> dict[str, Any]:
        """The draft as JSON-ready data, with unset optional keys left out."""
        out = {
            k: v for k, v in asdict(self).items() if k not in ("fields", "check", "auth") and v is not None
        }
        out["fields"] = [
            {k: v for k, v in asdict(f).items() if v is not None} for f in self.fields
        ]
        out["auth"] = self.auth.to_dict()
        if self.check is not None:
            out["check"] = self.check.to_dict()
        return out


def _looks_like_token(text: str, run: re.Pattern[str] = _TOKEN_RUN) -> bool:
    """Whether ``text`` holds a long letters-and-digits run with both kinds.

    Args:
        text: The text to check.
        run: What counts as one unbroken run.
    """
    for match in run.finditer(text):
        found = match.group()
        if any(c.isdigit() for c in found) and any(c.isalpha() for c in found):
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
    # rule applies to every text, the notes included. NFKC folds fullwidth
    # forms, which would otherwise pass both.
    folded = unicodedata.normalize("NFKC", text)
    for form in (text,) if folded == text else (text, folded):
        if scrub_output(form) != form:
            raise ConnectorDraftError(
                f"{where} looks like it holds a credential value: it matches a known "
                "key shape. Describe the credential (its name, where to find it) without the value."
            )
        if _looks_like_token(form):
            raise ConnectorDraftError(
                f"{where} looks like it holds a credential value: it has a run of 20 or "
                "more letters, digits, _ or - that mixes letters and digits. Describe the "
                "credential without the value, or shorten or split that run."
            )
    return text or None


def _check_identifier(name: str, where: str) -> None:
    """Refuse an id or field name that reads as a key, without quoting it.

    Raises:
        ConnectorDraftError: when ``name`` matches a key shape the scrubber
            knows, or one of its ``_``-separated parts is a run of 20 or more
            letters and digits with both kinds.
    """
    if scrub_output(name) != name or _looks_like_token(name, _ID_RUN):
        raise ConnectorDraftError(
            f"{where} looks like a credential value: it has a part of 20 or more letters "
            "and digits, or a known key shape. Use short words joined by underscores."
        )


def check_draft_text(key: str, value: Any) -> str | None:
    """Check one top-level text of a draft with the rule the parser applies to it.

    Lets a host check an edit to a saved connector's label, description or
    category without building a whole draft around it.

    Args:
        key: label, description, category or usage_notes.
        value: The text as the client sent it.

    Returns:
        ``value`` stripped, or None when absent or empty.

    Raises:
        ConnectorDraftError: for any other key, or a value the parser would
            refuse. The message names the key, never the value.
    """
    cap = _TOP_TEXT_CAPS.get(key)
    if cap is None:
        raise ConnectorDraftError(f"{key} is not a text of a connector draft.")
    return _check_text(value, key, cap, long_text=key in _LONG_TEXTS)


def check_header_name(name: Any, where: str = "The template") -> None:
    """Refuse a header name a connector template may not set.

    A grammar plus a list of refused names, not an allowlist: a name outside
    the list passes. The other services that apply a template's headers keep
    a copy of this rule and run ``tests/fixtures/header_name_cases.json``.

    Args:
        name: The header name as the template gives it.
        where: Where the name sits, for the message.

    Raises:
        ConnectorDraftError: when ``name`` breaks the grammar, reads as a
            token, or is refused. The message never quotes the name, which
            is model text.
    """
    if (
        not isinstance(name, str)
        or not _HEADER_NAME.fullmatch(name)
        or _looks_like_token(name)
        or scrub_output(name) != name
    ):
        raise ConnectorDraftError(f"{where} has a header name that is not allowed.")
    lowered = name.lower()
    if lowered in _REFUSED_HEADERS or lowered.startswith(_REFUSED_HEADER_PREFIXES):
        raise ConnectorDraftError(f"{where} has a header name that is not allowed.")


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


def _check_template(
    value: Any, where: str, declared: frozenset[str], *, may_be_empty: bool = False
) -> tuple[str, set[str]]:
    """Return an auth template and the field names its ``{field}`` slots place.

    The token and scrub rules read the template with its slots removed, so a
    long field name in a slot is not taken for a key.

    Args:
        value: The template as the draft gives it.
        where: Its place in the draft, for messages.
        declared: The draft's field names.
        may_be_empty: Whether an empty template is allowed.

    Raises:
        ConnectorDraftError: when it is not printable ASCII on one line, too
            long, empty when it may not be, has a brace outside a slot, names
            an undeclared field, or holds what reads as a credential value.
    """
    if not isinstance(value, str):
        raise ConnectorDraftError(f"{where} must be text.")
    text = value.strip()
    if not text and not may_be_empty:
        raise ConnectorDraftError(f"{where} is required.")
    if len(text) > _AUTH_TEMPLATE_MAX:
        raise ConnectorDraftError(f"{where} must be at most {_AUTH_TEMPLATE_MAX} characters.")
    if not _is_printable_ascii(text, space=True):
        raise ConnectorDraftError(f"{where} must be printable text on one line.")
    literal = _PLACEHOLDER.sub("", text)
    if "{" in literal or "}" in literal:
        raise ConnectorDraftError(f"{where} may use braces only as {{field_name}}.")
    placed = set(_PLACEHOLDER.findall(text))
    if not placed <= declared:
        raise ConnectorDraftError(f"{where} names a field the draft does not declare.")
    if scrub_output(literal) != literal or _looks_like_token(literal):
        raise ConnectorDraftError(
            f"{where} looks like it holds a credential value. Place the field as "
            "{field_name} instead of writing its value."
        )
    return text, placed


def _parse_auth_headers(value: Any, declared: frozenset[str]) -> tuple[tuple[tuple[str, str], ...], set[str]]:
    """Return the ``header`` type's headers and the fields they place."""
    if not isinstance(value, dict) or not 1 <= len(value) <= MAX_AUTH_HEADERS:
        raise ConnectorDraftError(f"auth.headers must be an object of 1 to {MAX_AUTH_HEADERS} headers.")
    seen: set[str] = set()
    headers: list[tuple[str, str]] = []
    placed: set[str] = set()
    for name, template in value.items():
        # A header name is model text: one that reads as a token is refused before any message names it.
        check_header_name(name, "auth.headers")
        if name.lower() in seen:
            raise ConnectorDraftError("auth.headers has a header name that is not allowed.")
        seen.add(name.lower())
        text, names = _check_template(template, f"auth.headers.{name}", declared)
        # Basic needs base64 of user:password, which a header template cannot build.
        if name.lower() == "authorization" and text.lower().startswith("basic"):
            raise ConnectorDraftError("auth.headers.Authorization must not be Basic; use auth type 'basic'.")
        headers.append((name, text))
        placed |= names
    return tuple(headers), placed


def _parse_auth(data: Any, fields: tuple[ConnectorDraftField, ...]) -> ConnectorDraftAuth:
    """Parse the required ``auth`` section against the draft's fields.

    It must place at least one secret field and every declared field: a
    field it never uses would ask the member for a value that goes nowhere.
    """
    if data is None:
        raise ConnectorDraftError("auth is required: say how the fields go on a request.")
    if not isinstance(data, dict):
        raise ConnectorDraftError("auth must be an object.")
    kind = data.get("type")
    if not isinstance(kind, str) or kind not in _AUTH_KEYS:
        raise ConnectorDraftError(f"auth.type must be one of: {', '.join(sorted(_AUTH_KEYS))}.")
    _check_keys(data, _AUTH_KEYS[kind], "auth")
    declared = frozenset(f.name for f in fields)
    if kind == "header":
        headers, placed = _parse_auth_headers(data.get("headers"), declared)
        auth = ConnectorDraftAuth(type=kind, headers=headers)
    else:
        username, placed = _check_template(data.get("username"), "auth.username", declared)
        if not placed or ":" in _PLACEHOLDER.sub("", username):
            raise ConnectorDraftError("auth.username must place a field and hold no ':' outside it.")
        if "password" not in data:
            raise ConnectorDraftError("auth.password is required; use an empty text when the API takes none.")
        password, in_password = _check_template(data["password"], "auth.password", declared, may_be_empty=True)
        placed |= in_password
        auth = ConnectorDraftAuth(type=kind, username=username, password=password)
    secrets = {f.name for f in fields if f.secret}
    if not placed & secrets:
        raise ConnectorDraftError("auth must place a secret field, such as {api_key}.")
    unused = [i for i, f in enumerate(fields) if f.name not in placed]
    if unused:
        raise ConnectorDraftError(f"auth must place every field; fields[{unused[0]}] is never used.")
    return auth


def _parse_check(data: Any) -> ConnectorDraftCheck | None:
    """Parse the optional ``check``, or return None when absent."""
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ConnectorDraftError("check must be an object.")
    _check_keys(data, _CHECK_KEYS, "check")
    return ConnectorDraftCheck(path=_check_path(data.get("path")))


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
    _check_identifier(name, f"{where}.name")
    if name in RESERVED_FIELD_NAMES:
        raise ConnectorDraftError(f"{where}.name is reserved; the host stores a value under that name.")
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
    _check_identifier(connector_id, "connector_id")
    texts = {key: check_draft_text(key, data.get(key)) for key in _TOP_TEXT_CAPS}
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
        auth=_parse_auth(data.get("auth"), fields),
        description=texts["description"],
        category=texts["category"],
        help_url=_check_help_url(data.get("help_url")),
        usage_notes=texts["usage_notes"],
        check=_parse_check(data.get("check")),
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
    "the user will fill in (for example api_key, or email and api_token), "
    "auth: how those fields go on every request, usage notes, and a check: "
    "the path of one GET that answers only with valid credentials (for "
    "example /v1/me). auth is type 'header' with headers such as "
    "Authorization: Bearer {api_key}, or type 'basic' with a username and "
    "password such as {email}/token and {api_token}; {field_name} places a "
    "field's value and every field must be placed. The user types the API's "
    "address when connecting, so leave it out. Never pass a credential "
    "value: no key, token, password or secret goes into any field of this "
    "tool, and never ask the user for one. The template is staged, not "
    "saved: the user sees it as a card in the chat and saves it, then enters "
    "their credentials in Connect Apps and Data. Only API connectors that "
    "take keys or tokens in headers are supported; not database drivers, "
    "not keys in the URL, and not sign-in flows that redirect to the provider."
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
            "description": (
                "How to use the API once connected: which paths and query parameters answer what. "
                "Calls go through call_minds_api(connection_id, path, query, headers), which adds "
                "the credentials, so leave out auth headers, keys and the API's address. No values."
            ),
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
    "required": ["connector_id", "label", "fields", "auth"],
    "additionalProperties": False,
}
# One flat object with a type, not a oneOf: providers refuse union schemas in tool inputs.
_INPUT_SCHEMA["properties"]["auth"] = {
    "type": "object",
    "description": (
        "How the fields go on every request. type 'header': headers maps a header name to a "
        "template, e.g. {'Authorization': 'Bearer {api_key}'}, or a literal such as "
        "{'Notion-Version': '2022-06-28'}. type 'basic': username and password templates, e.g. "
        "'{email}/token' and '{api_token}'; the password may be a literal such as 'X' or empty. "
        "{field_name} places a field's value; place every field and at least one secret."
    ),
    "properties": {
        "type": {"type": "string", "enum": sorted(_AUTH_KEYS)},
        "headers": {
            "type": "object",
            "description": "For type 'header': header name to template.",
            "additionalProperties": {"type": "string"},
            "maxProperties": MAX_AUTH_HEADERS,
        },
        "username": {"type": "string", "description": "For type 'basic'."},
        "password": {"type": "string", "description": "For type 'basic'."},
    },
    "required": ["type"],
    "additionalProperties": False,
}
_INPUT_SCHEMA["properties"]["check"] = {
    "type": "object",
    "description": "How to test the credentials: one GET of this path, on the address the user types, sent with auth.",
    "properties": {
        "path": {"type": "string", "description": "The request path, e.g. '/v1/me'. No host."},
    },
    "required": ["path"],
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
