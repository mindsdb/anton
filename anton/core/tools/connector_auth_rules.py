"""Rules for a connector template's auth section: how a member's values go on a request.

Standard library only. Every service that builds, stores or applies a
template's auth carries this file byte for byte, and pins its SHA-256, so
they all check it with the same code. Each service passes its own
``looks_secret`` detector for values it holds or key shapes it knows.
"""

from __future__ import annotations

import base64
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import unquote_plus

MAX_AUTH_HEADERS = 5
MAX_AUTH_PARAMS = 5
AUTH_TEMPLATE_MAX = 200
#: The keys each auth type may hold; later types join this list.
AUTH_KEYS = {
    "header": frozenset({"type", "headers"}),
    "basic": frozenset({"type", "username", "password"}),
    "query": frozenset({"type", "params"}),
}
#: The type of a connection's own section when it holds no key to place. Never
#: part of a template, so it stays out of ``AUTH_KEYS`` and the tool schema.
AUTH_NONE = "none"

HEADER_NAME = re.compile(r"[A-Za-z][A-Za-z0-9-]{0,40}")
#: No ``.``: it would split the token-run rule, and some frameworks read it as ``_``.
PARAM_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,40}")
#: Lowercase query parameter names a template may not place: method overrides.
REFUSED_PARAMS = frozenset({"x-http-method-override", "x-http-method", "x-method-override"})
#: Characters a query value's literal part may not hold: they would split or end the value.
QUERY_LITERAL_REFUSED = frozenset("&#+= ")
PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_]{0,39})\}")
#: An unbroken run this long of the characters keys are made of, holding both
#: letters and digits, reads as a key or token. Words and digitless ids pass.
TOKEN_RUN = re.compile(r"[A-Za-z0-9_-]{20,}")
#: Lowercase names a template may not set: transport-owned, read by proxies as
#: client, host, path or method, or passed by the call helper. Others pass.
REFUSED_HEADERS = frozenset(
    {
        "host",
        "cookie",
        "via",
        "max-forwards",
        "range",
        "expect",
        "accept-encoding",
        "connection",
        "keep-alive",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "forwarded",
        "x-real-ip",
        "x-client-ip",
        "x-cluster-client-ip",
        "true-client-ip",
        "cf-connecting-ip",
        "fastly-client-ip",
        "x-original-url",
        "x-rewrite-url",
        "x-original-host",
        "x-host",
        "x-http-host-override",
        "x-http-method-override",
        "x-http-method",
        "x-method-override",
        "accept",
        "accept-language",
        "if-none-match",
        "if-modified-since",
    }
)
REFUSED_HEADER_PREFIXES = ("proxy-", "x-forwarded-", "x-envoy-", "content-")

LooksSecret = Callable[[str], bool]


class AuthRuleError(ValueError):
    """An auth section that cannot be kept. The message names the place and
    never quotes a value, a slot name or a header name, which are model text."""


def looks_like_token(text: str, run: re.Pattern[str] = TOKEN_RUN) -> bool:
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


def is_printable_ascii(text: str, *, space: bool) -> bool:
    """Whether every character of ``text`` is printable ASCII, the space only when allowed."""
    low = 0x20 if space else 0x21
    return all(low <= ord(c) <= 0x7E for c in text)


def _secret(text: str, looks_secret: LooksSecret | None) -> bool:
    """Whether the host's detector reads ``text`` as holding a secret."""
    return looks_secret is not None and looks_secret(text)


def check_header_name(name: Any, where: str = "The template", looks_secret: LooksSecret | None = None) -> None:
    """Refuse a header name a connector template may not set.

    A grammar plus a list of refused names, not an allowlist: a name outside
    the list passes. The shared ``header_name_cases.json`` pins it.

    Args:
        name: The header name as the template gives it.
        where: Where the name sits, for the message.
        looks_secret: The host's detector for a value it holds or a known key shape.

    Raises:
        AuthRuleError: when ``name`` breaks the grammar, reads as a token or
            a secret, or is refused.
    """
    if (
        not isinstance(name, str)
        or not HEADER_NAME.fullmatch(name)
        or looks_like_token(name)
        or _secret(name, looks_secret)
    ):
        raise AuthRuleError(f"{where} has a header name that is not allowed.")
    lowered = name.lower()
    if lowered in REFUSED_HEADERS or lowered.startswith(REFUSED_HEADER_PREFIXES):
        raise AuthRuleError(f"{where} has a header name that is not allowed.")


def check_param_name(name: Any, where: str = "The template", looks_secret: LooksSecret | None = None) -> None:
    """Refuse a query parameter name a connector template may not place.

    Args:
        name: The parameter name as the template gives it.
        where: Where the name sits, for the message.
        looks_secret: The host's detector for a value it holds or a known key shape.

    Raises:
        AuthRuleError: when ``name`` breaks the grammar, reads as a token or
            a secret, or is a method override.
    """
    if (
        not isinstance(name, str)
        or not PARAM_NAME.fullmatch(name)
        or looks_like_token(name)
        or _secret(name, looks_secret)
        or name.lower() in REFUSED_PARAMS
    ):
        raise AuthRuleError(f"{where} has a parameter name that is not allowed.")


def check_template(
    value: Any,
    where: str,
    declared: frozenset[str],
    *,
    may_be_empty: bool = False,
    looks_secret: LooksSecret | None = None,
) -> tuple[str, set[str]]:
    """Return an auth template and the field names its ``{field}`` slots place.

    The token and secret rules read the template with its slots removed, so a
    long field name in a slot is not taken for a key.

    Args:
        value: The template as the draft gives it.
        where: Its place in the draft, for messages.
        declared: The field names a slot may name.
        may_be_empty: Whether an empty template is allowed.
        looks_secret: The host's detector for a value it holds or a known key shape.

    Raises:
        AuthRuleError: when it is not printable ASCII on one line, too long,
            empty when it may not be, has a brace outside a slot, names an
            undeclared field, or holds what reads as a credential value.
    """
    if not isinstance(value, str):
        raise AuthRuleError(f"{where} must be text.")
    text = value.strip()
    if not text and not may_be_empty:
        raise AuthRuleError(f"{where} is required.")
    if len(text) > AUTH_TEMPLATE_MAX:
        raise AuthRuleError(f"{where} must be at most {AUTH_TEMPLATE_MAX} characters.")
    if not is_printable_ascii(text, space=True):
        raise AuthRuleError(f"{where} must be printable text on one line.")
    literal = PLACEHOLDER.sub("", text)
    if "{" in literal or "}" in literal:
        raise AuthRuleError(f"{where} may use braces only as {{field_name}}.")
    placed = set(PLACEHOLDER.findall(text))
    if not placed <= declared:
        raise AuthRuleError(f"{where} names a field the draft does not declare.")
    if _secret(literal, looks_secret) or looks_like_token(literal):
        raise AuthRuleError(
            f"{where} looks like it holds a credential value. Place the field as "
            "{field_name} instead of writing its value."
        )
    return text, placed


def _check_keys(data: dict, allowed: frozenset[str], where: str) -> None:
    """Refuse keys outside ``allowed`` without naming them; a key is model text too."""
    if not set(data) <= allowed:
        raise AuthRuleError(f"{where} has a key that is not allowed. Allowed keys: {', '.join(sorted(allowed))}.")


def _check_optional_placement(names: set[str], optional: frozenset[str], where: str) -> None:
    """Refuse a template that places an optional field together with another field.

    Dropping the optional field must drop the whole header or parameter, so
    nothing else may ride in it.
    """
    if names & optional and len(names) > 1:
        raise AuthRuleError(f"{where} places an optional field with another field; give it a place of its own.")


def _parse_headers(
    value: Any, declared: frozenset[str], looks_secret: LooksSecret | None, optional: frozenset[str]
) -> tuple[dict[str, str], set[str]]:
    """Return the ``header`` type's headers and the fields they place."""
    if not isinstance(value, dict) or not 1 <= len(value) <= MAX_AUTH_HEADERS:
        raise AuthRuleError(f"auth.headers must be an object of 1 to {MAX_AUTH_HEADERS} headers.")
    seen: set[str] = set()
    headers: dict[str, str] = {}
    placed: set[str] = set()
    for index, (name, template) in enumerate(value.items()):
        # Messages name the position, never the header: a name is model text.
        check_header_name(name, "auth.headers", looks_secret)
        if name.lower() in seen:
            raise AuthRuleError("auth.headers has a header name that is not allowed.")
        seen.add(name.lower())
        text, names = check_template(template, f"auth.headers[{index}]", declared, looks_secret=looks_secret)
        _check_optional_placement(names, optional, f"auth.headers[{index}]")
        # Basic needs base64 of user:password, which a header template cannot build.
        if text.lower().startswith("basic"):
            raise AuthRuleError("auth.headers must not hold a Basic value; use auth type 'basic'.")
        headers[name] = text
        placed |= names
    return headers, placed


def _parse_params(
    value: Any, declared: frozenset[str], looks_secret: LooksSecret | None, optional: frozenset[str]
) -> tuple[dict[str, str], set[str]]:
    """Return the ``query`` type's parameters and the fields they place."""
    if not isinstance(value, dict) or not 1 <= len(value) <= MAX_AUTH_PARAMS:
        raise AuthRuleError(f"auth.params must be an object of 1 to {MAX_AUTH_PARAMS} parameters.")
    seen: set[str] = set()
    params: dict[str, str] = {}
    placed: set[str] = set()
    for index, (name, template) in enumerate(value.items()):
        check_param_name(name, "auth.params", looks_secret)
        if name.lower() in seen:
            raise AuthRuleError("auth.params has a parameter name that is not allowed.")
        seen.add(name.lower())
        where = f"auth.params[{index}]"
        text, names = check_template(template, where, declared, looks_secret=looks_secret)
        _check_optional_placement(names, optional, where)
        if QUERY_LITERAL_REFUSED & set(PLACEHOLDER.sub("", text)):
            raise AuthRuleError(f"{where} may not hold &, #, +, = or a space outside {{field_name}}.")
        params[name] = text
        placed |= names
    return params, placed


def _fill(template: str, values: Mapping[str, str]) -> str:
    """Put each slot's value into ``template``; a missing or unsafe value is refused without naming it."""

    def value_of(match: re.Match[str]) -> str:
        """The value for one slot, refused when absent or holding a control character."""
        name = match.group(1)
        if name not in values:
            raise AuthRuleError("A field the auth section places has no value.")
        value = values[name]
        # A control or line-separator character would split a header or a URL.
        if any(unicodedata.category(c) in ("Cc", "Zl", "Zp") for c in value):
            raise AuthRuleError("A value the auth section places holds a control character.")
        return value

    return PLACEHOLDER.sub(value_of, template)


def render(auth: Mapping[str, Any], values: Mapping[str, str]) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Fill a checked auth section with the member's values, as headers and query parameters.

    The one place slots become values, so the connection check and the
    gateway build the same request. URL encoding is left to the caller's
    HTTP client.

    Args:
        auth: An auth section as ``parse_auth`` returned it; ``none`` places nothing.
        values: Each placed field's value.

    Returns:
        The headers to set and the query parameters to append.

    Raises:
        AuthRuleError: when the type is not one this file knows, or a placed
            field has no value or holds a control character.
    """
    if auth["type"] == AUTH_NONE:
        return {}, []
    if auth["type"] == "basic":
        pair = f"{_fill(auth['username'], values)}:{_fill(auth['password'], values)}"
        return {"Authorization": "Basic " + base64.b64encode(pair.encode()).decode("ascii")}, []
    if auth["type"] == "query":
        return {}, [(name, _fill(template, values)) for name, template in auth["params"].items()]
    if auth["type"] == "header":
        return {name: _fill(template, values) for name, template in auth["headers"].items()}, []
    raise AuthRuleError("auth.type is not one these rules can apply.")


def query_names(query: str) -> list[str]:
    """The parameter names in a raw query string, split on ``&`` and ``;`` as servers do, decoded."""
    return [unquote_plus(part.split("=", 1)[0]) for part in re.split(r"[&;]", query) if part]


def query_collides(auth: Mapping[str, Any], names: Iterable[str]) -> bool:
    """Whether a request's own query parameter names may not go out beside the auth section's.

    For a ``query`` section every name must be a plain ``PARAM_NAME``, and
    none may equal a placed name ignoring case. Anything else is refused
    rather than interpreted: servers rewrite brackets, dots, spaces and
    separators in ways no check can follow. Other types accept any name.
    ``auth`` is a section ``parse_auth`` returned; ``names`` are strings.
    """
    if auth["type"] != "query":
        return False
    placed = {name.lower() for name in auth["params"]}
    return any(not PARAM_NAME.fullmatch(name) or name.lower() in placed for name in names)


def parse_auth(
    data: Any,
    fields: Sequence[tuple[str, bool]],
    looks_secret: LooksSecret | None = None,
    *,
    optional: frozenset[str] = frozenset(),
    stored: bool = False,
) -> dict[str, Any]:
    """Check an auth section against the template's fields and return it canonical.

    A template's section must place at least one secret field and every
    field: a field it never uses would ask the member for a value that goes
    nowhere. A header or parameter that places an optional field places only
    that field. In Basic only the password may be optional, as the whole
    password template, and only beside a username that places a required
    secret field, so a skipped password never leaves a bare identity.

    A stored section is a connection's own: the template's section with the
    placements of skipped optional fields taken out. It may be ``none``, may
    place no secret field and may be literal only, but must still place every
    field the connection holds.

    Args:
        data: The auth section as the template gives it.
        fields: Each field's name and whether it is secret, in template order.
        looks_secret: The host's detector for a value it holds or a known key shape.
        optional: The names of the fields a member may leave out; each must be secret.
        stored: Whether ``data`` is a connection's section, not a template's.

    Returns:
        The section rebuilt from what was checked, holding only its type's keys.

    Raises:
        AuthRuleError: for any shape or value the rules refuse.
    """
    if data is None:
        raise AuthRuleError("auth is required: say how the fields go on a request.")
    if not isinstance(data, dict):
        raise AuthRuleError("auth must be an object.")
    kind = data.get("type")
    stored_none = stored and kind == AUTH_NONE
    if not stored_none and (not isinstance(kind, str) or kind not in AUTH_KEYS):
        types = sorted({*AUTH_KEYS, AUTH_NONE} if stored else AUTH_KEYS)
        raise AuthRuleError(f"auth.type must be one of: {', '.join(types)}.")
    _check_keys(data, frozenset({"type"}) if stored_none else AUTH_KEYS[kind], "auth")
    declared = frozenset(name for name, _ in fields)
    secrets = {name for name, secret in fields if secret}
    optional = frozenset(optional)
    if not optional <= secrets:
        raise AuthRuleError("An optional field must be a secret field the template declares.")
    if stored_none:
        placed: set[str] = set()
        canonical: dict[str, Any] = {"type": kind}
    elif kind == "header":
        headers, placed = _parse_headers(data.get("headers"), declared, looks_secret, optional)
        canonical = {"type": kind, "headers": headers}
    elif kind == "query":
        params, placed = _parse_params(data.get("params"), declared, looks_secret, optional)
        canonical = {"type": kind, "params": params}
    else:
        username, placed = check_template(data.get("username"), "auth.username", declared, looks_secret=looks_secret)
        if not placed or ":" in PLACEHOLDER.sub("", username):
            raise AuthRuleError("auth.username must place a field and hold no ':' outside it.")
        if "password" not in data:
            raise AuthRuleError("auth.password is required; use an empty text when the API takes none.")
        password, in_password = check_template(
            data["password"], "auth.password", declared, may_be_empty=True, looks_secret=looks_secret
        )
        _check_basic_optional(placed, password, in_password, optional, secrets)
        placed |= in_password
        canonical = {"type": kind, "username": username, "password": password}
    # Basic is kept to a secret even stored, so no request carries an identity with an empty password.
    if (not stored or kind == "basic") and not placed & secrets:
        raise AuthRuleError("auth must place a secret field, such as {api_key}.")
    unused = [i for i, (name, _) in enumerate(fields) if name not in placed]
    if unused:
        raise AuthRuleError(f"auth must place every field; fields[{unused[0]}] is never used.")
    return canonical


def _check_basic_optional(
    in_username: set[str], password: str, in_password: set[str], optional: frozenset[str], secrets: set[str]
) -> None:
    """Refuse a Basic section whose optional fields sit anywhere but the whole password.

    The username must place a required secret field whenever the password may
    be skipped, so a keyless request never carries an identity with no password.
    """
    if in_username & optional:
        raise AuthRuleError("auth.username may not place an optional field; only the password may be optional.")
    if not in_password & optional:
        return
    if len(in_password) > 1 or password != "{" + next(iter(in_password)) + "}":
        raise AuthRuleError("auth.password must be only the optional field, with nothing around it.")
    if not (in_username & secrets) - optional:
        raise AuthRuleError("An optional auth.password needs a username that places a required secret field.")


def _check_drops_only_missing(placed: set[str], missing: set[str]) -> None:
    """Refuse dropping a placement that also places a field the connection holds."""
    if placed & missing and placed - missing:
        raise AuthRuleError("A placement of a missing field also places a field the connection holds.")


def without_missing(auth: Mapping[str, Any], held: Iterable[str], optional: Iterable[str]) -> dict[str, Any]:
    """Take out of a template's auth section the placements of fields a connection does not hold.

    A header or parameter that places a missing field is dropped, and so
    must place nothing else the connection holds. A missing
    Basic password becomes empty, and a section left with no header or
    parameter becomes ``none``. The result is the section ``parse_auth`` takes
    with ``stored=True``. Computed once from the stored template, never from
    a request.

    Args:
        auth: The template's section as ``parse_auth`` returned it.
        held: The names of the fields the connection holds.
        optional: The names of the fields the template lets a member leave out.

    Returns:
        A new section; ``auth`` is not changed.

    Raises:
        AuthRuleError: when a missing field is not optional, a placement of a
            missing field also places a held one, or the type is not one these
            rules can apply.
    """
    held = set(held)
    optional = set(optional)
    kind = auth["type"]
    if kind == "basic":
        names = set(PLACEHOLDER.findall(auth["username"])) | set(PLACEHOLDER.findall(auth["password"]))
    elif kind in ("header", "query"):
        key = "headers" if kind == "header" else "params"
        names = {n for template in auth[key].values() for n in PLACEHOLDER.findall(template)}
    elif kind == AUTH_NONE:
        return {"type": AUTH_NONE}
    else:
        raise AuthRuleError("auth.type is not one these rules can apply.")
    if names - held - optional:
        raise AuthRuleError("A field that is not optional has no value.")
    missing = names - held
    if kind == "basic":
        if set(PLACEHOLDER.findall(auth["username"])) & missing:
            raise AuthRuleError("A field that is not optional has no value.")
        in_password = set(PLACEHOLDER.findall(auth["password"]))
        _check_drops_only_missing(in_password, missing)
        password = "" if in_password & missing else auth["password"]
        return {"type": kind, "username": auth["username"], "password": password}
    kept = {}
    for name, template in auth[key].items():
        placed = set(PLACEHOLDER.findall(template))
        _check_drops_only_missing(placed, missing)
        if not placed & missing:
            kept[name] = template
    return {"type": kind, key: kept} if kept else {"type": AUTH_NONE}
