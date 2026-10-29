"""Embed observed JSON without making a model transcribe the records again.

Only the artifact's canonical observed-data.json handoff is embedded. Values
remain data: script delimiters are escaped before insertion into the document.
"""
from __future__ import annotations

import hashlib
import json
import re
from html.parser import HTMLParser
from urllib.parse import urlsplit

from anton.utils.datasources import scrub_output
from anton.utils.nofollow import read_bytes_nofollow

DATA_ID = "cowork-observed-json"
PLACES_ID = "cowork-places-json"
MARKER = "data-cowork-observed"
def _invalid_constant(value):
    raise ValueError(f"Non-JSON constant: {value}")


# Compared after lowercasing and dropping "-" and "_", so apiKey, x-api-key and API_KEY all match.
CREDENTIAL_KEYS = {
    "accesstoken", "refreshtoken", "idtoken", "token", "bearer", "authorization", "password", "passwd",
    "secret", "clientsecret", "apikey", "xapikey", "privatekey", "secretkey", "accesskey",
    "secretaccesskey", "awssecretaccesskey", "sessiontoken", "connectionstring", "dsn",
}
_CREDENTIAL_WORDS = {"password", "passwd", "secret", "token", "credential", "credentials",
                     "authorization", "auth", "bearer", "dsn"}
_KEY_QUALIFIERS = {"api", "auth", "access", "private", "secret", "client", "session",
                   "signing", "encryption", "aws", "ssh"}
_URL = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s<>\"']+", re.I)


def _credential_key(key):
    key = str(key)
    compact = re.sub(r"[-_]", "", key).lower()
    if compact in CREDENTIAL_KEYS:
        return True
    words = re.findall(r"[a-z0-9]+", re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key).lower())
    # Only the last word names what the value is: token_count and auth_status are ordinary columns.
    return bool(words) and (words[-1] in _CREDENTIAL_WORDS or
                            words[-1] == "key" and len(words) > 1 and words[-2] in _KEY_QUALIFIERS)


def _credential_url(text):
    for match in _URL.finditer(text):
        try:
            if urlsplit(match.group().rstrip(".,;)]}")).password is not None:
                return True
        except ValueError:
            continue
    return False


def read_handoff(folder):
    """The data channel stays inside this artifact and has an explicit size cap."""
    raw = read_bytes_nofollow(folder, folder / "observed-data.json", limit=4 * 1024 * 1024)
    value = json.loads(raw, parse_constant=_invalid_constant)
    if not isinstance(value, (dict, list)):
        raise ValueError("Structured source data must be an object or array.")
    return value


def _credential_structure(value):
    if isinstance(value, dict):
        return any(_credential_key(k) or _credential_structure(v)
                   for k, v in value.items())
    if isinstance(value, list):
        return any(_credential_structure(v) for v in value)
    if isinstance(value, str):
        return _credential_url(value)
    return False


def credential_bearing(value):
    if _credential_structure(value):
        return True
    try:
        wire = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return True
    return scrub_output(wire) != wire


def identity(value):
    wire = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return "data-" + hashlib.sha256(wire.encode()).hexdigest()[:12]


def contract(value, depth=0):
    """Describe the shape, small values and sample rows without repeating a dataset."""
    if depth > 6:
        return {"type": type(value).__name__}
    if isinstance(value, dict):
        return {k: contract(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) <= 8:
            return [contract(v, depth + 1) for v in value]
        result = {"array_length": len(value), "sample": [contract(v, depth + 1) for v in value[:2]]}
        if all(isinstance(v, dict) for v in value):
            result["record_keys"] = list(dict.fromkeys(k for v in value for k in v))
            # Column descriptors and other small enumerations need their names
            # as well as their schema. Large/free-text fields remain sampled.
            enumerations = {}
            for key in result["record_keys"]:
                entries = [v.get(key) for v in value]
                if all(isinstance(v, (str, int, float, bool, type(None))) for v in entries):
                    distinct = list(dict.fromkeys(entries))
                    if len(distinct) <= 64 and sum(len(str(v)) for v in distinct) < 800:
                        enumerations[key] = distinct
            if enumerations:
                result["distinct_values"] = enumerations
        return result
    if isinstance(value, str) and len(value) > 800:
        return {"string_length": len(value), "sample": value[:160]}
    return value


def runtime_contract(value, path, depth=0, *, compact=False):
    """Describe actual access expressions without making arrays look like wrapper objects."""
    if depth > 6:
        return [f"{path}: {type(value).__name__}; nested value retained in full."]
    if isinstance(value, dict):
        if compact:
            # Small summaries remain actual objects, without repeated paths.
            # Arrays retain their explicit runtime type and full-data binding.
            def has_array(item):
                return isinstance(item, list) or isinstance(item, dict) and any(has_array(v) for v in item.values())
            rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            if len(rendered) <= 6000 and not has_array(value):
                return [f"{path} is an OBJECT with these exact observed values: {rendered}"]
        lines = [f"{path} is an object with keys {json.dumps(list(value), ensure_ascii=False)}."]
        for key, item in value.items():
            lines.extend(runtime_contract(item, path + "[" + json.dumps(key, ensure_ascii=False) + "]", depth + 1, compact=compact))
        return lines
    if isinstance(value, list):
        example = value if len(value) <= 8 else value[:2]
        rendered = json.dumps(example, ensure_ascii=False)
        lines = [f"{path} is an ARRAY with {len(value)} elements; iterate this array directly. "
                 f"Example elements (descriptive only): {rendered[:4000]}" +
                 (" [example excerpt truncated]" if len(rendered) > 4000 else "")]
        summary = contract(value)
        if isinstance(summary, dict) and summary.get("record_keys"):
            lines.append("Each object element has source fields " + json.dumps(summary["record_keys"], ensure_ascii=False) + ".")
            if summary.get("distinct_values"):
                lines.append("Observed low-cardinality field values: " + json.dumps(summary["distinct_values"], ensure_ascii=False))
        return lines
    rendered = json.dumps(value, ensure_ascii=False)
    return [f"{path} = {rendered[:1000]}" + (" [value excerpt truncated]" if len(rendered) > 1000 else "")]


def strip_binding(html):
    html = re.sub(r'<script\b(?=[^>]*\bdata-cowork-observed\b)[^>]*>.*?</script\s*>', "", html,
                  flags=re.S | re.I)
    return re.sub(r'<meta\b(?=[^>]*\bdata-cowork-observed\b)[^>]*>', "", html, flags=re.I)


def previous_data(html):
    match = re.search(r'<script\b(?=[^>]*\bid=["\']cowork-observed-json["\'])[^>]*>(.*?)</script\s*>',
                      html, re.S | re.I)
    if match:
        try:
            value = json.loads(match.group(1), parse_constant=_invalid_constant)
            if isinstance(value, dict):
                return {key: item for key, item in value.items() if key == identity(item) and not credential_bearing(item)}
        except (ValueError, RecursionError):
            pass
    return {}


def collect(handoff, existing=""):
    """Bind a fresh handoff; an edit without one keeps only the page's previous data."""
    data = {identity(handoff): handoff} if handoff is not None else previous_data(existing)
    schemas = []
    for key, value in data.items():
        schemas.extend(runtime_contract(value, "window.ARTIFACT_DATA[" + json.dumps(key) + "]", compact=True))
    return data, "Embedded JSON contracts (actual runtime access paths):\n" + "\n".join(schemas)


def _json_script(script_id, name, value):
    # Escaping '<' also protects against </script>, HTML comments and tags in
    # untrusted data. JSON remains lossless when read through textContent.
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    return (f'<script id="{script_id}" type="application/json" {MARKER}>{payload}</script>'
            f'<script {MARKER}>window.{name}=JSON.parse(document.getElementById("{script_id}").textContent);</script>')


def bind(html, data, places=None):
    """Insert the observed data, and the places it names (places.index), before the page's scripts."""
    if not data:
        return html
    scripts = (f'<meta charset="utf-8" {MARKER}>' + _json_script(DATA_ID, "ARTIFACT_DATA", data)
               + (_json_script(PLACES_ID, "ARTIFACT_PLACES", places) if places else ""))
    html = strip_binding(html)
    line_starts = [0] + [match.end() for match in re.finditer("\n", html)]
    # HTMLParser ignores comments and understands quoted '>' in attributes;
    # a regex can otherwise insert the data inside a comment or an attribute.
    class Locations(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=False)
            self.ends = {}

        def handle_starttag(self, tag, attrs):
            if tag in {"html", "head"} and tag not in self.ends:
                line, column = self.getpos()
                self.ends[tag] = line_starts[line - 1] + column + len(self.get_starttag_text())

    locations = Locations()
    locations.feed(html)
    if end := locations.ends.get("head"):
        return html[:end] + scripts + html[end:]
    if end := locations.ends.get("html"):
        return html[:end] + "<head>" + scripts + "</head>" + html[end:]
    raise ValueError("A complete HTML document is required for data binding.")
