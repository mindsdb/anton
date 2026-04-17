"""Local parsers for pasted credential inputs.

Dispatches pasted user text (a URI, JSON blob, `$ENV_VAR` reference, or
file path) to a format-specific parser and returns structured field
values keyed by engine field name. Pure and read-only: no environment
writes, no file writes, no network calls.

Returning ``None`` means no local format matched — the caller should
fall through to its existing flow (LLM extraction or field-by-field
prompting).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from anton.core.datasources.datasource_registry import DatasourceEngine


MAX_FILE_SIZE = 1024 * 1024  # 1 MiB


SCHEME_ALIASES: dict[str, set[str]] = {
    "postgres": {"postgres", "postgresql"},
    "postgresql": {"postgres", "postgresql"},
    "mysql": {"mysql", "mariadb"},
    "mariadb": {"mysql", "mariadb"},
    "redshift": {"redshift"},
    "pgvector": {"pgvector"},
    "timescaledb": {"timescaledb"},
}

ENV_REF_RE = re.compile(r"^\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?$")

QUERY_PARAM_KEYS = ("schema", "ssl", "sslmode")


@dataclass
class ParseResult:
    fields: dict[str, str]
    source: str


def parse_credential_input(
    text: str,
    engine_def: DatasourceEngine,
) -> ParseResult | None:
    """Return parsed fields or None if no local parser matched."""
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None

    if stripped.startswith(("/", "~/", "./", "../")):
        return parse_file(stripped, engine_def)
    if stripped.startswith("$"):
        return parse_env_ref(stripped, engine_def)
    if stripped.startswith("{"):
        return parse_json(stripped, engine_def)

    tokens = stripped.split()
    if tokens and "://" in tokens[0]:
        return parse_uri(stripped, engine_def)

    return None


def collect_field_names(engine_def: DatasourceEngine) -> set[str]:
    names = {f.name for f in engine_def.fields}
    for method in engine_def.auth_methods:
        for f in method.fields:
            names.add(f.name)
    return names


def parse_uri(
    text: str, engine_def: DatasourceEngine
) -> ParseResult | None:
    tokens = text.strip().split()
    if not tokens:
        return None
    token = tokens[0]
    if "://" not in token:
        return None
    try:
        parts = urlsplit(token)
    except ValueError:
        return None

    scheme = (parts.scheme or "").lower()
    if not scheme:
        return None

    engine_slug = (engine_def.engine or "").lower()
    allowed = SCHEME_ALIASES.get(engine_slug, {engine_slug})
    if scheme not in allowed:
        return None

    try:
        hostname = parts.hostname
        port = parts.port
        username = parts.username
        password = parts.password
    except ValueError:
        return None

    valid_names = collect_field_names(engine_def)
    fields: dict[str, str] = {}

    if hostname and "host" in valid_names:
        fields["host"] = hostname
    if port is not None and "port" in valid_names:
        fields["port"] = str(port)
    if username and "user" in valid_names:
        fields["user"] = unquote(username)
    if password and "password" in valid_names:
        fields["password"] = unquote(password)
    database = parts.path.lstrip("/") if parts.path else ""
    if database and "database" in valid_names:
        fields["database"] = database

    if parts.query:
        query = parse_qs(parts.query, keep_blank_values=False)
        for key in QUERY_PARAM_KEYS:
            values = query.get(key)
            if values and key in valid_names:
                fields[key] = values[0]

    if not fields:
        return None
    return ParseResult(fields=fields, source="uri")


def parse_json(
    text: str, engine_def: DatasourceEngine
) -> ParseResult | None:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    valid_names = collect_field_names(engine_def)
    fields: dict[str, str] = {}
    for key, value in data.items():
        if not isinstance(key, str) or key not in valid_names:
            continue
        if value is None:
            continue
        fields[key] = str(value)

    if not fields:
        return None
    return ParseResult(fields=fields, source="json")


def parse_env_ref(
    text: str, engine_def: DatasourceEngine
) -> ParseResult | None:
    match = ENV_REF_RE.match(text.strip())
    if not match:
        return None
    value = os.environ.get(match.group(1))
    if not value or not value.strip():
        return None
    
    inner = parse_uri(value, engine_def) or parse_json(value, engine_def)
    if inner is None:
        return None
    return ParseResult(fields=inner.fields, source="env")


def parse_file(
    text: str, engine_def: DatasourceEngine
) -> ParseResult | None:
    try:
        path = Path(text.strip()).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if not path.exists() or not path.is_file():
        return None
    try:
        if path.stat().st_size > MAX_FILE_SIZE:
            return None
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None

    inner = parse_json(content, engine_def) or parse_uri(content, engine_def)
    if inner is None:
        return None
    return ParseResult(fields=inner.fields, source="file")
