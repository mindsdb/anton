"""Smart variable collection for the /connect flow.

Provides:
- `ConnectionCollector` — a state machine that tracks which credential
  fields have been filled vs. are still missing for a specific engine.
- `extract_variables()` — an LLM-driven parser that reads free-form user
  input and returns (a) the structured variables detected and (b) whether
  the user is redirecting (changing datasource, cancelling, etc).

The LLM handles all the messy cases naturally: natural language
("my host is db.example.com"), connection strings
(`postgres://u:p@host:5432/db`), aliases (pwd→password, hostname→host),
comma-separated lists, and redirect phrasing ("actually it's mysql").

This mirrors the LLM-returns-JSON pattern already used by
`handle_add_custom_datasource()` in anton/commands/datasource.py.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from anton.core.datasources.datasource_registry import (
    AuthMethod,
    DatasourceEngine,
    DatasourceField,
)

if TYPE_CHECKING:
    from rich.console import Console

    from anton.core.session import ChatSession


@dataclass
class ExtractedData:
    """Result of running extract_variables() on a user response."""

    variables: dict[str, str] = field(default_factory=dict)
    is_redirect: bool = False
    redirect_engine: str | None = None
    redirect_reason: str = ""


@dataclass
class ConnectionCollector:
    """Tracks the puzzle state of a single connection attempt.

    Holds the engine definition and which fields have been filled in so
    far. Use `fill_many()` to apply extracted variables and the
    `missing_*` / `is_complete` / `next_field` properties to drive the
    smart prompt loop.
    """

    engine_def: DatasourceEngine
    auth_method: AuthMethod | None = None
    collected: dict[str, str] = field(default_factory=dict)
    redirect_message: str = ""

    @property
    def active_fields(self) -> list[DatasourceField]:
        if self.auth_method is not None:
            return self.auth_method.fields
        return self.engine_def.fields

    @property
    def field_names(self) -> set[str]:
        return {f.name for f in self.active_fields}

    @property
    def missing_required(self) -> list[DatasourceField]:
        return [
            f for f in self.active_fields
            if f.required and not self.collected.get(f.name)
        ]

    @property
    def missing_optional(self) -> list[DatasourceField]:
        return [
            f for f in self.active_fields
            if not f.required and not self.collected.get(f.name)
        ]

    @property
    def is_complete(self) -> bool:
        return not self.missing_required

    @property
    def next_field(self) -> DatasourceField | None:
        """The next field to ask about — first missing required, else first missing optional."""
        if self.missing_required:
            return self.missing_required[0]
        if self.missing_optional:
            return self.missing_optional[0]
        return None

    def fill(self, key: str, value: str) -> bool:
        """Store value for a field. Returns True if accepted, False if unknown field."""
        if key not in self.field_names:
            return False
        if value:
            self.collected[key] = value
        return True

    def fill_many(self, pairs: dict[str, str]) -> list[str]:
        """Bulk-fill from a dict. Returns list of keys actually accepted."""
        accepted: list[str] = []
        for k, v in pairs.items():
            if self.fill(k, v):
                accepted.append(k)
        return accepted

    def format_status(self, console: "Console") -> None:
        """Print a Rich-formatted summary of what's filled vs. missing."""
        filled_active = [
            f.name for f in self.active_fields if self.collected.get(f.name)
        ]
        if filled_active:
            console.print(
                "        [anton.muted]Filled:[/] " + ", ".join(filled_active)
            )
        if self.missing_required:
            console.print(
                "        [anton.muted]Still needed:[/] "
                + ", ".join(f.name for f in self.missing_required)
            )

    def to_redirect_result(self) -> dict:
        """Serializable summary for the main agent when the user changes direction."""
        return {
            "status": "redirect",
            "engine": self.engine_def.engine,
            "engine_display": self.engine_def.display_name,
            "collected_variables": dict(self.collected),
            "missing_required": [f.name for f in self.missing_required],
            "redirect_message": self.redirect_message,
        }


_SYSTEM_PROMPT = (
    "You extract structured connection credentials from user messages. "
    "You are helping fill out a form for a specific datasource. "
    "Return ONLY valid JSON — no commentary, no markdown fences."
)


async def extract_variables(
    raw_input: str,
    *,
    expected_fields: list[DatasourceField],
    current_engine: str,
    current_engine_display: str,
    known_engine_slugs: list[str],
    session: "ChatSession",
) -> ExtractedData:
    """Use the LLM to parse free-form user input into connection variables.

    Returns an `ExtractedData` with:
      - `variables`: field name → value for any credentials detected
      - `is_redirect`: True if the user is changing direction
      - `redirect_engine`: the new engine slug if they named one
      - `redirect_reason`: a short description of the redirect

    Trusts the LLM to handle aliases (hostname→host, pwd→password),
    connection strings (postgres://user:pass@host:5432/db), natural
    language ("my host is db.example.com"), and free-form redirect
    phrasing ("actually let's do mysql instead"). Falls back to an empty
    result on any parse error — callers should treat an empty result as
    "treat the raw input as the next field's value".
    """
    result = ExtractedData()
    text = (raw_input or "").strip()
    if not text:
        return result

    field_lines = "\n".join(
        f"  - {f.name}{' (secret)' if f.secret else ''}: "
        f"{f.description or '(no description)'}"
        for f in expected_fields
    )
    other_engines = ", ".join(s for s in known_engine_slugs if s != current_engine)

    user_prompt = (
        f"Current datasource: {current_engine_display} (slug: {current_engine})\n"
        f"Expected fields for this datasource:\n{field_lines}\n\n"
        f"Other known datasource slugs: {other_engines}\n\n"
        f"The user was asked to provide credentials and wrote:\n"
        f"{text!r}\n\n"
        "Return ONLY valid JSON with this exact shape:\n"
        '{\n'
        '  "variables": {"<field_name>": "<value>", ...},\n'
        '  "is_redirect": true or false,\n'
        '  "redirect_engine": "<slug or empty string>",\n'
        '  "redirect_reason": "<short phrase or empty string>"\n'
        '}\n\n'
        "Rules:\n"
        "- Only include fields from the expected list above. Use the exact "
        "field names (snake_case).\n"
        "- Recognize common aliases (hostname→host, pwd→password, "
        "db→database, username→user, etc.) and map to the canonical name.\n"
        "- If the user pasted a connection string (e.g. "
        "postgres://u:p@host:5432/db), extract host/port/user/password/"
        "database from it.\n"
        "- Set `is_redirect` to true ONLY if the user is clearly trying to "
        "cancel or switch to a DIFFERENT datasource (e.g. \"actually it's "
        "mysql\", \"never mind\", \"cancel\"). Providing credentials is NOT "
        "a redirect.\n"
        "- If they mention a different datasource by name (from the other "
        "known slugs list), set `redirect_engine` to that slug.\n"
        "- If the user just provided a plain value for one field (e.g. "
        "typed \"localhost\" when asked for host), and did NOT mention a "
        "field name, leave `variables` empty — the caller will treat the "
        "raw text as the next field's value.\n"
        "- Never invent values. Only extract what the user explicitly wrote."
    )

    try:
        response = await session._llm.plan(
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=512,
        )
        content = (response.content or "").strip()
        # Strip optional markdown fences, same pattern as
        # handle_add_custom_datasource().
        content = re.sub(
            r"^```[^\n]*\n|```\s*$", "", content, flags=re.MULTILINE
        ).strip()
        data = json.loads(content)
    except Exception:
        return result

    if not isinstance(data, dict):
        return result

    raw_vars = data.get("variables") or {}
    if isinstance(raw_vars, dict):
        valid_names = {f.name for f in expected_fields}
        for k, v in raw_vars.items():
            if not isinstance(v, (str, int, float)):
                continue
            key = str(k).strip()
            if key in valid_names:
                value = str(v).strip()
                if value:
                    result.variables[key] = value

    result.is_redirect = bool(data.get("is_redirect"))
    redirect_engine = data.get("redirect_engine")
    if isinstance(redirect_engine, str) and redirect_engine.strip():
        result.redirect_engine = redirect_engine.strip()
    redirect_reason = data.get("redirect_reason")
    if isinstance(redirect_reason, str):
        result.redirect_reason = redirect_reason.strip()

    return result
