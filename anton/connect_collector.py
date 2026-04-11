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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from anton.datasource_registry import AuthMethod, DatasourceEngine, DatasourceField

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
    "You are helping fill out a form for a specific datasource."
)


# ─────────────────────────────────────────────────────────────────────────────
# LLM-facing schema (Pydantic) — used by LLMClient.generate_object
# ─────────────────────────────────────────────────────────────────────────────


class _ExtractionResult(BaseModel):
    """Structured output of extract_variables.

    The LLM is forced to call a tool whose input matches this schema,
    so the call site never has to parse JSON, strip markdown fences,
    or guard against non-dict responses.
    """

    variables: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Mapping of canonical field name (snake_case) to extracted "
            "value. Only include fields from the expected list above. "
            "Recognize common aliases (hostname→host, pwd→password, "
            "db→database, username→user) and map to the canonical name. "
            "If the user pasted a connection string (e.g. "
            "postgres://u:p@host:5432/db), extract host/port/user/"
            "password/database from it. If the user just provided a "
            "plain value for one field without naming it (e.g. typed "
            "'localhost' when asked for host), leave this empty — the "
            "caller will treat the raw text as the next field's value. "
            "Never invent values."
        ),
    )
    is_redirect: bool = Field(
        default=False,
        description=(
            "True ONLY if the user is clearly trying to cancel or switch "
            "to a DIFFERENT datasource (e.g. \"actually it's mysql\", "
            "\"never mind\", \"cancel\"). Providing credentials is NOT "
            "a redirect."
        ),
    )
    redirect_engine: str = Field(
        default="",
        description=(
            "If the user mentioned a different datasource by name (from "
            "the 'other known slugs' list), set this to that slug. "
            "Otherwise empty string."
        ),
    )
    redirect_reason: str = Field(
        default="",
        description="Short phrase describing the redirect, or empty string.",
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
    phrasing ("actually let's do mysql instead").

    Uses `LLMClient.generate_object` for forced-schema structured
    output — no manual JSON parsing or fence stripping. Falls back to
    an empty result on any LLM/validation error so the caller can
    treat the raw input as the next field's value.
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
        "Extract any credential values, then determine whether the user is "
        "trying to redirect to a different datasource."
    )

    try:
        extraction: _ExtractionResult = await session._llm.generate_object(
            _ExtractionResult,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=512,
        )
    except Exception:
        return result

    # Filter and normalize variables — only keep keys that match the
    # expected field list (the LLM might hallucinate field names).
    valid_names = {f.name for f in expected_fields}
    for k, v in extraction.variables.items():
        key = str(k).strip()
        value = str(v).strip()
        if key in valid_names and value:
            result.variables[key] = value

    result.is_redirect = extraction.is_redirect
    if extraction.redirect_engine.strip():
        result.redirect_engine = extraction.redirect_engine.strip()
    result.redirect_reason = extraction.redirect_reason.strip()

    return result
