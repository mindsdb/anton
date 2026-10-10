"""Parsing and serialisation of the agentskills.io SKILL.md format.

SKILL.md layout:
    ---
    name: my-cat
    description: Short description (1-1024 chars)
    license: MIT                        # optional
    compatibility: requires network     # optional
    allowed-tools: read_file write_file # optional, space-separated
    metadata:
      display_name: My Cat
      provenance: manual
      created_at: "2026-06-15T15:20:42+00:00"
    ---
    Step-by-step body (= former declarative.md content)

Parsing is intentionally lenient — no field validation is applied when
reading, since files may be authored by external tools.  Validators on
AgentSkill exist for write/creation paths only and must be called
explicitly via AgentSkill.model_validate().

Unknown top-level YAML keys are folded into `metadata` automatically.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from anton.core.utils.yaml_cache import safe_load_cached

logger = logging.getLogger(__name__)

# ─── spec constraints (used by validators, not enforced on read) ──────────────

SKILL_FILE = "SKILL.md"
DESC_MAX = 1024
_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")
_NAME_MAX = 64
_COMPAT_MAX = 500

# The longest frontmatter parse_skill_dir parses. PyYAML takes about 300 bytes
# per character to parse, and a shipped skill's frontmatter is under 900
# characters.
_FRONTMATTER_MAX_CHARS = 64 * 1024

# A line that opens or closes the frontmatter: "---" and any trailing
# whitespace, up to the newline or the end of the text. Matched at the start of
# a line, it accepts the lines whose rstrip() is "---".
_DELIMITER_LINE = re.compile(r"---[^\S\n]*(?:\n|\Z)")

# canonical YAML keys defined by the spec
_SPEC_KEYS = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}


# ─── model ────────────────────────────────────────────────────────────────────

def validate_name(v: str) -> str:
    if len(v) > _NAME_MAX:
        raise ValueError(f"name exceeds {_NAME_MAX} chars")
    if "--" in v:
        raise ValueError("name must not contain '--'")
    if not _NAME_RE.match(v):
        raise ValueError(
            f"name {v!r} must match {_NAME_RE.pattern}"
        )
    return v


def normalize_name(value: str) -> str:
    slug = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)  # non-alnum runs -> single hyphen
    slug = re.sub(r"-{2,}", "-", slug).strip("-")

    return slug[:_NAME_MAX].rstrip("-")


class AgentSkill(BaseModel):
    """In-memory representation of a SKILL.md file (frontmatter + body).

    Validators are active when you call model_validate() (write / creation
    path).  parse_skill_dir() uses model_construct() so validators are skipped.
    """

    name: str = ""
    instructions: str
    description: str = ""
    license: str | None = None
    compatibility: str | None = None
    allowed_tools: str | None = Field(None, alias="allowed-tools")
    metadata: dict[str, str] = {}

    # ── validators (write / creation path only) ───────────────────────

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        return validate_name(v)

    @field_validator("description")
    @classmethod
    def _validate_description(cls, v: str) -> str:
        if not v:
            raise ValueError("description must not be empty")
        if len(v) > DESC_MAX:
            raise ValueError(f"description exceeds {DESC_MAX} chars")
        return v

    @field_validator("compatibility")
    @classmethod
    def _validate_compatibility(cls, v: str | None) -> str | None:
        if v is not None and len(v) > _COMPAT_MAX:
            raise ValueError(f"compatibility exceeds {_COMPAT_MAX} chars")
        return v

    @field_validator("metadata", mode="before")
    @classmethod
    def _coerce_metadata(cls, v: Any) -> dict[str, str]:
        if not isinstance(v, dict):
            return {}
        return {str(k): str(val) for k, val in v.items()}


# ─── parse ────────────────────────────────────────────────────────────────────


def parse_skill_dir(skill_dir: Path) -> AgentSkill | None:
    """Read ``<skill_dir>/SKILL.md`` into a ``Skill``"""
    folder_name = skill_dir.name
    md_path = skill_dir / "SKILL.md"

    try:
        text = md_path.read_text(encoding="utf-8")
    except OSError:
        return None
    except UnicodeDecodeError as exc:
        logger.warning(
            "parse_skill_md: SKILL.md is not UTF-8: error_type=%s", type(exc).__name__
        )
        return None

    opening = _DELIMITER_LINE.match(text)
    if opening is None:
        logger.debug("parse_skill_md: no opening '---' delimiter")
        return None

    # Look for the closing line only as far as the longest frontmatter this
    # reads, a line at a time, so a long file is never split into lines.
    yaml_start = opening.end()
    last_line_start = yaml_start + _FRONTMATTER_MAX_CHARS + 1
    line_start = yaml_start
    closing = _DELIMITER_LINE.match(text, line_start)
    while closing is None:
        newline = text.find("\n", line_start, last_line_start)
        if newline == -1:
            break
        line_start = newline + 1
        closing = _DELIMITER_LINE.match(text, line_start)

    if closing is None:
        if last_line_start < len(text):
            logger.warning(
                "parse_skill_md: frontmatter is longer than %d characters",
                _FRONTMATTER_MAX_CHARS,
            )
        else:
            logger.debug("parse_skill_md: no closing '---' delimiter")
        return None

    yaml_text = text[yaml_start : closing.start()].removesuffix("\n")
    body = text[closing.end() :]

    # safe_load_cached raises YAMLError when aliases would expand the
    # frontmatter without bound, so every str() below stays small. PyYAML
    # raises ValueError on a date or time zone out of range, or an integer too
    # long to read, and RecursionError on nesting deeper than the stack.
    try:
        props = safe_load_cached(yaml_text)
    except (yaml.YAMLError, ValueError, RecursionError) as exc:
        # PyYAML's message quotes the offending frontmatter lines, which can
        # hold a token, so the log names only the error class and position.
        # The mark counts from 0 within the frontmatter, which starts on the
        # file's second line.
        mark = getattr(exc, "problem_mark", None)
        logger.warning(
            "parse_skill_md: YAML error in SKILL.md: error_type=%s line=%s column=%s",
            type(exc).__name__,
            mark.line + 2 if mark is not None else "unknown",
            mark.column + 1 if mark is not None else "unknown",
        )
        return None

    if not isinstance(props, dict):
        logger.warning("parse_skill_md: frontmatter is not a YAML mapping")
        return None

    # Collect metadata from the spec field, then fold in unknown top-level keys.
    # str() raises ValueError on an integer too long to print in decimal.
    try:
        meta: dict[str, str] = {}
        spec_meta = props.get("metadata")
        if isinstance(spec_meta, dict):
            meta = {str(k): str(v) for k, v in spec_meta.items()}
        for k, v in props.items():
            if k not in _SPEC_KEYS:
                meta.setdefault(str(k), str(v))

        # check name
        name = props.get("name")
        if not name:
            name = folder_name
        name = str(name)
        description = str(props.get("description", ""))
    except ValueError as exc:
        logger.warning(
            "parse_skill_md: frontmatter value cannot be read as text: error_type=%s",
            type(exc).__name__,
        )
        return None

    return AgentSkill.model_construct(
        name=normalize_name(name),
        instructions=body,
        description=description,
        license=props.get("license"),
        compatibility=props.get("compatibility"),
        allowed_tools=props.get("allowed-tools"),
        metadata=meta,
    )


# ─── dump ─────────────────────────────────────────────────────────────────────


def dump_skill(skill: AgentSkill) -> str:
    """Serialize an AgentSkill back to SKILL.md text."""
    data: dict[str, Any] = {
        "name": skill.name,
        "description": skill.description,
    }
    if skill.license is not None:
        data["license"] = skill.license
    if skill.compatibility is not None:
        data["compatibility"] = skill.compatibility
    if skill.allowed_tools is not None:
        data["allowed-tools"] = skill.allowed_tools
    if skill.metadata:
        data["metadata"] = dict(skill.metadata)

    yaml_text = yaml.dump(
        data,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
    return f"---\n{yaml_text}---\n{skill.instructions}"


__all__ = ["AgentSkill", "parse_skill_dir", "dump_skill"]
