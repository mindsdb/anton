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

logger = logging.getLogger(__name__)

# ─── spec constraints (used by validators, not enforced on read) ──────────────

SKILL_FILE = "SKILL.md"
DESC_MAX = 1024
_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")
_NAME_MAX = 64
_COMPAT_MAX = 500

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
    except OSError as e:
        return None

    lines = text.split("\n")
    if not lines or lines[0].rstrip() != "---":
        logger.debug("parse_skill_md: no opening '---' delimiter")
        return None

    close_idx: int | None = None
    for i, line in enumerate(lines[1:], 1):
        if line.rstrip() == "---":
            close_idx = i
            break

    if close_idx is None:
        logger.debug("parse_skill_md: no closing '---' delimiter")
        return None

    yaml_text = "\n".join(lines[1:close_idx])
    body = "\n".join(lines[close_idx + 1 :])

    try:
        props = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        logger.warning("parse_skill_md: YAML error: %s", exc)
        return None

    if not isinstance(props, dict):
        logger.warning("parse_skill_md: frontmatter is not a YAML mapping")
        return None

    # Collect metadata from the spec field, then fold in unknown top-level keys
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

    return AgentSkill.model_construct(
        name=normalize_name(name),
        instructions=body,
        description=str(props.get("description", "")),
        license=props.get("license"),
        compatibility=props.get("compatibility"),
        allowed_tools=props.get("allowed-tools"),
        metadata=meta,
    )


# ─── dump ─────────────────────────────────────────────────────────────────────


def dump_skill(skill: AgentSkill) -> str:
    """Serialise an AgentSkill back to SKILL.md text."""
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
