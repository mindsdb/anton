"""Procedural memory: skills as multi-stage directories.

A *skill* is one concept with multiple representations that coexist:

  - Stage 1 (SKILL.md body)  — step-by-step procedure the LLM reads. Always present.
  - Stage 2 (references/)    — higher-level recipes/macros. Emerges from use.
  - Stage 3 (scripts/)       — runnable helper modules. Emerges from reliability.

Each skill lives at `~/.anton/skills/<label>/` as a directory:

    ~/.anton/skills/csv-summary/
    ├── SKILL.md           # agentskills.io format: frontmatter + declarative body
    ├── references/        # Stage 2 — optional
    ├── scripts/           # Stage 3 — optional
    └── stats.json         # per-stage usage counters (internal sidecar)

Legacy format (meta.json + declarative.md) is migrated transparently on first
read via check_migrate().

Labels use hyphens (e.g. `my-cat`), not underscores.

Relationship to Engram (anton/core/memory/hippocampus.py):
    Engram is the unit of *declarative* memory — a single fact, rule, or
    lesson stored as a flat bullet in rules.md / lessons.md / profile.md.
    Engrams are loaded into every prompt unconditionally because they're cheap.

    Skill is the unit of *procedural* memory — a multi-step workflow stored
    as a directory of staged representations. Skills are NOT loaded into every
    prompt; the LLM sees only a compact label + description line and explicitly
    retrieves the full procedure via `recall_skill` when it recognizes a match.

    Both systems coexist the way they do in the brain — declarative and
    procedural memory are dissociable (H.M. lost the former but kept the
    latter). Engrams hold facts; Skills hold procedures.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from anton.core.tools.skill_format import AgentSkill, dump_skill, parse_skill_dir, normalize_name, DESC_MAX

logger = logging.getLogger(__name__)

_DEFAULT_SKILLS_ROOT = Path("~/.anton/skills").expanduser()


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StageStats:
    recommended: int = 0
    used: int = 0
    last_used: str = ""  # ISO timestamp
    confidence: float = 0.0


@dataclass
class SkillStats:
    total_recalls: int = 0
    stage_1: StageStats = field(default_factory=StageStats)
    stage_2: StageStats = field(default_factory=StageStats)
    stage_3: StageStats = field(default_factory=StageStats)


@dataclass
class Skill:
    """In-memory representation of a skill directory.

    `label` is the hyphen-case identifier (= SKILL.md `name` field = dir name).
    `name` is the human-readable display name (metadata.display_name).
    `stage_*_present` flags are derived from directory presence on disk, not
    stored explicitly: stage_1 is always True, stage_2 requires `references/`,
    stage_3 requires `scripts/`.
    """

    label: str
    name: str
    description: str
    declarative_md: str
    created_at: str
    provenance: str  # "manual" | "consolidator"
    when_to_use: str = ""
    stage_1_present: bool = True
    stage_2_present: bool = False
    stage_3_present: bool = False
    stats: SkillStats = field(default_factory=SkillStats)

    def to_stats_dict(self) -> dict:
        return {
            "total_recalls": self.stats.total_recalls,
            "stage_1": _stage_stats_to_dict(self.stats.stage_1),
            "stage_2": _stage_stats_to_dict(self.stats.stage_2),
            "stage_3": _stage_stats_to_dict(self.stats.stage_3),
        }


def _stage_stats_to_dict(s: StageStats) -> dict:
    return {
        "recommended": s.recommended,
        "used": s.used,
        "last_used": s.last_used,
        "confidence": s.confidence,
    }


def _stage_stats_from_dict(d: dict) -> StageStats:
    return StageStats(
        recommended=int(d.get("recommended", 0)),
        used=int(d.get("used", 0)),
        last_used=str(d.get("last_used", "")),
        confidence=float(d.get("confidence", 0.0)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Slug helpers
# ─────────────────────────────────────────────────────────────────────────────


_SLUG_RE = re.compile(r"[^a-z0-9-]+")


def slugify(text: str) -> str:
    """Normalize arbitrary text into a hyphen-case label.

    Strips non-alphanumerics (except hyphens), lowercases, collapses runs.
    Empty input becomes 'skill'.
    """
    s = text.strip().lower().replace("_", "-").replace(" ", "-")
    s = _SLUG_RE.sub("-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "skill"


def make_unique_label(base: str, store: "SkillStore") -> str:
    """Return a label that doesn't collide with any existing skill.

    Appends `-2`, `-3`, ... until a free slot is found.
    """
    candidate = slugify(base)
    if store.load(candidate) is None:
        return candidate
    n = 2
    while True:
        next_candidate = f"{candidate}-{n}"
        if store.load(next_candidate) is None:
            return next_candidate
        n += 1


# ─────────────────────────────────────────────────────────────────────────────
# Migration
# ─────────────────────────────────────────────────────────────────────────────


def check_migrate(skill_dir: Path, store_root: Path) -> Path | None:
    """Migrate a legacy skill directory to SKILL.md format in place.

    Old format: meta.json + declarative.md
    New format: SKILL.md (agentskills.io spec)

    Also renames the directory from snake_case to kebab-case and moves:
      code/      → scripts/

    Returns the (possibly renamed) directory path, or None if the directory
    contains neither legacy files nor a SKILL.md (unrecognised dir, skip it).
    Raises OSError on IO failures.
    Idempotent: returns skill_dir immediately if SKILL.md already exists.
    """
    skill_md_path = skill_dir / "SKILL.md"
    if skill_md_path.is_file():
        return skill_dir

    meta_path = skill_dir / "meta.json"
    decl_path = skill_dir / "declarative.md"
    if not meta_path.is_file() and not decl_path.is_file():
        return None

    # Load legacy data
    meta: dict = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    declarative = decl_path.read_text(encoding="utf-8") if decl_path.is_file() else ""

    old_label = skill_dir.name
    new_label = normalize_name(old_label)

    # Resolve collision when renaming
    final_label = new_label
    if new_label != old_label:
        target = store_root / new_label
        if target.exists():
            n = 2
            while True:
                candidate = f"{new_label}-{n}"
                if not (store_root / candidate).exists():
                    final_label = candidate
                    break
                n += 1

    description = str(meta.get("description", "")).strip()
    when_to_use = str(meta.get("when_to_use", "")).strip()
    if when_to_use:
        description = f"{description}. {when_to_use}" if description else when_to_use
    description = description[:DESC_MAX]

    metadata = {k: str(v) for k, v in {
        "display_name": meta.get("name", ""),
        "provenance": meta.get("provenance", "manual"),
        "created_at": meta.get("created_at", ""),
    }.items() if v}

    fm = AgentSkill.model_construct(
        name=final_label,
        description=description,
        instructions=declarative,
        metadata=metadata,
    )
    skill_md_path.write_text(dump_skill(fm), encoding="utf-8")

    # code/ → scripts/
    code_dir = skill_dir / "code"
    scripts_dir = skill_dir / "scripts"
    if code_dir.is_dir() and not scripts_dir.exists():
        shutil.move(str(code_dir), str(scripts_dir))

    # Remove legacy files
    meta_path.unlink(missing_ok=True)
    decl_path.unlink(missing_ok=True)

    # Rename directory
    if final_label != old_label:
        final_dir = store_root / final_label
        skill_dir.rename(final_dir)
        return final_dir

    return skill_dir


# ─────────────────────────────────────────────────────────────────────────────
# Store
# ─────────────────────────────────────────────────────────────────────────────


class SkillStore:
    """File-backed store of skills under `~/.anton/skills/` (by default).

    Each skill is a directory whose name is its `label` (hyphen-case).
    The store is stateless — reads from disk on demand.
    Legacy directories (meta.json + declarative.md) are migrated transparently
    on first access via check_migrate().
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else _DEFAULT_SKILLS_ROOT

    # ── internal helpers ─────────────────────────────────────────────

    def _skill_dir(self, label: str) -> Path:
        return self.root / label

    def _ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def _find_dir(self, label: str) -> Path | None:
        """Locate the directory for a label, tolerating legacy underscore names."""
        norm = label.replace("_", "-")
        d = self.root / norm
        if d.is_dir():
            return d
        if norm != label:
            old = self.root / label
            if old.is_dir():
                return old
        return None

    def _skill_from_dir(self, d: Path) -> Skill | None:
        """Build a Skill from a (already-migrated) directory."""
        fm = parse_skill_dir(d)
        if fm is None:
            return None

        return Skill(
            label=fm.name,
            name=fm.metadata.get("display_name", fm.name),
            description=fm.description,
            declarative_md=fm.instructions,
            created_at=fm.metadata.get("created_at", ""),
            provenance=fm.metadata.get("provenance", "manual"),
            stage_1_present=True,
            stage_2_present=(d / "references").is_dir(),
            stage_3_present=(d / "scripts").is_dir(),
            stats=self._load_stats(d.name),
        )

    # ── reading ─────────────────────────────────────────────────────

    def load(self, label: str) -> Skill | None:
        """Read a single skill by label. Returns None if absent or unreadable."""
        if not self.root.is_dir():
            return None
        d = self._find_dir(label)
        if d is None:
            return None

        d = check_migrate(d, self.root)
        if d is None:
            return None

        return self._skill_from_dir(d)

    def _load_stats(self, label: str) -> SkillStats:
        path = self._skill_dir(label) / "stats.json"
        if not path.is_file():
            return SkillStats()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return SkillStats()
        return SkillStats(
            total_recalls=int(data.get("total_recalls", 0)),
            stage_1=_stage_stats_from_dict(data.get("stage_1", {})),
            stage_2=_stage_stats_from_dict(data.get("stage_2", {})),
            stage_3=_stage_stats_from_dict(data.get("stage_3", {})),
        )

    def list_all(self) -> list[Skill]:
        """Return every loadable skill, sorted by label."""
        if not self.root.is_dir():
            return []
        out: list[Skill] = []
        for child in sorted(self.root.iterdir()):
            if not child.is_dir():
                continue
            try:
                child = check_migrate(child, self.root)
                if child is None:
                    continue
            except OSError:
                continue
            skill = self._skill_from_dir(child)
            if skill is not None:
                out.append(skill)
        return out

    def list_summaries(self) -> list[dict]:
        """Lightweight listing for prompt-building.

        Returns dicts with keys: label, name, description.
        Reads only SKILL.md frontmatter, skips the body.
        """
        if not self.root.is_dir():
            return []
        out: list[dict] = []
        for child in sorted(self.root.iterdir()):
            if not child.is_dir():
                continue
            try:
                child = check_migrate(child, self.root)
                if child is None:
                    continue
            except OSError:
                continue
            fm = parse_skill_dir(child)
            if fm is None:
                continue

            out.append({
                "label": fm.name,
                "name": fm.metadata.get("display_name", fm.name),
                "description": fm.description,
            })
        return out

    # ── writing ─────────────────────────────────────────────────────

    def save(self, skill: Skill) -> Path:
        """Write the skill directory to disk. Overwrites SKILL.md in place.

        Returns the directory path. Stage 2/3 files are not touched here.
        stats.json is only initialized if it doesn't already exist.
        """
        self._ensure_root()

        skill.label = normalize_name(skill.label)
        d = self._skill_dir(skill.label)
        d.mkdir(parents=True, exist_ok=True)

        metadata = {k: v for k, v in {
            "display_name": skill.name,
            "provenance": skill.provenance,
            "created_at": skill.created_at,
        }.items() if v}

        description = skill.description
        if skill.when_to_use:
            description = f"{description}. {skill.when_to_use}" if description else skill.when_to_use
        description = description[:DESC_MAX]

        fm = AgentSkill.model_construct(
            name=skill.label,
            description=description,
            instructions=skill.declarative_md,
            metadata=metadata,
        )
        (d / "SKILL.md").write_text(dump_skill(fm), encoding="utf-8")

        stats_path = d / "stats.json"
        if not stats_path.is_file():
            stats_path.write_text(
                json.dumps(skill.to_stats_dict(), indent=2) + "\n",
                encoding="utf-8",
            )
        return d

    def delete(self, label: str) -> bool:
        """Remove a skill directory. Returns True if it existed."""
        d = self._skill_dir(label)
        if not d.is_dir():
            return False
        shutil.rmtree(d)
        return True

    # ── stats updates ───────────────────────────────────────────────

    def increment_recommended(self, label: str, *, stage: int = 1) -> None:
        """Bump the per-stage `recommended` counter. Best-effort, no-ops on error."""
        d = self._find_dir(label)
        if d is None:
            return
        stats = self._load_stats(d.name)
        stats.total_recalls += 1
        target = self._stage_for(stats, stage)
        target.recommended += 1
        target.last_used = datetime.now(timezone.utc).isoformat()
        try:
            (d / "stats.json").write_text(
                json.dumps(
                    {
                        "total_recalls": stats.total_recalls,
                        "stage_1": _stage_stats_to_dict(stats.stage_1),
                        "stage_2": _stage_stats_to_dict(stats.stage_2),
                        "stage_3": _stage_stats_to_dict(stats.stage_3),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass

    @staticmethod
    def _stage_for(stats: SkillStats, stage: int) -> StageStats:
        if stage == 1:
            return stats.stage_1
        if stage == 2:
            return stats.stage_2
        if stage == 3:
            return stats.stage_3
        raise ValueError(f"Unknown stage: {stage}")

    # ── search ──────────────────────────────────────────────────────

    def closest_match(self, bad_label: str, *, cutoff: float = 0.6) -> str | None:
        """Find the existing label closest to `bad_label`, or None."""
        bad = slugify(bad_label)
        candidates = [s["label"] for s in self.list_summaries()]
        if not candidates:
            return None
        if bad in candidates:
            return bad
        matches = difflib.get_close_matches(bad, candidates, n=1, cutoff=cutoff)
        return matches[0] if matches else None


__all__ = [
    "Skill",
    "SkillStats",
    "SkillStore",
    "StageStats",
    "check_migrate",
    "make_unique_label",
    "slugify",
]
