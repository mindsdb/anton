"""Procedural memory: skills as multi-stage directories.

A *skill* is one concept with multiple representations that coexist:

  - Stage 1 (declarative.md) — step-by-step procedure the LLM reads. Always present.
  - Stage 2 (chunks.md)      — higher-level recipes/macros. Emerges from use. (v2+)
  - Stage 3 (code/)          — runnable helper modules. Emerges from reliability. (v2+)

Each skill lives at `~/.anton/skills/<label>/` as a directory:

    ~/.anton/skills/csv_summary/
    ├── meta.json          # label, name, description, when_to_use, provenance, presence flags
    ├── declarative.md     # Stage 1 — required
    ├── chunks.md          # Stage 2 — optional
    ├── code/              # Stage 3 — optional
    │   └── __init__.py
    ├── requirements.txt   # Stage 3 deps — optional
    └── stats.json         # per-stage usage counters

This module is the storage layer only — read/write/search. The classifier
(`recall_skill` tool) and the LLM-driven save command live elsewhere.

Brain analogue: cortico-striatal procedural memory. The executive (PFC)
recognizes a familiar pattern in the user's request, retrieves the
matching procedure from the striatum, and executes it. Stages coexist
rather than graduating — the executive picks the highest stage that's
reliable enough for the current context.

Relationship to `Engram` (anton/core/memory/hippocampus.py):
    `Engram` is the unit of *declarative* memory in Anton — a single
    fact, rule, or lesson stored as a flat bullet in rules.md / lessons.md /
    profile.md. Engrams are loaded into every prompt unconditionally because
    they're cheap (one line each). The brain-region analogue is the
    hippocampus → neocortex consolidation pathway.

    `Skill` is the unit of *procedural* memory in Anton — a multi-step
    workflow stored as a directory of staged representations. Skills are
    NOT loaded into every prompt; the LLM sees only their compact label +
    when_to_use line and explicitly retrieves the full procedure via the
    `recall_skill` tool when it recognizes a match. The brain-region
    analogue is the hippocampus → striatum / cerebellum pathway.

    Both systems coexist in the brain (declarative and procedural memory
    are dissociable — H.M. lost the former but kept the latter), and they
    coexist in Anton. Engrams hold facts; Skills hold procedures.

Naming note:
    The unique identifier for a skill is called its `label`. In cognitive
    psychology this is the declarative handle by which a procedural
    memory is addressed in working memory — the verbal token the
    executive holds when deciding to invoke a stored procedure. It is
    deliberately distinct from `name` (the human-readable display) and
    `when_to_use` (the retrieval cue describing the matching context).
"""

from __future__ import annotations

import difflib
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


_DEFAULT_SKILLS_ROOT = Path("~/.anton/skills").expanduser()


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StageStats:
    """Per-stage usage tracking.

    `recommended` increments every time the classifier (recall_skill tool)
    pulls this stage into context. `used` increments when scratchpad code
    actually imports a Stage 3 helper — Stage 1 and Stage 2 only have a
    `recommended` signal because there's no mechanical way to detect
    whether the LLM "followed" a markdown procedure.
    """

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

    Always carries the metadata. The Stage 1 markdown is loaded eagerly
    because it's small and almost always needed. Stage 2 and Stage 3
    content (when present) is loaded on demand by callers.

    The `label` is the declarative handle for this procedural memory —
    the snake_case identifier the LLM uses when it calls
    `recall_skill(label)`. It is the directory name on disk.
    """

    label: str
    name: str
    description: str
    when_to_use: str
    declarative_md: str
    created_at: str
    provenance: str  # "manual" | "consolidator" (future)
    stage_1_present: bool = True
    stage_2_present: bool = False
    stage_3_present: bool = False
    stats: SkillStats = field(default_factory=SkillStats)

    def to_meta_dict(self) -> dict:
        """Serialize the meta.json payload (excludes declarative content + stats)."""
        return {
            "label": self.label,
            "name": self.name,
            "description": self.description,
            "when_to_use": self.when_to_use,
            "created_at": self.created_at,
            "provenance": self.provenance,
            "stage_1_present": self.stage_1_present,
            "stage_2_present": self.stage_2_present,
            "stage_3_present": self.stage_3_present,
        }

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


_SLUG_RE = re.compile(r"[^a-z0-9_]+")


def slugify(text: str) -> str:
    """Normalize arbitrary text into a snake_case identifier.

    Strips non-alphanumerics, lowercases, collapses runs of underscores.
    Empty input becomes 'skill'. Used to produce path/URL-safe labels;
    the term "slugify" refers to the formatting operation, not to the
    semantic role of the result (which we call a `label`).
    """
    s = text.strip().lower().replace("-", "_").replace(" ", "_")
    s = _SLUG_RE.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "skill"


def make_unique_label(base: str, store: "SkillStore") -> str:
    """Return a label that doesn't collide with any existing skill.

    If `base` (after slugify normalization) is already unique, return it
    as-is. Otherwise append `_2`, `_3`, ... until a free slot is found.
    """
    candidate = slugify(base)
    if store.load(candidate) is None:
        return candidate
    n = 2
    while True:
        next_candidate = f"{candidate}_{n}"
        if store.load(next_candidate) is None:
            return next_candidate
        n += 1


# ─────────────────────────────────────────────────────────────────────────────
# Store
# ─────────────────────────────────────────────────────────────────────────────


class SkillStore:
    """File-backed store of skills under `~/.anton/skills/` (by default).

    Each skill is a directory whose name is its `label` (the snake_case
    declarative handle). The store is stateless — it reads from disk on
    demand. Callers should not cache Skill instances long-term, since
    stats are mutated through the store's increment helpers and a stale
    in-memory copy will drift.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else _DEFAULT_SKILLS_ROOT

    # ── reading ─────────────────────────────────────────────────────

    def _skill_dir(self, label: str) -> Path:
        return self.root / label

    def _ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def load(self, label: str) -> Skill | None:
        """Read a single skill by label. Returns None if absent or malformed."""
        d = self._skill_dir(label)
        meta_path = d / "meta.json"
        decl_path = d / "declarative.md"
        if not meta_path.is_file() or not decl_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        try:
            declarative = decl_path.read_text(encoding="utf-8")
        except OSError:
            return None
        stats = self._load_stats(label)
        return Skill(
            label=str(meta.get("label", label)),
            name=str(meta.get("name", label)),
            description=str(meta.get("description", "")),
            when_to_use=str(meta.get("when_to_use", "")),
            declarative_md=declarative,
            created_at=str(meta.get("created_at", "")),
            provenance=str(meta.get("provenance", "manual")),
            stage_1_present=bool(meta.get("stage_1_present", True)),
            stage_2_present=bool(meta.get("stage_2_present", False)),
            stage_3_present=bool(meta.get("stage_3_present", False)),
            stats=stats,
        )

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
            skill = self.load(child.name)
            if skill is not None:
                out.append(skill)
        return out

    def list_summaries(self) -> list[dict]:
        """Lightweight listing for prompt-building — label + when_to_use only.

        Avoids reading declarative.md for skills the LLM won't recall this turn.
        """
        if not self.root.is_dir():
            return []
        out: list[dict] = []
        for child in sorted(self.root.iterdir()):
            if not child.is_dir():
                continue
            meta_path = child / "meta.json"
            if not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            out.append(
                {
                    "label": str(meta.get("label", child.name)),
                    "name": str(meta.get("name", child.name)),
                    "when_to_use": str(meta.get("when_to_use", "")),
                }
            )
        return out

    # ── writing ─────────────────────────────────────────────────────

    def save(self, skill: Skill) -> Path:
        """Write the skill directory to disk. Overwrites in place.

        Returns the directory path. Creates the parent root if needed.
        Stage 2 and Stage 3 files are NOT touched here — they have their
        own writers (consolidator, future).
        """
        self._ensure_root()
        d = self._skill_dir(skill.label)
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(
            json.dumps(skill.to_meta_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        (d / "declarative.md").write_text(skill.declarative_md, encoding="utf-8")
        # Only initialize stats.json if it doesn't already exist — we
        # never want save() to wipe accumulated counts.
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
        """Atomic-ish bump of the per-stage `recommended` counter.

        Reads the existing stats.json, mutates the right field, writes
        back. Best-effort — if the skill doesn't exist or the file is
        unwritable, silently no-ops. Concurrent writers may race; that's
        acceptable for a counter that's used for guidance, not billing.
        """
        d = self._skill_dir(label)
        if not d.is_dir():
            return
        stats = self._load_stats(label)
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
        """Find the existing label closest to `bad_label`, or None.

        Used by the recall_skill tool to recover from typos and guesses.
        Cutoff is intentionally generous — we'd rather suggest a wrong
        match the LLM can reject than return nothing.
        """
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
    "make_unique_label",
    "slugify",
]
