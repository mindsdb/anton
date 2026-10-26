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

A second, read-only root ships inside the package (`builtin_skills/` next to
this module) for skills every install must carry from the first session (e.g.
the backend/fullstack and HTML-dashboard contracts). Built-ins appear in
listings and `recall_skill` lookups like user skills; a user skill with the
same label shadows the built-in (logged), and an unreadable user dir falls
back to the built-in.

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

# Read-only skills shipped inside the package. They appear in listings and
# recall_skill lookups like user skills, but live in the codebase so every
# install has them from the first session. A user/consolidator skill with the
# same label shadows the built-in.
_BUILTIN_SKILLS_ROOT = Path(__file__).parent / "builtin_skills"

# One value of a skill's `applies_to` list.
_APPLIES_TO_VALUE_RE = re.compile(r"[a-z0-9_-]+")


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
    provenance: str  # "manual" | "consolidator" | "builtin" | "host"
    when_to_use: str = ""
    # The SKILL.md `applies_to` key, at the top level or under `metadata`
    # (the parser folds the first into the second), as written. See
    # `for_artifacts`.
    applies_to: str = ""
    stage_1_present: bool = True
    stage_2_present: bool = False
    stage_3_present: bool = False
    stats: SkillStats = field(default_factory=SkillStats)

    @property
    def for_artifacts(self) -> bool:
        """Whether `applies_to` names `artifacts`.

        One value, a comma- or space-separated list, or a YAML list, which
        the parser keeps as its `str()` ("['artifacts', 'chat']").
        """
        return "artifacts" in _APPLIES_TO_VALUE_RE.findall(self.applies_to.lower())

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
    contains neither legacy files nor a SKILL.md, or a declarative.md that
    isn't UTF-8 (unrecognised dir, skip it). A meta.json that isn't a JSON
    object is read as empty. Raises OSError on IO failures.
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
    # A walk of the store runs this on every folder, and a store built with
    # host-named roots walks at construction, so a folder that can't be read
    # must be skipped rather than raise: it would fail every turn.
    meta: dict = {}
    if meta_path.is_file():
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):  # JSONDecodeError, UnicodeDecodeError
            loaded = None
        if isinstance(loaded, dict):
            meta = loaded

    try:
        declarative = decl_path.read_text(encoding="utf-8") if decl_path.is_file() else ""
    except UnicodeDecodeError:
        return None

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

    def __init__(
        self,
        root: Path | None = None,
        builtin_root: Path | None = None,
        extra_roots: list[Path] | None = None,
        artifact_roots: list[Path] | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else _DEFAULT_SKILLS_ROOT
        # Resolved at construction so tests can monkeypatch the module default.
        self.builtin_root = (
            Path(builtin_root) if builtin_root is not None else _BUILTIN_SKILLS_ROOT
        )
        # Read-only skill roots supplied by the host (e.g. cowork-server ships
        # skills for its own tools). Take precedence over anton's packaged
        # built-ins; user skills under `root` still shadow everything.
        self.extra_roots: list[Path] = [Path(p) for p in extra_roots or []]
        # Folders the host trusts for artifact builds (`list_for_artifacts`);
        # None when it named none.
        self.artifact_roots: list[Path] | None = (
            None if artifact_roots is None else [Path(p) for p in artifact_roots]
        )
        # Read now, when the host names the trusted folders at the start of
        # its turn: they are paths, so a skill deleted during the turn and
        # made again under its name by someone else would otherwise be
        # followed. None without host-named folders, read live instead.
        self._pinned_artifact_skills: list[Skill] | None = (
            None if self.artifact_roots is None else self._trusted_artifact_skills()
        )

    def _readonly_roots(self) -> list[Path]:
        """Host roots first (they win on label collisions), then built-ins."""
        return [*self.extra_roots, self.builtin_root]

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
            applies_to=fm.metadata.get("applies_to", ""),
            stage_1_present=True,
            stage_2_present=(d / "references").is_dir(),
            stage_3_present=(d / "scripts").is_dir(),
            stats=self._load_stats(d.name),
        )

    # ── reading ─────────────────────────────────────────────────────

    def dir_for(self, label: str) -> Path | None:
        """The user-root directory holding `label`, or None.

        Built-ins and `extra_roots` are deliberately not searched: this exists
        for callers that copy a skill's files to edit them, and those roots are
        read-only.
        """
        return self._find_dir(label)

    def load(self, label: str) -> Skill | None:
        """Read a single skill by label. Returns None if absent or unreadable.

        User skills (under `root`) shadow built-in skills with the same label;
        the shadow is logged so a silent swap of a built-in contract is at
        least visible. A user dir that exists but cannot be parsed falls back
        to the built-in rather than dead-ending the label.
        """
        if self.root.is_dir():
            d = self._find_dir(label)
            if d is not None:
                try:
                    migrated = check_migrate(d, self.root)
                except OSError:
                    migrated = None
                skill = self._skill_from_dir(migrated) if migrated is not None else None
                if skill is not None:
                    if (self.builtin_root / skill.label).is_dir():
                        logger.info(
                            "skill %r: user skill shadows the built-in skill "
                            "of the same label", skill.label,
                        )
                    return skill
                fallback = self._load_builtin(label)
                if fallback is not None:
                    logger.warning(
                        "skill dir %r exists but is unreadable; falling back "
                        "to the built-in skill", label,
                    )
                    return fallback
                return None
        return self._load_builtin(label)

    def _load_builtin(self, label: str) -> Skill | None:
        norm = label.replace("_", "-")
        for base in self._readonly_roots():
            d = base / norm
            if not d.is_dir():
                continue
            skill = self._skill_from_dir(d)
            if skill is None:
                continue
            # Only anton's own root is force-labelled "builtin"; host roots
            # keep whatever provenance their SKILL.md declares.
            if base == self.builtin_root:
                skill.provenance = "builtin"
            return skill
        return None

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
        """Return every loadable skill (built-ins included), sorted by label."""
        out: list[Skill] = []
        for child in self._iter_skill_dirs():
            skill = self._skill_from_dir(child)
            if skill is not None:
                if child.parent == self.builtin_root:
                    skill.provenance = "builtin"
                out.append(skill)
        out.sort(key=lambda s: s.label)
        return out

    def list_for_artifacts(self) -> list[Skill]:
        """The skills an artifact build follows without a recall, by label.

        A skill qualifies when it is marked (`Skill.for_artifacts`) and its
        folder, with links resolved, sits under a trusted folder: anton's
        built-ins always, plus `artifact_roots` when the host named them.
        Resolving matters because a host names the folders its own links
        point into, so a real folder planted where one of its links was
        resolves elsewhere and is left out.

        With no `artifact_roots`, a store with host roots trusts nothing
        else: a host's writable root can hold other people's skills, and
        only the host can tell them apart. A store without host roots (the
        CLI) also trusts every skill listed in `root`, the user's own
        library, links to folders elsewhere included.

        With `artifact_roots`, the skills are the ones read when the store
        was built, so a turn follows what its host trusted when it began.
        """
        if self._pinned_artifact_skills is not None:
            return list(self._pinned_artifact_skills)
        return self._trusted_artifact_skills()

    def _trusted_artifact_skills(self) -> list[Skill]:
        """`list_for_artifacts`, read from disk now."""
        out: list[Skill] = []
        for d in self._iter_skill_dirs():
            if not self._trusts_for_artifacts(d):
                continue
            skill = self._skill_from_dir(d)
            if skill is not None and skill.for_artifacts:
                out.append(skill)
        out.sort(key=lambda s: s.label)
        return out

    def _trusts_for_artifacts(self, d: Path) -> bool:
        """Whether an artifact build may follow the skill in folder `d`
        without a recall: the rule `list_for_artifacts` describes."""
        # Nobody else writes the CLI's library, so where its links lead is
        # the user's choice, not a planted folder.
        if self.artifact_roots is None and not self.extra_roots and d.parent == self.root:
            return True
        resolved = d.resolve()
        return any(
            resolved.is_relative_to(t.resolve())
            for t in [self.builtin_root, *(self.artifact_roots or [])]
        )

    def load_for_artifacts(self, label: str) -> Skill | None:
        """`load(label)` kept to folders an artifact build trusts
        (`_trusts_for_artifacts`), marked or not, or None.

        For a skill a generator step reads by name, such as the fetch step's
        `public-data-sources`. Through `load`, any folder under that name in
        `root`, which on a host can link another person's skill, would
        replace anton's built-in. Here an untrusted folder is passed over and
        the next one in `load`'s order is read: the host roots, then the
        built-ins.
        """
        candidates: list[Path] = []
        if self.root.is_dir():
            own = self._find_dir(label)
            if own is not None:
                candidates.append(own)
        norm = label.replace("_", "-")
        candidates += [base / norm for base in self._readonly_roots()]
        for d in candidates:
            if not d.is_dir() or not self._trusts_for_artifacts(d):
                continue
            if d.parent == self.root:
                try:
                    migrated = check_migrate(d, self.root)
                except OSError:
                    continue
                if migrated is None:
                    continue
                d = migrated
            skill = self._skill_from_dir(d)
            if skill is None:
                continue
            if d.parent == self.builtin_root:
                skill.provenance = "builtin"
            return skill
        return None

    def list_summaries(self) -> list[dict]:
        """Lightweight listing for prompt-building.

        Returns dicts with keys: label, name, description.
        Reads only SKILL.md frontmatter, skips the body.
        """
        out: list[dict] = []
        for child in self._iter_skill_dirs():
            fm = parse_skill_dir(child)
            if fm is None:
                continue
            out.append({
                "label": fm.name,
                "name": fm.metadata.get("display_name", fm.name),
                "description": fm.description,
            })
        out.sort(key=lambda s: s["label"])
        return out

    def _readonly_dirs(self) -> list[Path]:
        """All read-only skill dirs (host roots + built-ins), deduped by name.

        Host roots are walked first, so a host skill shadows an anton built-in
        sharing its label — matching `_load_builtin`'s precedence.
        """
        dirs: list[Path] = []
        seen: set[str] = set()
        for base in self._readonly_roots():
            if not base.is_dir():
                continue
            for d in sorted(base.iterdir()):
                if d.is_dir() and (d / "SKILL.md").is_file() and d.name not in seen:
                    dirs.append(d)
                    seen.add(d.name)
        return dirs

    def _iter_skill_dirs(self) -> list[Path]:
        """User skill dirs (migrated) plus built-in dirs not shadowed by one.

        Shared walk for list_all()/list_summaries(); load() resolves single
        labels itself so it can fall back when a user dir is unreadable.
        """
        dirs: list[Path] = []
        seen: set[str] = set()
        if self.root.is_dir():
            for child in sorted(self.root.iterdir()):
                if not child.is_dir():
                    continue
                try:
                    child = check_migrate(child, self.root)
                except OSError:
                    continue
                if child is None:
                    continue
                dirs.append(child)
                seen.add(child.name)
        dirs.extend(d for d in self._readonly_dirs() if d.name not in seen)
        return dirs

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
            "applies_to": skill.applies_to,
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
