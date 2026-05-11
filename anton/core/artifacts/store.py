"""ArtifactStore — CRUD over `<workspace>/artifacts/<slug>/`.

One folder per artifact. Each folder owns:
  - `metadata.json` — Pydantic-validated source of truth
  - `README.md`     — rendered from metadata, never authored by hand
  - The artifact's actual files (HTML, datasets, etc.)

Provenance accumulates across conversations: every turn that
touches files in the folder appends a `TurnEntry` (or upserts the
matching `ProvenanceEntry` for that conversation).

Slug naming follows the same convention as `projects_store.py`:
sanitize the name, suffix `-2` / `-3` / … on collision.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from anton.core.artifacts.models import (
    Artifact,
    ArtifactType,
    FileEntry,
    ProvenanceEntry,
    TurnEntry,
)


logger = logging.getLogger(__name__)


METADATA_FILENAME = "metadata.json"
README_FILENAME = "README.md"

# Same character whitelist projects_store uses — keeps slug shapes
# consistent across antontron's project names AND artifact slugs.
_NAME_DISALLOWED = re.compile(r"[^A-Za-z0-9._-]+")
_NAME_HYPHEN_RUNS = re.compile(r"-{2,}")
_NAME_MAX_LEN = 64
_NAME_FALLBACK = "untitled-artifact"

# Maximum turn-summary length stored in provenance. Long user
# prompts get truncated with an ellipsis; the full text always
# lives in the conversation history.
_SUMMARY_MAX = 240


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def _sanitize_slug(value: str) -> str:
    """Map any name to a folder-safe slug.

    Always returns a non-empty string. Strange characters collapse
    to hyphens; runs are deduped; leading/trailing punctuation is
    stripped. Lowercased so the slug reads consistently (matters for
    case-insensitive filesystems on macOS / Windows).
    """
    raw = (value or "").strip().lower()
    cleaned = _NAME_DISALLOWED.sub("-", raw)
    cleaned = _NAME_HYPHEN_RUNS.sub("-", cleaned)
    cleaned = cleaned.strip("-._")
    if len(cleaned) > _NAME_MAX_LEN:
        cleaned = cleaned[:_NAME_MAX_LEN].rstrip("-._")
    return cleaned or _NAME_FALLBACK


def _truncate_summary(text: str) -> str:
    text = (text or "").strip()
    if len(text) <= _SUMMARY_MAX:
        return text
    return text[: _SUMMARY_MAX - 1].rstrip() + "…"


class ArtifactStore:
    """File-backed artifact store rooted at `<workspace>/artifacts/`.

    Stateless beyond the root path — every method reads + writes the
    on-disk metadata.json directly, so concurrent calls (e.g. two
    chat sessions in the same workspace) Just Work as long as they
    don't race on the same artifact slug.
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)

    # ── Path helpers ────────────────────────────────────────────

    @property
    def root(self) -> Path:
        return self._root

    def ensure_root(self) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        return self._root

    def folder_for(self, slug: str) -> Path:
        return self._root / slug

    def metadata_path(self, slug: str) -> Path:
        return self.folder_for(slug) / METADATA_FILENAME

    def readme_path(self, slug: str) -> Path:
        return self.folder_for(slug) / README_FILENAME

    # ── Slug uniqueness ─────────────────────────────────────────

    def _unique_slug(self, base: str) -> str:
        """Append `-2`, `-3`, … on collision. Mirrors
        `projects_store.unique_name` semantics."""
        if not self.folder_for(base).exists():
            return base
        i = 2
        while True:
            candidate = f"{base}-{i}"
            if not self.folder_for(candidate).exists():
                return candidate
            i += 1

    # ── CRUD ────────────────────────────────────────────────────

    def create(
        self,
        *,
        name: str,
        description: str,
        type: ArtifactType,
        primary: str | None = None,
    ) -> Artifact:
        """Create a fresh artifact folder + metadata.json + README.

        Slug derives from the name (sanitised + collision suffix).
        Returns the populated `Artifact`. The folder is empty other
        than the two metadata files — the agent writes its own
        files into it.

        `primary` (optional) is the relative path of the artifact's
        entry-point file. The renderer reads this to decide what to
        open by default. Falls back to a heuristic when None.
        Stored as-is — we don't validate it against the (empty)
        folder at create time, since the agent is about to write
        the file in the next scratchpad cell.
        """
        self.ensure_root()
        slug_base = _sanitize_slug(name)
        slug = self._unique_slug(slug_base)
        now = _utc_now()
        artifact = Artifact(
            id=_new_id(),
            slug=slug,
            createdAt=now,
            updatedAt=now,
            name=name.strip() or slug,
            description=description.strip(),
            type=type,
            primary=(primary.strip() if isinstance(primary, str) and primary.strip() else None),
            files=[],
            provenance=[],
        )
        folder = self.folder_for(slug)
        folder.mkdir(parents=True, exist_ok=True)
        self._save(artifact)
        return artifact

    def set_primary(self, slug: str, primary: str | None) -> Artifact | None:
        """Update the primary-file pointer on an existing artifact.

        Used when the agent created with no `primary` and decided
        later, or when the primary file got renamed. Pass `None` to
        clear (the renderer reverts to the heuristic). Returns the
        updated artifact, or None when the slug is missing.
        """
        artifact = self._load_silent(slug)
        if artifact is None:
            return None
        artifact.primary = (
            primary.strip() if isinstance(primary, str) and primary.strip() else None
        )
        artifact.updatedAt = _utc_now()
        self._save(artifact)
        return artifact

    def list(self) -> list[Artifact]:
        """Every artifact under the root, sorted by `updatedAt` desc.

        Folders without a valid `metadata.json` are skipped (they're
        either incomplete writes mid-flight or user-dropped folders
        the agent never claimed). A warning logs once so we notice
        if it happens repeatedly.
        """
        self.ensure_root()
        out: list[Artifact] = []
        for child in self._root.iterdir():
            if not child.is_dir():
                continue
            artifact = self._load_silent(child.name)
            if artifact is not None:
                out.append(artifact)
        out.sort(key=lambda a: a.updatedAt, reverse=True)
        return out

    def open(self, slug: str) -> Artifact | None:
        """Load an artifact by slug. None when the folder doesn't
        exist or the metadata file is missing/corrupt."""
        return self._load_silent(slug)

    # ── Provenance + per-turn updates ───────────────────────────

    def record_turn(
        self,
        slug: str,
        *,
        conversation_id: str,
        conversation_title: str | None,
        turn_index: int,
        summary: str,
        files_touched: list[str],
    ) -> Artifact | None:
        """Append a turn to the artifact's provenance.

        Upserts the matching `ProvenanceEntry` for `conversation_id`:
        first call for a conversation creates the entry, subsequent
        calls within the same conversation append to its `turns[]`.
        Files are deduped per-turn (a turn that writes the same path
        twice still yields a single `files_touched` entry).

        Returns the updated artifact, or None when the slug is
        missing on disk.
        """
        artifact = self._load_silent(slug)
        if artifact is None:
            return None
        prov_entry = next(
            (p for p in artifact.provenance if p.conversation == conversation_id),
            None,
        )
        if prov_entry is None:
            prov_entry = ProvenanceEntry(
                conversation=conversation_id,
                title=conversation_title,
                turns=[],
            )
            artifact.provenance.append(prov_entry)
        elif conversation_title and prov_entry.title != conversation_title:
            # Conversation got renamed since the last turn — keep
            # the latest title so the README stays current.
            prov_entry.title = conversation_title
        prov_entry.turns.append(
            TurnEntry(
                index=turn_index,
                timestamp=_utc_now(),
                summary=_truncate_summary(summary),
                files_touched=sorted(set(files_touched)),
            )
        )
        artifact.updatedAt = _utc_now()
        self._save(artifact)
        return artifact

    def rescan_files(self, slug: str) -> Artifact | None:
        """Refresh `files[]` from disk. Skips `metadata.json` and
        `README.md` — those are housekeeping, not artifact content."""
        artifact = self._load_silent(slug)
        if artifact is None:
            return None
        folder = self.folder_for(slug)
        entries: list[FileEntry] = []
        for p in sorted(folder.rglob("*")):
            if not p.is_file() or p.is_symlink():
                continue
            rel = str(p.relative_to(folder))
            if rel in (METADATA_FILENAME, README_FILENAME):
                continue
            try:
                stat = p.stat()
            except OSError:
                continue
            mtime_iso = datetime.fromtimestamp(
                stat.st_mtime, timezone.utc
            ).isoformat(timespec="seconds")
            entries.append(FileEntry(path=rel, bytes=stat.st_size, modifiedAt=mtime_iso))
        artifact.files = entries
        artifact.updatedAt = _utc_now()
        self._save(artifact)
        return artifact

    def render_readme(self, slug: str) -> str | None:
        """Re-render README.md from the current metadata. Returns the
        rendered text, or None when the slug is missing."""
        artifact = self._load_silent(slug)
        if artifact is None:
            return None
        text = self._render_readme_text(artifact)
        self.readme_path(slug).write_text(text, encoding="utf-8")
        return text

    # ── Internals ───────────────────────────────────────────────

    def _save(self, artifact: Artifact) -> None:
        """Atomic write of metadata.json + re-render of README.md."""
        folder = self.folder_for(artifact.slug)
        folder.mkdir(parents=True, exist_ok=True)
        metadata_path = self.metadata_path(artifact.slug)
        # Pydantic v2: model_dump_json renders the JSON; round-trip
        # through json.loads → dump for indented output (writeable).
        payload = json.loads(artifact.model_dump_json())
        tmp = metadata_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(metadata_path)
        readme = self._render_readme_text(artifact)
        self.readme_path(artifact.slug).write_text(readme, encoding="utf-8")

    def _load_silent(self, slug: str) -> Artifact | None:
        path = self.metadata_path(slug)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return Artifact.model_validate(data)
        except Exception:
            logger.warning("Could not load artifact metadata at %s", path, exc_info=True)
            return None

    @staticmethod
    def _render_readme_text(artifact: Artifact) -> str:
        """Render the human-readable README from the metadata.

        Format mirrors what we agreed in the design — title, type +
        file-count line, description paragraph, file list, then a
        provenance section grouped by conversation. The rendering is
        purely deterministic (no LLM) so a re-render is idempotent
        as long as the metadata didn't change.
        """
        lines: list[str] = []
        lines.append(f"# {artifact.name}")
        file_count = len(artifact.files)
        meta_line = f"*{artifact.type} · {file_count} file{'s' if file_count != 1 else ''} · last updated {artifact.updatedAt}*"
        lines.append(meta_line)
        lines.append("")
        if artifact.description:
            lines.append(artifact.description)
            lines.append("")
        if artifact.files:
            lines.append("## Files")
            for f in artifact.files:
                size_kb = max(1, round(f.bytes / 1024))
                lines.append(f"- `{f.path}` ({size_kb} KB)")
            lines.append("")
        if artifact.provenance:
            lines.append("## Provenance")
            for entry in artifact.provenance:
                title = entry.title or entry.conversation
                lines.append(f"**Conversation: {title}**")
                for turn in entry.turns:
                    lines.append(f"- Turn {turn.index} — {turn.summary}")
                lines.append("")
        return "\n".join(lines).rstrip() + "\n"
