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
    ARTIFACT_TYPES,
    METADATA_SCHEMA_VERSION,
    Artifact,
    ArtifactType,
    DatasourceRef,
    FileEntry,
    ProvenanceEntry,
    TurnEntry,
)
from anton.core.artifacts.internal_files import GENERATION_INPUT_FILES


logger = logging.getLogger(__name__)


METADATA_FILENAME = "metadata.json"
README_FILENAME = "README.md"
PUBLISHED_FILENAME = ".published.json"
BACKEND_LOG_FILENAME = "backend.log"

# Files the store owns, hold publish-state, or belong to a running backend —
# not artifact content the agent authored. Mirrors cowork-server's
# artifacts-service housekeeping set (`cowork/services/artifacts.py:132`) so the
# agent's view and the UI agree on what counts as an artifact file; the same set
# appears in `anton/publish_access.py` and `publisher._FULLSTACK_EXCLUDED`.
# `backend.log` is here for that agreement: it is the launched backend's runtime
# log, written into the artifact folder by `launch_artifact_backend`, and every
# other copy of this set already excluded it.
# The `.anton_state.db*` trio is the local STATE driver's SQLite database (the
# -wal/-shm side files carry the freshest writes) and
# `.state_manifest.published.json` is the publisher's schema snapshot — all
# runtime/publish bookkeeping of a stateful backend, never authored content.
# `state_manifest.json` itself is NOT here: it is a deliverable the publisher
# bundles. NOTE: cowork-server's copy of this set does not know these names yet.
_HOUSEKEEPING_FILES = {
    METADATA_FILENAME, README_FILENAME, PUBLISHED_FILENAME, BACKEND_LOG_FILENAME,
    ".anton_state.db", ".anton_state.db-wal", ".anton_state.db-shm",
    ".state_manifest.published.json",
}

# Kept separate from the housekeeping set rather than merged into it: these are
# authored by the generation tools, not owned by the store, and the set above
# mirrors cowork-server's — folding these in would quietly make that claim
# false. Both are excluded from `files[]`; only the reason differs.
_EXCLUDED_FROM_FILES = _HOUSEKEEPING_FILES | set(GENERATION_INPUT_FILES)

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


#: Width of `_new_id()`, plus the hyphen joining it to the name. Every slug ends
#: in `-<id>` (see `create`), and the name is trimmed by this much so the whole
#: slug still respects `_NAME_MAX_LEN`.
_ID_SUFFIX_LEN = 8 + 1


_UNSET = object()


def _sanitize_slug(value: str, max_len: int = _NAME_MAX_LEN) -> str:
    """Map any name to a folder-safe slug.

    Always returns a non-empty string. Strange characters collapse
    to hyphens; runs are deduped; leading/trailing punctuation is
    stripped. Lowercased so the slug reads consistently (matters for
    case-insensitive filesystems on macOS / Windows).

    The whitelist is ASCII, so a name in a non-Latin script collapses
    entirely and falls back to `_NAME_FALLBACK` — every artifact a
    Russian- or Chinese-speaking user names gets the SAME base. That is
    why `create` appends an id rather than relying on the name to
    distinguish folders.
    """
    raw = (value or "").strip().lower()
    cleaned = _NAME_DISALLOWED.sub("-", raw)
    cleaned = _NAME_HYPHEN_RUNS.sub("-", cleaned)
    cleaned = cleaned.strip("-._")
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip("-._")
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
        # `open`/`update` take the slug straight from tool input. Reject anything
        # that resolves outside the root (``..``, absolute, or symlink escape);
        # nested-but-contained paths are fine (create never emits them).
        candidate = (self._root / slug).resolve()
        root = self._root.resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError(f"artifact slug escapes the workspace: {slug!r}")
        return self._root / slug

    def metadata_path(self, slug: str) -> Path:
        return self.folder_for(slug) / METADATA_FILENAME

    def readme_path(self, slug: str) -> Path:
        return self.folder_for(slug) / README_FILENAME

    # ── Slug uniqueness ─────────────────────────────────────────

    def _unique_slug(self, base: str) -> str:
        """Append `-2`, `-3`, … on collision. Mirrors
        `projects_store.unique_name` semantics.

        A backstop since `create` began ending every slug in a random id:
        it now fires only on an id collision within one store, and cannot
        help across stores (see `create`), which is why it is not the
        thing keeping slugs distinct.
        """
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

        Slug is `<sanitised name>-<id>`, e.g. `sales-report-a1b2c3d4`.
        Returns the populated `Artifact`. The folder is empty other
        than the two metadata files — the agent writes its own
        files into it.

        The id is IN the folder name, not just in metadata.json, because
        the name alone does not identify an artifact:

        * `_sanitize_slug`'s whitelist is ASCII, so every artifact named
          in a non-Latin script collapses to the same `untitled-artifact`
          base;
        * `_unique_slug`'s `-2`/`-3` counter only sees the store it is
          rooted at, and on the cloud deployment each conversation gets
          its own store (the agent's workspace is a conversation, not a
          project), so the counter restarts per conversation and two
          conversations happily produce the same folder name.

        Together those made `project + slug` — which is how cowork-server
        addresses an artifact once paths stop being usable — ambiguous in
        the common case rather than the rare one.

        Only NEW artifacts get the suffix; existing folders keep their
        slugs, so consumers still cannot assume every slug carries one.

        `primary` (optional) is the relative path of the artifact's
        entry-point file. The renderer reads this to decide what to
        open by default. Falls back to a heuristic when None.
        Stored as-is — we don't validate it against the (empty)
        folder at create time, since the agent is about to write
        the file in the next scratchpad cell.
        """
        self.ensure_root()
        # The id is part of the slug, so it has to exist before it.
        artifact_id = _new_id()
        slug_base = _sanitize_slug(name, max_len=_NAME_MAX_LEN - _ID_SUFFIX_LEN)
        slug = self._unique_slug(f"{slug_base}-{artifact_id}")
        now = _utc_now()
        artifact = Artifact(
            schemaVersion=METADATA_SCHEMA_VERSION,
            id=artifact_id,
            slug=slug,
            createdAt=now,
            updatedAt=now,
            # `slug_base`, not `slug`: this is the human-facing name, and it
            # surfaces as the card title wherever the UI falls back to it
            # (cowork-server's card_for_folder, the chat stream adapter). The
            # id belongs in the folder name, not on screen.
            name=name.strip() or slug_base,
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

    def update(
        self,
        slug: str,
        *,
        primary: str | None = _UNSET,  # type: ignore[assignment]
        port: int | None = _UNSET,  # type: ignore[assignment]
        datasources: list[DatasourceRef] | None = _UNSET,  # type: ignore[assignment]
        type: ArtifactType | None = _UNSET,  # type: ignore[assignment]  # noqa: A002 (shadows builtin `type` — matches `create()`'s existing param name below)
    ) -> Artifact | None:
        """Update mutable agent-supplied fields on an existing artifact.

        Only fields explicitly passed are modified; omitted fields are
        left unchanged. Pass `primary=None` or `primary=""` to clear
        the entry-point pointer. Pass `port=None` to clear the port.
        Pass `datasources=[]` to clear the datasource list.

        `type` is validated against `ARTIFACT_TYPES` after the slug is
        confirmed to exist, but before anything is mutated — Pydantic's
        `Artifact` model has no `validate_assignment`, so an invalid value
        assigned directly would write corrupt JSON that only fails on the
        *next* load. A missing slug always returns `None` regardless of
        `type`'s validity (checked first): `None` already means "slug not
        found", so a bad `type` on top of a missing slug must not escalate
        that into a `ValueError` — the two causes stay distinguishable by
        checking existence before validity.

        Returns the updated artifact, or None when the slug is missing.
        """
        artifact = self._load_silent(slug)
        if artifact is None:
            return None
        if type is not _UNSET and type not in ARTIFACT_TYPES:
            raise ValueError(
                f"`type` must be one of {ARTIFACT_TYPES}. Got: {type!r}."
            )
        if primary is not _UNSET:
            artifact.primary = (
                primary.strip() if isinstance(primary, str) and primary.strip() else None
            )
        if port is not _UNSET:
            artifact.port = int(port) if port is not None else None
        if datasources is not _UNSET:
            artifact.datasources = list(datasources or [])
        if type is not _UNSET:
            artifact.type = type
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
                # Reconcile against disk so file counts are accurate even
                # though scratchpad writes bypass the store (see _reconcile_files).
                out.append(self._reconcile_files(artifact))
        out.sort(key=lambda a: a.updatedAt, reverse=True)
        return out

    def open(self, slug: str) -> Artifact | None:
        """Load an artifact by slug, reconciling `files[]` against disk.

        None when the folder doesn't exist or the metadata file is
        missing/corrupt. Reconciliation (see `_reconcile_files`) is what
        makes scratchpad-written files show up in `files[]` — the agent
        writes them directly to the folder, bypassing the store.
        """
        artifact = self._load_silent(slug)
        if artifact is None:
            return None
        return self._reconcile_files(artifact)

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
        """Reconcile `files[]` with what's actually on disk, by slug.

        Returns the (possibly refreshed) artifact, or None when the slug
        is missing. See `_reconcile_files` for the why.
        """
        artifact = self._load_silent(slug)
        if artifact is None:
            return None
        return self._reconcile_files(artifact)

    def _reconcile_files(self, artifact: Artifact) -> Artifact:
        """Re-derive `files[]` from disk for an already-loaded artifact.

        Scratchpad code writes artifact files straight into the folder via
        plain ``open()``, bypassing the store — so without this, `files[]`
        stays frozen at whatever ``create()``/``update()`` last set (usually
        empty), and ``open()``/``list()`` report file_count 0 for artifacts
        that are fully written on disk. The agent then concludes the file is
        missing and burns turns in a recovery loop.

        Persists (and bumps ``updatedAt``) ONLY when the on-disk file set
        actually changed, so this is safe and cheap to call on every read —
        no metadata/README churn and no spurious ``updatedAt`` bumps when
        nothing moved. Skips everything in ``_EXCLUDED_FROM_FILES``: the
        store's own files, the backend log, and the generation pipeline's
        inputs (`prd.md` / `spec.md` / `openapi.json`) — the last group sits
        in the folder but is not what the user asked to be built, and listing
        it invites the agent to present a spec as a deliverable.
        """
        folder = self.folder_for(artifact.slug)
        entries: list[FileEntry] = []
        for p in sorted(folder.rglob("*")):
            if not p.is_file() or p.is_symlink():
                continue
            # POSIX separators regardless of platform: FileEntry.path is
            # persisted to metadata.json and compared against the stored
            # fingerprint, so a Windows-written artifact must not disagree
            # with the same artifact written anywhere else.
            rel = p.relative_to(folder).as_posix()
            if rel in _EXCLUDED_FROM_FILES:
                continue
            try:
                stat = p.stat()
            except OSError:
                continue
            mtime_iso = datetime.fromtimestamp(
                stat.st_mtime, timezone.utc
            ).isoformat(timespec="seconds")
            entries.append(FileEntry(path=rel, bytes=stat.st_size, modifiedAt=mtime_iso))

        def _fingerprint(files: list[FileEntry]) -> list[tuple]:
            return sorted((f.path, f.bytes, f.modifiedAt) for f in files)

        if _fingerprint(entries) == _fingerprint(artifact.files):
            return artifact  # nothing changed on disk — don't rewrite metadata

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
        try:
            path = self.metadata_path(slug)
        except ValueError:
            # Escaping slug → treat as "no such artifact" so open/update stay graceful.
            logger.warning("Rejected out-of-workspace artifact slug %r", slug)
            return None
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
        if artifact.datasources:
            lines.append("## Data sources")
            for d in artifact.datasources:
                lines.append(f"- `{d.slug}` ({d.engine}) — env prefix `{d.env_prefix}`")
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
