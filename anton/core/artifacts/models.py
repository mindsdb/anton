"""Pydantic models for artifact metadata.

Schema split:
  Server-managed (deterministic):
    schemaVersion, id, slug, createdAt, updatedAt, files[], provenance[]
  Agent-supplied (validated at create_artifact / update_artifact time):
    name, description, type, primary, port, datasources[]

The `Artifact` model is the on-disk source of truth — the README
that sits alongside it is rendered FROM the metadata, not the other
way around.

`schemaVersion` tags the on-disk layout so future format changes can
be migrated deterministically. Bump `METADATA_SCHEMA_VERSION` whenever
the shape changes incompatibly; records written before this field
existed load as version 1 (the field default).
"""

from __future__ import annotations

import re
from typing import Literal
from uuid import UUID, uuid5

from pydantic import BaseModel, Field, model_validator


# On-disk metadata.json layout version. Bump on incompatible changes
# and add a migration keyed off the loaded `schemaVersion`.
METADATA_SCHEMA_VERSION = 1

# A legacy artifact was created when `id` was only eight hex characters — too
# narrow to be a global identity. Widen it from fields that already survive
# folder/project moves instead of inventing a new value on every read. New
# artifacts get a full random id in the store.
_LEGACY_ARTIFACT_NAMESPACE = UUID("4ba9bdf8-3f0e-4ce5-beb0-8f00a8d955e7")

#: Width of the canonical `id`: a UUID spelled as bare lowercase hex.
ARTIFACT_ID_LEN = 32
#: Width of the `id` prefix that ends every slug `create()` mints.
ARTIFACT_ID_SLUG_PREFIX_LEN = 8

_LEGACY_ID_RE = re.compile(rf"^[0-9a-f]{{{ARTIFACT_ID_SLUG_PREFIX_LEN}}}$")

#: Hex-only and wider than a legacy id, yet not parseable as a UUID: a
#: truncated or padded identity, not a name. Widening it would mint a new one.
_DAMAGED_ID_RE = re.compile(rf"^[0-9a-fA-F]{{{ARTIFACT_ID_SLUG_PREFIX_LEN + 1},}}$")


def canonical_artifact_id(value: str) -> str:
    """Normalize any accepted spelling of an id to 32 lowercase hex chars.

    Raises `ValueError` on anything that is not a UUID — a corrupted identity
    must not be silently replaced, because that detaches the artifact from its
    published versions and comment threads.
    """
    return UUID(str(value)).hex


def extend_legacy_id(artifact_id: str, created_at: str) -> str:
    """Widen a short legacy id to a full 32-hex identity, deterministically.

    The old eight characters stay as the *prefix*, so the `-<id[:8]>` suffix
    already baked into the folder slug keeps addressing the same artifact. The
    24-character tail is derived, never random: anton widens in memory only
    while cowork-server persists, so both sides have to reach the same value
    without coordinating — otherwise the identity would be minted by whoever
    touched the artifact first and comment threads would fork.
    """
    derived = uuid5(_LEGACY_ARTIFACT_NAMESPACE, f"{artifact_id}:{created_at}").hex
    prefix = (artifact_id or "").strip().lower()
    if _LEGACY_ID_RE.match(prefix):
        return prefix + derived[ARTIFACT_ID_SLUG_PREFIX_LEN:]
    # An id that never was eight hex characters carries no slug contract worth
    # preserving; take the derived value whole.
    return derived


def resolve_artifact_id(raw_id: str, inherited_id: str, created_at: str) -> str:
    """Pick the canonical id for one metadata record's identity fields.

    `raw_id` is the `id` field, `inherited_id` the `stableId` field written by
    the short-lived two-field era.

    An `id` that already parses wins outright. That makes a stale `stableId`
    written by an older build inert, instead of letting it re-stamp an identity
    that published versions are already keyed under. Whenever `id` does NOT
    parse — the short legacy form, a name, or a damaged value — `stableId`
    decides if it is present: it already keyed those published versions, auth
    rules and comment threads, and keeping them bound is worth more than the
    folder slug's readable suffix (and more than re-deriving a value that
    nothing out there is keyed under).

    With no `stableId` to fall back on, everything else is widened rather than
    rejected: hand-written and very old records carry names in `id`
    (`"static-art"`), and refusing them would drop the artifact from every
    listing, which reads as a deletion. The one shape that does raise is a
    value that plausibly IS a damaged identity — hex-only and wider than a
    legacy id — because re-minting that would silently detach the artifact
    from its published versions, auth rules and comment threads.
    """
    raw = (raw_id or "").strip()
    try:
        return canonical_artifact_id(raw)
    except ValueError:
        pass
    inherited = (inherited_id or "").strip()
    if inherited:
        return canonical_artifact_id(inherited)
    if _DAMAGED_ID_RE.match(raw):
        raise ValueError(f"artifact id looks like a damaged UUID: {raw!r}")
    return extend_legacy_id(raw, created_at)


def artifact_key(artifact_id: str) -> str:
    """The `artifact/<uuid>` key drafts, published versions and comments share.

    Canonical dashed spelling: the upload lambda normalizes what it stores in
    `_meta.json` the same way, so both sides of the comments API agree.
    """
    return f"artifact/{UUID(str(artifact_id))}"


# Closed enum of artifact shapes. The renderer uses this to pick
# the right preview affordance (iframe sandbox for html-app /
# fullstack-stateless-app, "open" for documents, table preview for
# datasets, etc.).
ArtifactType = Literal[
    "html-app",
    "document",
    "dataset",
    "image",
    "mixed",
    "fullstack-stateless-app",
    "fullstack-stateful-app",
]

ARTIFACT_TYPES: tuple[str, ...] = (
    "html-app",
    "document",
    "dataset",
    "image",
    "mixed",
    "fullstack-stateless-app",
    "fullstack-stateful-app",
)


class FileEntry(BaseModel):
    """One file inside the artifact folder.

    Re-derived from disk on read (`ArtifactStore._reconcile_files`, called
    by `open()` / `list()`) rather than mutated in place. The agent never
    populates this directly — it writes the files into the folder via the
    scratchpad, and the store reconciles `files[]` against disk on access.
    """

    path: str  # relative to the artifact folder (e.g. "dashboard.html", "data/prices.csv")
    bytes: int
    modifiedAt: str  # ISO 8601 UTC


class TurnEntry(BaseModel):
    """A single conversation turn that touched the artifact.

    `summary` is the user's prompt for that turn (truncated) — NOT
    an LLM rewrite. Provenance is deterministic by design.
    """

    index: int  # turn index within the conversation (0-based)
    timestamp: str  # ISO 8601 UTC
    summary: str
    files_touched: list[str] = Field(default_factory=list)


class DatasourceRef(BaseModel):
    """A data-source connection that the artifact's backend reads from.

    Declared by the agent at backend-build time so the metadata can
    record which vault connections a fullstack artifact depends on.
    `engine` and `name` match a `~/.anton/data_vault/<engine>-<name>`
    record and are the only stored fields. `slug` and `env_prefix`
    are derived on access (not persisted): `slug` is `<engine>-<name>`;
    `env_prefix` is the `DS_<ENGINE>_<NAME>` token used to namespace the
    field-level env vars handed to the backend subprocess.
    """

    engine: str  # e.g. "postgres"
    name: str  # e.g. "prod_db"

    @property
    def slug(self) -> str:
        """`<engine>-<name>` — the vault connection identifier."""
        return f"{self.engine}-{self.name}"

    @property
    def env_prefix(self) -> str:
        """`DS_<ENGINE>_<NAME>` env-var namespace (special chars sanitized)."""
        from anton.core.datasources.data_vault import _slug_env_prefix

        return _slug_env_prefix(self.engine, self.name)


class ProvenanceEntry(BaseModel):
    """Provenance for a single conversation that contributed to the artifact.

    A given artifact may be modified across multiple conversations
    over time; we accumulate one ProvenanceEntry per conversation
    that ever touched it. Per-turn detail lives in `turns[]`.
    """

    conversation: str  # conversation id
    title: str | None = None
    turns: list[TurnEntry] = Field(default_factory=list)


class Artifact(BaseModel):
    """The full metadata.json contents.

    Pydantic-validated end-to-end so a corrupted record raises on
    load instead of silently round-tripping bad data.
    """

    # ── Server-managed identity / timestamps ─────────────────────
    # On-disk layout version. Records predating this field load as 1
    # (the default); `create()` stamps the current
    # `METADATA_SCHEMA_VERSION` on fresh artifacts.
    schemaVersion: int = 1
    # `uuid4().hex` — the artifact's one identity, stable across folder renames
    # and re-publishes. Drafts, published versions, revisions and comments all
    # key off it; `id[:8]` is the suffix carried by the folder slug.
    # Constrained so a record `_widen_identity` could not widen is rejected here
    # rather than surfacing later as `UUID('')` inside `artifact_key`.
    id: str = Field(pattern=rf"^[0-9a-f]{{{ARTIFACT_ID_LEN}}}$")
    slug: str  # matches folder name; sanitized from `name` with collision suffix
    createdAt: str
    updatedAt: str

    # ── Agent-supplied (Pydantic-validated at create_artifact) ──
    name: str
    description: str
    type: ArtifactType
    # Relative path (within the artifact folder) of the file that
    # acts as the artifact's entry point — `dashboard.html`,
    # `index.html`, `report.pdf`, etc. Optional: when None, the
    # renderer falls back to a heuristic (`index.html` →
    # newest `.html` → newest non-housekeeping file). Lets the
    # agent commit to a primary up front when it knows (which is
    # most cases — they generally know the filename they're going
    # to write).
    primary: str | None = None
    port: int | None = None

    # ── Agent-declared datasources (fullstack apps) ─────────────
    # Connections the backend reads from at runtime. Agent-supplied
    # via `update_artifact(datasources=[...])` — typically right
    # after writing `backend.py`, so the metadata stays in sync with
    # the env-var references in the code.
    datasources: list[DatasourceRef] = Field(default_factory=list)

    # ── Server-managed contents ─────────────────────────────────
    files: list[FileEntry] = Field(default_factory=list)
    provenance: list[ProvenanceEntry] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _widen_identity(cls, data: object) -> object:
        """Give pre-widening records their full id without a write-on-read.

        Runs before field validation so the retired `stableId` field is still
        visible in the raw document. Nothing is written back here — the value is
        derived deterministically, so recomputing it on every load is free of
        the mtime side effects a metadata rewrite would carry.
        """
        if not isinstance(data, dict):
            return data
        raw_id = str(data.get("id") or "")
        inherited = str(data.get("stableId") or "")
        if not raw_id and not inherited:
            # Nothing to widen from. Let the field constraint reject the record
            # rather than invent an identity — a wrong id detaches the artifact
            # from its published versions and comment threads.
            return data
        resolved = resolve_artifact_id(raw_id, inherited, str(data.get("createdAt") or ""))
        if resolved == data.get("id"):
            return data
        return {**data, "id": resolved}
