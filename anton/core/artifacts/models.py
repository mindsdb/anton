"""Pydantic models for artifact metadata.

Schema split:
  Server-managed (deterministic):
    id, slug, createdAt, updatedAt, files[], provenance[]
  Agent-supplied (validated at create_artifact time):
    name, description, type

The `Artifact` model is the on-disk source of truth — the README
that sits alongside it is rendered FROM the metadata, not the other
way around.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


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

    Re-derived from disk on every save (`ArtifactStore.rescan_files`)
    rather than mutated in place. The agent never writes here
    directly — it just creates the files; the server records them.
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
    Values are derived from the connection slug — `engine` and `name`
    match a `~/.anton/data_vault/<engine>-<name>` record; `env_prefix`
    is the `DS_<ENGINE>_<NAME>` token used to namespace the field-level
    env vars handed to the backend subprocess.
    """

    slug: str  # e.g. "postgres-prod_db"
    engine: str  # e.g. "postgres"
    name: str  # e.g. "prod_db"
    env_prefix: str  # e.g. "DS_POSTGRES_PROD_DB"


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
    id: str  # short hex (uuid4().hex[:8]) — stable across folder renames
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
