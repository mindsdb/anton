"""Project-visible artifacts.

Each artifact is a folder under `<workspace>/artifacts/<slug>/` carrying:
  - `metadata.json` — structured truth (Pydantic-validated)
  - `README.md`     — human-readable rendering of the metadata
  - The artifact's own files (HTML, datasets, etc.)

Replaces the legacy flat `<workspace>/.anton/output/` dump:
  - One subfolder per artifact (multi-file outputs cluster)
  - Per-folder metadata + provenance (which conversation, which turns)
  - Visible at the project root (not hidden under `.anton/`)

Provenance is server-managed (deterministic). Only `name`,
`description`, and `type` are agent-supplied at creation time.
"""

from anton.core.artifacts.models import (
    ARTIFACT_TYPES,
    Artifact,
    ArtifactType,
    FileEntry,
    ProvenanceEntry,
    TurnEntry,
)
from anton.core.artifacts.snapshot import (
    DirSnapshot,
    diff_snapshots,
    snapshot_dir,
)
from anton.core.artifacts.store import ArtifactStore

__all__ = [
    "ARTIFACT_TYPES",
    "Artifact",
    "ArtifactStore",
    "ArtifactType",
    "DirSnapshot",
    "FileEntry",
    "ProvenanceEntry",
    "TurnEntry",
    "diff_snapshots",
    "snapshot_dir",
]
