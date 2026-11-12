"""The artifact editor: edit a generated artifact's files with a diff."""

from anton.core.artifact_editor.workspace import (
    ArtifactWorkspace,
    MapEntry,
    SearchHit,
    Snapshot,
    WorkspaceError,
)

__all__ = [
    "ArtifactWorkspace",
    "MapEntry",
    "SearchHit",
    "Snapshot",
    "WorkspaceError",
]
