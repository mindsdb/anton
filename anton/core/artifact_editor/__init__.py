"""Artifact editor v2 (ENG-4207): edit a small artifact in one pass.

`patch` reads the patches a model writes and places them by content;
`workspace` is the artifact folder the editor reads and writes through.
"""

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
