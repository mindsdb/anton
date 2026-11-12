"""Artifact editor v2 (ENG-4207): edit a small artifact in one pass.

`patch` reads the patches a model writes and places them by content;
`workspace` is the artifact folder the editor reads and writes through;
`editor` is the one-conversation edit loop and `checkers` how it finds out
whether the artifact still works.
"""

from anton.core.artifact_editor.checkers import CheckResult, Checker, Finding, select_checker
from anton.core.artifact_editor.editor import ArtifactEditor, EditUsage, Outcome
from anton.core.artifact_editor.workspace import (
    ArtifactWorkspace,
    MapEntry,
    SearchHit,
    Snapshot,
    WorkspaceError,
)

__all__ = [
    "ArtifactEditor",
    "ArtifactWorkspace",
    "CheckResult",
    "Checker",
    "EditUsage",
    "Finding",
    "MapEntry",
    "Outcome",
    "SearchHit",
    "Snapshot",
    "WorkspaceError",
    "select_checker",
]
