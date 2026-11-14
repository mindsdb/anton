"""Artifact editor v2 (ENG-4207): edit a small artifact in one pass.

`patch` reads the patches a model writes and places them by content;
`workspace` is the artifact folder the editor reads and writes through;
`editor` is the one-conversation edit loop and `checkers` how it finds out
whether the artifact still works. `router` and `quick` are the artifact
viewer's quick-edit bar (ENG-4227): route a message, then edit directly.
"""

from anton.core.artifact_editor.checkers import CheckResult, Checker, Finding, select_checker
from anton.core.artifact_editor.editor import ArtifactEditor, EditUsage, Outcome
from anton.core.artifact_editor.quick import QuickEditResult, quick_edit_llm_from_settings, run_quick_edit
from anton.core.artifact_editor.router import (
    RouteDecision,
    decisions_available,
    decisions_config,
    describe_artifact,
    route_request,
)
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
    "QuickEditResult",
    "RouteDecision",
    "SearchHit",
    "Snapshot",
    "WorkspaceError",
    "decisions_available",
    "decisions_config",
    "describe_artifact",
    "quick_edit_llm_from_settings",
    "route_request",
    "run_quick_edit",
    "select_checker",
]
