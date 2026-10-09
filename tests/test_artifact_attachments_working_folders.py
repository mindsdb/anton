"""generate_artifact attachments from the session's working folders."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from anton.core.tools.generate_artifact import attachments as att_mod
from anton.core.tools.generate_artifact import engine, orchestrator
from anton.core.tools.generate_artifact.attachments import refusal_reason, resolve_attachments
from anton.core.tools.tool_defs import _ATTACHMENTS_OUTSIDE, GENERATE_ARTIFACT_TOOL
from anton.core.tools.working_folders import working_folder_roots
from anton.workspace import Workspace


@pytest.fixture()
def layout(tmp_path):
    workspace = tmp_path / "project"
    folder = tmp_path / "docs"
    outside = tmp_path / "elsewhere"
    for d in (workspace, folder, outside):
        d.mkdir()
    return workspace, folder.resolve(), outside


def test_a_file_in_a_working_folder_is_accepted(layout):
    workspace, folder, _ = layout
    sheet = folder / "q3" / "sales.csv"
    sheet.parent.mkdir()
    sheet.write_text("a,b\n")

    kept, dropped = resolve_attachments([str(sheet)], workspace=workspace, extra_roots=(folder,))

    assert dropped == []
    assert [a.path for a in kept] == [str(sheet.resolve())]


@pytest.mark.parametrize("hidden", [".env", ".git/config", ".anton/notes.md"])
def test_a_hidden_file_in_a_working_folder_is_refused(layout, hidden):
    workspace, folder, _ = layout
    target = folder / hidden
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x")

    assert refusal_reason(target.resolve(), workspace, (folder,)) == att_mod.REFUSED_HIDDEN


def test_a_file_outside_every_root_is_refused(layout):
    workspace, folder, outside = layout
    stray = outside / "contract.pdf"
    stray.write_text("x")

    assert refusal_reason(stray.resolve(), workspace, (folder,)) == att_mod.REFUSED_OUTSIDE


def test_a_link_in_a_working_folder_pointing_outside_is_refused(layout):
    workspace, folder, outside = layout
    stray = outside / "contract.pdf"
    stray.write_text("x")
    (folder / "contract.pdf").symlink_to(stray)

    kept, dropped = resolve_attachments(
        [str(folder / "contract.pdf")], workspace=workspace, extra_roots=(folder,)
    )

    assert kept == []
    assert dropped == [f"{folder / 'contract.pdf'}: {att_mod.REFUSED_OUTSIDE}"]


def test_a_mock_session_has_no_working_folders():
    assert working_folder_roots(AsyncMock()) == ()


async def test_generate_passes_the_sessions_working_folders(layout, monkeypatch):
    workspace, folder, _ = layout
    sheet = folder / "sales.csv"
    sheet.write_text("a,b\n")
    art = workspace / "art"
    art.mkdir()
    seen = {}

    async def fake_run(state, *, entry):
        seen["attachments"] = list(state.attachments)
        return {"status": "generated", "files_written": [], "internal_files": [], "trace": []}

    monkeypatch.setattr(orchestrator, "run", fake_run)
    monkeypatch.setattr(engine, "_scratchpads_context", lambda session: "")
    monkeypatch.setattr(orchestrator, "_datasource_catalog", lambda session: None)
    session = AsyncMock()
    session._workspace = SimpleNamespace(base=workspace)
    session._working_folders = (folder,)

    await engine.generate(
        session=session, slug="a", artifact_path=art, artifact_type="html-app",
        user_request="r", agent_understanding="u", attachments=[str(sheet)],
    )

    assert [a.name for a in seen["attachments"]] == ["sales.csv"]


def _registered_generate_artifact(session):
    session._build_tools()
    return next(t for t in session.tool_registry.get_tool_defs() if t.name == "generate_artifact")


def test_the_widened_phrase_exists_in_the_definition():
    assert _ATTACHMENTS_OUTSIDE in GENERATE_ARTIFACT_TOOL.input_schema["properties"]["attachments"]["description"]


def test_without_working_folders_the_registered_tool_is_the_module_definition(make_session, tmp_path):
    tool = _registered_generate_artifact(make_session(workspace=Workspace(tmp_path)))

    assert tool is GENERATE_ARTIFACT_TOOL


def test_with_working_folders_the_attachments_text_allows_them(make_session, layout):
    workspace, folder, _ = layout
    pristine = copy.deepcopy(GENERATE_ARTIFACT_TOOL)

    tool = _registered_generate_artifact(
        make_session(workspace=Workspace(workspace), working_folders=(folder,))
    )

    text = tool.input_schema["properties"]["attachments"]["description"]
    assert "outside the workspace and this session's working folders" in text
    assert text.endswith("A file inside one of this session's working folders may be attached.")
    assert GENERATE_ARTIFACT_TOOL == pristine


def test_a_working_folder_holding_the_workspace_never_refuses_what_the_workspace_allows(tmp_path):
    """Working folders only widen: an upload under the workspace's `.anton/uploads`
    stays accepted when a working folder happens to contain the workspace."""
    workspace = tmp_path / "proj"
    upload = workspace / ".anton" / "uploads" / "a.png"
    upload.parent.mkdir(parents=True)
    upload.write_bytes(b"x")

    assert refusal_reason(upload.resolve(), workspace, ()) is None
    assert refusal_reason(upload.resolve(), workspace, (tmp_path.resolve(),)) is None
