"""handle_read_text_file: roots from the session, the policy, and every verdict."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from anton.core.artifacts import ArtifactStore
from anton.core.tools.registry import ToolOutcome
from anton.core.tools.tool_handlers import handle_create_artifact, handle_read_text_file


class _Workspace:
    def __init__(self, base: Path) -> None:
        self.base = base
        self.artifacts_dir = base / ".anton" / "artifacts"


class _Session:
    def __init__(self, base: Path) -> None:
        self._workspace = _Workspace(base)
        self._session_id = "conv-1"
        self._turn_count = 0
        self._artifacts_touched: set[str] = set()
        self._data_vault = None


@pytest.fixture
def project(tmp_path: Path) -> Path:
    base = tmp_path / "project"
    base.mkdir()
    return base


@pytest.fixture
def session(project: Path) -> _Session:
    return _Session(project)


async def _read(session, **tc_input) -> ToolOutcome:
    return await handle_read_text_file(session, tc_input)


async def test_relative_path_is_read_from_the_project_root(session, project):
    (project / "notes.md").write_text("hello\nworld\n")
    out = await _read(session, path="notes.md")
    assert out.ok is True
    assert out.content == f"{project / 'notes.md'} — lines 1-2 of 2\nhello\nworld"


async def test_line_numbers_flag_is_parsed_explicitly(session, project):
    (project / "a.txt").write_text("x\n")
    for value in (True, "true", "TRUE", 1, "1"):
        assert "     1\tx" in (await _read(session, path="a.txt", line_numbers=value)).content
    for value in (False, "false", 0, None):
        assert "\t" not in (await _read(session, path="a.txt", line_numbers=value)).content


async def test_an_artifact_file_is_readable_and_reading_tracks_nothing(session):
    await handle_create_artifact(session, {"name": "Dash", "description": "d", "type": "html-app"})
    (artifact,) = ArtifactStore(session._workspace.artifacts_dir).list()
    folder = session._workspace.artifacts_dir / artifact.slug
    (folder / "index.html").write_text("<html></html>\n")
    session._artifacts_touched.clear()
    metadata_before = (folder / "metadata.json").read_text()

    out = await _read(session, path=str(folder / "index.html"))

    assert out.ok is True
    assert session._artifacts_touched == set()
    assert (folder / "metadata.json").read_text() == metadata_before


async def test_skill_drafts_are_readable_by_convention(session, project):
    draft = project / ".anton" / "skill_drafts" / "s" / "SKILL.md"
    draft.parent.mkdir(parents=True)
    draft.write_text("# Skill\n")
    assert (await _read(session, path=str(draft))).ok is True


async def test_outside_the_roots_is_denied_with_attach_advice(session, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("x")
    out = await _read(session, path=str(outside))
    assert (out.ok, out.reason) == (False, "access_denied")
    assert "ask them to attach it to the conversation" in out.content


async def test_a_symlink_leading_out_names_its_target(session, project, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("x")
    os.symlink(outside, project / "link.txt")
    out = await _read(session, path="link.txt")
    assert out.reason == "access_denied"
    assert f"(it resolves to {outside.resolve()})" in out.content


async def test_a_dot_file_in_the_project_is_denied_without_attach_advice(session, project):
    env = project / ".anton" / ".env"
    env.parent.mkdir(parents=True)
    env.write_text("KEY=1")
    out = await _read(session, path=".anton/.env")
    assert out.reason == "access_denied"
    assert "dot-file" in out.content
    assert "attach" not in out.content


async def test_policy_is_checked_before_existence(session, tmp_path):
    out = await _read(session, path=str(tmp_path / "missing" / "x.txt"))
    assert out.reason == "access_denied"


@pytest.mark.parametrize(
    "tc_input,reason",
    [
        ({}, "missing_path"),
        ({"path": "   "}, "missing_path"),
        ({"path": 5}, "missing_path"),
        ({"path": "bad\x00name"}, "invalid_path"),
        ({"path": "absent.txt"}, "path_not_found"),
        ({"path": "folder"}, "not_a_file"),
        ({"path": "a.txt", "start_line": "ten"}, "invalid_range"),
        ({"path": "a.txt", "start_line": 5}, "invalid_range"),
        ({"path": "blob.bin"}, "not_text"),
    ],
)
async def test_failure_reasons(session, project, tc_input, reason):
    (project / "folder").mkdir()
    (project / "a.txt").write_text("one\n")
    (project / "blob.bin").write_bytes(b"\x00\x01")
    out = await handle_read_text_file(session, tc_input)
    assert (out.ok, out.reason) == (False, reason), out.content


async def test_the_header_shows_the_path_as_given_not_resolved(tmp_path):
    real = tmp_path / "real-project"
    real.mkdir()
    (real / "notes.md").write_text("hi\n")
    link = tmp_path / "project-link"
    os.symlink(real, link)
    out = await handle_read_text_file(_Session(link), {"path": "notes.md"})
    assert out.ok is True
    assert out.content.splitlines()[0] == f"{link / 'notes.md'} — lines 1-1 of 1"


async def test_a_large_file_is_refused(session, project, monkeypatch):
    from anton.core.tools import text_file

    monkeypatch.setattr(text_file, "MAX_FILE_BYTES", 4)
    (project / "big.txt").write_text("12345")
    assert (await _read(session, path="big.txt")).reason == "file_too_large"


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
async def test_an_unreadable_directory_gives_a_verdict_not_an_exception(session, project):
    locked = project / "locked"
    locked.mkdir()
    (locked / "secret.txt").write_text("x\n")
    locked.chmod(0)
    try:
        out = await _read(session, path="locked/secret.txt")
    finally:
        locked.chmod(0o700)
    assert out.ok is False
    assert out.reason == "read_failed"


async def test_a_line_number_that_is_not_a_finite_integer_is_an_invalid_range(session, project):
    (project / "a.txt").write_text("x\n")
    for value in (float("inf"), float("-inf"), float("nan")):
        out = await _read(session, path="a.txt", start_line=value)
        assert out.ok is False
        assert out.reason == "invalid_range", value


async def test_a_fractional_line_number_is_an_invalid_range(session, project):
    (project / "a.txt").write_text("1\n2\n3\n")
    out = await _read(session, path="a.txt", start_line=2.5)
    assert (out.ok, out.reason) == (False, "invalid_range")
    out = await _read(session, path="a.txt", start_line=2.0, end_line=2.0)
    assert out.content == f"{project / 'a.txt'} — lines 2-2 of 3\n2"


async def test_display_root_names_files_inside_it_relative_to_it(session, project):
    folder = project / "site"
    (folder / "css").mkdir(parents=True)
    (folder / "css" / "a.css").write_text("x\n")
    out = await handle_read_text_file(
        session, {"path": str(folder / "css" / "a.css")}, display_root=folder
    )
    assert out.content.splitlines()[0] == "css/a.css — lines 1-1 of 1"
    out = await handle_read_text_file(
        session, {"path": str(folder / "missing.css")}, display_root=folder
    )
    assert out.content == "Error: file not found: missing.css"
    (project / "notes.md").write_text("n\n")
    out = await handle_read_text_file(
        session, {"path": str(project / "notes.md")}, display_root=folder
    )
    assert out.content.splitlines()[0] == f"{project / 'notes.md'} — lines 1-1 of 1"


async def test_a_path_component_that_is_too_long_gives_a_verdict(session):
    out = await _read(session, path="a" * 300)
    assert out.ok is False
    assert out.reason in ("read_failed", "invalid_path")


async def test_a_partial_session_falls_back_to_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("x\n")
    out = await handle_read_text_file(AsyncMock(), {"path": "a.txt"})
    assert out.ok is True


def test_the_tool_is_defined_and_allowed_in_the_cloud():
    from anton.cloud_turn.session import CLOUD_TOOL_ALLOWLIST
    from anton.core.tools.tool_defs import READ_TEXT_FILE_TOOL

    assert READ_TEXT_FILE_TOOL.name == "read_text_file"
    assert set(READ_TEXT_FILE_TOOL.input_schema["properties"]) == {
        "path", "start_line", "end_line", "line_numbers",
    }
    assert "read_text_file" in CLOUD_TOOL_ALLOWLIST
