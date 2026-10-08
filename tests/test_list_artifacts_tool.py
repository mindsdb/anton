"""handle_list_artifacts: default fields, match, explicit fields and errors."""
from __future__ import annotations

from pathlib import Path

import pytest

from anton.core.artifacts import ArtifactStore
from anton.core.tools.tool_handlers import handle_create_artifact, handle_list_artifacts


class _Workspace:
    def __init__(self, root: Path) -> None:
        self.artifacts_dir = root


class _Session:
    def __init__(self, root: Path) -> None:
        self._workspace = _Workspace(root)
        self._session_id = "conv-1"
        self._turn_count = 0
        self._artifacts_touched: set[str] = set()
        self._data_vault = None


@pytest.fixture
def session(tmp_path: Path) -> _Session:
    return _Session(tmp_path / "artifacts")


async def _create(session, name: str) -> str:
    await handle_create_artifact(session, {"name": name, "description": "d", "type": "html-app"})
    return next(a.slug for a in ArtifactStore(session._workspace.artifacts_dir).list() if a.name == name)


async def test_summary_by_default(session):
    await _create(session, "Alpha")
    out = await handle_list_artifacts(session, {})
    assert out.ok is True
    assert "primary: " in out.content
    assert "id: " not in out.content
    assert "files:" not in out.content


async def test_the_root_is_shown_as_create_artifact_returns_it(session):
    await _create(session, "Alpha")
    out = await handle_list_artifacts(session, {})
    assert f"Artifacts root: {session._workspace.artifacts_dir}" in out.content.splitlines()


async def test_match_returns_the_full_record(session):
    slug = await _create(session, "Alpha")
    await _create(session, "Beta")
    out = await handle_list_artifacts(session, {"match": [slug]})
    assert out.content.count("\n## ") == 1
    for label in ("id: ", "files:", "service_files:"):
        assert label in out.content


async def test_match_accepts_a_string_and_reports_non_strings_unmatched(session):
    slug = await _create(session, "Alpha")
    assert f"## {slug}" in (await handle_list_artifacts(session, {"match": slug})).content
    out = await handle_list_artifacts(session, {"match": [slug, 123]})
    assert out.content.splitlines()[-1] == "unmatched: 123"


async def test_empty_match_lists_everything(session):
    await _create(session, "Alpha")
    await _create(session, "Beta")
    assert (await handle_list_artifacts(session, {"match": []})).content.count("\n## ") == 2


async def test_fields_accepts_a_single_string(session):
    slug = await _create(session, "Alpha")
    out = await handle_list_artifacts(session, {"match": [slug], "fields": "files"})
    assert out.content.splitlines()[-1] == "files: none"
    assert "name: " not in out.content


async def test_explicit_fields(session):
    slug = await _create(session, "Alpha")
    out = await handle_list_artifacts(session, {"match": [slug], "fields": ["name"]})
    assert out.content.splitlines()[-1] == "name: Alpha"


@pytest.mark.parametrize(
    "fields,needle",
    [([], "fields is empty"), (["path"], 'unknown field "path"'), ([1], 'unknown field "1"')],
)
async def test_invalid_fields(session, fields, needle):
    out = await handle_list_artifacts(session, {"fields": fields})
    assert (out.ok, out.reason) == (False, "invalid_fields")
    assert needle in out.content
    assert "Valid fields: id, name, type, updatedAt, description, primary, files, service_files." in out.content


async def test_no_artifacts_yet(session):
    assert (await handle_list_artifacts(session, {})).content == "No artifacts yet."


async def test_no_workspace():
    class _Bare:
        _workspace = None

    out = await handle_list_artifacts(_Bare(), {})
    assert (out.ok, out.reason) == (False, "store_unavailable")
