"""open_artifact returns a small text primary file with the descriptor."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from anton.core.artifacts import ArtifactStore
from anton.core.tools import tool_handlers
from anton.core.tools.tool_handlers import handle_create_artifact, handle_open_artifact


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


async def _open_with(session, primary, content: bytes | None) -> dict:
    await handle_create_artifact(session, {"name": "Report", "description": "d", "type": "html-app",
                                           **({"primary": primary} if primary else {})})
    (artifact,) = ArtifactStore(session._workspace.artifacts_dir).list()
    folder = session._workspace.artifacts_dir / artifact.slug
    if content is not None:
        (folder / (primary or "index.html")).write_bytes(content)
    result = await handle_open_artifact(session, {"slug": artifact.slug})
    return json.loads(result.content)


async def test_small_text_primary_comes_back_with_the_descriptor(session):
    opened = await _open_with(session, "report.html", b"<!doctype html><p>Total 90</p>")
    assert opened["primary_content"] == "<!doctype html><p>Total 90</p>"
    assert opened["path"].endswith(opened["slug"])


async def test_a_large_primary_is_named_but_left_out(session, monkeypatch):
    monkeypatch.setattr(tool_handlers, "OPEN_ARTIFACT_CONTENT_MAX_BYTES", 10)
    opened = await _open_with(session, "report.html", b"x" * 11)
    assert "primary_content" not in opened
    assert "11 bytes" in opened["primary_content_omitted"]


async def test_binary_primary_is_left_out(session):
    opened = await _open_with(session, "chart.png", b"\x89PNG\r\n\x1a\n\xff\xfe")
    assert "primary_content" not in opened and "not a text file" in opened["primary_content_omitted"]


@pytest.mark.parametrize("primary,content", [(None, b"<p>no primary set</p>"), ("missing.html", None)])
async def test_no_primary_file_means_no_content_fields(session, primary, content):
    opened = await _open_with(session, primary, content)
    assert "primary_content" not in opened and "primary_content_omitted" not in opened


def test_a_primary_outside_the_folder_is_never_read(tmp_path):
    (tmp_path / "secret.txt").write_text("secret")
    folder = tmp_path / "artifacts" / "report"
    folder.mkdir(parents=True)
    assert tool_handlers._primary_content(folder, "../../secret.txt") == {}
