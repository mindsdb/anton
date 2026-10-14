"""read_refusal: which files read_text_file may read."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from anton.core.tools.file_access import REFUSED_HIDDEN, REFUSED_OUTSIDE, read_refusal


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    return root


def _file(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path.resolve()


def _refusal(path: Path, project: Path, owned=()):
    return read_refusal(path, workspace=project, owned_roots=owned)


def test_a_project_file_is_readable(project):
    assert _refusal(_file(project / "data" / "sales.csv"), project) is None


@pytest.mark.parametrize("rel", [".anton/.env", ".git/config", ".env"])
def test_dot_paths_in_the_project_are_hidden(project, rel):
    assert _refusal(_file(project / rel), project) == REFUSED_HIDDEN


@pytest.mark.parametrize("rel", ["attachments/brief.md", ".anton/uploads/paste.txt"])
def test_uploads_in_the_workspace_are_readable(project, rel):
    assert _refusal(_file(project / rel), project) is None


def test_a_cowork_upload_outside_the_project_is_readable(tmp_path, project):
    upload = _file(tmp_path / ".cowork" / "files" / "1234" / "notes.txt")
    assert _refusal(upload, project) is None


def test_a_file_outside_every_root_is_refused(tmp_path, project):
    assert _refusal(_file(tmp_path / "elsewhere" / "secret.txt"), project) == REFUSED_OUTSIDE


def test_an_artifact_under_dot_anton_is_readable_through_its_owned_root(project):
    artifacts = project / ".anton" / "artifacts"
    path = _file(artifacts / "dash" / "index.html")
    assert _refusal(path, project) == REFUSED_HIDDEN
    assert _refusal(path, project, owned=(artifacts,)) is None


def test_a_dot_file_inside_an_artifact_is_hidden(project):
    artifacts = project / ".anton" / "artifacts"
    path = _file(artifacts / "dash" / "static" / ".published.json")
    assert _refusal(path, project, owned=(artifacts,)) == REFUSED_HIDDEN


def test_an_artifacts_root_outside_the_workspace(tmp_path, project):
    shared = tmp_path / "shared-artifacts"
    path = _file(shared / "dash" / "index.html")
    assert _refusal(path, project) == REFUSED_OUTSIDE
    assert _refusal(path, project, owned=(shared,)) is None


def test_an_owned_root_given_through_a_symlink(tmp_path, project):
    real = tmp_path / "real-artifacts"
    path = _file(real / "dash" / "index.html")
    link = tmp_path / "link-artifacts"
    os.symlink(real, link)
    assert _refusal(path, project, owned=(link,)) is None


def test_skill_drafts_are_readable(project):
    drafts = project / ".anton" / "skill_drafts"
    path = _file(drafts / "my-skill" / "SKILL.md")
    assert _refusal(path, project, owned=(drafts,)) is None
