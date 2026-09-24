from __future__ import annotations

import io
import zipfile

from anton.core.artifacts.internal_files import GENERATION_INPUT_FILES
from anton.publish_access import _pick_primary, _user_files
from anton.publisher import _zip_html


def test_directory_publish_never_bundles_revision_journal(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "index.html").write_text("<h1>Current</h1>")
    revisions = artifact / ".revisions" / "entries"
    revisions.mkdir(parents=True)
    (revisions / "old.html").write_text("<h1>Private old source</h1>")

    with zipfile.ZipFile(io.BytesIO(_zip_html(artifact))) as bundle:
        assert bundle.namelist() == ["index.html"]


def test_directory_publish_never_bundles_generation_inputs(tmp_path):
    """prd.md / spec.md / discovery.json / openapi.json are generation inputs
    (hidden from the artifact's files[]) and must not ship in a public bundle."""
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "index.html").write_text("<h1>Current</h1>")
    for name in GENERATION_INPUT_FILES:
        (artifact / name).write_text("internal")

    with zipfile.ZipFile(io.BytesIO(_zip_html(artifact))) as bundle:
        assert bundle.namelist() == ["index.html"]


def test_user_files_skip_generation_inputs_so_primary_is_real_content(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    for name in GENERATION_INPUT_FILES:
        (artifact / name).write_text("internal")
    (artifact / "dashboard.html").write_text("<h1>UI</h1>")

    files = _user_files(artifact)
    assert [p.name for p in files] == ["dashboard.html"]
    assert _pick_primary(artifact, files, primary_hint="missing.html").name == "dashboard.html"
