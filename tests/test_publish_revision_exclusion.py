from __future__ import annotations

import io
import zipfile

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
