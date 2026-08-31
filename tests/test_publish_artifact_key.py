"""The publish payload must carry `artifact_key` for static artifacts too.

A static artifact publishes its primary *file*, so the identity has to be read
from the metadata.json next to it. Without that the upload lambda locks the
report to the legacy `{user_dir}/{report_id}` key and the artifact loses the
key its draft, comment threads and access rule are grouped under.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock
from uuid import UUID

from anton import publisher
from anton.core.artifacts import ArtifactStore, artifact_key
from anton.publisher import publish


def _capture(target: Path, **kwargs) -> dict:
    captured: dict = {}

    def fake_request(url, api_key, *, method="POST", payload=None, verify=True, timeout=30):
        captured["payload"] = json.loads(payload.decode())
        return json.dumps(
            {"user_prefix": "u", "report_id": "r", "md5": "m", "view_url": "url", "version": 1, "files": []}
        )

    with mock.patch.object(publisher, "minds_request", fake_request):
        publish(target, api_key="k", **kwargs)
    return captured["payload"]


def _static_artifact(tmp_path: Path) -> tuple[Path, str]:
    """A real html-app artifact; returns (primary file, artifact id)."""
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = store.create(name="Sales report", description="x", type="html-app")
    folder = store.folder_for(artifact.slug)
    primary = folder / "index.html"
    primary.write_text("<html>hi</html>", encoding="utf-8")
    return primary, artifact.id


def test_static_publish_sends_the_artifact_key(tmp_path: Path):
    primary, artifact_id = _static_artifact(tmp_path)
    payload = _capture(primary)
    assert payload["artifact_key"] == artifact_key(artifact_id)
    assert payload["artifact_key"] == f"artifact/{UUID(artifact_id)}"


def test_explicit_key_wins_over_the_derived_one(tmp_path: Path):
    """cowork-server derives the key itself and passes it in; it must not be
    second-guessed by the folder anton happens to see."""
    primary, _artifact_id = _static_artifact(tmp_path)
    given = "artifact/11111111-1111-1111-1111-111111111111"
    assert _capture(primary, artifact_key=given)["artifact_key"] == given


def test_loose_file_publish_sends_no_key(tmp_path: Path):
    """A legacy loose file has no metadata — it keeps the service-generated key."""
    loose = tmp_path / "index.html"
    loose.write_text("<html>hi</html>", encoding="utf-8")
    assert "artifact_key" not in _capture(loose)


def test_page_inside_an_artifact_subfolder_sends_no_key(tmp_path: Path):
    """Only the artifact root is consulted. Publishing one nested page would
    otherwise mint a second report under the same key, and the auth rule the
    upload lambda upserts is per key — the two reports would fight over it."""
    primary, _artifact_id = _static_artifact(tmp_path)
    nested = primary.parent / "pages"
    nested.mkdir()
    page = nested / "detail.html"
    page.write_text("<html>detail</html>", encoding="utf-8")
    assert "artifact_key" not in _capture(page)
