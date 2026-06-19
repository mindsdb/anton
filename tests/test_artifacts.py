"""Unit coverage for the artifacts module — Pydantic models,
ArtifactStore CRUD, slug uniqueness, provenance accumulation,
file rescan, and snapshot/diff helpers.

These tests are filesystem-only — no scratchpad subprocess, no
LLM. The store works against a tmp_path root passed straight into
the constructor, so coverage is fast and deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anton.core.artifacts import (
    ARTIFACT_TYPES,
    Artifact,
    ArtifactStore,
    diff_snapshots,
    snapshot_dir,
)


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts")


# ─── Type enum ──────────────────────────────────────────────────────────────


def test_artifact_types_match_design():
    """The closed enum is part of the agent contract — pin the exact
    membership so any future drift is intentional."""
    assert ARTIFACT_TYPES == (
        "html-app",
        "document",
        "dataset",
        "image",
        "mixed",
        "fullstack-stateless-app",
        "fullstack-stateful-app",
    )


# ─── Create ─────────────────────────────────────────────────────────────────


def test_create_writes_metadata_and_readme(store: ArtifactStore):
    artifact = store.create(
        name="NVDA BTC Dashboard",
        description="Compares NVDA and BTC.",
        type="html-app",
    )
    assert artifact.slug == "nvda-btc-dashboard"
    assert len(artifact.id) == 8
    folder = store.folder_for(artifact.slug)
    assert folder.is_dir()
    assert (folder / "metadata.json").is_file()
    assert (folder / "README.md").is_file()


def test_create_validates_type(store: ArtifactStore):
    with pytest.raises(Exception):
        store.create(name="bad", description="x", type="not-a-real-type")  # type: ignore[arg-type]


def test_create_persists_round_trip(store: ArtifactStore):
    artifact = store.create(
        name="My Doc",
        description="A doc.",
        type="document",
    )
    on_disk = json.loads(store.metadata_path(artifact.slug).read_text())
    assert on_disk["name"] == "My Doc"
    assert on_disk["type"] == "document"
    assert on_disk["files"] == []
    assert on_disk["provenance"] == []


# ─── Slug uniqueness ────────────────────────────────────────────────────────


def test_slug_collision_appends_suffix(store: ArtifactStore):
    a = store.create(name="Dashboard", description="x", type="html-app")
    b = store.create(name="Dashboard", description="x", type="html-app")
    c = store.create(name="Dashboard", description="x", type="html-app")
    assert a.slug == "dashboard"
    assert b.slug == "dashboard-2"
    assert c.slug == "dashboard-3"


def test_slug_lowercases_and_sanitizes(store: ArtifactStore):
    artifact = store.create(name="Hello, World!", description="x", type="document")
    # Punctuation collapses to hyphens; runs deduped; lowercased.
    assert artifact.slug == "hello-world"


def test_slug_falls_back_when_name_is_garbage(store: ArtifactStore):
    artifact = store.create(name="!!!", description="x", type="document")
    assert artifact.slug == "untitled-artifact"


# ─── List + open ────────────────────────────────────────────────────────────


def test_list_empty_when_no_artifacts(store: ArtifactStore):
    assert store.list() == []


def test_list_returns_artifacts_newest_first(store: ArtifactStore):
    """Both creates land in the same wall-clock second, so we can't
    rely on the natural timestamp to disambiguate. Manually bump
    `b.updatedAt` to a later second-level value so the sort is
    deterministic regardless of wall-clock granularity."""
    a = store.create(name="First", description="x", type="document")
    b = store.create(name="Second", description="x", type="document")
    # Re-load + re-save with a bumped updatedAt — the simplest way
    # to inject a future timestamp without sleeping in the test.
    record = store.open(b.slug)
    assert record is not None
    record.updatedAt = "2099-01-01T00:00:00+00:00"
    store._save(record)  # type: ignore[attr-defined]
    listing = store.list()
    assert listing[0].slug == "second"
    assert {x.slug for x in listing} == {"first", "second"}


def test_open_returns_none_for_missing_slug(store: ArtifactStore):
    assert store.open("does-not-exist") is None


def test_open_returns_artifact(store: ArtifactStore):
    created = store.create(name="X", description="x", type="document")
    loaded = store.open(created.slug)
    assert loaded is not None
    assert loaded.id == created.id
    assert loaded.name == "X"


def test_list_skips_folders_without_metadata(store: ArtifactStore, tmp_path: Path):
    """A bare folder under artifacts/ (user-dropped or partial write)
    is silently ignored, not treated as a corrupt artifact."""
    store.ensure_root()
    (store.root / "stranger").mkdir()
    assert store.list() == []


# ─── Provenance ─────────────────────────────────────────────────────────────


def test_record_turn_creates_first_provenance_entry(store: ArtifactStore):
    artifact = store.create(name="X", description="x", type="document")
    updated = store.record_turn(
        artifact.slug,
        conversation_id="conv_1",
        conversation_title="My Task",
        turn_index=0,
        summary="first turn",
        files_touched=["report.html"],
    )
    assert updated is not None
    assert len(updated.provenance) == 1
    entry = updated.provenance[0]
    assert entry.conversation == "conv_1"
    assert entry.title == "My Task"
    assert len(entry.turns) == 1
    assert entry.turns[0].summary == "first turn"
    assert entry.turns[0].files_touched == ["report.html"]


def test_record_turn_upserts_within_same_conversation(store: ArtifactStore):
    """Two turns of the same conversation accumulate under one
    ProvenanceEntry, not two."""
    artifact = store.create(name="X", description="x", type="document")
    store.record_turn(
        artifact.slug,
        conversation_id="conv_1", conversation_title="Task",
        turn_index=0, summary="t0", files_touched=["a.txt"],
    )
    updated = store.record_turn(
        artifact.slug,
        conversation_id="conv_1", conversation_title="Task",
        turn_index=2, summary="t2", files_touched=["b.txt"],
    )
    assert len(updated.provenance) == 1
    assert len(updated.provenance[0].turns) == 2


def test_record_turn_multiple_conversations(store: ArtifactStore):
    """Different conversations get their own provenance entries."""
    artifact = store.create(name="X", description="x", type="document")
    store.record_turn(
        artifact.slug,
        conversation_id="conv_1", conversation_title="A",
        turn_index=0, summary="from A", files_touched=[],
    )
    updated = store.record_turn(
        artifact.slug,
        conversation_id="conv_2", conversation_title="B",
        turn_index=0, summary="from B", files_touched=[],
    )
    assert len(updated.provenance) == 2
    assert {p.conversation for p in updated.provenance} == {"conv_1", "conv_2"}


def test_record_turn_truncates_long_summary(store: ArtifactStore):
    artifact = store.create(name="X", description="x", type="document")
    long_text = "x" * 1000
    updated = store.record_turn(
        artifact.slug,
        conversation_id="c", conversation_title=None,
        turn_index=0, summary=long_text, files_touched=[],
    )
    summary = updated.provenance[0].turns[0].summary
    assert len(summary) <= 240
    assert summary.endswith("…")


def test_record_turn_dedupes_files_touched(store: ArtifactStore):
    """Same file written twice in a turn → one provenance entry per file."""
    artifact = store.create(name="X", description="x", type="document")
    updated = store.record_turn(
        artifact.slug,
        conversation_id="c", conversation_title=None,
        turn_index=0, summary="x",
        files_touched=["a.txt", "b.txt", "a.txt"],
    )
    assert updated.provenance[0].turns[0].files_touched == ["a.txt", "b.txt"]


def test_record_turn_returns_none_for_missing_slug(store: ArtifactStore):
    result = store.record_turn(
        "does-not-exist",
        conversation_id="c", conversation_title=None,
        turn_index=0, summary="x", files_touched=[],
    )
    assert result is None


# ─── File rescan ────────────────────────────────────────────────────────────


def test_rescan_picks_up_new_files(store: ArtifactStore):
    artifact = store.create(name="X", description="x", type="html-app")
    folder = store.folder_for(artifact.slug)
    (folder / "dashboard.html").write_text("<html></html>")
    (folder / "data").mkdir()
    (folder / "data" / "prices.csv").write_text("a,b\n1,2")
    updated = store.rescan_files(artifact.slug)
    paths = {f.path for f in updated.files}
    assert paths == {"dashboard.html", "data/prices.csv"}


def test_rescan_excludes_metadata_and_readme(store: ArtifactStore):
    """metadata.json + README.md are housekeeping, not artifact content."""
    artifact = store.create(name="X", description="x", type="document")
    folder = store.folder_for(artifact.slug)
    (folder / "real-file.md").write_text("x")
    updated = store.rescan_files(artifact.slug)
    paths = {f.path for f in updated.files}
    assert paths == {"real-file.md"}


# ─── Reconcile-on-read (ENG-372) ────────────────────────────────────────────
# Scratchpad code writes artifact files straight into the folder via plain
# open(), bypassing the store. open()/list() must reconcile files[] against
# disk so the agent never sees file_count 0 for a fully-written artifact.


def test_open_reflects_scratchpad_written_files(store: ArtifactStore):
    artifact = store.create(name="Dash", description="x", type="html-app")
    assert artifact.files == []  # create starts empty
    folder = store.folder_for(artifact.slug)
    (folder / "dashboard.html").write_text("<html>" + "x" * 1000 + "</html>")
    opened = store.open(artifact.slug)
    assert [f.path for f in opened.files] == ["dashboard.html"]
    assert opened.files[0].bytes > 1000


def test_list_reflects_scratchpad_written_files(store: ArtifactStore):
    artifact = store.create(name="Dash", description="x", type="html-app")
    (store.folder_for(artifact.slug) / "dashboard.html").write_text("<html></html>")
    match = next(a for a in store.list() if a.slug == artifact.slug)
    assert {f.path for f in match.files} == {"dashboard.html"}


def test_reconcile_excludes_published_json(store: ArtifactStore):
    """.published.json is publish-state housekeeping, not artifact content."""
    artifact = store.create(name="Dash", description="x", type="html-app")
    folder = store.folder_for(artifact.slug)
    (folder / "dashboard.html").write_text("<html></html>")
    (folder / ".published.json").write_text("{}")
    opened = store.open(artifact.slug)
    assert {f.path for f in opened.files} == {"dashboard.html"}


def test_reconcile_on_read_is_idempotent(store: ArtifactStore):
    """A read with no on-disk change must not re-save or bump updatedAt."""
    artifact = store.create(name="Dash", description="x", type="html-app")
    (store.folder_for(artifact.slug) / "dashboard.html").write_text("<html></html>")
    first = store.open(artifact.slug)   # reconciles + persists
    second = store.open(artifact.slug)  # no disk change → no re-save
    assert {f.path for f in second.files} == {"dashboard.html"}
    assert second.updatedAt == first.updatedAt


def test_reconcile_re_saves_and_persists_when_files_change(store: ArtifactStore):
    """A changed on-disk file set must reconcile AND persist to metadata.json."""
    artifact = store.create(name="Dash", description="x", type="html-app")
    folder = store.folder_for(artifact.slug)
    (folder / "a.html").write_text("<html></html>")
    assert {f.path for f in store.open(artifact.slug).files} == {"a.html"}
    # A second file lands on disk → next read reconciles and persists it.
    (folder / "b.html").write_text("<html></html>")
    assert {f.path for f in store.open(artifact.slug).files} == {"a.html", "b.html"}
    on_disk = json.loads(store.metadata_path(artifact.slug).read_text())
    assert {f["path"] for f in on_disk["files"]} == {"a.html", "b.html"}


# ─── README rendering ───────────────────────────────────────────────────────


def test_readme_renders_provenance_section(store: ArtifactStore):
    artifact = store.create(name="My Dash", description="A dashboard.", type="html-app")
    folder = store.folder_for(artifact.slug)
    (folder / "index.html").write_text("<html></html>")
    store.rescan_files(artifact.slug)
    store.record_turn(
        artifact.slug,
        conversation_id="conv_1", conversation_title="Build dashboard",
        turn_index=2, summary="rendered the dashboard", files_touched=["index.html"],
    )
    readme = store.readme_path(artifact.slug).read_text()
    assert "My Dash" in readme
    assert "html-app" in readme
    assert "A dashboard." in readme
    assert "index.html" in readme
    assert "Build dashboard" in readme
    assert "rendered the dashboard" in readme


def test_readme_re_render_is_idempotent(store: ArtifactStore):
    artifact = store.create(name="X", description="x", type="document")
    first = store.render_readme(artifact.slug)
    second = store.render_readme(artifact.slug)
    assert first == second


# ─── Snapshot + diff ────────────────────────────────────────────────────────


def test_snapshot_empty_when_dir_missing(tmp_path: Path):
    assert snapshot_dir(tmp_path / "nope") == {}


def test_snapshot_lists_files(tmp_path: Path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("y")
    snap = snapshot_dir(tmp_path)
    assert "a.txt" in snap
    assert "sub/b.txt" in snap


def test_diff_picks_up_new_and_changed(tmp_path: Path):
    (tmp_path / "a.txt").write_text("x")
    before = snapshot_dir(tmp_path)
    # Add a new file + modify the existing one (size change forces a diff
    # even if mtime resolution is coarse on the host filesystem).
    (tmp_path / "a.txt").write_text("xyz")
    (tmp_path / "b.txt").write_text("y")
    after = snapshot_dir(tmp_path)
    changes = diff_snapshots(before, after)
    assert changes == ["a.txt", "b.txt"]


def test_diff_ignores_deletions(tmp_path: Path):
    """diff_snapshots tracks creations and modifications. A file that
    existed before and is gone after is NOT flagged — provenance is
    about what got produced, not what got cleaned up."""
    (tmp_path / "a.txt").write_text("x")
    before = snapshot_dir(tmp_path)
    (tmp_path / "a.txt").unlink()
    after = snapshot_dir(tmp_path)
    assert diff_snapshots(before, after) == []


def test_diff_empty_when_unchanged(tmp_path: Path):
    (tmp_path / "a.txt").write_text("x")
    before = snapshot_dir(tmp_path)
    after = snapshot_dir(tmp_path)
    assert diff_snapshots(before, after) == []


# ─── Primary file pointer ───────────────────────────────────────────────────


def test_create_with_primary(store: ArtifactStore):
    """Agent-declared primary lands on the metadata as-is."""
    artifact = store.create(
        name="Dashboard", description="x", type="html-app",
        primary="dashboard.html",
    )
    assert artifact.primary == "dashboard.html"
    on_disk = json.loads(store.metadata_path(artifact.slug).read_text())
    assert on_disk["primary"] == "dashboard.html"


def test_create_without_primary_defaults_none(store: ArtifactStore):
    artifact = store.create(name="X", description="x", type="document")
    assert artifact.primary is None
    on_disk = json.loads(store.metadata_path(artifact.slug).read_text())
    # Field is present in JSON (Pydantic dumps null) — whether it
    # appears as null or omitted depends on Pydantic version, but
    # we don't depend on the absence either way; the model loads
    # back to None.
    assert on_disk.get("primary") is None


def test_create_strips_blank_primary(store: ArtifactStore):
    """Whitespace-only primary normalizes to None — keeps the
    'agent didn't pick' signal honest."""
    artifact = store.create(
        name="X", description="x", type="document", primary="   ",
    )
    assert artifact.primary is None


def test_update_primary(store: ArtifactStore):
    artifact = store.create(name="X", description="x", type="html-app")
    updated = store.update(artifact.slug, primary="main.html")
    assert updated is not None
    assert updated.primary == "main.html"
    # Persisted: re-loading the same slug returns the new value.
    reloaded = store.open(artifact.slug)
    assert reloaded.primary == "main.html"


def test_update_primary_clears_with_none(store: ArtifactStore):
    artifact = store.create(
        name="X", description="x", type="html-app",
        primary="dashboard.html",
    )
    cleared = store.update(artifact.slug, primary=None)
    assert cleared.primary is None
    # Empty string is also treated as "clear".
    artifact2 = store.create(
        name="Y", description="x", type="html-app",
        primary="dashboard.html",
    )
    cleared2 = store.update(artifact2.slug, primary="   ")
    assert cleared2.primary is None


def test_update_port(store: ArtifactStore):
    artifact = store.create(name="App", description="x", type="fullstack-stateful-app")
    updated = store.update(artifact.slug, port=8080)
    assert updated is not None
    assert updated.port == 8080
    reloaded = store.open(artifact.slug)
    assert reloaded.port == 8080


def test_update_primary_and_port_together(store: ArtifactStore):
    artifact = store.create(name="App", description="x", type="fullstack-stateful-app")
    updated = store.update(artifact.slug, primary="index.html", port=5000)
    assert updated.primary == "index.html"
    assert updated.port == 5000


def test_update_omitted_field_unchanged(store: ArtifactStore):
    artifact = store.create(
        name="App", description="x", type="fullstack-stateful-app",
        primary="index.html",
    )
    # Updating only port must not touch primary.
    updated = store.update(artifact.slug, port=3000)
    assert updated.primary == "index.html"
    assert updated.port == 3000


def test_update_returns_none_for_missing_slug(store: ArtifactStore):
    assert store.update("does-not-exist", primary="main.html") is None
