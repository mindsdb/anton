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
from uuid import UUID

import pytest

from anton.core.artifacts import (
    ARTIFACT_TYPES,
    Artifact,
    ArtifactStore,
    artifact_key,
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
    assert artifact.slug == f"nvda-btc-dashboard-{artifact.id[:8]}"
    assert len(artifact.id) == 32
    assert UUID(artifact.id).hex == artifact.id
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
    assert on_disk["id"] == artifact.id
    assert "stableId" not in on_disk


LEGACY_METADATA = {
    "id": "a1b2c3d4",
    "slug": "legacy-a1b2c3d4",
    "createdAt": "2026-08-25T12:00:00+00:00",
    "updatedAt": "2026-08-25T12:00:00+00:00",
    "name": "Legacy",
    "description": "",
    "type": "document",
}


def test_legacy_metadata_widens_to_a_repeatable_full_id():
    """anton widens in memory while cowork-server persists, so two independent
    readers have to land on the same value without coordinating."""
    first = Artifact.model_validate(LEGACY_METADATA)
    second = Artifact.model_validate(LEGACY_METADATA)

    assert first.id == second.id
    assert len(first.id) == 32
    assert UUID(first.id).hex == first.id


def test_legacy_widening_keeps_the_slug_suffix_addressable():
    """The old eight characters stay the id prefix, so `<name>-<id[:8]>` folders
    keep resolving after the widening."""
    artifact = Artifact.model_validate(LEGACY_METADATA)

    assert artifact.id[:8] == "a1b2c3d4"
    assert artifact.slug.endswith(f"-{artifact.id[:8]}")


def test_persisted_stable_id_wins_over_rederivation():
    """Records from the two-field era already keyed published versions and
    comment threads by `stableId`; adopting it keeps those keys bound."""
    minted = "11111111-2222-4333-8444-555555555555"
    artifact = Artifact.model_validate({**LEGACY_METADATA, "stableId": minted})

    assert artifact.id == UUID(minted).hex


@pytest.mark.parametrize("damaged_id", [
    "7db94eb8f0a54c7e9c1d2b3a4f5e6d7",   # one char short of a UUID
    "7db94eb8f0a54c7e9c1d2b3a4f5e6d700",  # one char long
    "7db94eb8f0a5",                       # hex, wider than a legacy id
])
def test_a_damaged_id_is_rejected_not_re_minted(damaged_id):
    """Hex-only and wider than a legacy id: a truncated or padded identity.
    Re-minting it would detach the artifact from its published versions, auth
    rules and comment threads."""
    with pytest.raises(Exception):
        Artifact.model_validate({**LEGACY_METADATA, "id": damaged_id})


@pytest.mark.parametrize("name_like_id", ["static-art", "legacy", "z" * 32])
def test_a_name_in_the_id_field_widens_rather_than_dropping_the_artifact(name_like_id):
    """Hand-written and very old records carry names in `id`. Refusing them
    would drop the artifact from every listing, which reads as a deletion."""
    artifact = Artifact.model_validate({**LEGACY_METADATA, "id": name_like_id})

    assert UUID(artifact.id).hex == artifact.id


def test_an_empty_id_is_rejected():
    """The early return in the widening validator leaves nothing to widen from;
    the field constraint has to catch it, or `artifact_key('')` blows up later."""
    with pytest.raises(Exception):
        Artifact.model_validate({**LEGACY_METADATA, "id": ""})


def test_a_malformed_persisted_stable_id_is_rejected():
    with pytest.raises(Exception):
        Artifact.model_validate({**LEGACY_METADATA, "stableId": "not-a-uuid"})


def test_an_already_widened_id_ignores_a_stale_stable_id():
    """An older build could re-add `stableId` to a record whose `id` is already
    the identity that published versions are keyed under. `id` has to win, or
    that stale write would re-stamp the artifact and orphan its comments."""
    widened = "7db94eb8f0a54c7e9c1d2b3a4f5e6d70"

    artifact = Artifact.model_validate({
        **LEGACY_METADATA,
        "id": widened,
        "stableId": "080ee44e-9ebd-5f7f-ab07-cccfc6b9d56e",
    })

    assert artifact.id == widened


def test_artifact_key_is_the_canonical_dashed_form():
    """The upload lambda normalizes what it stores in `_meta.json` the same way,
    so the browser shell and cowork agree on one comments key."""
    artifact = Artifact.model_validate(LEGACY_METADATA)

    assert artifact_key(artifact.id) == f"artifact/{UUID(artifact.id)}"
    assert artifact_key(artifact.id) == artifact_key(str(UUID(artifact.id)))


# ─── Slug uniqueness ────────────────────────────────────────────────────────


def test_same_name_yields_distinct_slugs(store: ArtifactStore):
    """The id, not the `-2`/`-3` counter, is what separates them now. The
    counter only ever saw one store, and on the cloud each conversation has its
    own — so identical names in two conversations used to collide."""
    slugs = {
        store.create(name="Dashboard", description="x", type="html-app").slug
        for _ in range(3)
    }
    assert len(slugs) == 3
    assert all(s.startswith("dashboard-") for s in slugs)


def test_slug_lowercases_and_sanitizes(store: ArtifactStore):
    artifact = store.create(name="Hello, World!", description="x", type="document")
    # Punctuation collapses to hyphens; runs deduped; lowercased.
    assert artifact.slug == f"hello-world-{artifact.id[:8]}"


def test_slug_falls_back_when_name_is_garbage(store: ArtifactStore):
    artifact = store.create(name="!!!", description="x", type="document")
    assert artifact.slug == f"untitled-artifact-{artifact.id[:8]}"


def test_non_latin_names_still_get_distinct_folders(store: ArtifactStore):
    """The character whitelist is ASCII, so a Cyrillic name sanitises to
    nothing and every such artifact shares the `untitled-artifact` base. Before
    the id suffix that made same-base collisions the NORM for a non-English
    user, not an edge case."""
    a = store.create(name="Текущее время", description="x", type="html-app")
    b = store.create(name="Другой отчёт", description="x", type="html-app")

    assert a.slug != b.slug
    assert a.slug.startswith("untitled-artifact-")
    assert b.slug.startswith("untitled-artifact-")


def test_display_name_carries_no_id(store: ArtifactStore):
    """The id belongs in the folder name, not on screen: `name` is what the UI
    falls back to for a title when the caller supplied none."""
    artifact = store.create(name="", description="x", type="document")

    assert artifact.name == "untitled-artifact"
    assert artifact.slug == f"untitled-artifact-{artifact.id[:8]}"


def test_slug_stays_within_the_length_budget(store: ArtifactStore):
    """The name is trimmed by the suffix width, so a long name plus the id
    still respects the same 64-char ceiling folders had before."""
    artifact = store.create(name="x" * 200, description="x", type="document")

    assert len(artifact.slug) <= 64
    assert artifact.slug.endswith(f"-{artifact.id[:8]}")


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
    assert listing[0].slug == b.slug
    assert {x.slug for x in listing} == {a.slug, b.slug}


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


def test_reconcile_excludes_the_backend_log(store: ArtifactStore):
    """`launch_artifact_backend` writes backend.log into the artifact folder.
    Every other copy of the housekeeping set already excluded it — cowork-server's
    artifacts service, `publish_access`, and the publish bundle — so listing it
    here made the agent's view disagree with the UI's."""
    artifact = store.create(
        name="App", description="x", type="fullstack-stateless-app"
    )
    folder = store.folder_for(artifact.slug)
    (folder / "backend.py").write_text("app = 1")
    (folder / "backend.log").write_text("INFO: started")
    opened = store.open(artifact.slug)
    assert {f.path for f in opened.files} == {"backend.py"}


def test_reconcile_excludes_generation_inputs(store: ArtifactStore):
    """prd.md / spec.md / openapi.json are what the generator built FROM, not
    what it built. Listed among `files[]` they inflate file_count, show up in
    the README, and invite the agent to hand the user a spec as a deliverable."""
    artifact = store.create(
        name="App", description="x", type="fullstack-stateless-app"
    )
    folder = store.folder_for(artifact.slug)
    (folder / "backend.py").write_text("app = 1")
    (folder / "requirements.txt").write_text("fastapi")
    (folder / "prd.md").write_text("## Goal\nx")
    (folder / "spec.md").write_text("# Spec")
    (folder / "openapi.json").write_text("{}")
    opened = store.open(artifact.slug)
    assert {f.path for f in opened.files} == {"backend.py", "requirements.txt"}


def test_excluded_set_covers_every_generation_input():
    """The exclusion list is derived from the same constants the generators
    write through — a fourth generation input added there must not need a
    second edit here to stay out of `files[]`."""
    from anton.core.artifacts.internal_files import GENERATION_INPUT_FILES
    from anton.core.artifacts.store import _EXCLUDED_FROM_FILES

    assert set(GENERATION_INPUT_FILES) <= _EXCLUDED_FROM_FILES


def test_housekeeping_set_still_mirrors_cowork_server():
    """The set is documented as mirroring cowork-server's artifacts service
    (`cowork/services/artifacts.py:132`), and `publish_access` plus the publish
    bundle carry their own copies. Drift means the agent and the UI disagree on
    what an artifact contains — which is exactly how backend.log ended up
    counted here and nowhere else. The STATE runtime entries (PR #259) are in
    anton's three copies; cowork-server's copy does not know them yet."""
    from anton.core.artifacts.store import _HOUSEKEEPING_FILES

    assert _HOUSEKEEPING_FILES == {
        "metadata.json", "README.md", "backend.log", ".published.json",
        ".anton_state.db", ".anton_state.db-wal", ".anton_state.db-shm",
        ".state_manifest.published.json",
    }


def test_housekeeping_set_matches_publish_access_copy():
    """anton's own two copies must never drift from each other."""
    from anton.core.artifacts.store import _HOUSEKEEPING_FILES
    from anton.publish_access import _HOUSEKEEPING_FILES as ACCESS_COPY

    assert _HOUSEKEEPING_FILES == ACCESS_COPY


def test_housekeeping_dirs_match_publish_access_copy():
    """Reserved DIRECTORIES are a second set, and it drifts just as quietly.

    `publish_access` matches on the path's first component, so a directory name
    also "works" inside `_HOUSEKEEPING_FILES` there — which is how `.revisions`
    arrived in the staging merge, breaking the lock above. The store matches
    whole relative paths and cannot fold directories in, so both sides keep the
    split and both sides are locked.
    """
    from anton.core.artifacts.store import _HOUSEKEEPING_DIRS
    from anton.publish_access import _HOUSEKEEPING_DIRS as ACCESS_COPY

    assert _HOUSEKEEPING_DIRS == ACCESS_COPY


def test_reconcile_excludes_state_runtime_files(store: ArtifactStore):
    """A locally-run stateful backend writes its SQLite store (and WAL/SHM side
    files) into the artifact folder; publishing adds the schema snapshot. All
    of it is runtime bookkeeping. `state_manifest.json` itself is a deliverable
    the publisher bundles — it stays visible."""
    artifact = store.create(
        name="App", description="x", type="fullstack-stateful-app"
    )
    folder = store.folder_for(artifact.slug)
    (folder / "backend.py").write_text("app = 1")
    (folder / "state_manifest.json").write_text('{"version": 1}')
    (folder / ".anton_state.db").write_text("sqlite")
    (folder / ".anton_state.db-wal").write_text("wal")
    (folder / ".anton_state.db-shm").write_text("shm")
    (folder / ".state_manifest.published.json").write_text("{}")
    opened = store.open(artifact.slug)
    assert {f.path for f in opened.files} == {"backend.py", "state_manifest.json"}


def test_a_nested_file_named_like_a_generation_input_is_kept(store: ArtifactStore):
    """The exclusion is by exact relative path, not by basename: a
    `static/openapi.json` the artifact genuinely serves is its own content."""
    artifact = store.create(
        name="App", description="x", type="fullstack-stateless-app"
    )
    folder = store.folder_for(artifact.slug)
    (folder / "static").mkdir()
    (folder / "static" / "openapi.json").write_text("{}")
    (folder / "openapi.json").write_text("{}")
    opened = store.open(artifact.slug)
    assert {f.path for f in opened.files} == {"static/openapi.json"}

def test_reconcile_excludes_revision_journal(store: ArtifactStore):
    artifact = store.create(name="Dash", description="x", type="html-app")
    folder = store.folder_for(artifact.slug)
    (folder / "dashboard.html").write_text("<html></html>")
    journal = folder / ".revisions" / "entries"
    journal.mkdir(parents=True)
    (journal / "private-source.md").write_text("must not surface")

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


def test_snapshot_keys_use_posix_separators(tmp_path: Path):
    """Snapshot keys must be POSIX-separated on every platform.

    The keys are split on "/" by `_files_by_artifact` and persisted in
    provenance records, so a native-separator key silently groups to
    nothing on Windows instead of failing loudly.
    """
    nested = tmp_path / "dashboard" / "data"
    nested.mkdir(parents=True)
    (nested / "prices.csv").write_text("a,b")
    snap = snapshot_dir(tmp_path)
    assert "dashboard/data/prices.csv" in snap
    assert not any("\\" in key for key in snap)
    assert diff_snapshots({}, snap) == ["dashboard/data/prices.csv"]


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


def test_update_type(store: ArtifactStore):
    artifact = store.create(name="X", description="x", type="html-app")
    updated = store.update(artifact.slug, type="fullstack-stateless-app")
    assert updated is not None
    assert updated.type == "fullstack-stateless-app"
    reloaded = store.open(artifact.slug)
    assert reloaded.type == "fullstack-stateless-app"


def test_update_type_together_with_other_fields(store: ArtifactStore):
    artifact = store.create(name="X", description="x", type="html-app")
    updated = store.update(artifact.slug, type="document", primary="report.pdf")
    assert updated.type == "document"
    assert updated.primary == "report.pdf"


def test_update_invalid_type_raises(store: ArtifactStore):
    artifact = store.create(name="X", description="x", type="html-app")
    with pytest.raises(ValueError, match="type"):
        store.update(artifact.slug, type="not-a-real-type")
    # Rejected before any write — the artifact's type is untouched.
    reloaded = store.open(artifact.slug)
    assert reloaded.type == "html-app"


def test_update_missing_slug_still_returns_none_not_valueerror(store: ArtifactStore):
    """A missing slug and a bad `type` are different failure causes — the
    ValueError must not fire before the existence check, or a caller testing
    for a missing artifact via `is None` would see an exception instead."""
    assert store.update("does-not-exist", type="not-a-real-type") is None


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
