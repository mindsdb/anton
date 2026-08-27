"""Per-turn artifact attribution recorded by the tool handlers.

A host (cowork-server) surfaces the artifacts a turn produced as cards on that
turn's reply. It used to work that out by snapshotting the artifacts directory
before the turn and diffing it after — which breaks as soon as two turns share
one artifacts directory, because a concurrent turn's brand-new artifact lands
in the other turn's diff and gets attributed to the wrong conversation
(ENG-1933).

These tests pin the replacement: the artifact tools record what they touched as
they run, so attribution comes from the turn that actually did the work rather
than from whatever else changed on disk meanwhile. Two records, two readers —
an in-memory per-turn set for the host, and durable `provenance` in the
artifact's own metadata for everyone after that.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from anton.core.artifacts import ArtifactStore
from anton.core.tools.tool_handlers import (
    _snapshot_existing_artifact_mtimes,
    _track_edits_since,
    handle_create_artifact,
    handle_open_artifact,
    handle_update_artifact_metadata,
)


class FakeWorkspace:
    def __init__(self, root: Path) -> None:
        self.artifacts_dir = root


class FakeSession:
    """The handful of ChatSession attributes the artifact handlers read."""

    def __init__(self, root: Path, *, session_id: str | None = "conv-1", turn_count: int = 0) -> None:
        self._workspace = FakeWorkspace(root)
        self._session_id = session_id
        self._turn_count = turn_count
        self._artifacts_touched: set[str] = set()
        self._data_vault = None


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"


@pytest.fixture
def session(root: Path) -> FakeSession:
    return FakeSession(root)


def _provenance(root: Path, slug: str) -> list[dict]:
    return json.loads((root / slug / "metadata.json").read_text())["provenance"]


async def _create(session: FakeSession, name: str = "Dash") -> str:
    """Create an artifact and return its slug.

    Read back off the store rather than parsed out of the tool's own result,
    so these tests don't depend on the outcome payload's shape.
    """
    await handle_create_artifact(
        session, {"name": name, "description": "d", "type": "html-app"}
    )
    root = session._workspace.artifacts_dir
    match = [a for a in ArtifactStore(root).list() if a.name == name]
    assert match, f"create_artifact did not write an artifact named {name!r}"
    return match[0].slug


# ─── the in-memory per-turn set ─────────────────────────────────────────────


async def test_create_tracks_the_slug_for_this_turn(session, root):
    slug = await _create(session)

    assert session._artifacts_touched == {slug}


async def test_open_tracks_the_slug_for_this_turn(session, root):
    """Opening is how the agent gets a path to write into, so it counts as
    intent to modify — the writes themselves happen in scratchpad cells the
    tool layer never observes."""
    slug = await _create(session)
    session._artifacts_touched.clear()

    await handle_open_artifact(session, {"slug": slug})

    assert session._artifacts_touched == {slug}


async def test_update_metadata_tracks_the_slug(session, root):
    slug = await _create(session)
    session._artifacts_touched.clear()

    await handle_update_artifact_metadata(session, {"slug": slug, "primary": "index.html"})

    assert session._artifacts_touched == {slug}


async def test_untouched_artifact_is_not_tracked(session, root):
    """The whole point: an artifact sitting in the same directory that this
    turn never touched must not be attributed to it. This is the case a
    directory diff gets wrong when two turns share one artifacts folder."""
    sibling = FakeSession(root, session_id="conv-2")
    sibling_slug = await _create(sibling, name="Someone Elses Work")

    slug = await _create(session, name="My Work")

    assert session._artifacts_touched == {slug}
    assert sibling_slug not in session._artifacts_touched


async def test_open_of_a_missing_slug_tracks_nothing(session, root):
    await _create(session)
    session._artifacts_touched.clear()

    await handle_open_artifact(session, {"slug": "no-such-artifact"})

    assert session._artifacts_touched == set()


# ─── durable provenance ─────────────────────────────────────────────────────


async def test_create_stamps_provenance_with_the_conversation(session, root):
    slug = await _create(session)

    entries = _provenance(root, slug)
    assert [e["conversation"] for e in entries] == ["conv-1"]
    assert len(entries[0]["turns"]) == 1


async def test_turn_index_follows_the_session(root):
    session = FakeSession(root, turn_count=4)

    slug = await _create(session)

    # _turn_count is turns COMPLETED, so the turn in flight is the next one.
    assert _provenance(root, slug)[0]["turns"][0]["index"] == 5


async def test_two_conversations_accumulate_separate_provenance(session, root):
    """The same artifact worked on by two conversations keeps one entry each —
    that is the record of who has ever touched it, and the seed for any later
    version history."""
    slug = await _create(session)

    other = FakeSession(root, session_id="conv-2")
    await handle_open_artifact(other, {"slug": slug})

    assert [e["conversation"] for e in _provenance(root, slug)] == ["conv-1", "conv-2"]


async def test_no_conversation_id_skips_provenance_but_still_tracks(root):
    """A bare CLI session has no host conversation to attribute to. Provenance
    keyed by nothing is worse than none, but the in-memory set is still useful
    to whatever is driving the session."""
    session = FakeSession(root, session_id=None)

    slug = await _create(session)

    assert session._artifacts_touched == {slug}
    assert _provenance(root, slug) == []


async def test_tracking_failure_never_fails_the_tool_call(session, root, monkeypatch):
    """Attribution is bookkeeping. If it breaks, the agent's actual work must
    still succeed — a lost card is recoverable, a failed create is not."""
    def boom(*_a, **_k):
        raise OSError("disk gone")

    monkeypatch.setattr(ArtifactStore, "record_turn", boom)

    slug = await _create(session)

    assert slug
    assert (root / slug / "metadata.json").is_file()


# ─── edits made without re-opening (scratchpad mtime fallback) ─────────────
#
# `open_artifact` is how attribution is SUPPOSED to work, but nothing forces
# the agent to call it again once it already has an artifact's path from
# earlier in the conversation — it can (and in practice does) just write
# straight into a remembered folder via the scratchpad. `_track_edits_since`
# is the fallback: a before/after mtime diff scoped to one scratchpad cell's
# own execution window, so that edit still gets attributed.


def _bump_mtime(path: Path, delta_s: int = 2) -> None:
    st = path.stat()
    os.utime(path, (st.st_atime, st.st_mtime + delta_s))


async def test_edit_without_reopening_is_tracked_via_mtime(session, root):
    slug = await _create(session)
    session._artifacts_touched.clear()

    before = _snapshot_existing_artifact_mtimes(ArtifactStore(root))
    # The scratchpad writing straight into the folder, bypassing open_artifact.
    index = root / slug / "index.html"
    index.write_text("<html>v2</html>")
    _bump_mtime(index)

    _track_edits_since(session, ArtifactStore(root), before)

    assert session._artifacts_touched == {slug}


async def test_housekeeping_only_mtime_bump_is_not_an_edit(session, root):
    """metadata.json/README.md are the store's own bookkeeping, not artifact
    content — touching only those must not read as the agent having edited
    the artifact."""
    slug = await _create(session)
    session._artifacts_touched.clear()

    before = _snapshot_existing_artifact_mtimes(ArtifactStore(root))
    _bump_mtime(root / slug / "README.md")

    _track_edits_since(session, ArtifactStore(root), before)

    assert session._artifacts_touched == set()


async def test_edit_without_reopening_stamps_provenance_too(session, root):
    slug = await _create(session)
    session._artifacts_touched.clear()

    before = _snapshot_existing_artifact_mtimes(ArtifactStore(root))
    index = root / slug / "index.html"
    index.write_text("<html>v2</html>")
    _bump_mtime(index)

    _track_edits_since(session, ArtifactStore(root), before)

    assert [e["conversation"] for e in _provenance(root, slug)] == ["conv-1"]


async def test_untouched_artifact_is_not_tracked_by_mtime_fallback(session, root):
    slug = await _create(session)
    session._artifacts_touched.clear()

    before = _snapshot_existing_artifact_mtimes(ArtifactStore(root))
    # Nothing written this cell — the folder is exactly as it was.

    _track_edits_since(session, ArtifactStore(root), before)

    assert session._artifacts_touched == set()


async def test_a_slug_that_did_not_exist_before_is_not_claimed_as_an_edit(session, root):
    """A folder the scratchpad creates from scratch (bypassing create_artifact
    too) isn't in `before` at all, so the mtime fallback correctly leaves it
    alone — this path only ever claims EDITS to already-existing artifacts."""
    before: dict[str, float] = {}
    (root / "brand-new").mkdir(parents=True)
    (root / "brand-new" / "metadata.json").write_text("{}")
    (root / "brand-new" / "index.html").write_text("<html></html>")

    _track_edits_since(session, ArtifactStore(root), before)

    assert session._artifacts_touched == set()


async def test_reopened_slug_is_not_double_tracked_by_the_fallback(session, root):
    """When open_artifact WAS called this turn, the fallback is a no-op for
    that slug — it only fills in what the tools missed."""
    slug = await _create(session)
    await handle_open_artifact(session, {"slug": slug})
    assert session._artifacts_touched == {slug}

    before = _snapshot_existing_artifact_mtimes(ArtifactStore(root))
    index = root / slug / "index.html"
    index.write_text("<html>v2</html>")
    _bump_mtime(index)

    _track_edits_since(session, ArtifactStore(root), before)

    assert session._artifacts_touched == {slug}
