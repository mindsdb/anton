"""ArtifactStore.find: exact slug or id, reconciling only what matched."""
from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from anton.core.artifacts import ArtifactStore


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts")


def test_find_by_slug_and_by_id(store):
    a = store.create(name="Alpha", description="d", type="html-app")
    b = store.create(name="Beta", description="d", type="document")
    store.create(name="Gamma", description="d", type="dataset")

    found, unmatched = store.find([a.slug, b.id])

    assert {x.slug for x in found} == {a.slug, b.slug}
    assert unmatched == []


def test_an_id_matches_in_any_accepted_spelling(store):
    a = store.create(name="Alpha", description="d", type="html-app")
    found, unmatched = store.find([str(uuid.UUID(a.id)).upper()])
    assert [x.slug for x in found] == [a.slug]
    assert unmatched == []


def test_unknown_and_escaping_keys_are_unmatched(store):
    store.create(name="Alpha", description="d", type="html-app")
    found, unmatched = store.find(["nope", "../outside", ""])
    assert found == []
    assert unmatched == ["nope", "../outside", ""]


def test_an_artifact_matched_twice_is_listed_once(store):
    a = store.create(name="Alpha", description="d", type="html-app")
    found, unmatched = store.find([a.slug, a.id, a.slug])
    assert [x.slug for x in found] == [a.slug]
    assert unmatched == []


def test_only_matches_are_reconciled(store, monkeypatch):
    a = store.create(name="Alpha", description="d", type="html-app")
    store.create(name="Beta", description="d", type="html-app")
    seen: list[str] = []
    original = ArtifactStore._reconcile_files

    def spy(self, artifact):
        seen.append(artifact.slug)
        return original(self, artifact)

    monkeypatch.setattr(ArtifactStore, "_reconcile_files", spy)
    store.find([a.id])
    assert seen == [a.slug]


def test_a_key_that_cannot_be_an_id_does_not_scan_the_store(store, monkeypatch):
    store.create(name="Alpha", description="d", type="html-app")
    store.create(name="Beta", description="d", type="html-app")
    loaded: list[str] = []
    original = ArtifactStore._load_silent

    def spy(self, slug):
        loaded.append(slug)
        return original(self, slug)

    monkeypatch.setattr(ArtifactStore, "_load_silent", spy)
    assert store.find(["alpha-typo"]) == ([], ["alpha-typo"])
    assert loaded == ["alpha-typo"]


def test_find_on_a_missing_root(tmp_path):
    found, unmatched = ArtifactStore(tmp_path / "none").find(["x"])
    assert (found, unmatched) == ([], ["x"])
