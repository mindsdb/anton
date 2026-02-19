from __future__ import annotations

import json

import pytest

from anton.memory.learnings import LearningStore


@pytest.fixture()
def store(tmp_path):
    return LearningStore(tmp_path)


class TestRecord:
    async def test_creates_topic_file_and_updates_index(self, store, tmp_path):
        await store.record("file_ops", "Always check if file exists", "Check file existence")

        # Topic file created
        topic_file = tmp_path / "learnings" / "file_ops.md"
        assert topic_file.exists()
        assert "Always check if file exists" in topic_file.read_text()

        # Index updated
        index = json.loads((tmp_path / "learnings" / "index.json").read_text())
        assert "file_ops" in index
        assert index["file_ops"]["summary"] == "Check file existence"
        assert index["file_ops"]["updated_at"] > 0

    async def test_appends_to_existing_topic(self, store, tmp_path):
        await store.record("file_ops", "First learning", "Summary 1")
        await store.record("file_ops", "Second learning", "Summary 2")

        topic_file = tmp_path / "learnings" / "file_ops.md"
        content = topic_file.read_text()
        assert "First learning" in content
        assert "Second learning" in content

        # Index summary should be updated to latest
        index = json.loads((tmp_path / "learnings" / "index.json").read_text())
        assert index["file_ops"]["summary"] == "Summary 2"

    async def test_slugifies_topic_name(self, store, tmp_path):
        await store.record("File Operations!", "content", "summary")

        topic_file = tmp_path / "learnings" / "file_operations.md"
        assert topic_file.exists()


class TestFindRelevant:
    async def test_returns_matches_by_keyword_overlap(self, store):
        await store.record("file_ops", "Reading files safely", "File read operations")
        await store.record("network", "HTTP request patterns", "Network HTTP patterns")

        results = store.find_relevant("read a file", limit=3)
        assert len(results) >= 1
        # file_ops should match due to "file" overlap
        topics = [r["topic"] for r in results]
        assert "file_ops" in topics

    async def test_returns_empty_for_no_matches(self, store):
        await store.record("file_ops", "Reading files safely", "File read operations")

        results = store.find_relevant("quantum computing", limit=3)
        assert results == []

    async def test_limits_results(self, store):
        await store.record("topic_a", "Content a about files", "Files summary a")
        await store.record("topic_b", "Content b about files", "Files summary b")
        await store.record("topic_c", "Content c about files", "Files summary c")

        results = store.find_relevant("files", limit=2)
        assert len(results) <= 2

    async def test_includes_full_content(self, store):
        await store.record("file_ops", "Detailed learning content here", "File operations")

        results = store.find_relevant("file operations")
        assert len(results) == 1
        assert "Detailed learning content here" in results[0]["content"]


class TestListAll:
    async def test_returns_all_topics(self, store):
        await store.record("file_ops", "Content 1", "Summary 1")
        await store.record("network", "Content 2", "Summary 2")

        items = store.list_all()
        assert len(items) == 2
        topics = {item["topic"] for item in items}
        assert topics == {"file_ops", "network"}

    async def test_returns_empty_when_no_learnings(self, store):
        assert store.list_all() == []
