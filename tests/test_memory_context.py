from __future__ import annotations

import pytest

from anton.memory.context import MemoryContext
from anton.memory.learnings import LearningStore
from anton.memory.store import SessionStore


@pytest.fixture()
def session_store(tmp_path):
    return SessionStore(tmp_path)


@pytest.fixture()
def learning_store(tmp_path):
    return LearningStore(tmp_path)


@pytest.fixture()
def ctx(session_store, learning_store):
    return MemoryContext(session_store, learning_store)


class TestBuild:
    async def test_no_history_returns_empty_string(self, ctx):
        result = ctx.build("some task")
        assert result == ""

    async def test_with_summaries_includes_recent_activity(self, ctx, session_store):
        s1 = await session_store.start_session("task 1")
        await session_store.complete_session(s1, "Completed task 1 successfully")

        result = ctx.build("new task")
        assert "## Recent Activity" in result
        assert "Completed task 1 successfully" in result

    async def test_with_relevant_learnings_includes_them(self, ctx, learning_store):
        await learning_store.record("file_ops", "Always check existence", "File operations safety")

        result = ctx.build("read a file safely")
        assert "## Relevant Learnings" in result
        assert "Always check existence" in result

    async def test_with_both_includes_both_sections(self, ctx, session_store, learning_store):
        s1 = await session_store.start_session("task 1")
        await session_store.complete_session(s1, "Completed task 1")
        await learning_store.record("file_ops", "Check existence", "File operations")

        result = ctx.build("file operations task")
        assert "## Recent Activity" in result
        assert "## Relevant Learnings" in result

    async def test_no_matching_learnings_omits_section(self, ctx, learning_store):
        await learning_store.record("network", "HTTP patterns", "Network stuff")

        result = ctx.build("quantum computing")
        assert "## Relevant Learnings" not in result
