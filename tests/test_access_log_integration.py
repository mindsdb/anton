"""
Integration tests for the Memory Access Log feature.

Tests the full flow:
  - access_log.jsonl is created and appended to when memories are delivered
  - Log entries contain correct fields (session_id, memory_id, scope, kind, topic)
  - Same memory delivered twice in a session produces two entries
  - get_session_entries filters correctly by session_id
  - No entries when session_id is None
  - AccessLog integrates with Cortex.build_memory_context()

Uses temporary directories — never touches ~/.anton/memory/.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from anton.core.memory.access_log import AccessLog
from anton.core.memory.hippocampus import Hippocampus


SESSION_A = "20260422_100000"
SESSION_B = "20260422_110000"


@pytest.fixture()
def mem_dir(tmp_path):
    d = tmp_path / "memory"
    d.mkdir()
    return d


@pytest.fixture()
def hc(mem_dir):
    return Hippocampus(mem_dir)


@pytest.fixture()
def access_log(mem_dir):
    return AccessLog(mem_dir)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ── AccessLog unit tests ──────────────────────────────────────────────────────

class TestAccessLogWrite:
    def test_creates_file_on_first_write(self, mem_dir, access_log):
        records = [{"id": "m_abc12345", "kind": "always", "topic": ""}]
        access_log.log_delivered(records, scope="project", session_id=SESSION_A)
        assert (mem_dir / "access_log.jsonl").exists()

    def test_log_entry_fields(self, mem_dir, access_log):
        records = [{"id": "m_abc12345", "text": "Always do X.", "kind": "always", "topic": "db"}]
        access_log.log_delivered(records, scope="project", session_id=SESSION_A)
        entries = read_jsonl(mem_dir / "access_log.jsonl")
        assert len(entries) == 1
        e = entries[0]
        assert e["session_id"] == SESSION_A
        assert e["memory_id"] == "m_abc12345"
        assert e["memory_text"] == "Always do X."
        assert e["memory_scope"] == "project"
        assert e["memory_kind"] == "always"
        assert e["memory_topic"] == "db"
        assert "delivered_at" in e
        assert "T" in e["delivered_at"]  # ISO 8601

    def test_multiple_records_appended(self, mem_dir, access_log):
        records = [
            {"id": "m_aaa00001", "kind": "always", "topic": ""},
            {"id": "m_bbb00002", "kind": "never", "topic": ""},
        ]
        access_log.log_delivered(records, scope="project", session_id=SESSION_A)
        entries = read_jsonl(mem_dir / "access_log.jsonl")
        assert len(entries) == 2

    def test_appends_across_calls(self, mem_dir, access_log):
        access_log.log_delivered(
            [{"id": "m_aaa00001", "kind": "always", "topic": ""}],
            scope="project",
            session_id=SESSION_A,
        )
        access_log.log_delivered(
            [{"id": "m_bbb00002", "kind": "lesson", "topic": "api"}],
            scope="project",
            session_id=SESSION_A,
        )
        entries = read_jsonl(mem_dir / "access_log.jsonl")
        assert len(entries) == 2

    def test_no_write_when_session_id_none(self, mem_dir, access_log):
        access_log.log_delivered(
            [{"id": "m_aaa00001", "kind": "always", "topic": ""}],
            scope="project",
            session_id=None,
        )
        assert not (mem_dir / "access_log.jsonl").exists()

    def test_no_write_when_records_empty(self, mem_dir, access_log):
        access_log.log_delivered([], scope="project", session_id=SESSION_A)
        assert not (mem_dir / "access_log.jsonl").exists()


# ── get_session_entries ───────────────────────────────────────────────────────

class TestGetSessionEntries:
    def test_returns_entries_for_session(self, mem_dir, access_log):
        access_log.log_delivered(
            [{"id": "m_aaa00001", "kind": "always", "topic": ""}],
            scope="project",
            session_id=SESSION_A,
        )
        access_log.log_delivered(
            [{"id": "m_bbb00002", "kind": "lesson", "topic": ""}],
            scope="project",
            session_id=SESSION_B,
        )
        entries = access_log.get_session_entries(SESSION_A)
        assert len(entries) == 1
        assert entries[0]["memory_id"] == "m_aaa00001"

    def test_returns_empty_when_no_entries_for_session(self, mem_dir, access_log):
        access_log.log_delivered(
            [{"id": "m_aaa00001", "kind": "always", "topic": ""}],
            scope="project",
            session_id=SESSION_B,
        )
        assert access_log.get_session_entries(SESSION_A) == []

    def test_returns_empty_when_file_missing(self, mem_dir, access_log):
        assert access_log.get_session_entries(SESSION_A) == []

    def test_multiple_deliveries_of_same_memory(self, mem_dir, access_log):
        """Same memory delivered twice → two entries (deduplicate at export time)."""
        record = {"id": "m_aaa00001", "kind": "always", "topic": ""}
        access_log.log_delivered([record], scope="project", session_id=SESSION_A)
        access_log.log_delivered([record], scope="project", session_id=SESSION_A)
        entries = access_log.get_session_entries(SESSION_A)
        assert len(entries) == 2


# ── Hippocampus record-listing methods ───────────────────────────────────────

class TestHippocampusListRecords:
    def test_list_rule_records_returns_all_rules(self, hc, mem_dir):
        hc.encode_rule("Use httpx", kind="always", session_id=SESSION_A)
        hc.encode_rule("No sleep()", kind="never", session_id=SESSION_A)
        records = hc.list_rule_records()
        assert len(records) == 2
        texts = [r["text"] for r in records]
        assert "Use httpx" in texts
        assert "No sleep()" in texts

    def test_list_lesson_records_matches_recall_budget(self, hc, mem_dir):
        hc.encode_lesson("Fact one", session_id=SESSION_A)
        hc.encode_lesson("Fact two", session_id=SESSION_A)
        records = hc.list_lesson_records(token_budget=1000)
        # Both facts should fit well within budget
        assert len(records) == 2

    def test_list_rule_records_empty_when_no_rules(self, hc):
        assert hc.list_rule_records() == []

    def test_list_lesson_records_empty_when_no_lessons(self, hc):
        assert hc.list_lesson_records() == []

    def test_list_lesson_records_respects_budget(self, hc, mem_dir):
        """Tiny budget should exclude most lessons."""
        for i in range(20):
            hc.encode_lesson(f"Fact number {i} with some extra text to consume budget", session_id=SESSION_A)
        all_records = hc.list_lesson_records(token_budget=999999)
        limited_records = hc.list_lesson_records(token_budget=1)
        assert len(limited_records) < len(all_records)


# ── Cortex integration: access log wired through build_memory_context ─────────

class TestCortexAccessLogIntegration:
    @pytest.mark.asyncio
    async def test_project_rules_logged_on_delivery(self, mem_dir):
        from anton.core.memory.access_log import AccessLog
        from anton.core.memory.cortex import Cortex
        from anton.core.memory.hippocampus import Hippocampus

        project_mem = mem_dir / "project"
        global_mem = mem_dir / "global"
        project_mem.mkdir()
        global_mem.mkdir()

        project_hc = Hippocampus(project_mem)
        global_hc = Hippocampus(global_mem)
        log = AccessLog(project_mem)

        project_hc.encode_rule("Use httpx", kind="always", session_id=SESSION_A)

        cortex = Cortex(
            global_hc=global_hc,
            project_hc=project_hc,
            access_log=log,
        )
        await cortex.build_memory_context("hello", session_id=SESSION_A)

        entries = log.get_session_entries(SESSION_A)
        assert any(e["memory_kind"] == "always" for e in entries)
        assert all(e["memory_scope"] == "project" for e in entries)

    @pytest.mark.asyncio
    async def test_project_lessons_logged_on_delivery(self, mem_dir):
        from anton.core.memory.access_log import AccessLog
        from anton.core.memory.cortex import Cortex
        from anton.core.memory.hippocampus import Hippocampus

        project_mem = mem_dir / "project"
        global_mem = mem_dir / "global"
        project_mem.mkdir()
        global_mem.mkdir()

        project_hc = Hippocampus(project_mem)
        global_hc = Hippocampus(global_mem)
        log = AccessLog(project_mem)

        project_hc.encode_lesson("CoinGecko rate limit is 50/min", topic="api", session_id=SESSION_A)

        cortex = Cortex(
            global_hc=global_hc,
            project_hc=project_hc,
            access_log=log,
        )
        await cortex.build_memory_context("hello", session_id=SESSION_A)

        entries = log.get_session_entries(SESSION_A)
        assert any(e["memory_kind"] == "lesson" for e in entries)
        assert any(e["memory_topic"] == "api" for e in entries)

    @pytest.mark.asyncio
    async def test_no_log_when_session_id_none(self, mem_dir):
        from anton.core.memory.access_log import AccessLog
        from anton.core.memory.cortex import Cortex
        from anton.core.memory.hippocampus import Hippocampus

        project_mem = mem_dir / "project"
        global_mem = mem_dir / "global"
        project_mem.mkdir()
        global_mem.mkdir()

        project_hc = Hippocampus(project_mem)
        global_hc = Hippocampus(global_mem)
        log = AccessLog(project_mem)

        project_hc.encode_rule("Use httpx", kind="always")

        cortex = Cortex(
            global_hc=global_hc,
            project_hc=project_hc,
            access_log=log,
        )
        await cortex.build_memory_context("hello", session_id=None)

        assert not (project_mem / "access_log.jsonl").exists()

    @pytest.mark.asyncio
    async def test_no_log_when_access_log_not_set(self, mem_dir):
        from anton.core.memory.cortex import Cortex
        from anton.core.memory.hippocampus import Hippocampus

        project_mem = mem_dir / "project"
        global_mem = mem_dir / "global"
        project_mem.mkdir()
        global_mem.mkdir()

        project_hc = Hippocampus(project_mem)
        global_hc = Hippocampus(global_mem)

        project_hc.encode_rule("Use httpx", kind="always")

        cortex = Cortex(global_hc=global_hc, project_hc=project_hc)
        # Should not raise even without access_log
        await cortex.build_memory_context("hello", session_id=SESSION_A)

    @pytest.mark.asyncio
    async def test_different_sessions_logged_separately(self, mem_dir):
        from anton.core.memory.access_log import AccessLog
        from anton.core.memory.cortex import Cortex
        from anton.core.memory.hippocampus import Hippocampus

        project_mem = mem_dir / "project"
        global_mem = mem_dir / "global"
        project_mem.mkdir()
        global_mem.mkdir()

        project_hc = Hippocampus(project_mem)
        global_hc = Hippocampus(global_mem)
        log = AccessLog(project_mem)

        project_hc.encode_rule("Use httpx", kind="always", session_id=SESSION_A)

        cortex = Cortex(
            global_hc=global_hc,
            project_hc=project_hc,
            access_log=log,
        )
        await cortex.build_memory_context("hello", session_id=SESSION_A)
        await cortex.build_memory_context("world", session_id=SESSION_B)

        a_entries = log.get_session_entries(SESSION_A)
        b_entries = log.get_session_entries(SESSION_B)
        assert len(a_entries) > 0
        assert len(b_entries) > 0
        assert a_entries[0]["session_id"] == SESSION_A
        assert b_entries[0]["session_id"] == SESSION_B
