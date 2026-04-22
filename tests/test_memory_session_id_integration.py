"""
Integration test for the session_id memory feature.

Tests the full flow end-to-end:
  - Engrams are created with session_id
  - session_id is stored in JSONL
  - Migration from .md preserves all content, sets session_id=None
  - recall_* methods return correct markdown output
  - Dedup, topic files, and scratchpad wisdom all work correctly

Uses a temporary directory — never touches ~/.anton/memory/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anton.core.memory.hippocampus import Engram, Hippocampus, _migrate_md_to_jsonl


SESSION_A = "20260421_100000"
SESSION_B = "20260421_110000"


@pytest.fixture()
def mem_dir(tmp_path):
    d = tmp_path / "memory"
    d.mkdir()
    return d


@pytest.fixture()
def hc(mem_dir):
    return Hippocampus(mem_dir)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ── session_id is stored ──────────────────────────────────────────────────────

class TestSessionIdStored:
    def test_rule_stores_session_id(self, hc, mem_dir):
        hc.encode_rule("Use httpx", kind="always", session_id=SESSION_A)
        records = read_jsonl(mem_dir / "rules.jsonl")
        assert records[0]["session_id"] == SESSION_A

    def test_lesson_stores_session_id(self, hc, mem_dir):
        hc.encode_lesson("CoinGecko rate limit is 50/min", topic="api", session_id=SESSION_A)
        records = read_jsonl(mem_dir / "lessons.jsonl")
        assert records[0]["session_id"] == SESSION_A

    def test_topic_file_also_stores_session_id(self, hc, mem_dir):
        hc.encode_lesson("CoinGecko rate limit is 50/min", topic="api", session_id=SESSION_A)
        records = read_jsonl(mem_dir / "topics" / "api.jsonl")
        assert records[0]["session_id"] == SESSION_A

    def test_different_sessions_stored_correctly(self, hc, mem_dir):
        hc.encode_rule("Always use httpx", kind="always", session_id=SESSION_A)
        hc.encode_rule("Never use sleep()", kind="never", session_id=SESSION_B)
        records = read_jsonl(mem_dir / "rules.jsonl")
        assert records[0]["session_id"] == SESSION_A
        assert records[1]["session_id"] == SESSION_B

    def test_null_session_id_when_not_provided(self, hc, mem_dir):
        hc.encode_rule("Use httpx", kind="always")
        records = read_jsonl(mem_dir / "rules.jsonl")
        assert records[0]["session_id"] is None


# ── id and created_at are always set ─────────────────────────────────────────

class TestEngramMetadata:
    def test_id_is_set(self, hc, mem_dir):
        hc.encode_rule("Use httpx", kind="always", session_id=SESSION_A)
        records = read_jsonl(mem_dir / "rules.jsonl")
        assert records[0]["id"].startswith("m_")
        assert len(records[0]["id"]) == 10  # "m_" + 8 hex chars

    def test_created_at_is_set(self, hc, mem_dir):
        hc.encode_lesson("Some fact", session_id=SESSION_A)
        records = read_jsonl(mem_dir / "lessons.jsonl")
        assert records[0]["created_at"]  # non-empty
        assert "T" in records[0]["created_at"]  # ISO 8601 format

    def test_each_engram_gets_unique_id(self, hc, mem_dir):
        hc.encode_rule("Rule one", kind="always", session_id=SESSION_A)
        hc.encode_rule("Rule two", kind="never", session_id=SESSION_A)
        records = read_jsonl(mem_dir / "rules.jsonl")
        ids = [r["id"] for r in records]
        assert len(ids) == len(set(ids))  # all unique


# ── output contract unchanged ─────────────────────────────────────────────────

class TestOutputContract:
    def test_recall_rules_returns_markdown_sections(self, hc, mem_dir):
        hc.encode_rule("Use httpx", kind="always", session_id=SESSION_A)
        hc.encode_rule("No sleep()", kind="never", session_id=SESSION_A)
        hc.encode_rule("If paginated → use progress()", kind="when", session_id=SESSION_A)
        result = hc.recall_rules()
        assert "## Always" in result
        assert "## Never" in result
        assert "## When" in result
        assert "Use httpx" in result
        assert "No sleep()" in result
        assert "If paginated" in result

    def test_recall_lessons_returns_markdown_list(self, hc, mem_dir):
        hc.encode_lesson("Fact one", session_id=SESSION_A)
        hc.encode_lesson("Fact two", session_id=SESSION_B)
        result = hc.recall_lessons()
        assert "- Fact one" in result or "- Fact two" in result

    def test_recall_identity_returns_markdown_list(self, hc, mem_dir):
        hc.rewrite_identity(["Name: Jorge", "TZ: PST"])
        result = hc.recall_identity()
        assert "- Name: Jorge" in result
        assert "- TZ: PST" in result

    def test_recall_topic_returns_markdown(self, hc, mem_dir):
        hc.encode_lesson("Rate limit: 50/min", topic="api-coingecko", session_id=SESSION_A)
        result = hc.recall_topic("api-coingecko")
        assert "Rate limit: 50/min" in result

    def test_scratchpad_wisdom_returns_when_rules_and_lessons(self, hc, mem_dir):
        hc.encode_rule("If paginated → use progress()", kind="when", session_id=SESSION_A)
        hc.encode_lesson("Scratchpad cells timeout at 30s", topic="scratchpad", session_id=SESSION_A)
        result = hc.recall_scratchpad_wisdom()
        assert "paginated" in result
        assert "Scratchpad cells timeout" in result


# ── migration ─────────────────────────────────────────────────────────────────

class TestMigrationIntegration:
    def test_existing_user_memories_are_preserved(self, tmp_path):
        """Simulates an existing user upgrading — their memories survive."""
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        (mem_dir / "rules.md").write_text(
            "# Rules\n\n## Always\n- Use httpx <!-- confidence:high source:user ts:2026-01-01 -->\n\n## Never\n- Use sleep\n"
        )
        (mem_dir / "lessons.md").write_text(
            "# Lessons\n- CoinGecko rate limit is 50/min <!-- topic:api ts:2026-01-01 -->\n"
        )

        hc = Hippocampus(mem_dir)

        assert "Use httpx" in hc.recall_rules()
        assert "Use sleep" in hc.recall_rules()
        assert "CoinGecko rate limit" in hc.recall_lessons()

    def test_migrated_records_have_null_session_id(self, tmp_path):
        """Migrated memories have session_id=None — expected, documented behavior."""
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        (mem_dir / "lessons.md").write_text("# Lessons\n- Old fact\n")
        Hippocampus(mem_dir)
        records = read_jsonl(mem_dir / "lessons.jsonl")
        assert all(r["session_id"] is None for r in records)

    def test_new_memories_after_migration_get_session_id(self, tmp_path):
        """After migration, new memories written in a session get session_id."""
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        (mem_dir / "lessons.md").write_text("# Lessons\n- Old fact\n")
        hc = Hippocampus(mem_dir)
        hc.encode_lesson("New fact from session", session_id=SESSION_A)

        records = read_jsonl(mem_dir / "lessons.jsonl")
        old = next(r for r in records if r["text"] == "Old fact")
        new = next(r for r in records if r["text"] == "New fact from session")
        assert old["session_id"] is None
        assert new["session_id"] == SESSION_A

    def test_bak_file_created_for_rollback(self, tmp_path):
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        (mem_dir / "lessons.md").write_text("# Lessons\n- Old fact\n")
        Hippocampus(mem_dir)
        assert (mem_dir / "lessons.md.bak").exists()
        assert not (mem_dir / "lessons.md").exists()


# ── session filtering (foundation for /share export) ─────────────────────────

class TestSessionFiltering:
    def test_can_filter_memories_by_session_id(self, hc, mem_dir):
        """The access log uses session_id to filter — verify the data is queryable."""
        hc.encode_rule("Rule from session A", kind="always", session_id=SESSION_A)
        hc.encode_rule("Rule from session B", kind="never", session_id=SESSION_B)
        hc.encode_lesson("Lesson from session A", session_id=SESSION_A)
        hc.encode_lesson("Lesson no session")

        all_rules = read_jsonl(mem_dir / "rules.jsonl")
        all_lessons = read_jsonl(mem_dir / "lessons.jsonl")

        session_a_memories = [
            r for r in all_rules + all_lessons
            if r.get("session_id") == SESSION_A
        ]
        assert len(session_a_memories) == 2
        texts = [r["text"] for r in session_a_memories]
        assert "Rule from session A" in texts
        assert "Lesson from session A" in texts

    def test_null_session_memories_excluded_from_session_filter(self, hc, mem_dir):
        hc.encode_lesson("No session lesson")
        hc.encode_lesson("Session A lesson", session_id=SESSION_A)

        all_lessons = read_jsonl(mem_dir / "lessons.jsonl")
        session_a = [r for r in all_lessons if r.get("session_id") == SESSION_A]
        assert len(session_a) == 1
        assert session_a[0]["text"] == "Session A lesson"
