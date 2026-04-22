from __future__ import annotations

import json
from pathlib import Path

import pytest

from anton.core.memory.hippocampus import Hippocampus


@pytest.fixture()
def mem_dir(tmp_path):
    d = tmp_path / "memory"
    d.mkdir()
    return d


@pytest.fixture()
def hc(mem_dir):
    return Hippocampus(mem_dir)


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Helper: write JSONL records to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def make_record(text: str, kind: str = "lesson", topic: str = "") -> dict:
    return {"id": "m_test", "text": text, "kind": kind, "scope": "project",
            "confidence": "medium", "topic": topic, "source": "llm",
            "session_id": None, "created_at": "2026-04-21T10:00:00Z"}


class TestRecallIdentity:
    def test_empty_returns_empty(self, hc):
        assert hc.recall_identity() == ""

    def test_reads_profile(self, hc, mem_dir):
        write_jsonl(mem_dir / "profile.jsonl", [
            make_record("Name: Jorge", kind="profile"),
            make_record("TZ: PST", kind="profile"),
        ])
        result = hc.recall_identity()
        assert "Name: Jorge" in result
        assert "TZ: PST" in result

    def test_nonexistent_dir(self, tmp_path):
        hc = Hippocampus(tmp_path / "nonexistent")
        assert hc.recall_identity() == ""


class TestRecallRules:
    def test_empty_returns_empty(self, hc):
        assert hc.recall_rules() == ""

    def test_reads_rules(self, hc, mem_dir):
        write_jsonl(mem_dir / "rules.jsonl", [
            make_record("Use httpx", kind="always"),
            make_record("Use sleep", kind="never"),
        ])
        result = hc.recall_rules()
        assert "Use httpx" in result
        assert "Use sleep" in result


class TestRecallLessons:
    def test_empty_returns_empty(self, hc):
        assert hc.recall_lessons() == ""

    def test_reads_lessons(self, hc, mem_dir):
        write_jsonl(mem_dir / "lessons.jsonl", [
            make_record("Fact one"),
            make_record("Fact two"),
        ])
        result = hc.recall_lessons()
        assert "Fact one" in result or "Fact two" in result

    def test_budget_limits_output(self, hc, mem_dir):
        records = [make_record(f"Lesson number {i} with some extra words") for i in range(50)]
        write_jsonl(mem_dir / "lessons.jsonl", records)
        result = hc.recall_lessons(token_budget=10)
        entry_count = result.count("- Lesson")
        assert entry_count < 50


class TestRecallTopic:
    def test_empty_returns_empty(self, hc):
        assert hc.recall_topic("nonexistent") == ""

    def test_reads_topic(self, hc, mem_dir):
        topics = mem_dir / "topics"
        topics.mkdir()
        write_jsonl(topics / "api-coingecko.jsonl", [
            make_record("Rate limit: 50/min", topic="api-coingecko"),
        ])
        result = hc.recall_topic("api-coingecko")
        assert "Rate limit: 50/min" in result


class TestRecallScratchpadWisdom:
    def test_empty_returns_empty(self, hc):
        assert hc.recall_scratchpad_wisdom() == ""

    def test_extracts_when_rules(self, hc, mem_dir):
        write_jsonl(mem_dir / "rules.jsonl", [
            make_record("Be fast", kind="always"),
            make_record("If paginated → use progress()", kind="when"),
        ])
        result = hc.recall_scratchpad_wisdom()
        assert "paginated" in result

    def test_includes_scratchpad_lessons(self, hc, mem_dir):
        write_jsonl(mem_dir / "lessons.jsonl", [
            make_record("Scratchpad cells timeout at 30s", topic="scratchpad"),
            make_record("Unrelated fact"),
        ])
        result = hc.recall_scratchpad_wisdom()
        assert "Scratchpad cells timeout" in result
        assert "Unrelated fact" not in result

    def test_includes_scratchpad_topic_files(self, hc, mem_dir):
        topics = mem_dir / "topics"
        topics.mkdir()
        write_jsonl(topics / "scratchpad-tips.jsonl", [
            make_record("Always re-import modules", topic="scratchpad-tips"),
        ])
        result = hc.recall_scratchpad_wisdom()
        assert "Always re-import" in result


class TestEncodeRule:
    def test_creates_rules_file(self, hc, mem_dir):
        hc.encode_rule("Use httpx", kind="always", confidence="high", source="user")
        assert (mem_dir / "rules.jsonl").exists()
        records = [json.loads(l) for l in (mem_dir / "rules.jsonl").read_text().splitlines() if l]
        texts = [r["text"] for r in records]
        assert "Use httpx" in texts
        assert any(r["kind"] == "always" for r in records)

    def test_appends_to_correct_section(self, hc, mem_dir):
        hc.encode_rule("Use httpx", kind="always")
        hc.encode_rule("No sleep", kind="never")
        hc.encode_rule("If slow → batch", kind="when")
        result = hc.recall_rules()
        assert "Use httpx" in result
        assert "No sleep" in result
        assert "If slow" in result

    def test_skips_duplicate(self, hc, mem_dir):
        hc.encode_rule("Use httpx", kind="always")
        hc.encode_rule("Use httpx", kind="always")
        records = [json.loads(l) for l in (mem_dir / "rules.jsonl").read_text().splitlines() if l]
        assert sum(1 for r in records if r["text"] == "Use httpx") == 1

    def test_includes_metadata(self, hc, mem_dir):
        hc.encode_rule("Use httpx", kind="always", confidence="high", source="user")
        records = [json.loads(l) for l in (mem_dir / "rules.jsonl").read_text().splitlines() if l]
        r = records[0]
        assert r["confidence"] == "high"
        assert r["source"] == "user"

    def test_allows_superstring_of_existing(self, hc, mem_dir):
        """A longer, more specific rule should NOT be blocked by a shorter one."""
        hc.encode_rule("Use httpx", kind="always")
        hc.encode_rule("Use httpx with timeout=15", kind="always")
        records = [json.loads(l) for l in (mem_dir / "rules.jsonl").read_text().splitlines() if l]
        texts = [r["text"] for r in records]
        assert "Use httpx with timeout=15" in texts

    def test_allows_substring_of_existing(self, hc, mem_dir):
        """A shorter rule should NOT be blocked by a longer one containing it."""
        hc.encode_rule("Use httpx with timeout=15", kind="always")
        hc.encode_rule("Use httpx", kind="always")
        records = [json.loads(l) for l in (mem_dir / "rules.jsonl").read_text().splitlines() if l]
        texts = [r["text"] for r in records]
        assert "Use httpx" in texts
        assert "Use httpx with timeout=15" in texts


class TestEncodeLesson:
    def test_creates_lessons_file(self, hc, mem_dir):
        hc.encode_lesson("CoinGecko limits at 50/min", topic="api-coingecko")
        assert (mem_dir / "lessons.jsonl").exists()
        records = [json.loads(l) for l in (mem_dir / "lessons.jsonl").read_text().splitlines() if l]
        assert any("CoinGecko limits at 50/min" in r["text"] for r in records)

    def test_creates_topic_file(self, hc, mem_dir):
        hc.encode_lesson("CoinGecko limits at 50/min", topic="api-coingecko")
        topic_path = mem_dir / "topics" / "api-coingecko.jsonl"
        assert topic_path.exists()
        records = [json.loads(l) for l in topic_path.read_text().splitlines() if l]
        assert any("CoinGecko limits at 50/min" in r["text"] for r in records)

    def test_skips_duplicate(self, hc, mem_dir):
        hc.encode_lesson("Fact one")
        hc.encode_lesson("Fact one")
        records = [json.loads(l) for l in (mem_dir / "lessons.jsonl").read_text().splitlines() if l]
        assert sum(1 for r in records if r["text"] == "Fact one") == 1

    def test_no_topic_no_topic_file(self, hc, mem_dir):
        hc.encode_lesson("Simple fact")
        assert not (mem_dir / "topics").exists() or not any((mem_dir / "topics").iterdir())

    def test_allows_superstring_of_existing_lesson(self, hc, mem_dir):
        hc.encode_lesson("CoinGecko limits at 50/min")
        hc.encode_lesson("CoinGecko limits at 50/min for free tier accounts")
        records = [json.loads(l) for l in (mem_dir / "lessons.jsonl").read_text().splitlines() if l]
        texts = [r["text"] for r in records]
        assert "CoinGecko limits at 50/min for free tier accounts" in texts

    def test_skips_exact_duplicate_with_metadata(self, hc, mem_dir):
        hc.encode_lesson("Fact one", topic="api")
        hc.encode_lesson("Fact one", topic="other")
        records = [json.loads(l) for l in (mem_dir / "lessons.jsonl").read_text().splitlines() if l]
        assert sum(1 for r in records if r["text"] == "Fact one") == 1


class TestRewriteIdentity:
    def test_creates_profile(self, hc, mem_dir):
        hc.rewrite_identity(["Name: Jorge", "TZ: PST"])
        assert (mem_dir / "profile.jsonl").exists()
        result = hc.recall_identity()
        assert "Name: Jorge" in result
        assert "TZ: PST" in result

    def test_overwrites_existing(self, hc, mem_dir):
        hc.rewrite_identity(["Name: Old"])
        hc.rewrite_identity(["Name: New"])
        result = hc.recall_identity()
        assert "Name: New" in result
        assert "Name: Old" not in result


class TestEntryCount:
    def test_empty_returns_zero(self, hc):
        assert hc.entry_count() == 0

    def test_counts_entries(self, hc, mem_dir):
        hc.encode_rule("Rule 1", kind="always")
        hc.encode_lesson("Lesson 1")
        assert hc.entry_count() == 2


class TestSanitizeSlug:
    def test_simple(self):
        assert Hippocampus._sanitize_slug("hello world") == "hello-world"

    def test_special_chars(self):
        assert Hippocampus._sanitize_slug("API: CoinGecko!") == "api-coingecko"

    def test_empty(self):
        assert Hippocampus._sanitize_slug("") == "general"


class TestMigration:
    def test_migrates_lessons_md(self, tmp_path):
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        (mem_dir / "lessons.md").write_text("# Lessons\n- Fact one\n- Fact two\n")
        hc = Hippocampus(mem_dir)
        result = hc.recall_lessons()
        assert "Fact one" in result
        assert "Fact two" in result
        assert (mem_dir / "lessons.jsonl").exists()
        assert (mem_dir / "lessons.md.bak").exists()

    def test_migrates_rules_md(self, tmp_path):
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        (mem_dir / "rules.md").write_text(
            "# Rules\n\n## Always\n- Use httpx\n\n## Never\n- Use sleep\n"
        )
        hc = Hippocampus(mem_dir)
        result = hc.recall_rules()
        assert "Use httpx" in result
        assert "Use sleep" in result
        assert (mem_dir / "rules.md.bak").exists()

    def test_migration_is_idempotent(self, tmp_path):
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        (mem_dir / "lessons.md").write_text("# Lessons\n- Fact one\n")
        Hippocampus(mem_dir)
        Hippocampus(mem_dir)  # second init should not re-migrate
        assert (mem_dir / "lessons.jsonl").exists()

    def test_session_id_is_null_after_migration(self, tmp_path):
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        (mem_dir / "lessons.md").write_text("# Lessons\n- Fact one\n")
        Hippocampus(mem_dir)
        records = [json.loads(l) for l in (mem_dir / "lessons.jsonl").read_text().splitlines() if l]
        assert all(r["session_id"] is None for r in records)
