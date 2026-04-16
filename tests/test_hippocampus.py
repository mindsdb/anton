from __future__ import annotations

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


class TestRecallIdentity:
    def test_empty_returns_empty(self, hc):
        assert hc.recall_identity() == ""

    def test_reads_profile(self, hc, mem_dir):
        (mem_dir / "profile.md").write_text("# Profile\n- Name: Jorge\n- TZ: PST")
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
        (mem_dir / "rules.md").write_text("# Rules\n\n## Always\n- Use httpx\n\n## Never\n- Use sleep\n")
        result = hc.recall_rules()
        assert "Use httpx" in result
        assert "Use sleep" in result


class TestRecallLessons:
    def test_empty_returns_empty(self, hc):
        assert hc.recall_lessons() == ""

    def test_reads_lessons(self, hc, mem_dir):
        (mem_dir / "lessons.md").write_text("# Lessons\n- Fact one\n- Fact two\n")
        result = hc.recall_lessons()
        assert "Fact one" in result or "Fact two" in result

    def test_budget_limits_output(self, hc, mem_dir):
        # Each entry is ~30 chars. Budget of 10 tokens = ~40 chars
        entries = [f"- Lesson number {i} with some extra words" for i in range(50)]
        (mem_dir / "lessons.md").write_text("# Lessons\n" + "\n".join(entries))
        result = hc.recall_lessons(token_budget=10)
        # Should have fewer entries than the original 50
        entry_count = result.count("- Lesson")
        assert entry_count < 50


class TestRecallTopic:
    def test_empty_returns_empty(self, hc):
        assert hc.recall_topic("nonexistent") == ""

    def test_reads_topic(self, hc, mem_dir):
        topics = mem_dir / "topics"
        topics.mkdir()
        (topics / "api-coingecko.md").write_text("# CoinGecko\n- Rate limit: 50/min")
        result = hc.recall_topic("api-coingecko")
        assert "Rate limit: 50/min" in result


class TestRecallScratchpadWisdom:
    def test_empty_returns_empty(self, hc):
        assert hc.recall_scratchpad_wisdom() == ""

    def test_extracts_when_rules(self, hc, mem_dir):
        (mem_dir / "rules.md").write_text(
            "# Rules\n\n## Always\n- Be fast\n\n## When\n- If paginated → use progress()\n"
        )
        result = hc.recall_scratchpad_wisdom()
        assert "paginated" in result

    def test_includes_scratchpad_lessons(self, hc, mem_dir):
        (mem_dir / "lessons.md").write_text(
            "# Lessons\n- Scratchpad cells timeout at 30s\n- Unrelated fact\n"
        )
        result = hc.recall_scratchpad_wisdom()
        assert "Scratchpad cells timeout" in result
        assert "Unrelated fact" not in result

    def test_includes_scratchpad_topic_files(self, hc, mem_dir):
        topics = mem_dir / "topics"
        topics.mkdir()
        (topics / "scratchpad-tips.md").write_text("Always re-import modules")
        result = hc.recall_scratchpad_wisdom()
        assert "Always re-import" in result


class TestEncodeRule:
    def test_creates_rules_file(self, hc, mem_dir):
        hc.encode_rule("Use httpx", kind="always", confidence="high", source="user")
        assert (mem_dir / "rules.md").exists()
        content = (mem_dir / "rules.md").read_text()
        assert "Use httpx" in content
        assert "## Always" in content

    def test_appends_to_correct_section(self, hc, mem_dir):
        hc.encode_rule("Use httpx", kind="always")
        hc.encode_rule("No sleep", kind="never")
        hc.encode_rule("If slow → batch", kind="when")

        content = (mem_dir / "rules.md").read_text()
        assert "Use httpx" in content
        assert "No sleep" in content
        assert "If slow" in content

    def test_skips_duplicate(self, hc, mem_dir):
        hc.encode_rule("Use httpx", kind="always")
        hc.encode_rule("Use httpx", kind="always")

        content = (mem_dir / "rules.md").read_text()
        assert content.count("Use httpx") == 1

    def test_includes_metadata(self, hc, mem_dir):
        hc.encode_rule("Use httpx", kind="always", confidence="high", source="user")
        content = (mem_dir / "rules.md").read_text()
        assert "confidence:high" in content
        assert "source:user" in content

    def test_allows_superstring_of_existing(self, hc, mem_dir):
        """A longer, more specific rule should NOT be blocked by a shorter one."""
        hc.encode_rule("Use httpx", kind="always")
        hc.encode_rule("Use httpx with timeout=15", kind="always")
        content = (mem_dir / "rules.md").read_text()
        assert "Use httpx with timeout=15" in content

    def test_allows_substring_of_existing(self, hc, mem_dir):
        """A shorter rule should NOT be blocked by a longer one containing it."""
        hc.encode_rule("Use httpx with timeout=15", kind="always")
        hc.encode_rule("Use httpx", kind="always")
        content = (mem_dir / "rules.md").read_text()
        assert content.count("Use httpx") == 2  # both present


class TestEncodeLesson:
    def test_creates_lessons_file(self, hc, mem_dir):
        hc.encode_lesson("CoinGecko limits at 50/min", topic="api-coingecko")
        assert (mem_dir / "lessons.md").exists()
        content = (mem_dir / "lessons.md").read_text()
        assert "CoinGecko limits at 50/min" in content

    def test_creates_topic_file(self, hc, mem_dir):
        hc.encode_lesson("CoinGecko limits at 50/min", topic="api-coingecko")
        topic_path = mem_dir / "topics" / "api-coingecko.md"
        assert topic_path.exists()
        assert "CoinGecko limits at 50/min" in topic_path.read_text()

    def test_skips_duplicate(self, hc, mem_dir):
        hc.encode_lesson("Fact one")
        hc.encode_lesson("Fact one")
        content = (mem_dir / "lessons.md").read_text()
        assert content.count("Fact one") == 1

    def test_no_topic_no_topic_file(self, hc, mem_dir):
        hc.encode_lesson("Simple fact")
        assert not (mem_dir / "topics").exists() or not any((mem_dir / "topics").iterdir())

    def test_allows_superstring_of_existing_lesson(self, hc, mem_dir):
        """A more detailed lesson should NOT be blocked by a shorter one."""
        hc.encode_lesson("CoinGecko limits at 50/min")
        hc.encode_lesson("CoinGecko limits at 50/min for free tier accounts")
        content = (mem_dir / "lessons.md").read_text()
        assert "for free tier accounts" in content

    def test_skips_exact_duplicate_with_metadata(self, hc, mem_dir):
        """Exact same text should be blocked even when metadata differs."""
        hc.encode_lesson("Fact one", topic="api")
        hc.encode_lesson("Fact one", topic="other")
        content = (mem_dir / "lessons.md").read_text()
        assert content.count("Fact one") == 1


class TestRewriteIdentity:
    def test_creates_profile(self, hc, mem_dir):
        hc.rewrite_identity(["Name: Jorge", "TZ: PST"])
        profile = (mem_dir / "profile.md").read_text()
        assert "Name: Jorge" in profile
        assert "TZ: PST" in profile

    def test_overwrites_existing(self, hc, mem_dir):
        hc.rewrite_identity(["Name: Old"])
        hc.rewrite_identity(["Name: New"])
        profile = (mem_dir / "profile.md").read_text()
        assert "Name: New" in profile
        assert "Name: Old" not in profile


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


