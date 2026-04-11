"""Tests for the procedural-memory section in the chat system prompt."""

from __future__ import annotations

from pathlib import Path

import pytest

from anton.core.llm.prompt_builder import ChatSystemPromptBuilder
from anton.core.memory.skills import Skill, SkillStore


@pytest.fixture()
def empty_store(tmp_path: Path) -> SkillStore:
    return SkillStore(root=tmp_path / "skills_empty")


@pytest.fixture()
def populated_store(tmp_path: Path) -> SkillStore:
    s = SkillStore(root=tmp_path / "skills_populated")
    for label, when in [
        ("csv_summary", "User asks to explore or summarize a CSV file."),
        ("web_scraping", "User asks to scrape data from a website."),
        ("api_fetcher", "User asks to fetch data from a JSON API."),
    ]:
        s.save(
            Skill(
                label=label,
                name=label.replace("_", " ").title(),
                description="",
                when_to_use=when,
                declarative_md="step 1\nstep 2",
                created_at="2026-04-10T12:00:00+00:00",
                provenance="manual",
            )
        )
    return s


def _build_prompt(builder: ChatSystemPromptBuilder, **overrides) -> str:
    defaults = dict(
        current_datetime="2026-04-10T12:00:00+00:00",
        runtime_context="test runtime",
        proactive_dashboards=False,
        output_dir="/tmp/anton_out",
    )
    defaults.update(overrides)
    return builder.build(**defaults)


class TestProceduralMemorySection:
    def test_no_store_omits_section(self):
        builder = ChatSystemPromptBuilder()
        prompt = _build_prompt(builder, skill_store=None)
        assert "Procedural memory" not in prompt

    def test_empty_store_omits_section(self, empty_store: SkillStore):
        builder = ChatSystemPromptBuilder()
        prompt = _build_prompt(builder, skill_store=empty_store)
        assert "Procedural memory" not in prompt

    def test_populated_store_renders_section(self, populated_store: SkillStore):
        builder = ChatSystemPromptBuilder()
        prompt = _build_prompt(builder, skill_store=populated_store)
        assert "## Procedural memory" in prompt
        # All labels are listed
        assert "`csv_summary`" in prompt
        assert "`web_scraping`" in prompt
        assert "`api_fetcher`" in prompt
        # And their when_to_use
        assert "explore or summarize a CSV" in prompt
        assert "scrape data from a website" in prompt
        assert "fetch data from a JSON API" in prompt

    def test_section_mentions_recall_skill_tool(
        self, populated_store: SkillStore
    ):
        builder = ChatSystemPromptBuilder()
        prompt = _build_prompt(builder, skill_store=populated_store)
        # The section instructs the LLM how to use them
        assert "recall_skill" in prompt

    def test_section_appears_after_other_contexts(
        self, populated_store: SkillStore
    ):
        builder = ChatSystemPromptBuilder()
        prompt = _build_prompt(
            builder,
            memory_context="\n\n## Memory context\nMEMORY HERE",
            datasource_context="\n\n## Datasources\nDS HERE",
            skill_store=populated_store,
        )
        # Procedural memory should appear AFTER datasource_context
        memory_pos = prompt.find("MEMORY HERE")
        ds_pos = prompt.find("DS HERE")
        proc_pos = prompt.find("## Procedural memory")
        assert memory_pos != -1
        assert ds_pos != -1
        assert proc_pos != -1
        assert proc_pos > ds_pos
        assert proc_pos > memory_pos

    def test_section_is_compact(self, populated_store: SkillStore):
        """Sanity check: ~50 tokens per skill or less.

        We don't want the procedural memory section to dominate the
        prompt — it's a navigational index, not the actual procedures.
        """
        builder = ChatSystemPromptBuilder()
        prompt = _build_prompt(builder, skill_store=populated_store)
        section_start = prompt.find("## Procedural memory")
        section = prompt[section_start:]
        # Three skills, header + intro + bullets — keep it under ~600 chars
        assert len(section) < 1000

    def test_handles_skill_without_when_to_use(self, tmp_path: Path):
        s = SkillStore(root=tmp_path / "skills_partial")
        s.save(
            Skill(
                label="bare",
                name="Bare",
                description="",
                when_to_use="",
                declarative_md="x",
                created_at="2026-04-10T12:00:00+00:00",
                provenance="manual",
            )
        )
        builder = ChatSystemPromptBuilder()
        prompt = _build_prompt(builder, skill_store=s)
        assert "`bare`" in prompt
        # No crash, even with no when_to_use

    def test_skip_section_when_store_raises(self, tmp_path: Path, monkeypatch):
        """If the store blows up at read time, the section is omitted gracefully."""
        s = SkillStore(root=tmp_path / "skills_broken")

        def boom(self):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(SkillStore, "list_summaries", boom)
        builder = ChatSystemPromptBuilder()
        prompt = _build_prompt(builder, skill_store=s)
        assert "Procedural memory" not in prompt
