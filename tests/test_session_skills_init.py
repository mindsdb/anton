"""Light wiring tests for skills integration in ChatSession.

These tests don't construct a full ChatSession (it requires a live LLM
client and many other dependencies). Instead they verify the *contract*
points where session and skills meet:

- The `recall_skill` tool is exported from its module and registers without error
- The prompt builder accepts `skill_store` and renders the section
- The store + tool dispatch round-trip works (this is what session.py wires)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from anton.core.llm.prompt_builder import ChatSystemPromptBuilder, SystemPromptContext
from anton.core.memory.skills import Skill, SkillStore
from anton.core.tools.recall_skill import RECALL_SKILL_TOOL, handle_recall_skill
from anton.core.tools.registry import ToolRegistry


@pytest.fixture()
def store_with_one_skill(tmp_path: Path) -> SkillStore:
    s = SkillStore(root=tmp_path / "skills")
    s.save(
        Skill(
            label="csv_summary",
            name="CSV Summary",
            description="Load a CSV, infer schema, compute stats.",
            when_to_use="User asks to explore or summarize a CSV file.",
            declarative_md="1. Load CSV\n2. Describe\n3. Plot",
            created_at="2026-04-10T12:00:00+00:00",
            provenance="manual",
        )
    )
    return s


class TestRegistryRegistration:
    def test_recall_skill_registers_without_collision(self):
        registry = ToolRegistry()
        registry.register_tool(RECALL_SKILL_TOOL)
        names = [t.name for t in registry.get_tool_defs()]
        assert "recall_skill" in names

    def test_double_registration_is_idempotent(self):
        registry = ToolRegistry()
        registry.register_tool(RECALL_SKILL_TOOL)
        registry.register_tool(RECALL_SKILL_TOOL)
        names = [t.name for t in registry.get_tool_defs()]
        assert names.count("recall_skill") == 1

    def test_recall_skill_appears_in_dump(self):
        registry = ToolRegistry()
        registry.register_tool(RECALL_SKILL_TOOL)
        dumped = registry.dump()
        assert any(t["name"] == "recall_skill" for t in dumped)


class TestPromptBuilderReceivesStore:
    def test_section_appears_when_store_passed(
        self, store_with_one_skill: SkillStore
    ):
        builder = ChatSystemPromptBuilder()
        prompt = builder.build(
            current_datetime="2026-04-10",
            system_prompt_context=SystemPromptContext(runtime_context="test"),
            proactive_dashboards=False,
            skill_store=store_with_one_skill,
        )
        assert "## Procedural memory" in prompt
        assert "csv_summary" in prompt

    def test_section_omitted_when_no_store(self):
        builder = ChatSystemPromptBuilder()
        prompt = builder.build(
            current_datetime="2026-04-10",
            system_prompt_context=SystemPromptContext(runtime_context="test"),
            proactive_dashboards=False,
            skill_store=None,
        )
        assert "Procedural memory" not in prompt


class TestDispatchRoundtrip:
    """The end-to-end path: registry dispatches → handler reads store → counter bumps."""

    @pytest.mark.asyncio
    async def test_dispatch_recall_skill_through_registry(
        self, store_with_one_skill: SkillStore
    ):
        registry = ToolRegistry()
        registry.register_tool(RECALL_SKILL_TOOL)
        # Minimal session-like object — only `_skill_store` is read by the handler.
        session = SimpleNamespace(_skill_store=store_with_one_skill)

        result = await registry.dispatch_tool(
            session, "recall_skill", {"label": "csv_summary"}
        )

        assert "CSV Summary" in result
        assert "Load CSV" in result
        loaded = store_with_one_skill.load("csv_summary")
        assert loaded is not None
        assert loaded.stats.stage_1.recommended == 1

    @pytest.mark.asyncio
    async def test_dispatch_unknown_label_through_registry(
        self, store_with_one_skill: SkillStore
    ):
        registry = ToolRegistry()
        registry.register_tool(RECALL_SKILL_TOOL)
        session = SimpleNamespace(_skill_store=store_with_one_skill)

        result = await registry.dispatch_tool(
            session, "recall_skill", {"label": "nonexistent_xyz"}
        )

        assert "NO MATCH" in result
        # Counter should NOT have moved
        loaded = store_with_one_skill.load("csv_summary")
        assert loaded is not None
        assert loaded.stats.stage_1.recommended == 0
