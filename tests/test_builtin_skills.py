"""Tests for built-in skills (ENG-648 Phase 2).

The big how-to-build tutorials moved out of the always-on system prompt
into built-in skills loaded via recall_skill (or thalamus preload). These
tests pin the three load-bearing properties: the base prompt no longer
carries the tutorials but does list the built-ins, recall resolves
built-ins (shadowing the store), and the thalamus preload path injects
built-in content.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from anton.core.llm.builtin_skills import (
    BUILTIN_SKILLS,
    builtin_skill_summaries,
    get_builtin_skill,
)
from anton.core.llm.prompt_builder import ChatSystemPromptBuilder, SystemPromptContext
from anton.core.llm.provider import LLMResponse, ToolCall, Usage
from anton.core.session import ChatSession, ChatSessionConfig
from anton.core.tools.recall_skill import handle_recall_skill
from tests.conftest import make_mock_llm


def _build_prompt(*, proactive_dashboards: bool = True, skill_store=None) -> str:
    return ChatSystemPromptBuilder().build(
        conversation_started="Monday, July 06, 2026",
        current_datetime="Monday, July 06, 2026 at 01:00 PM",
        system_prompt_context=SystemPromptContext(runtime_context="test"),
        proactive_dashboards=proactive_dashboards,
        output_dir=".anton/output",
        skill_store=skill_store,
    )


class TestBuiltinRegistry:
    def test_expected_builtins_exist(self):
        assert set(BUILTIN_SKILLS) == {
            "html-dashboards",
            "backend-apps",
            "public-data-sources",
        }

    def test_render_collapses_brace_escapes(self):
        for label, skill in BUILTIN_SKILLS.items():
            rendered = skill.render(output_dir="/tmp/out")
            assert "{{" not in rendered, label
            assert "}}" not in rendered, label

    def test_lookup_strips_and_misses(self):
        assert get_builtin_skill(" html-dashboards ") is BUILTIN_SKILLS["html-dashboards"]
        assert get_builtin_skill("nope") is None
        assert get_builtin_skill("") is None

    def test_summaries_shape(self):
        summaries = builtin_skill_summaries()
        assert {s["label"] for s in summaries} == set(BUILTIN_SKILLS)
        assert all(s["description"] for s in summaries)


class TestSystemPromptSlimming:
    def test_tutorials_not_inlined(self):
        for dash in (True, False):
            prompt = _build_prompt(proactive_dashboards=dash)
            assert "BACKEND & FULLSTACK APPLICATION GENERATION" not in prompt
            assert "REROUND DISCIPLINE" not in prompt
            assert "news.google.com/rss/search" not in prompt

    def test_builtins_listed_even_without_store(self):
        prompt = _build_prompt(skill_store=None)
        assert "## Procedural memory (skills available)" in prompt
        for label in BUILTIN_SKILLS:
            assert f"`{label}`" in prompt

    def test_html_mode_points_at_skill(self):
        prompt = _build_prompt(proactive_dashboards=True)
        assert 'recall_skill("html-dashboards")' in prompt

    def test_prompt_is_halved(self):
        # The whole point of Phase 2: keep the always-on prompt lean.
        # Pre-change this was ~45.5K chars; fail loudly if it creeps back.
        prompt = _build_prompt(proactive_dashboards=True)
        assert len(prompt) < 25_000, f"system prompt regrew to {len(prompt)} chars"


class TestRecallBuiltin:
    async def test_recall_returns_rendered_builtin(self):
        session = ChatSession(ChatSessionConfig(llm_client=make_mock_llm()))
        result = await handle_recall_skill(session, {"label": "backend-apps"})
        assert result.startswith("# Skill: backend-apps (built-in)")
        assert "BACKEND & FULLSTACK APPLICATION GENERATION" in result
        assert "{{" not in result

    async def test_store_skills_still_resolve(self):
        session = ChatSession(ChatSessionConfig(llm_client=make_mock_llm()))
        # A label that is neither builtin nor in the store falls through
        # to the store's no-match handling, not the builtin path.
        result = await handle_recall_skill(session, {"label": "definitely-not-a-skill"})
        assert "NO MATCH" in result or "closest" in result.lower()


class TestThalamusBuiltinPreload:
    async def test_delegation_preloads_builtin(self):
        llm = make_mock_llm()
        llm.gate = AsyncMock(
            return_value=LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc_1",
                        name="delegate",
                        input={"reason": "build", "skills": ["html-dashboards"]},
                    )
                ],
                usage=Usage(input_tokens=5, output_tokens=5),
                stop_reason="tool_use",
            )
        )
        llm.plan = AsyncMock(
            return_value=LLMResponse(
                content="Building it.",
                usage=Usage(input_tokens=5, output_tokens=5),
                stop_reason="end_turn",
            )
        )
        session = ChatSession(ChatSessionConfig(llm_client=llm, router_enabled=True))
        reply = await session.turn("build me a sales dashboard")
        assert reply == "Building it."
        tool_use = session.history[1]["content"][0]
        assert tool_use["input"] == {"label": "html-dashboards"}
        assert "REROUND DISCIPLINE" in session.history[2]["content"][0]["content"]

    async def test_thalamus_prompt_lists_builtins(self):
        llm = make_mock_llm()
        llm.gate = AsyncMock(
            return_value=LLMResponse(
                content="hi", usage=Usage(), stop_reason="end_turn"
            )
        )
        session = ChatSession(ChatSessionConfig(llm_client=llm, router_enabled=True))
        await session.turn("hello")
        system = llm.gate.call_args.kwargs["system"]
        for label in BUILTIN_SKILLS:
            assert f"`{label}`" in system
