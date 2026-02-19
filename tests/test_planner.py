from __future__ import annotations

from unittest.mock import AsyncMock

from anton.core.planner import Plan, PlanStep, Planner
from anton.llm.provider import LLMResponse, ToolCall, Usage
from anton.skill.registry import SkillRegistry


def _make_plan_response(skills_to_create: list[str] | None = None) -> LLMResponse:
    tool_input: dict = {
        "reasoning": "Need to read and then write",
        "steps": [
            {
                "skill_name": "read_file",
                "description": "Read input",
                "parameters": {"path": "in.txt"},
            },
            {
                "skill_name": "write_file",
                "description": "Write output",
                "parameters": {"path": "out.txt", "content": "done"},
                "depends_on": [0],
            },
        ],
        "estimated_time_seconds": 3.0,
    }
    if skills_to_create is not None:
        tool_input["skills_to_create"] = skills_to_create
    return LLMResponse(
        content="",
        tool_calls=[
            ToolCall(
                id="tc_1",
                name="create_plan",
                input=tool_input,
            )
        ],
        usage=Usage(input_tokens=100, output_tokens=200),
        stop_reason="tool_use",
    )


def _make_text_response() -> LLMResponse:
    return LLMResponse(
        content="I'll just do it directly",
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="end_turn",
    )


class TestPlanner:
    async def test_plan_from_tool_call(self, skill_registry):
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_make_plan_response())

        planner = Planner(mock_llm, skill_registry)
        plan = await planner.plan("read in.txt and write out.txt")

        assert isinstance(plan, Plan)
        assert plan.reasoning == "Need to read and then write"
        assert len(plan.steps) == 2
        assert plan.steps[0].skill_name == "read_file"
        assert plan.steps[1].skill_name == "write_file"
        assert plan.steps[1].depends_on == [0]
        assert plan.estimated_time_seconds == 3.0

    async def test_plan_text_fallback(self, skill_registry):
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_make_text_response())

        planner = Planner(mock_llm, skill_registry)
        plan = await planner.plan("do something")

        assert isinstance(plan, Plan)
        assert plan.reasoning == "I'll just do it directly"
        assert len(plan.steps) == 1
        assert plan.steps[0].skill_name == "run_command"

    async def test_plan_with_skills_to_create(self, skill_registry):
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            return_value=_make_plan_response(skills_to_create=["count lines in a file"])
        )

        planner = Planner(mock_llm, skill_registry)
        plan = await planner.plan("count lines and write output")

        assert plan.skills_to_create == ["count lines in a file"]

    async def test_plan_without_skills_to_create_defaults_empty(self, skill_registry):
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_make_plan_response())

        planner = Planner(mock_llm, skill_registry)
        plan = await planner.plan("read and write")

        assert plan.skills_to_create == []

    async def test_plan_passes_catalog_in_system(self, skill_registry):
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_make_plan_response())

        planner = Planner(mock_llm, skill_registry)
        await planner.plan("test task")

        call_kwargs = mock_llm.plan.call_args.kwargs
        assert "Available skills:" in call_kwargs["system"]
        assert "read_file" in call_kwargs["system"]

    async def test_memory_context_injected_into_system(self, skill_registry):
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_make_plan_response())

        planner = Planner(mock_llm, skill_registry)
        await planner.plan("test task", memory_context="## Recent Activity\nDid stuff")

        call_kwargs = mock_llm.plan.call_args.kwargs
        assert "## Recent Activity" in call_kwargs["system"]
        assert "Did stuff" in call_kwargs["system"]
        # Memory context should appear before the skills catalog
        system = call_kwargs["system"]
        assert system.index("Recent Activity") < system.index("Available skills:")

    async def test_empty_memory_context_doesnt_affect_prompt(self, skill_registry):
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_make_plan_response())

        planner = Planner(mock_llm, skill_registry)
        await planner.plan("test task", memory_context="")

        call_kwargs = mock_llm.plan.call_args.kwargs
        # Empty memory context should not add an extra section
        assert "## Recent Activity" not in call_kwargs["system"]
        assert "## Relevant Learnings" not in call_kwargs["system"]
