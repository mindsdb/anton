from __future__ import annotations

from unittest.mock import AsyncMock

from anton.core.executor import ExecutionResult, Executor, StepResult
from anton.core.planner import Plan, PlanStep
from anton.events.bus import EventBus
from anton.events.types import Phase, StatusUpdate
from anton.skill.base import SkillInfo, SkillResult
from anton.skill.registry import SkillRegistry


def _make_skill(name: str, output: str = "done") -> SkillInfo:
    execute = AsyncMock(return_value=SkillResult(output=output))
    return SkillInfo(
        name=name,
        description=f"{name} skill",
        parameters={"type": "object", "properties": {}, "required": []},
        execute=execute,
    )


class TestExecutor:
    async def test_happy_path(self):
        registry = SkillRegistry()
        s = _make_skill("read_file", "file content")
        registry.register(s)
        bus = EventBus()

        plan = Plan(
            reasoning="test",
            steps=[
                PlanStep(
                    skill_name="read_file",
                    description="read it",
                    parameters={"path": "test.txt"},
                )
            ],
        )

        executor = Executor(registry, bus)
        result = await executor.execute_plan(plan)

        assert isinstance(result, ExecutionResult)
        assert len(result.step_results) == 1
        assert result.step_results[0].skill_name == "read_file"
        assert result.step_results[0].result.output == "file content"
        assert result.total_duration_seconds >= 0
        s.execute.assert_awaited_once_with(path="test.txt")

    async def test_unknown_skill(self):
        registry = SkillRegistry()
        bus = EventBus()

        plan = Plan(
            reasoning="test",
            steps=[
                PlanStep(
                    skill_name="nonexistent",
                    description="won't work",
                    parameters={},
                )
            ],
        )

        executor = Executor(registry, bus)
        result = await executor.execute_plan(plan)

        assert len(result.step_results) == 1
        sr = result.step_results[0]
        assert sr.result.output is None
        assert "Skill not found" in sr.result.metadata["error"]
        assert sr.duration_seconds == 0

    async def test_multi_step(self):
        registry = SkillRegistry()
        registry.register(_make_skill("read_file", "content"))
        registry.register(_make_skill("write_file", "written"))
        bus = EventBus()

        plan = Plan(
            reasoning="two steps",
            steps=[
                PlanStep(skill_name="read_file", description="read", parameters={}),
                PlanStep(skill_name="write_file", description="write", parameters={}),
            ],
        )

        executor = Executor(registry, bus)
        result = await executor.execute_plan(plan)

        assert len(result.step_results) == 2
        assert result.step_results[0].skill_name == "read_file"
        assert result.step_results[1].skill_name == "write_file"

    async def test_publishes_status_events(self):
        registry = SkillRegistry()
        registry.register(_make_skill("read_file"))
        bus = EventBus()
        queue = bus.subscribe()

        plan = Plan(
            reasoning="test",
            steps=[
                PlanStep(skill_name="read_file", description="reading", parameters={}),
            ],
        )

        executor = Executor(registry, bus)
        await executor.execute_plan(plan)

        event = queue.get_nowait()
        assert isinstance(event, StatusUpdate)
        assert event.phase == Phase.EXECUTING
        assert "Step 1/1" in event.message

    async def test_publishes_eta_from_initial_estimate(self):
        """First step uses the provided eta_seconds."""
        registry = SkillRegistry()
        registry.register(_make_skill("read_file"))
        bus = EventBus()
        queue = bus.subscribe()

        plan = Plan(
            reasoning="test",
            steps=[
                PlanStep(skill_name="read_file", description="reading", parameters={}),
            ],
        )

        executor = Executor(registry, bus)
        await executor.execute_plan(plan, eta_seconds=10.0)

        event = queue.get_nowait()
        assert isinstance(event, StatusUpdate)
        assert event.eta_seconds == 10.0

    async def test_multi_step_computes_eta_from_pace(self):
        """After first step, ETA is computed from observed pace."""
        registry = SkillRegistry()
        registry.register(_make_skill("read_file"))
        registry.register(_make_skill("write_file"))
        bus = EventBus()
        queue = bus.subscribe()

        plan = Plan(
            reasoning="test",
            steps=[
                PlanStep(skill_name="read_file", description="read", parameters={}),
                PlanStep(skill_name="write_file", description="write", parameters={}),
            ],
        )

        executor = Executor(registry, bus)
        await executor.execute_plan(plan)

        # First step: no ETA (no history, no initial estimate)
        event1 = queue.get_nowait()
        assert event1.eta_seconds is None

        # Second step: ETA computed from elapsed time
        event2 = queue.get_nowait()
        assert isinstance(event2, StatusUpdate)
        assert event2.eta_seconds is not None
        assert event2.eta_seconds >= 0
