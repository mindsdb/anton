from __future__ import annotations

import time
from dataclasses import dataclass, field

from anton.core.planner import Plan
from anton.events.bus import EventBus
from anton.events.types import Phase, StatusUpdate
from anton.skill.base import SkillResult
from anton.skill.registry import SkillRegistry


@dataclass
class StepResult:
    step_index: int
    skill_name: str
    result: SkillResult
    duration_seconds: float


@dataclass
class ExecutionResult:
    step_results: list[StepResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0


class Executor:
    def __init__(self, registry: SkillRegistry, bus: EventBus) -> None:
        self._registry = registry
        self._bus = bus

    async def execute_plan(self, plan: Plan) -> ExecutionResult:
        results: list[StepResult] = []
        total_start = time.monotonic()

        for i, step in enumerate(plan.steps):
            await self._bus.publish(
                StatusUpdate(
                    phase=Phase.EXECUTING,
                    message=f"Step {i + 1}/{len(plan.steps)}: {step.description}",
                )
            )

            skill = self._registry.get(step.skill_name)
            if skill is None:
                result = SkillResult(
                    output=None,
                    metadata={"error": f"Skill not found: {step.skill_name}"},
                )
                results.append(
                    StepResult(
                        step_index=i,
                        skill_name=step.skill_name,
                        result=result,
                        duration_seconds=0,
                    )
                )
                continue

            step_start = time.monotonic()
            result = await skill.execute(**step.parameters)
            step_duration = time.monotonic() - step_start

            results.append(
                StepResult(
                    step_index=i,
                    skill_name=step.skill_name,
                    result=result,
                    duration_seconds=step_duration,
                )
            )

        total_duration = time.monotonic() - total_start
        return ExecutionResult(step_results=results, total_duration_seconds=total_duration)
