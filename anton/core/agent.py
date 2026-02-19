from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from anton.core.estimator import TimeEstimator
from anton.core.executor import Executor
from anton.core.planner import Planner
from anton.events.bus import EventBus
from anton.events.types import (
    AntonEvent,
    Phase,
    StatusUpdate,
    TaskComplete,
    TaskFailed,
)
from anton.skill.spec import SkillSpec

if TYPE_CHECKING:
    from anton.channel.base import Channel
    from anton.core.planner import Plan, PlanStep
    from anton.llm.client import LLMClient
    from anton.memory.learnings import LearningStore
    from anton.memory.store import SessionStore
    from anton.skill.registry import SkillRegistry


class Agent:
    def __init__(
        self,
        *,
        channel: Channel,
        llm_client: LLMClient,
        registry: SkillRegistry,
        user_skills_dir: Path | None = None,
        memory: SessionStore | None = None,
        learnings: LearningStore | None = None,
    ) -> None:
        self._channel = channel
        self._llm = llm_client
        self._registry = registry
        self._user_skills_dir = user_skills_dir
        self._memory = memory
        self._learnings = learnings
        self._bus = EventBus()
        self._estimator = TimeEstimator()
        self._planner = Planner(llm_client, registry)
        self._executor = Executor(registry, self._bus)

    async def run(self, task: str) -> None:
        # Wire the event bus to the channel
        queue = self._bus.subscribe()
        relay_task = asyncio.create_task(self._relay_events(queue))

        session_id: str | None = None

        try:
            # Start session if memory enabled
            if self._memory is not None:
                session_id = await self._memory.start_session(task)

            # Phase 0: Memory recall
            memory_context = ""
            if self._memory is not None and self._learnings is not None:
                await self._bus.publish(
                    StatusUpdate(phase=Phase.MEMORY_RECALL, message="Loading memory context...")
                )
                from anton.memory.context import MemoryContext

                ctx_builder = MemoryContext(self._memory, self._learnings)
                memory_context = ctx_builder.build(task)

            # Phase 1: Skill discovery
            await self._bus.publish(
                StatusUpdate(phase=Phase.SKILL_DISCOVERY, message="Inspecting available skills...")
            )

            # Phase 2: Planning
            await self._bus.publish(
                StatusUpdate(phase=Phase.PLANNING, message="Analyzing task and creating plan...")
            )
            plan = await self._planner.plan(task, memory_context=memory_context)

            # Log plan to transcript
            if self._memory is not None and session_id is not None:
                await self._memory.append(session_id, {
                    "type": "plan",
                    "reasoning": plan.reasoning,
                    "steps": [s.model_dump() for s in plan.steps],
                })

            # Use estimator ETA if available, else fall back to LLM estimate
            eta = self._get_eta(plan)
            await self._bus.publish(
                StatusUpdate(
                    phase=Phase.PLANNING,
                    message=f"Plan ready — {len(plan.steps)} step(s)",
                    eta_seconds=eta,
                )
            )

            # Phase 2.5: Build missing skills
            needs_build = self._needs_skill_building(plan)
            if needs_build and self._user_skills_dir is not None:
                unknown_steps = [s for s in plan.steps if s.skill_name == "unknown"]
                await self._build_missing_skills(
                    unknown_steps, plan.skills_to_create, session_id=session_id
                )
                # Re-plan with newly available skills
                await self._bus.publish(
                    StatusUpdate(
                        phase=Phase.PLANNING,
                        message="Re-planning with newly built skills...",
                    )
                )
                plan = await self._planner.plan(task, memory_context=memory_context)

            # Phase 3: Execution
            execution_result = await self._executor.execute_plan(plan)

            # Log each step result to transcript
            if self._memory is not None and session_id is not None:
                for sr in execution_result.step_results:
                    await self._memory.append(session_id, {
                        "type": "step",
                        "index": sr.step_index,
                        "skill": sr.skill_name,
                        "output": str(sr.result.output)[:500] if sr.result.output else None,
                        "error": sr.result.metadata.get("error"),
                        "duration": sr.duration_seconds,
                    })

            # Record durations in estimator for future ETAs
            for sr in execution_result.step_results:
                if sr.duration_seconds > 0:
                    self._estimator.record(sr.skill_name, sr.duration_seconds)

            # Build summary
            summary = self._build_summary(plan, execution_result)

            # Complete session and extract learnings
            if self._memory is not None and session_id is not None:
                await self._memory.complete_session(session_id, summary)

            if self._learnings is not None:
                await self._extract_learnings(summary)

            await self._bus.publish(TaskComplete(summary=summary))

        except Exception as exc:
            if self._memory is not None and session_id is not None:
                await self._memory.fail_session(session_id, str(exc))
            await self._bus.publish(TaskFailed(error_summary=str(exc)))

        finally:
            # Stop relay
            relay_task.cancel()
            try:
                await relay_task
            except asyncio.CancelledError:
                pass
            self._bus.unsubscribe(queue)

    def _needs_skill_building(self, plan: Plan) -> bool:
        """Check if the plan requires building any skills."""
        if plan.skills_to_create:
            return True
        return any(s.skill_name == "unknown" for s in plan.steps)

    def _get_eta(self, plan: Plan) -> float | None:
        """Get the best available ETA for a plan."""
        skill_names = [s.skill_name for s in plan.steps]
        estimator_eta = self._estimator.estimate_plan(skill_names)
        if estimator_eta is not None:
            return estimator_eta
        if plan.estimated_time_seconds > 0:
            return plan.estimated_time_seconds
        return None

    async def _build_missing_skills(
        self,
        unknown_steps: list[PlanStep],
        skills_to_create: list[str],
        *,
        session_id: str | None = None,
    ) -> None:
        from anton.skill.builder import SkillBuilder

        builder = SkillBuilder(
            llm_client=self._llm,
            registry=self._registry,
            user_skills_dir=self._user_skills_dir,
            bus=self._bus,
        )

        seen: set[str] = set()

        # Build from explicit skills_to_create descriptions
        for description in skills_to_create:
            name = _slugify(description)
            if name in seen:
                continue
            seen.add(name)
            spec = SkillSpec(name=name, description=description)
            result = await builder.build(spec)

            # Log skill build to transcript
            if self._memory is not None and session_id is not None:
                await self._memory.append(session_id, {
                    "type": "skill_built",
                    "name": name,
                    "success": result is not None,
                })

        # Build from unknown steps (may overlap with skills_to_create)
        for step in unknown_steps:
            spec = self._extract_spec(step)
            if spec.name in seen:
                continue
            seen.add(spec.name)
            result = await builder.build(spec)

            if self._memory is not None and session_id is not None:
                await self._memory.append(session_id, {
                    "type": "skill_built",
                    "name": spec.name,
                    "success": result is not None,
                })

    def _extract_spec(self, step: PlanStep) -> SkillSpec:
        """Convert an unknown PlanStep into a SkillSpec for building."""
        name = _slugify(step.description)
        # Use parameters from the plan step as type hints (default to str)
        parameters = {k: type(v).__name__ for k, v in step.parameters.items()}
        return SkillSpec(
            name=name,
            description=step.description,
            parameters=parameters,
        )

    async def _extract_learnings(self, summary: str) -> None:
        """Call LLM to extract reusable learnings from a completed task."""
        from anton.llm.prompts import LEARNING_EXTRACT_PROMPT

        try:
            response = await self._llm.code(
                system=LEARNING_EXTRACT_PROMPT,
                messages=[{"role": "user", "content": summary}],
                max_tokens=2048,
            )
            text = response.content or ""
            # Parse JSON array from response (handle markdown fences)
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text
                text = text.rsplit("```", 1)[0]
                text = text.strip()

            learnings = json.loads(text) if text else []
            if not isinstance(learnings, list):
                return

            for item in learnings:
                if isinstance(item, dict) and all(k in item for k in ("topic", "content", "summary")):
                    await self._learnings.record(
                        topic=item["topic"],
                        content=item["content"],
                        summary=item["summary"],
                    )
        except Exception:
            # Learning extraction is best-effort — don't fail the task
            pass

    async def _relay_events(self, queue: asyncio.Queue[AntonEvent]) -> None:
        while True:
            event = await queue.get()
            await self._channel.emit(event)

    def _build_summary(self, plan, execution_result) -> str:
        lines: list[str] = [f"Task completed in {execution_result.total_duration_seconds:.1f}s"]
        lines.append(f"Plan: {plan.reasoning}")
        lines.append("")

        for sr in execution_result.step_results:
            error = sr.result.metadata.get("error")
            if error:
                lines.append(f"  [{sr.step_index + 1}] {sr.skill_name}: ERROR — {error}")
            else:
                output_preview = str(sr.result.output or "")
                if len(output_preview) > 200:
                    output_preview = output_preview[:200] + "..."
                lines.append(f"  [{sr.step_index + 1}] {sr.skill_name}: {output_preview}")

        return "\n".join(lines)


def _slugify(text: str) -> str:
    """Convert a description to a snake_case skill name."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    # Truncate to a reasonable length
    parts = text.split("_")[:6]
    return "_".join(parts)
