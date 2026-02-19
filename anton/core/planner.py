from __future__ import annotations

import json
from dataclasses import dataclass, field

from pydantic import BaseModel

from anton.llm.client import LLMClient
from anton.llm.prompts import PLANNER_PROMPT
from anton.skill.registry import SkillRegistry


class PlanStep(BaseModel):
    skill_name: str
    description: str
    parameters: dict = field(default_factory=dict)
    depends_on: list[int] = field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


class Plan(BaseModel):
    reasoning: str
    steps: list[PlanStep]
    skills_to_create: list[str] = field(default_factory=list)
    estimated_time_seconds: float = 0.0


# Tool definition sent to the LLM for structured plan output
CREATE_PLAN_TOOL = {
    "name": "create_plan",
    "description": "Create a structured execution plan for the given task.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Brief reasoning about how to approach the task",
            },
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": "Name of the skill to use",
                        },
                        "description": {
                            "type": "string",
                            "description": "What this step does",
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Parameters to pass to the skill",
                        },
                        "depends_on": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Indices of steps this depends on (0-based)",
                        },
                    },
                    "required": ["skill_name", "description", "parameters"],
                },
            },
            "skills_to_create": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Descriptions of skills that need to be created (not in catalog)",
            },
            "estimated_time_seconds": {
                "type": "number",
                "description": "Estimated total execution time in seconds",
            },
        },
        "required": ["reasoning", "steps", "estimated_time_seconds"],
    },
}


class Planner:
    def __init__(self, llm_client: LLMClient, registry: SkillRegistry) -> None:
        self._llm = llm_client
        self._registry = registry

    async def plan(self, task: str, memory_context: str = "") -> Plan:
        catalog = self._registry.catalog()
        if memory_context:
            system = f"{PLANNER_PROMPT}\n\n{memory_context}\n\nAvailable skills:\n{catalog}"
        else:
            system = f"{PLANNER_PROMPT}\n\nAvailable skills:\n{catalog}"

        messages = [{"role": "user", "content": task}]

        response = await self._llm.plan(
            system=system,
            messages=messages,
            tools=[CREATE_PLAN_TOOL],
            max_tokens=4096,
        )

        # Extract plan from tool call
        for tc in response.tool_calls:
            if tc.name == "create_plan":
                return Plan(
                    reasoning=tc.input.get("reasoning", ""),
                    steps=[PlanStep(**s) for s in tc.input.get("steps", [])],
                    skills_to_create=tc.input.get("skills_to_create", []),
                    estimated_time_seconds=tc.input.get("estimated_time_seconds", 0),
                )

        # Fallback: if model responded with text instead of tool call, create a
        # single-step plan using the response as guidance
        return Plan(
            reasoning=response.content or "Direct execution",
            steps=[
                PlanStep(
                    skill_name="run_command",
                    description=task,
                    parameters={"command": "echo 'No structured plan produced'"},
                )
            ],
            estimated_time_seconds=5,
        )
