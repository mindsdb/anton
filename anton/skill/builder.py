from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from anton.events.bus import EventBus
from anton.events.types import Phase, StatusUpdate
from anton.llm.prompts import BUILDER_PROMPT
from anton.skill.base import SkillInfo
from anton.skill.loader import load_skill_module
from anton.skill.spec import SkillSpec
from anton.skill.tester import SkillTester

if TYPE_CHECKING:
    from anton.llm.client import LLMClient
    from anton.skill.registry import SkillRegistry

MAX_ATTEMPTS = 3


class SkillBuilder:
    """Build-test-fix loop for generating new skills via LLM."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        registry: SkillRegistry,
        user_skills_dir: Path,
        bus: EventBus,
    ) -> None:
        self._llm = llm_client
        self._registry = registry
        self._user_skills_dir = user_skills_dir
        self._bus = bus
        self._tester = SkillTester()

    async def build(self, spec: SkillSpec) -> SkillInfo | None:
        """Attempt to build a skill from a spec. Returns SkillInfo on success, None after 3 failures."""
        skill_dir = self._user_skills_dir / spec.name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_path = skill_dir / "skill.py"

        system = BUILDER_PROMPT.format(
            name=spec.name,
            description=spec.description,
            parameters=", ".join(f"{k}: {v}" for k, v in spec.parameters.items()),
        )

        messages: list[dict] = [
            {
                "role": "user",
                "content": self._build_user_prompt(spec),
            }
        ]

        for attempt in range(1, MAX_ATTEMPTS + 1):
            await self._bus.publish(
                StatusUpdate(
                    phase=Phase.SKILL_BUILDING,
                    message=f"Building skill '{spec.name}' (attempt {attempt}/{MAX_ATTEMPTS})...",
                )
            )

            response = await self._llm.code(
                system=system,
                messages=messages,
                max_tokens=4096,
            )

            code = _extract_code(response.content)
            skill_path.write_text(code, encoding="utf-8")

            result = await self._tester.test_skill(skill_path, spec)

            if result.passed:
                await self._bus.publish(
                    StatusUpdate(
                        phase=Phase.SKILL_BUILDING,
                        message=f"Skill '{spec.name}' built successfully",
                    )
                )
                # Load and register
                skills = load_skill_module(skill_path)
                for skill in skills:
                    if skill.name == spec.name:
                        self._registry.register(skill)
                        return skill
                # Fallback: register first skill found
                if skills:
                    self._registry.register(skills[0])
                    return skills[0]

            # Failed — append error context for retry
            messages.append({"role": "assistant", "content": response.content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"The generated skill failed validation at stage '{result.stage}': "
                        f"{result.error}\n\nPlease fix the code and try again."
                    ),
                }
            )

        await self._bus.publish(
            StatusUpdate(
                phase=Phase.SKILL_BUILDING,
                message=f"Failed to build skill '{spec.name}' after {MAX_ATTEMPTS} attempts",
            )
        )
        return None

    def _build_user_prompt(self, spec: SkillSpec) -> str:
        lines = [
            f"Create a skill named '{spec.name}'.",
            f"Description: {spec.description}",
        ]
        if spec.parameters:
            params = ", ".join(f"{k} ({v})" for k, v in spec.parameters.items())
            lines.append(f"Parameters: {params}")
        if spec.expected_output:
            lines.append(f"Expected output: {spec.expected_output}")
        if spec.test_inputs:
            lines.append(f"Test inputs: {spec.test_inputs}")
        return "\n".join(lines)


def _extract_code(text: str) -> str:
    """Extract Python code from LLM response, handling ```python fences."""
    # Try ```python ... ``` first
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip() + "\n"

    # Try generic ``` ... ```
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip() + "\n"

    # No fences — return the whole text
    return text.strip() + "\n"
