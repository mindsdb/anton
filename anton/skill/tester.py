from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from anton.skill.base import SkillInfo
from anton.skill.loader import load_skill_module
from anton.skill.spec import SkillSpec


@dataclass
class TestResult:
    passed: bool
    stage: str  # "import", "signature", or "execution"
    error: str = ""


class SkillTester:
    """Three-stage validation of a generated skill file."""

    async def test_skill(self, path: Path, spec: SkillSpec) -> TestResult:
        # Stage 1: Import
        try:
            skills = load_skill_module(path)
        except Exception as exc:
            return TestResult(passed=False, stage="import", error=str(exc))

        if not skills:
            return TestResult(
                passed=False,
                stage="import",
                error="No @skill-decorated function found in module",
            )

        # Stage 2: Signature
        skill = self._find_skill(skills, spec.name)
        if skill is None:
            found = [s.name for s in skills]
            return TestResult(
                passed=False,
                stage="signature",
                error=f"Expected skill named '{spec.name}', found: {found}",
            )

        for param_name in spec.parameters:
            props = skill.parameters.get("properties", {})
            if param_name not in props:
                return TestResult(
                    passed=False,
                    stage="signature",
                    error=f"Missing parameter '{param_name}' in skill signature",
                )

        # Stage 3: Execution (only if test_inputs provided)
        if spec.test_inputs:
            try:
                result = await skill.execute(**spec.test_inputs)
            except Exception as exc:
                return TestResult(
                    passed=False, stage="execution", error=str(exc)
                )

            error = result.metadata.get("error")
            if error:
                return TestResult(
                    passed=False, stage="execution", error=f"Skill returned error: {error}"
                )

        return TestResult(passed=True, stage="execution" if spec.test_inputs else "signature")

    def _find_skill(self, skills: list[SkillInfo], name: str) -> SkillInfo | None:
        for s in skills:
            if s.name == name:
                return s
        return None
