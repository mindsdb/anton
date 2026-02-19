from __future__ import annotations

from pathlib import Path

from anton.skill.base import SkillInfo
from anton.skill.loader import load_skill_module


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, SkillInfo] = {}

    def discover(self, skills_dir: str | Path) -> None:
        skills_path = Path(skills_dir)
        if not skills_path.is_dir():
            return

        for skill_file in sorted(skills_path.glob("*/skill.py")):
            for info in load_skill_module(skill_file):
                self.register(info)

    def register(self, skill: SkillInfo) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> SkillInfo | None:
        return self._skills.get(name)

    def list_all(self) -> list[SkillInfo]:
        return list(self._skills.values())

    def catalog(self) -> str:
        if not self._skills:
            return "No skills available."

        lines: list[str] = []
        for info in self._skills.values():
            params_desc = ", ".join(
                f"{k}: {v.get('type', '?')}"
                for k, v in info.parameters.get("properties", {}).items()
            )
            lines.append(f"- {info.name}: {info.description} ({params_desc})")
        return "\n".join(lines)
