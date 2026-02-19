from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from anton.skill.base import SkillInfo


def load_skill_module(path: Path) -> list[SkillInfo]:
    module_name = f"anton_skill_{path.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return []

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    skills: list[SkillInfo] = []
    for attr_name in dir(module):
        obj = getattr(module, attr_name)
        info = getattr(obj, "_skill_info", None)
        if isinstance(info, SkillInfo):
            info.source_path = path
            skills.append(info)

    return skills
