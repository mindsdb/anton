from __future__ import annotations

from pathlib import Path

import pytest

from anton.skill.base import SkillInfo
from anton.skill.loader import load_skill_module

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


class TestLoadSkillModule:
    def test_load_read_file_skill(self):
        path = SKILLS_DIR / "read_file" / "skill.py"
        skills = load_skill_module(path)
        assert len(skills) >= 1
        info = skills[0]
        assert isinstance(info, SkillInfo)
        assert info.name == "read_file"
        assert info.source_path == path

    def test_load_write_file_skill(self):
        path = SKILLS_DIR / "write_file" / "skill.py"
        skills = load_skill_module(path)
        assert len(skills) >= 1
        assert skills[0].name == "write_file"

    def test_load_nonexistent_path(self):
        path = Path("/does/not/exist/skill.py")
        # spec_from_file_location returns a spec even for missing files,
        # but exec_module raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            load_skill_module(path)

    def test_loaded_skill_has_parameters(self):
        path = SKILLS_DIR / "run_command" / "skill.py"
        skills = load_skill_module(path)
        info = skills[0]
        assert "command" in info.parameters["properties"]
        assert "command" in info.parameters["required"]

    def test_loaded_skill_is_callable(self):
        path = SKILLS_DIR / "list_files" / "skill.py"
        skills = load_skill_module(path)
        info = skills[0]
        assert callable(info.execute)
