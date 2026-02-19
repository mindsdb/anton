from __future__ import annotations

from pathlib import Path

from anton.skill.base import SkillInfo, SkillResult
from anton.skill.registry import SkillRegistry

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


async def _dummy_execute(**kwargs) -> SkillResult:
    return SkillResult(output="ok")


def _make_skill(name: str = "test", desc: str = "test skill") -> SkillInfo:
    return SkillInfo(
        name=name,
        description=desc,
        parameters={"type": "object", "properties": {}, "required": []},
        execute=_dummy_execute,
    )


class TestSkillRegistry:
    def test_register_and_get(self):
        reg = SkillRegistry()
        s = _make_skill("my_skill")
        reg.register(s)
        assert reg.get("my_skill") is s

    def test_get_unknown_returns_none(self):
        reg = SkillRegistry()
        assert reg.get("nonexistent") is None

    def test_list_all_empty(self):
        reg = SkillRegistry()
        assert reg.list_all() == []

    def test_list_all_returns_registered(self):
        reg = SkillRegistry()
        reg.register(_make_skill("a"))
        reg.register(_make_skill("b"))
        names = [s.name for s in reg.list_all()]
        assert "a" in names
        assert "b" in names

    def test_register_overwrites(self):
        reg = SkillRegistry()
        reg.register(_make_skill("dup", "first"))
        reg.register(_make_skill("dup", "second"))
        assert reg.get("dup").description == "second"


class TestSkillRegistryDiscover:
    def test_discover_loads_built_in_skills(self):
        reg = SkillRegistry()
        reg.discover(str(SKILLS_DIR))
        names = [s.name for s in reg.list_all()]
        assert "read_file" in names
        assert "write_file" in names
        assert "list_files" in names
        assert "run_command" in names
        assert "search_code" in names

    def test_discover_nonexistent_dir(self):
        reg = SkillRegistry()
        reg.discover("/nonexistent/path")
        assert reg.list_all() == []


class TestSkillRegistryCatalog:
    def test_catalog_empty(self):
        reg = SkillRegistry()
        assert reg.catalog() == "No skills available."

    def test_catalog_contains_skill_info(self):
        reg = SkillRegistry()
        reg.register(
            SkillInfo(
                name="greet",
                description="Say hello",
                parameters={
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
                execute=_dummy_execute,
            )
        )
        cat = reg.catalog()
        assert "greet" in cat
        assert "Say hello" in cat
        assert "name: string" in cat
