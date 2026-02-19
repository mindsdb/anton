from __future__ import annotations

from pathlib import Path

import pytest

from anton.skill.spec import SkillSpec
from anton.skill.tester import SkillTester


@pytest.fixture()
def tester() -> SkillTester:
    return SkillTester()


@pytest.fixture()
def tmp_skill_dir(tmp_path: Path) -> Path:
    d = tmp_path / "my_skill"
    d.mkdir()
    return d


class TestImportStage:
    async def test_syntax_error(self, tester: SkillTester, tmp_skill_dir: Path):
        skill_file = tmp_skill_dir / "skill.py"
        skill_file.write_text("def broken(\n", encoding="utf-8")

        spec = SkillSpec(name="my_skill", description="test")
        result = await tester.test_skill(skill_file, spec)
        assert not result.passed
        assert result.stage == "import"

    async def test_no_decorator(self, tester: SkillTester, tmp_skill_dir: Path):
        skill_file = tmp_skill_dir / "skill.py"
        skill_file.write_text(
            "async def my_func():\n    return None\n",
            encoding="utf-8",
        )

        spec = SkillSpec(name="my_skill", description="test")
        result = await tester.test_skill(skill_file, spec)
        assert not result.passed
        assert result.stage == "import"
        assert "No @skill-decorated" in result.error


class TestSignatureStage:
    async def test_wrong_name(self, tester: SkillTester, tmp_skill_dir: Path):
        skill_file = tmp_skill_dir / "skill.py"
        skill_file.write_text(
            'from anton.skill.base import SkillResult, skill\n\n'
            '@skill("wrong_name", "desc")\n'
            'async def wrong_name() -> SkillResult:\n'
            '    return SkillResult(output="ok")\n',
            encoding="utf-8",
        )

        spec = SkillSpec(name="my_skill", description="test")
        result = await tester.test_skill(skill_file, spec)
        assert not result.passed
        assert result.stage == "signature"
        assert "my_skill" in result.error

    async def test_missing_parameter(self, tester: SkillTester, tmp_skill_dir: Path):
        skill_file = tmp_skill_dir / "skill.py"
        skill_file.write_text(
            'from anton.skill.base import SkillResult, skill\n\n'
            '@skill("my_skill", "desc")\n'
            'async def my_skill() -> SkillResult:\n'
            '    return SkillResult(output="ok")\n',
            encoding="utf-8",
        )

        spec = SkillSpec(name="my_skill", description="test", parameters={"path": "str"})
        result = await tester.test_skill(skill_file, spec)
        assert not result.passed
        assert result.stage == "signature"
        assert "path" in result.error


class TestExecutionStage:
    async def test_execution_error(self, tester: SkillTester, tmp_skill_dir: Path):
        skill_file = tmp_skill_dir / "skill.py"
        skill_file.write_text(
            'from anton.skill.base import SkillResult, skill\n\n'
            '@skill("my_skill", "desc")\n'
            'async def my_skill(path: str) -> SkillResult:\n'
            '    raise ValueError("boom")\n',
            encoding="utf-8",
        )

        spec = SkillSpec(
            name="my_skill",
            description="test",
            parameters={"path": "str"},
            test_inputs={"path": "test.txt"},
        )
        result = await tester.test_skill(skill_file, spec)
        assert not result.passed
        assert result.stage == "execution"
        assert "boom" in result.error

    async def test_happy_path(self, tester: SkillTester, tmp_skill_dir: Path):
        skill_file = tmp_skill_dir / "skill.py"
        skill_file.write_text(
            'from anton.skill.base import SkillResult, skill\n\n'
            '@skill("my_skill", "desc")\n'
            'async def my_skill(path: str) -> SkillResult:\n'
            '    return SkillResult(output=f"processed {path}")\n',
            encoding="utf-8",
        )

        spec = SkillSpec(
            name="my_skill",
            description="test",
            parameters={"path": "str"},
            test_inputs={"path": "test.txt"},
        )
        result = await tester.test_skill(skill_file, spec)
        assert result.passed

    async def test_happy_path_no_test_inputs(self, tester: SkillTester, tmp_skill_dir: Path):
        skill_file = tmp_skill_dir / "skill.py"
        skill_file.write_text(
            'from anton.skill.base import SkillResult, skill\n\n'
            '@skill("my_skill", "desc")\n'
            'async def my_skill(path: str) -> SkillResult:\n'
            '    return SkillResult(output=f"processed {path}")\n',
            encoding="utf-8",
        )

        spec = SkillSpec(name="my_skill", description="test", parameters={"path": "str"})
        result = await tester.test_skill(skill_file, spec)
        assert result.passed
        assert result.stage == "signature"
