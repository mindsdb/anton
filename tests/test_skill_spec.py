from __future__ import annotations

from anton.skill.spec import SkillSpec


class TestSkillSpec:
    def test_minimal_creation(self):
        spec = SkillSpec(name="count_lines", description="Count lines in a file")
        assert spec.name == "count_lines"
        assert spec.description == "Count lines in a file"

    def test_full_creation(self):
        spec = SkillSpec(
            name="count_lines",
            description="Count lines in a file",
            parameters={"path": "str"},
            expected_output="integer line count",
            test_inputs={"path": "pyproject.toml"},
        )
        assert spec.name == "count_lines"
        assert spec.parameters == {"path": "str"}
        assert spec.expected_output == "integer line count"
        assert spec.test_inputs == {"path": "pyproject.toml"}

    def test_default_values(self):
        spec = SkillSpec(name="foo", description="bar")
        assert spec.parameters == {}
        assert spec.expected_output == ""
        assert spec.test_inputs == {}
