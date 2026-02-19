from __future__ import annotations

from anton.skill.base import SkillInfo, SkillResult, _build_parameters_schema, skill


class TestSkillDecorator:
    def test_attaches_skill_info(self):
        @skill("test_skill", "A test skill")
        async def my_skill(name: str) -> SkillResult:
            return SkillResult(output=name)

        assert hasattr(my_skill, "_skill_info")
        info = my_skill._skill_info
        assert isinstance(info, SkillInfo)
        assert info.name == "test_skill"
        assert info.description == "A test skill"

    def test_decorated_function_still_callable(self):
        @skill("echo", "echoes")
        async def echo(msg: str) -> SkillResult:
            return SkillResult(output=msg)

        # The function should still be the original coroutine function
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(echo("hello"))
        assert result.output == "hello"


class TestBuildParametersSchema:
    def test_string_param(self):
        async def fn(name: str) -> SkillResult:
            return SkillResult()

        schema = _build_parameters_schema(fn)
        assert schema["properties"]["name"]["type"] == "string"
        assert "name" in schema["required"]

    def test_int_param(self):
        async def fn(count: int) -> SkillResult:
            return SkillResult()

        schema = _build_parameters_schema(fn)
        assert schema["properties"]["count"]["type"] == "integer"

    def test_float_param(self):
        async def fn(rate: float) -> SkillResult:
            return SkillResult()

        schema = _build_parameters_schema(fn)
        assert schema["properties"]["rate"]["type"] == "number"

    def test_bool_param(self):
        async def fn(flag: bool) -> SkillResult:
            return SkillResult()

        schema = _build_parameters_schema(fn)
        assert schema["properties"]["flag"]["type"] == "boolean"

    def test_default_value_not_required(self):
        async def fn(name: str, timeout: int = 30) -> SkillResult:
            return SkillResult()

        schema = _build_parameters_schema(fn)
        assert "name" in schema["required"]
        assert "timeout" not in schema["required"]

    def test_multiple_params(self):
        async def fn(path: str, content: str) -> SkillResult:
            return SkillResult()

        schema = _build_parameters_schema(fn)
        assert len(schema["properties"]) == 2
        assert len(schema["required"]) == 2

    def test_no_params(self):
        async def fn() -> SkillResult:
            return SkillResult()

        schema = _build_parameters_schema(fn)
        assert schema["properties"] == {}
        assert schema["required"] == []


class TestSkillResult:
    def test_defaults(self):
        r = SkillResult()
        assert r.output is None
        assert r.metadata == {}

    def test_with_values(self):
        r = SkillResult(output="hello", metadata={"key": "val"})
        assert r.output == "hello"
        assert r.metadata["key"] == "val"
