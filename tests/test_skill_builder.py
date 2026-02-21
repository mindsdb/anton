from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from anton.events.bus import EventBus
from anton.events.types import Phase, StatusUpdate
from anton.llm.provider import LLMResponse, Usage
from anton.skill.builder import SkillBuilder, _extract_code
from anton.skill.registry import SkillRegistry
from anton.skill.spec import SkillSpec

VALID_SKILL_CODE = '''\
```python
from anton.skill.base import SkillResult, skill


@skill("count_lines", "Count lines in a file")
async def count_lines(path: str) -> SkillResult:
    with open(path) as f:
        lines = f.readlines()
    return SkillResult(output=len(lines), metadata={"path": path})
```
'''

BROKEN_SKILL_CODE = '''\
```python
from anton.skill.base import SkillResult, skill

@skill("count_lines", "Count lines in a file")
async def count_lines(path: str) -> SkillResult:
    # missing return
    x = 1 / 0
```
'''


def _make_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="end_turn",
    )


@pytest.fixture()
def spec() -> SkillSpec:
    return SkillSpec(
        name="count_lines",
        description="Count lines in a file",
        parameters={"path": "str"},
    )


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


class TestBuildSuccess:
    async def test_build_succeeds_first_attempt(self, tmp_path: Path, spec: SkillSpec, bus: EventBus):
        mock_llm = AsyncMock()
        mock_llm.code = AsyncMock(return_value=_make_response(VALID_SKILL_CODE))
        registry = SkillRegistry()

        builder = SkillBuilder(
            llm_client=mock_llm,
            registry=registry,
            user_skills_dir=tmp_path,
            bus=bus,
        )
        result = await builder.build(spec)

        assert result is not None
        assert result.name == "count_lines"
        assert registry.get("count_lines") is not None
        mock_llm.code.assert_awaited_once()

    async def test_build_succeeds_on_retry(self, tmp_path: Path, spec: SkillSpec, bus: EventBus):
        mock_llm = AsyncMock()
        mock_llm.code = AsyncMock(
            side_effect=[
                _make_response(BROKEN_SKILL_CODE),
                _make_response(VALID_SKILL_CODE),
            ]
        )
        registry = SkillRegistry()

        # Need test_inputs to trigger execution failure on broken code
        spec_with_test = SkillSpec(
            name="count_lines",
            description="Count lines in a file",
            parameters={"path": "str"},
            test_inputs={"path": __file__},  # use this test file as input
        )

        builder = SkillBuilder(
            llm_client=mock_llm,
            registry=registry,
            user_skills_dir=tmp_path,
            bus=bus,
        )
        result = await builder.build(spec_with_test)

        assert result is not None
        assert result.name == "count_lines"
        assert mock_llm.code.await_count == 2


class TestBuildFailure:
    async def test_build_fails_all_attempts(self, tmp_path: Path, spec: SkillSpec, bus: EventBus):
        mock_llm = AsyncMock()
        # Return code with syntax error every time
        bad_response = _make_response("```python\ndef broken(\n```")
        mock_llm.code = AsyncMock(return_value=bad_response)
        registry = SkillRegistry()

        builder = SkillBuilder(
            llm_client=mock_llm,
            registry=registry,
            user_skills_dir=tmp_path,
            bus=bus,
        )
        result = await builder.build(spec)

        assert result is None
        assert mock_llm.code.await_count == 3
        assert registry.get("count_lines") is None

    async def test_failed_build_does_not_publish_broken_skill_file(
        self, tmp_path: Path, spec: SkillSpec, bus: EventBus
    ):
        mock_llm = AsyncMock()
        bad_response = _make_response("```python\ndef broken(\n```")
        mock_llm.code = AsyncMock(return_value=bad_response)
        registry = SkillRegistry()

        builder = SkillBuilder(
            llm_client=mock_llm,
            registry=registry,
            user_skills_dir=tmp_path,
            bus=bus,
        )
        result = await builder.build(spec)

        assert result is None
        assert (tmp_path / "count_lines" / "skill.py").exists() is False

    async def test_failed_build_preserves_existing_skill_file(
        self, tmp_path: Path, spec: SkillSpec, bus: EventBus
    ):
        skill_dir = tmp_path / "count_lines"
        skill_dir.mkdir(parents=True)
        existing_path = skill_dir / "skill.py"
        existing_content = "# existing skill\n"
        existing_path.write_text(existing_content, encoding="utf-8")

        mock_llm = AsyncMock()
        bad_response = _make_response("```python\ndef broken(\n```")
        mock_llm.code = AsyncMock(return_value=bad_response)
        registry = SkillRegistry()

        builder = SkillBuilder(
            llm_client=mock_llm,
            registry=registry,
            user_skills_dir=tmp_path,
            bus=bus,
        )
        result = await builder.build(spec)

        assert result is None
        assert existing_path.read_text(encoding="utf-8") == existing_content


class TestExtractCode:
    def test_python_fences(self):
        text = "Here is code:\n```python\nprint('hello')\n```\nDone."
        assert _extract_code(text) == "print('hello')\n"

    def test_plain_fences(self):
        text = "```\nprint('hello')\n```"
        assert _extract_code(text) == "print('hello')\n"

    def test_no_fences(self):
        text = "print('hello')"
        assert _extract_code(text) == "print('hello')\n"


class TestStatusEvents:
    async def test_events_published_during_build(self, tmp_path: Path, spec: SkillSpec, bus: EventBus):
        mock_llm = AsyncMock()
        mock_llm.code = AsyncMock(return_value=_make_response(VALID_SKILL_CODE))
        registry = SkillRegistry()
        queue = bus.subscribe()

        builder = SkillBuilder(
            llm_client=mock_llm,
            registry=registry,
            user_skills_dir=tmp_path,
            bus=bus,
        )
        await builder.build(spec)

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())

        status_events = [e for e in events if isinstance(e, StatusUpdate)]
        assert len(status_events) >= 2  # building + success
        assert all(e.phase == Phase.SKILL_BUILDING for e in status_events)
