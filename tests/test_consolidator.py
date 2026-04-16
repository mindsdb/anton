from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from anton.core.memory.consolidator import (
    Consolidator,
    _ConsolidatedLesson,
    _ConsolidatedLessons,
)


@dataclass
class MockCell:
    """Minimal Cell stand-in for consolidator tests."""
    code: str = ""
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    description: str = ""
    estimated_time: str = ""
    logs: str = ""


@pytest.fixture()
def consolidator():
    return Consolidator()


class TestShouldReplay:
    def test_too_short(self, consolidator):
        cells = [MockCell()]
        assert consolidator.should_replay(cells) is False

    def test_long_session(self, consolidator):
        cells = [MockCell(description=f"Cell {i}") for i in range(5)]
        assert consolidator.should_replay(cells) is True

    def test_error_triggers(self, consolidator):
        cells = [MockCell(), MockCell(error="NameError: x is not defined")]
        assert consolidator.should_replay(cells) is True

    def test_cancelled_triggers(self, consolidator):
        cells = [MockCell(), MockCell(stderr="Process was cancelled")]
        assert consolidator.should_replay(cells) is True

    def test_clean_short_session(self, consolidator):
        cells = [MockCell(stdout="ok"), MockCell(stdout="done")]
        assert consolidator.should_replay(cells) is False


class TestReplayAndExtract:
    async def test_extracts_lessons(self, consolidator):
        cells = [
            MockCell(description="Fetch data", stdout="Got 100 rows"),
            MockCell(description="Parse JSON", error="JSONDecodeError: ..."),
            MockCell(description="Retry with fix", stdout="Success"),
        ]

        mock_llm = AsyncMock()
        mock_llm.generate_object_code = AsyncMock(
            return_value=_ConsolidatedLessons(
                items=[
                    _ConsolidatedLesson(
                        text="Always validate JSON before parsing",
                        kind="always",
                        scope="global",
                        confidence="high",
                    )
                ]
            )
        )

        engrams = await consolidator.replay_and_extract(cells, mock_llm)
        assert len(engrams) == 1
        assert engrams[0].text == "Always validate JSON before parsing"
        assert engrams[0].kind == "always"
        assert engrams[0].source == "consolidation"

    async def test_handles_empty_response(self, consolidator):
        cells = [MockCell(), MockCell()]
        mock_llm = AsyncMock()
        mock_llm.generate_object_code = AsyncMock(
            return_value=_ConsolidatedLessons(items=[])
        )

        engrams = await consolidator.replay_and_extract(cells, mock_llm)
        assert engrams == []

    async def test_handles_llm_failure(self, consolidator):
        cells = [MockCell(), MockCell()]
        mock_llm = AsyncMock()
        mock_llm.generate_object_code = AsyncMock(side_effect=Exception("API error"))

        engrams = await consolidator.replay_and_extract(cells, mock_llm)
        assert engrams == []

    async def test_skips_blank_text_entries(self, consolidator):
        """Defensive: even with forced schema, blank text should be skipped."""
        cells = [MockCell(), MockCell()]
        mock_llm = AsyncMock()
        mock_llm.generate_object_code = AsyncMock(
            return_value=_ConsolidatedLessons(
                items=[
                    _ConsolidatedLesson(text="valid", kind="lesson", scope="global"),
                    _ConsolidatedLesson(text="   ", kind="lesson", scope="project"),
                ]
            )
        )

        engrams = await consolidator.replay_and_extract(cells, mock_llm)
        assert len(engrams) == 1
        assert engrams[0].text == "valid"

    async def test_caps_at_five_lessons(self, consolidator):
        cells = [MockCell(), MockCell()]
        mock_llm = AsyncMock()
        mock_llm.generate_object_code = AsyncMock(
            return_value=_ConsolidatedLessons(
                items=[
                    _ConsolidatedLesson(text=f"lesson {i}")
                    for i in range(10)
                ]
            )
        )

        engrams = await consolidator.replay_and_extract(cells, mock_llm)
        assert len(engrams) == 5
