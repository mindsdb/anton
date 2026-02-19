from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from anton.channel.base import Channel
from anton.llm.provider import LLMResponse, ToolCall, Usage
from anton.skill.registry import SkillRegistry

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


@pytest.fixture()
def mock_channel() -> AsyncMock:
    ch = AsyncMock(spec=Channel)
    ch.emit = AsyncMock()
    ch.prompt = AsyncMock(return_value="yes")
    ch.close = AsyncMock()
    return ch


@pytest.fixture()
def make_llm_response():
    def _factory(
        content: str = "",
        tool_calls: list[ToolCall] | None = None,
        input_tokens: int = 10,
        output_tokens: int = 20,
        stop_reason: str | None = "end_turn",
    ) -> LLMResponse:
        return LLMResponse(
            content=content,
            tool_calls=tool_calls or [],
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            stop_reason=stop_reason,
        )

    return _factory


@pytest.fixture()
def skill_registry() -> SkillRegistry:
    reg = SkillRegistry()
    reg.discover(str(SKILLS_DIR))
    return reg
