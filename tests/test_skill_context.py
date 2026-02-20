from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from anton.llm.provider import LLMResponse, ToolCall, Usage
from anton.skill.context import SkillLLM, get_llm, set_skill_llm


class TestSkillLLM:
    async def test_complete_delegates_to_provider(self):
        provider = AsyncMock()
        provider.complete = AsyncMock(
            return_value=LLMResponse(
                content="hello",
                tool_calls=[],
                usage=Usage(input_tokens=5, output_tokens=3),
                stop_reason="end_turn",
            )
        )
        llm = SkillLLM(provider, "test-model")

        response = await llm.complete(
            system="sys", messages=[{"role": "user", "content": "hi"}]
        )

        assert response.content == "hello"
        provider.complete.assert_called_once_with(
            model="test-model",
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            max_tokens=4096,
        )

    async def test_model_property(self):
        provider = AsyncMock()
        llm = SkillLLM(provider, "claude-opus-4-6")
        assert llm.model == "claude-opus-4-6"

    async def test_complete_passes_tools(self):
        provider = AsyncMock()
        provider.complete = AsyncMock(
            return_value=LLMResponse(content="ok", stop_reason="end_turn")
        )
        llm = SkillLLM(provider, "m")

        tools = [{"name": "t", "description": "d", "input_schema": {"type": "object"}}]
        await llm.complete(system="s", messages=[], tools=tools, max_tokens=1024)

        provider.complete.assert_called_once_with(
            model="m", system="s", messages=[], tools=tools, max_tokens=1024
        )


class TestGetLLM:
    def test_raises_without_context(self):
        # Reset context to ensure clean state
        from anton.skill.context import _current_llm
        _current_llm.set(None)

        with pytest.raises(RuntimeError, match="No LLM available"):
            get_llm()

    def test_returns_llm_after_set(self):
        provider = AsyncMock()
        set_skill_llm(provider, "test-model")

        llm = get_llm()
        assert llm.model == "test-model"

        # Clean up
        from anton.skill.context import _current_llm
        _current_llm.set(None)
