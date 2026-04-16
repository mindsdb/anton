"""Tests for `LLMClient.generate_object` — structured output via forced tool_choice."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from anton.core.llm.client import LLMClient
from anton.core.llm.provider import LLMProvider, LLMResponse, ToolCall, Usage


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas used by the tests
# ─────────────────────────────────────────────────────────────────────────────


class SimpleAnswer(BaseModel):
    answer: str
    confidence: float


class Lesson(BaseModel):
    text: str
    topic: str = "default"


class LessonBatch(BaseModel):
    lessons: list[Lesson]


# ─────────────────────────────────────────────────────────────────────────────
# Fake provider that records calls and returns canned responses
# ─────────────────────────────────────────────────────────────────────────────


class _FakePlanningProvider(LLMProvider):
    """Captures `complete` arguments and returns a pre-built LLMResponse."""

    def __init__(self, response: LLMResponse) -> None:
        self.response = response
        self.complete_mock = AsyncMock(return_value=response)

    async def complete(self, **kwargs):  # type: ignore[override]
        return await self.complete_mock(**kwargs)


def _make_client(provider: _FakePlanningProvider) -> LLMClient:
    """Build an LLMClient where the planning provider is our fake."""
    return LLMClient(
        planning_provider=provider,
        planning_model="test-model",
        coding_provider=provider,  # reuse — we don't exercise the coding path
        coding_model="test-model",
        max_tokens=8192,
    )


def _tool_call_response(tool_name: str, payload: dict) -> LLMResponse:
    """Build an LLMResponse that looks like the LLM made a forced tool call."""
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id="t1", name=tool_name, input=payload)],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="tool_use",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single-model generation
# ─────────────────────────────────────────────────────────────────────────────


class TestSingleModel:
    @pytest.mark.asyncio
    async def test_returns_validated_instance(self):
        provider = _FakePlanningProvider(
            _tool_call_response(
                "SimpleAnswer", {"answer": "42", "confidence": 0.95}
            )
        )
        client = _make_client(provider)

        result = await client.generate_object(
            SimpleAnswer,
            system="be terse",
            messages=[{"role": "user", "content": "what's the answer?"}],
        )

        assert isinstance(result, SimpleAnswer)
        assert result.answer == "42"
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_forces_tool_choice(self):
        provider = _FakePlanningProvider(
            _tool_call_response("SimpleAnswer", {"answer": "x", "confidence": 0.5})
        )
        client = _make_client(provider)

        await client.generate_object(
            SimpleAnswer,
            system="x",
            messages=[{"role": "user", "content": "y"}],
        )

        # Provider was called with tool_choice forcing the SimpleAnswer tool
        provider.complete_mock.assert_awaited_once()
        kwargs = provider.complete_mock.call_args.kwargs
        assert kwargs["tool_choice"] == {"type": "tool", "name": "SimpleAnswer"}
        # And the tool's schema came from the Pydantic model
        assert len(kwargs["tools"]) == 1
        tool = kwargs["tools"][0]
        assert tool["name"] == "SimpleAnswer"
        assert "input_schema" in tool
        # The schema mentions both fields
        schema_str = str(tool["input_schema"])
        assert "answer" in schema_str
        assert "confidence" in schema_str

    @pytest.mark.asyncio
    async def test_passes_system_and_messages_through(self):
        provider = _FakePlanningProvider(
            _tool_call_response("SimpleAnswer", {"answer": "x", "confidence": 0.5})
        )
        client = _make_client(provider)

        await client.generate_object(
            SimpleAnswer,
            system="custom system",
            messages=[{"role": "user", "content": "custom user"}],
        )

        kwargs = provider.complete_mock.call_args.kwargs
        assert kwargs["system"] == "custom system"
        assert kwargs["messages"] == [{"role": "user", "content": "custom user"}]
        assert kwargs["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_invalid_payload_raises_validation_error(self):
        # Provider returns a payload missing the required `confidence` field
        provider = _FakePlanningProvider(
            _tool_call_response("SimpleAnswer", {"answer": "x"})
        )
        client = _make_client(provider)

        with pytest.raises(Exception) as exc_info:
            await client.generate_object(
                SimpleAnswer,
                system="x",
                messages=[{"role": "user", "content": "y"}],
            )
        # Pydantic raises ValidationError, which we want surfaced
        assert "confidence" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_no_tool_call_raises_value_error(self):
        # Provider returns text-only response — no tool call at all
        provider = _FakePlanningProvider(
            LLMResponse(content="just text", tool_calls=[])
        )
        client = _make_client(provider)

        with pytest.raises(ValueError, match="did not return a tool call"):
            await client.generate_object(
                SimpleAnswer,
                system="x",
                messages=[{"role": "user", "content": "y"}],
            )

    @pytest.mark.asyncio
    async def test_max_tokens_uses_default_when_not_specified(self):
        provider = _FakePlanningProvider(
            _tool_call_response("SimpleAnswer", {"answer": "x", "confidence": 0.5})
        )
        client = _make_client(provider)

        await client.generate_object(
            SimpleAnswer,
            system="x",
            messages=[{"role": "user", "content": "y"}],
        )

        kwargs = provider.complete_mock.call_args.kwargs
        assert kwargs["max_tokens"] == 8192  # the client default

    @pytest.mark.asyncio
    async def test_max_tokens_override(self):
        provider = _FakePlanningProvider(
            _tool_call_response("SimpleAnswer", {"answer": "x", "confidence": 0.5})
        )
        client = _make_client(provider)

        await client.generate_object(
            SimpleAnswer,
            system="x",
            messages=[{"role": "user", "content": "y"}],
            max_tokens=512,
        )

        kwargs = provider.complete_mock.call_args.kwargs
        assert kwargs["max_tokens"] == 512


# ─────────────────────────────────────────────────────────────────────────────
# list[Model] generation
# ─────────────────────────────────────────────────────────────────────────────


class TestListModel:
    @pytest.mark.asyncio
    async def test_returns_typed_list(self):
        provider = _FakePlanningProvider(
            _tool_call_response(
                "Lesson_array",
                {
                    "items": [
                        {"text": "first lesson", "topic": "scratchpad"},
                        {"text": "second lesson", "topic": "scratchpad"},
                    ]
                },
            )
        )
        client = _make_client(provider)

        result = await client.generate_object(
            list[Lesson],
            system="x",
            messages=[{"role": "user", "content": "y"}],
        )

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(item, Lesson) for item in result)
        assert result[0].text == "first lesson"
        assert result[1].text == "second lesson"

    @pytest.mark.asyncio
    async def test_list_uses_array_tool_name(self):
        provider = _FakePlanningProvider(
            _tool_call_response("Lesson_array", {"items": []})
        )
        client = _make_client(provider)

        await client.generate_object(
            list[Lesson],
            system="x",
            messages=[{"role": "user", "content": "y"}],
        )

        kwargs = provider.complete_mock.call_args.kwargs
        assert kwargs["tool_choice"] == {"type": "tool", "name": "Lesson_array"}
        assert kwargs["tools"][0]["name"] == "Lesson_array"

    @pytest.mark.asyncio
    async def test_empty_list_is_valid(self):
        provider = _FakePlanningProvider(
            _tool_call_response("Lesson_array", {"items": []})
        )
        client = _make_client(provider)

        result = await client.generate_object(
            list[Lesson],
            system="x",
            messages=[{"role": "user", "content": "y"}],
        )

        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# Nested model (BaseModel containing list[BaseModel]) — the cerebellum case
# ─────────────────────────────────────────────────────────────────────────────


class TestNestedModel:
    @pytest.mark.asyncio
    async def test_lesson_batch_round_trip(self):
        """The shape the cerebellum will use: a wrapper model with a list."""
        provider = _FakePlanningProvider(
            _tool_call_response(
                "LessonBatch",
                {
                    "lessons": [
                        {
                            "text": "Use low_memory=False with mixed dtypes.",
                            "topic": "scratchpad",
                        }
                    ]
                },
            )
        )
        client = _make_client(provider)

        result = await client.generate_object(
            LessonBatch,
            system="extract lessons",
            messages=[{"role": "user", "content": "cell errored"}],
        )

        assert isinstance(result, LessonBatch)
        assert len(result.lessons) == 1
        assert isinstance(result.lessons[0], Lesson)
        assert "low_memory" in result.lessons[0].text

    @pytest.mark.asyncio
    async def test_empty_lessons_list_is_valid(self):
        provider = _FakePlanningProvider(
            _tool_call_response("LessonBatch", {"lessons": []})
        )
        client = _make_client(provider)

        result = await client.generate_object(
            LessonBatch,
            system="x",
            messages=[{"role": "user", "content": "y"}],
        )

        assert result.lessons == []
