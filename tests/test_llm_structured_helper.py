"""Unit tests for `anton.core.llm.structured` — pure helper functions.

These verify the schema-building and unwrapping logic in isolation,
without going through any LLM call. Both `LLMClient.generate_object`
and `_ScratchpadLLM.generate_object` delegate to these functions, so
locking their contract here is the foundation for both call sites.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from anton.core.llm.structured import (
    build_structured_tool,
    unwrap_structured_response,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas used by the tests
# ─────────────────────────────────────────────────────────────────────────────


class Answer(BaseModel):
    text: str
    confidence: float


class Lesson(BaseModel):
    text: str
    topic: str = "default"


class WrappedLessons(BaseModel):
    """A wrapper model with a list field — the cerebellum's exact pattern."""

    lessons: list[Lesson]


# ─────────────────────────────────────────────────────────────────────────────
# build_structured_tool — single model
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildStructuredToolSingle:
    def test_returns_three_tuple(self):
        result = build_structured_tool(Answer)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_tool_has_required_fields(self):
        tool, _, _ = build_structured_tool(Answer)
        assert "name" in tool
        assert "description" in tool
        assert "input_schema" in tool

    def test_tool_name_is_class_name(self):
        tool, _, _ = build_structured_tool(Answer)
        assert tool["name"] == "Answer"

    def test_input_schema_includes_field_names(self):
        tool, _, _ = build_structured_tool(Answer)
        schema_str = str(tool["input_schema"])
        assert "text" in schema_str
        assert "confidence" in schema_str

    def test_validator_class_is_input_class(self):
        _, validator_class, _ = build_structured_tool(Answer)
        assert validator_class is Answer

    def test_is_list_is_false_for_single_model(self):
        _, _, is_list = build_structured_tool(Answer)
        assert is_list is False


# ─────────────────────────────────────────────────────────────────────────────
# build_structured_tool — list[Model]
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildStructuredToolList:
    def test_tool_name_uses_array_suffix(self):
        tool, _, _ = build_structured_tool(list[Lesson])
        assert tool["name"] == "Lesson_array"

    def test_validator_class_is_wrapper_not_inner(self):
        _, validator_class, _ = build_structured_tool(list[Lesson])
        # Wrapper is a synthetic class — it should NOT be Lesson itself
        assert validator_class is not Lesson
        # And it should have an `items` field
        assert "items" in validator_class.model_fields

    def test_is_list_is_true(self):
        _, _, is_list = build_structured_tool(list[Lesson])
        assert is_list is True

    def test_input_schema_has_items_array(self):
        tool, _, _ = build_structured_tool(list[Lesson])
        schema_str = str(tool["input_schema"])
        assert "items" in schema_str


# ─────────────────────────────────────────────────────────────────────────────
# build_structured_tool — wrapper model with list field (cerebellum pattern)
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildStructuredToolWrapper:
    def test_wrapper_treated_as_single_model(self):
        tool, validator_class, is_list = build_structured_tool(WrappedLessons)
        # WrappedLessons IS a BaseModel that happens to contain a list,
        # but the helper should treat it as a single model (not list[T])
        # because the input wasn't a list[X] annotation.
        assert is_list is False
        assert validator_class is WrappedLessons
        assert tool["name"] == "WrappedLessons"


# ─────────────────────────────────────────────────────────────────────────────
# unwrap_structured_response — single model
# ─────────────────────────────────────────────────────────────────────────────


class TestUnwrapSingleModel:
    def test_valid_payload_returns_instance(self):
        _, validator_class, is_list = build_structured_tool(Answer)
        result = unwrap_structured_response(
            {"text": "42", "confidence": 0.95},
            validator_class,
            is_list,
        )
        assert isinstance(result, Answer)
        assert result.text == "42"
        assert result.confidence == 0.95

    def test_missing_required_field_raises(self):
        _, validator_class, is_list = build_structured_tool(Answer)
        with pytest.raises(ValidationError) as exc_info:
            unwrap_structured_response(
                {"text": "no confidence here"}, validator_class, is_list
            )
        assert "confidence" in str(exc_info.value).lower()

    def test_wrong_type_raises(self):
        _, validator_class, is_list = build_structured_tool(Answer)
        with pytest.raises(ValidationError):
            unwrap_structured_response(
                {"text": "x", "confidence": "not a number"},
                validator_class,
                is_list,
            )


# ─────────────────────────────────────────────────────────────────────────────
# unwrap_structured_response — list[Model]
# ─────────────────────────────────────────────────────────────────────────────


class TestUnwrapList:
    def test_valid_list_returns_typed_items(self):
        _, validator_class, is_list = build_structured_tool(list[Lesson])
        result = unwrap_structured_response(
            {
                "items": [
                    {"text": "first", "topic": "scratchpad"},
                    {"text": "second", "topic": "default"},
                ]
            },
            validator_class,
            is_list,
        )
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(item, Lesson) for item in result)
        assert result[0].text == "first"
        assert result[1].topic == "default"

    def test_empty_list_returns_empty(self):
        _, validator_class, is_list = build_structured_tool(list[Lesson])
        result = unwrap_structured_response(
            {"items": []}, validator_class, is_list
        )
        assert result == []

    def test_missing_items_field_raises(self):
        _, validator_class, is_list = build_structured_tool(list[Lesson])
        with pytest.raises(ValidationError):
            unwrap_structured_response({}, validator_class, is_list)


# ─────────────────────────────────────────────────────────────────────────────
# unwrap_structured_response — wrapper model with list (cerebellum pattern)
# ─────────────────────────────────────────────────────────────────────────────


class TestUnwrapWrapperModel:
    def test_round_trip(self):
        _, validator_class, is_list = build_structured_tool(WrappedLessons)
        result = unwrap_structured_response(
            {
                "lessons": [
                    {"text": "lesson 1", "topic": "scratchpad"},
                    {"text": "lesson 2", "topic": "scratchpad"},
                ]
            },
            validator_class,
            is_list,
        )
        # Returns the wrapper instance, NOT the items list (because the
        # original input wasn't a list[T] annotation — it was the
        # wrapper class directly)
        assert isinstance(result, WrappedLessons)
        assert len(result.lessons) == 2
        assert result.lessons[0].text == "lesson 1"

    def test_empty_lessons_list_is_valid(self):
        _, validator_class, is_list = build_structured_tool(WrappedLessons)
        result = unwrap_structured_response(
            {"lessons": []}, validator_class, is_list
        )
        assert isinstance(result, WrappedLessons)
        assert result.lessons == []


# ─────────────────────────────────────────────────────────────────────────────
# Round-trip — build then unwrap, mimicking the full flow
# ─────────────────────────────────────────────────────────────────────────────


class TestRoundTrip:
    def test_single_model_round_trip(self):
        tool, validator_class, is_list = build_structured_tool(Answer)
        # Simulate the LLM calling the tool with this input
        simulated_tool_input = {"text": "yes", "confidence": 0.7}
        result = unwrap_structured_response(
            simulated_tool_input, validator_class, is_list
        )
        assert result.text == "yes"
        assert result.confidence == 0.7
        # Sanity check the tool name was right
        assert tool["name"] == "Answer"

    def test_list_model_round_trip(self):
        tool, validator_class, is_list = build_structured_tool(list[Lesson])
        simulated = {
            "items": [
                {"text": "x", "topic": "a"},
                {"text": "y", "topic": "b"},
                {"text": "z", "topic": "c"},
            ]
        }
        result = unwrap_structured_response(simulated, validator_class, is_list)
        assert len(result) == 3
        assert [l.text for l in result] == ["x", "y", "z"]
        assert tool["name"] == "Lesson_array"
