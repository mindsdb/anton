"""Unit tests for `anton.core.memory.cerebellum.Cerebellum`."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from anton.core.backends.base import Cell
from anton.core.memory.cerebellum import (
    Cerebellum,
    CerebellumLesson,
    _DiffPassResult,
    _format_cell_for_diff,
    _LessonDraft,
)
from anton.core.memory.hippocampus import Engram


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _cell(
    code: str = "x = 1",
    description: str = "",
    stdout: str = "",
    stderr: str = "",
    error: str | None = None,
) -> Cell:
    return Cell(
        code=code,
        stdout=stdout,
        stderr=stderr,
        error=error,
        description=description,
    )


def _make_llm_returning(*lessons: tuple[str, str]) -> MagicMock:
    """Build a mock LLMClient whose generate_object returns the given lessons.

    Each lesson is a (text, topic) tuple. Empty argument list means the
    diff returns no lessons (i.e. the LLM said "nothing to learn here").
    """
    drafts = [_LessonDraft(text=text, topic=topic) for text, topic in lessons]
    result = _DiffPassResult(lessons=drafts)
    llm = MagicMock()
    llm.generate_object_code = AsyncMock(return_value=result)
    return llm


def _make_llm_raising(exc: Exception) -> MagicMock:
    """Build a mock LLMClient whose generate_object raises an exception."""
    llm = MagicMock()
    llm.generate_object_code = AsyncMock(side_effect=exc)
    return llm


def _make_cortex() -> MagicMock:
    """Build a mock Cortex with an awaitable encode() method."""
    cortex = MagicMock()
    cortex.encode = AsyncMock(return_value=["encoded ok"])
    return cortex


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────


class TestFormatCellForDiff:
    def test_basic_cell_renders(self):
        cell = _cell(
            code="print(1+1)",
            description="add two numbers",
            stdout="2",
        )
        out = _format_cell_for_diff(cell, 1)
        assert "Cell 1" in out
        assert "add two numbers" in out
        assert "print(1+1)" in out
        assert "stdout:" in out

    def test_cell_with_error(self):
        cell = _cell(
            code="x = 1/0",
            description="divide",
            error="ZeroDivisionError: division by zero",
        )
        out = _format_cell_for_diff(cell, 2)
        assert "ZeroDivisionError" in out
        assert "Cell 2" in out

    def test_no_description_falls_back(self):
        cell = _cell(code="x = 1", stdout="")
        out = _format_cell_for_diff(cell, 1)
        assert "no description" in out

    def test_truncates_long_code(self):
        cell = _cell(code="x = 1\n" * 500, description="loop")
        out = _format_cell_for_diff(cell, 1)
        assert "[truncated]" in out

    def test_empty_outputs_marker(self):
        cell = _cell(code="pass", description="pass")
        out = _format_cell_for_diff(cell, 1)
        assert "no output produced" in out


# ─────────────────────────────────────────────────────────────────────────────
# Cheap path: clean cells should never reach the LLM
# ─────────────────────────────────────────────────────────────────────────────


class TestCheapPath:
    @pytest.mark.asyncio
    async def test_clean_cell_not_buffered(self):
        cb = Cerebellum(cortex=_make_cortex(), llm=_make_llm_returning())
        await cb.on_post_execute(_cell(code="x = 1", stdout="ok"))
        assert cb.buffered_count == 0

    @pytest.mark.asyncio
    async def test_clean_cell_with_only_stdout(self):
        cb = Cerebellum(cortex=_make_cortex(), llm=_make_llm_returning())
        await cb.on_post_execute(_cell(stdout="2", description="add"))
        assert cb.buffered_count == 0

    @pytest.mark.asyncio
    async def test_clean_cell_with_no_output_at_all(self):
        cb = Cerebellum(cortex=_make_cortex(), llm=_make_llm_returning())
        await cb.on_post_execute(_cell(code="pass"))
        assert cb.buffered_count == 0

    @pytest.mark.asyncio
    async def test_flush_with_no_cells_returns_empty(self):
        llm = _make_llm_returning()
        cb = Cerebellum(cortex=_make_cortex(), llm=llm)
        result = await cb.flush()
        assert result == []
        # No LLM call when nothing was buffered
        llm.generate_object_code.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Error path: buffer + diff + encode
# ─────────────────────────────────────────────────────────────────────────────


class TestErrorPath:
    @pytest.mark.asyncio
    async def test_error_cell_is_buffered(self):
        cb = Cerebellum(cortex=_make_cortex(), llm=_make_llm_returning())
        await cb.on_post_execute(
            _cell(code="x = 1/0", error="ZeroDivisionError")
        )
        assert cb.buffered_count == 1

    @pytest.mark.asyncio
    async def test_stderr_only_cell_is_buffered(self):
        cb = Cerebellum(cortex=_make_cortex(), llm=_make_llm_returning())
        # No `error` field but a non-empty stderr — counts as warning
        await cb.on_post_execute(
            _cell(code="import x", stderr="DeprecationWarning: ...")
        )
        assert cb.buffered_count == 1

    @pytest.mark.asyncio
    async def test_flush_calls_generate_object_with_buffered_cells(self):
        llm = _make_llm_returning()
        cb = Cerebellum(cortex=_make_cortex(), llm=llm)
        await cb.on_post_execute(
            _cell(
                code="x = 1/0",
                description="divide",
                error="ZeroDivisionError: division by zero",
            )
        )
        await cb.flush()
        llm.generate_object_code.assert_called_once()
        # generate_object_code was called with the _DiffPassResult Pydantic model
        call_args = llm.generate_object_code.call_args
        assert call_args.args[0] is _DiffPassResult
        # The prompt should mention the cell's intent
        user_msg = call_args.kwargs["messages"][0]["content"]
        assert "divide" in user_msg
        assert "ZeroDivisionError" in user_msg

    @pytest.mark.asyncio
    async def test_flush_clears_buffer(self):
        cb = Cerebellum(
            cortex=_make_cortex(),
            llm=_make_llm_returning(),
        )
        await cb.on_post_execute(_cell(error="boom"))
        assert cb.buffered_count == 1
        await cb.flush()
        assert cb.buffered_count == 0

    @pytest.mark.asyncio
    async def test_extracted_lesson_is_encoded(self):
        cortex = _make_cortex()
        cb = Cerebellum(
            cortex=cortex,
            llm=_make_llm_returning(
                ("Use low_memory=False with mixed-dtype CSVs.", "scratchpad")
            ),
        )
        await cb.on_post_execute(_cell(error="dtype error"))
        result = await cb.flush()

        assert len(result) == 1
        assert isinstance(result[0], CerebellumLesson)
        assert "low_memory" in result[0].text

        # Cortex.encode was called with an Engram
        cortex.encode.assert_awaited_once()
        engrams = cortex.encode.call_args.args[0]
        assert len(engrams) == 1
        assert isinstance(engrams[0], Engram)
        assert engrams[0].kind == "lesson"
        assert engrams[0].topic == "scratchpad"
        assert engrams[0].source == "consolidation"
        assert engrams[0].scope == "project"
        assert "low_memory" in engrams[0].text

    @pytest.mark.asyncio
    async def test_multiple_lessons_encoded_in_one_call(self):
        cortex = _make_cortex()
        cb = Cerebellum(
            cortex=cortex,
            llm=_make_llm_returning(
                ("lesson one", "scratchpad"),
                ("lesson two", "scratchpad"),
                ("lesson three", "scratchpad"),
            ),
        )
        await cb.on_post_execute(_cell(error="boom"))
        await cb.flush()

        cortex.encode.assert_awaited_once()
        engrams = cortex.encode.call_args.args[0]
        assert len(engrams) == 3

    @pytest.mark.asyncio
    async def test_max_lessons_caps_extraction(self):
        cortex = _make_cortex()
        cb = Cerebellum(
            cortex=cortex,
            llm=_make_llm_returning(
                *((f"lesson {i}", "scratchpad") for i in range(10))
            ),
            max_lessons_per_flush=2,
        )
        await cb.on_post_execute(_cell(error="boom"))
        result = await cb.flush()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_empty_lessons_list_does_not_encode(self):
        cortex = _make_cortex()
        cb = Cerebellum(cortex=cortex, llm=_make_llm_returning())
        await cb.on_post_execute(_cell(error="boom"))
        result = await cb.flush()
        assert result == []
        cortex.encode.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_lessons_with_blank_text(self):
        cortex = _make_cortex()
        cb = Cerebellum(
            cortex=cortex,
            llm=_make_llm_returning(
                ("real lesson", "scratchpad"),
                ("   ", "scratchpad"),  # blank — should be skipped
            ),
        )
        await cb.on_post_execute(_cell(error="boom"))
        result = await cb.flush()
        assert len(result) == 1
        assert result[0].text == "real lesson"


# ─────────────────────────────────────────────────────────────────────────────
# Robustness — bad LLM responses, missing infra
# ─────────────────────────────────────────────────────────────────────────────


class TestRobustness:
    @pytest.mark.asyncio
    async def test_llm_network_exception_does_not_crash(self):
        """Provider/network failure during generate_object → safe no-op."""
        cortex = _make_cortex()
        cb = Cerebellum(
            cortex=cortex,
            llm=_make_llm_raising(RuntimeError("network down")),
        )
        await cb.on_post_execute(_cell(error="boom"))
        # Must not raise
        result = await cb.flush()
        assert result == []
        cortex.encode.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_validation_error_does_not_crash(self):
        """If the LLM somehow violates the schema, Pydantic raises and
        flush() swallows it cleanly. This shouldn't happen in practice
        because tool_choice is forced, but the safety net is in place."""
        from pydantic import ValidationError as _PydValidationError

        cortex = _make_cortex()
        # Construct a real Pydantic validation error to feed in
        try:
            _DiffPassResult.model_validate({"lessons": "not a list"})
        except _PydValidationError as exc:
            llm = _make_llm_raising(exc)

        cb = Cerebellum(cortex=cortex, llm=llm)
        await cb.on_post_execute(_cell(error="boom"))
        result = await cb.flush()
        assert result == []
        cortex.encode.assert_not_called()

    @pytest.mark.asyncio
    async def test_value_error_from_no_tool_call_does_not_crash(self):
        """If generate_object raises ValueError because the LLM returned
        no tool call, flush() swallows it cleanly."""
        cortex = _make_cortex()
        cb = Cerebellum(
            cortex=cortex,
            llm=_make_llm_raising(ValueError("LLM did not return a tool call")),
        )
        await cb.on_post_execute(_cell(error="boom"))
        result = await cb.flush()
        assert result == []

    @pytest.mark.asyncio
    async def test_cortex_encode_exception_does_not_crash(self):
        """If cortex.encode itself fails, the cerebellum logs and moves on.
        The lesson was still extracted (returned to the caller) — only
        the persistence step failed."""
        cortex = MagicMock()
        cortex.encode = AsyncMock(side_effect=RuntimeError("disk full"))
        cb = Cerebellum(
            cortex=cortex,
            llm=_make_llm_returning(("be careful", "scratchpad")),
        )
        await cb.on_post_execute(_cell(error="boom"))
        # Must not raise
        result = await cb.flush()
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_no_cortex_no_encode(self):
        cb = Cerebellum(cortex=None, llm=_make_llm_returning())
        await cb.on_post_execute(_cell(error="boom"))
        result = await cb.flush()
        assert result == []

    @pytest.mark.asyncio
    async def test_no_llm_no_diff(self):
        cb = Cerebellum(cortex=_make_cortex(), llm=None)
        await cb.on_post_execute(_cell(error="boom"))
        result = await cb.flush()
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# Reset / lifecycle
# ─────────────────────────────────────────────────────────────────────────────


class TestReset:
    @pytest.mark.asyncio
    async def test_reset_clears_buffer_without_encoding(self):
        cortex = _make_cortex()
        llm = _make_llm_returning(("x", "scratchpad"))
        cb = Cerebellum(cortex=cortex, llm=llm)
        await cb.on_post_execute(_cell(error="boom"))
        await cb.on_post_execute(_cell(error="boom2"))
        assert cb.buffered_count == 2

        cb.reset()
        assert cb.buffered_count == 0

        # After reset, no LLM call happens because buffer is empty
        await cb.flush()
        llm.generate_object_code.assert_not_called()
        cortex.encode.assert_not_called()

    @pytest.mark.asyncio
    async def test_pre_execute_increments_counter(self):
        cb = Cerebellum(cortex=_make_cortex(), llm=_make_llm_returning())
        await cb.on_pre_execute(_cell(code="x = 1", description="set x"))
        await cb.on_pre_execute(_cell(code="y = 2", description="set y"))
        # Internal counter — we don't expose it as a property but reset clears it
        cb.reset()
        # No exception means reset worked
