"""End-to-end test of the cerebellum loop.

Verifies the full path with no real LLM and no real subprocess:

1. Build a fake session-like object with a real Cerebellum wired into
   `_scratchpad_observers`
2. Drive `handle_scratchpad` with an exec call where the (mocked) pad
   returns an errored Cell
3. Confirm the cerebellum's pre+post hooks fired via the dispatcher
4. Confirm the errored cell was buffered (clean cells would have been
   skipped via the cheap path)
5. Call cerebellum.flush() — verify the diff LLM was invoked, the
   lesson was extracted, and cortex.encode() received an Engram
6. The end-to-end loop runs without touching the runtime layer at all
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from anton.core.backends.base import Cell
from anton.core.memory.cerebellum import (
    Cerebellum,
    _DiffPassResult,
    _LessonDraft,
)
from anton.core.memory.hippocampus import Engram
from anton.core.tools.tool_handlers import handle_scratchpad


def _build_session_with_cerebellum(
    *,
    pad_returns: Cell,
    lessons: list[tuple[str, str]] | None = None,
) -> tuple[MagicMock, MagicMock, MagicMock, Cerebellum]:
    """Construct a fake session with a real Cerebellum wired in.

    Returns (session, mock_cortex, mock_llm, cerebellum) so the test
    can poke at any of them after the dispatch.

    `lessons` is a list of (text, topic) tuples that the mocked
    `generate_object` call will return as a `_DiffPassResult`. Empty
    list means the LLM extracted no lessons from the buffered cells.
    """
    cortex = MagicMock()
    cortex.encode = AsyncMock(return_value=["encoded"])

    drafts = [
        _LessonDraft(text=text, topic=topic) for text, topic in (lessons or [])
    ]
    llm = MagicMock()
    llm.generate_object_code = AsyncMock(
        return_value=_DiffPassResult(lessons=drafts)
    )

    cerebellum = Cerebellum(cortex=cortex, llm=llm)

    session = MagicMock()
    session._scratchpad_observers = [cerebellum]
    session._cerebellum = cerebellum
    session._cortex = cortex
    session._llm = llm
    session._record_cell_explainability = MagicMock()

    pad = MagicMock()
    pad.execute = AsyncMock(return_value=pad_returns)
    session._scratchpads = MagicMock()
    session._scratchpads.get_or_create = AsyncMock(return_value=pad)

    return session, cortex, llm, cerebellum


@pytest.mark.asyncio
async def test_full_cerebellum_loop_with_errored_cell():
    """Errored cell → dispatcher fires hooks → cerebellum buffers → flush
    runs diff → lesson encoded via cortex."""
    errored_cell = Cell(
        code="import pandas as pd\ndf = pd.read_csv('mixed.csv')",
        stdout="",
        stderr="",
        error=(
            "DtypeWarning: Columns have mixed types. "
            "Specify dtype option or set low_memory=False."
        ),
        description="Load mixed.csv into a DataFrame",
    )
    session, cortex, llm, cerebellum = _build_session_with_cerebellum(
        pad_returns=errored_cell,
        lessons=[
            (
                "When loading CSVs with mixed dtypes, pass low_memory=False to pd.read_csv.",
                "scratchpad",
            )
        ],
    )

    # Step 1: dispatch the exec call — this exercises the dispatcher's
    # observer firing path end-to-end
    result = await handle_scratchpad(
        session,
        {
            "action": "exec",
            "name": "main",
            "code": "import pandas as pd\ndf = pd.read_csv('mixed.csv')",
            "one_line_description": "Load mixed.csv into a DataFrame",
            "estimated_execution_time_seconds": 2,
        },
    )
    assert isinstance(result, str)

    # Step 2: cerebellum buffered the errored cell (cheap path skipped)
    assert cerebellum.buffered_count == 1

    # The diff LLM has NOT been called yet — that happens at flush
    llm.generate_object_code.assert_not_called()
    cortex.encode.assert_not_called()

    # Step 3: simulate end-of-turn flush
    lessons = await cerebellum.flush()

    # The diff LLM was called once with the buffered cell, using the
    # forced-tool-choice path via _DiffPassResult
    llm.generate_object_code.assert_called_once()
    call_args = llm.generate_object_code.call_args
    assert call_args.args[0] is _DiffPassResult
    diff_prompt = call_args.kwargs["messages"][0]["content"]
    assert "Load mixed.csv into a DataFrame" in diff_prompt
    assert "DtypeWarning" in diff_prompt

    # Lesson was extracted
    assert len(lessons) == 1
    assert "low_memory" in lessons[0].text

    # And encoded via cortex
    cortex.encode.assert_awaited_once()
    engrams = cortex.encode.call_args.args[0]
    assert len(engrams) == 1
    assert isinstance(engrams[0], Engram)
    assert engrams[0].kind == "lesson"
    assert engrams[0].topic == "scratchpad"
    assert "low_memory" in engrams[0].text

    # Buffer is now empty
    assert cerebellum.buffered_count == 0


@pytest.mark.asyncio
async def test_clean_cell_never_triggers_diff():
    """A successful cell should never reach the cerebellum's LLM diff."""
    clean_cell = Cell(
        code="print(1+1)",
        stdout="2",
        stderr="",
        error=None,
        description="add two",
    )
    session, cortex, llm, cerebellum = _build_session_with_cerebellum(
        pad_returns=clean_cell,
        lessons=[],
    )

    await handle_scratchpad(
        session,
        {
            "action": "exec",
            "name": "main",
            "code": "print(1+1)",
            "one_line_description": "add two",
            "estimated_execution_time_seconds": 1,
        },
    )

    # Cell was clean — never buffered
    assert cerebellum.buffered_count == 0

    # Even if we flush, no LLM call happens because buffer is empty
    await cerebellum.flush()
    llm.generate_object_code.assert_not_called()
    cortex.encode.assert_not_called()


@pytest.mark.asyncio
async def test_multiple_errored_cells_batched_into_single_diff():
    """Multiple cells in one turn → ONE diff call at flush, not N."""
    cell1 = Cell(
        code="x = bad_func()",
        stdout="",
        stderr="",
        error="NameError: name 'bad_func' is not defined",
        description="call bad_func",
    )
    cell2 = Cell(
        code="y = 1 / 0",
        stdout="",
        stderr="",
        error="ZeroDivisionError: division by zero",
        description="divide by zero",
    )
    session, cortex, llm, cerebellum = _build_session_with_cerebellum(
        pad_returns=cell1,  # first call returns cell1
        lessons=[
            ("Define functions before calling.", "scratchpad"),
            ("Guard against division by zero.", "scratchpad"),
        ],
    )

    # First exec returns cell1
    await handle_scratchpad(
        session,
        {
            "action": "exec",
            "name": "main",
            "code": "x = bad_func()",
            "one_line_description": "call bad_func",
            "estimated_execution_time_seconds": 1,
        },
    )

    # Re-mock pad.execute to return cell2 for the second call
    session._scratchpads.get_or_create.return_value.execute = AsyncMock(
        return_value=cell2
    )
    await handle_scratchpad(
        session,
        {
            "action": "exec",
            "name": "main",
            "code": "y = 1 / 0",
            "one_line_description": "divide by zero",
            "estimated_execution_time_seconds": 1,
        },
    )

    # Two cells buffered, zero LLM calls so far
    assert cerebellum.buffered_count == 2
    llm.generate_object_code.assert_not_called()

    # One flush → one generate_object call → both lessons encoded together
    await cerebellum.flush()
    llm.generate_object_code.assert_called_once()  # ← THE KEY ASSERTION: batched per turn
    cortex.encode.assert_awaited_once()
    engrams = cortex.encode.call_args.args[0]
    assert len(engrams) == 2
