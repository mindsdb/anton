from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

from anton.core.llm.provider import LLMResponse, ToolCall, Usage
from anton.core.tools.generate_artifact.engine import _run_loop


def _resp(tool_calls):
    return LLMResponse(content="", tool_calls=tool_calls,
                       usage=Usage(input_tokens=1, output_tokens=1), stop_reason="tool_use")


async def test_run_loop_allows_no_files_when_not_required(tmp_path: Path):
    session = AsyncMock()
    # Round 0 (plan): call finish immediately, writing nothing.
    session._llm.plan = AsyncMock(
        return_value=_resp([ToolCall(id="1", name="finish", input={"summary": "pad `a` cell 2: 100 rows"})])
    )
    result = await _run_loop(
        session=session,
        system="s",
        kickoff="k",
        artifact_path=tmp_path,
        require_files=False,
    )
    assert isinstance(result, dict)
    assert result["files_written"] == []
    assert result["summary"] == "pad `a` cell 2: 100 rows"


async def test_run_loop_still_requires_files_by_default(tmp_path: Path):
    session = AsyncMock()
    session._llm.plan = AsyncMock(
        return_value=_resp([ToolCall(id="1", name="finish", input={"summary": "done"})])
    )
    result = await _run_loop(session=session, system="s", kickoff="k", artifact_path=tmp_path)
    assert isinstance(result, str)
    assert "without writing any files" in result
