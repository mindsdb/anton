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
        node_label="fetch_data_sample",
    )
    assert isinstance(result, dict)
    assert result["files_written"] == []
    assert result["summary"] == "pad `a` cell 2: 100 rows"


async def test_run_loop_still_requires_files_by_default(tmp_path: Path):
    session = AsyncMock()
    session._llm.plan = AsyncMock(
        return_value=_resp([ToolCall(id="1", name="finish", input={"summary": "done"})])
    )
    result = await _run_loop(
        session=session, system="s", kickoff="k", artifact_path=tmp_path,
        node_label="generate_frontend",
    )
    assert isinstance(result, str)
    assert "without writing any files" in result


async def test_run_loop_records_scratchpad_execs(tmp_path: Path, monkeypatch):
    import anton.core.tools.tool_handlers as tool_handlers

    monkeypatch.setattr(
        tool_handlers, "handle_scratchpad", AsyncMock(return_value="cell 1 ok: 100 rows")
    )
    session = AsyncMock()
    session._llm.plan = AsyncMock(
        return_value=_resp([
            ToolCall(
                id="1", name="scratchpad",
                input={"action": "exec", "name": "pad", "code": "print(df.head())"},
            ),
            # Non-exec actions must not be recorded.
            ToolCall(id="2", name="scratchpad", input={"action": "view", "name": "pad"}),
            ToolCall(id="3", name="finish", input={"summary": "done"}),
        ])
    )
    result = await _run_loop(
        session=session, system="s", kickoff="k",
        artifact_path=tmp_path, require_files=False,
        node_label="fetch_data_sample",
    )
    assert isinstance(result, dict)
    assert result["scratchpad_execs"] == [
        {"name": "pad", "code": "print(df.head())", "output": "cell 1 ok: 100 rows"}
    ]
