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


def _resp_capped(tool_calls, *, output_tokens: int):
    """A reply whose output hit the cap — the truncation signal."""
    return LLMResponse(content="", tool_calls=tool_calls,
                       usage=Usage(input_tokens=1, output_tokens=output_tokens),
                       stop_reason="stop")


async def test_run_loop_rejects_write_file_without_content(tmp_path: Path):
    """Cut off before `content`: the key is absent. This used to write a 0-byte file and report success."""
    session = AsyncMock()
    session._llm.plan = AsyncMock(
        return_value=_resp([ToolCall(id="1", name="write_file", input={"path": "index.html"})])
    )
    session._llm.code = AsyncMock(
        return_value=_resp([ToolCall(id="2", name="finish", input={"summary": "gave up"})])
    )
    result = await _run_loop(
        session=session, system="s", kickoff="k", artifact_path=tmp_path,
        require_files=False, node_label="generate_frontend",
    )
    assert isinstance(result, dict)
    assert result["files_written"] == []
    assert not (tmp_path / "index.html").exists()


async def test_run_loop_rejects_write_file_with_empty_content(tmp_path: Path):
    """Cut off right after the opening quote: content == ""."""
    session = AsyncMock()
    session._llm.plan = AsyncMock(
        return_value=_resp([ToolCall(id="1", name="write_file",
                                     input={"path": "index.html", "content": ""})])
    )
    session._llm.code = AsyncMock(
        return_value=_resp([ToolCall(id="2", name="finish", input={"summary": "gave up"})])
    )
    result = await _run_loop(
        session=session, system="s", kickoff="k", artifact_path=tmp_path,
        require_files=False, node_label="generate_frontend",
    )
    assert not (tmp_path / "index.html").exists()


async def test_run_loop_rejects_only_the_last_call_of_a_truncated_response(tmp_path: Path):
    """The main form: the reply is truncated, but only the LAST call is incomplete.

    The JSON repair in provider.py stitches a valid-looking call and leaves
    parse_error unset, so the only signal is output_tokens hitting the client cap.
    Rejecting every call of the reply is not an option: if the model consistently
    spends its whole output budget (as the measurement shows), nothing would ever
    be written.
    """
    session = AsyncMock()
    session._llm._max_tokens = 100
    session._llm.plan = AsyncMock(
        return_value=_resp_capped(
            [
                ToolCall(id="1", name="write_file",
                         input={"path": "d.html", "content": "<head></head>", "mode": "w"}),
                ToolCall(id="2", name="write_file",
                         input={"path": "d.html", "content": "<body><div", "mode": "a"}),
            ],
            output_tokens=100,
        )
    )
    session._llm.code = AsyncMock(
        return_value=_resp([ToolCall(id="3", name="finish", input={"summary": "stopped"})])
    )
    result = await _run_loop(
        session=session, system="s", kickoff="k", artifact_path=tmp_path,
        require_files=False, node_label="generate_frontend",
    )
    assert isinstance(result, dict)
    # The first chunk arrived in full and must survive.
    assert (tmp_path / "d.html").read_text(encoding="utf-8") == "<head></head>"
    assert result["files_written"] == ["d.html"]


async def test_truncation_message_says_earlier_calls_landed(tmp_path: Path):
    """Otherwise the model re-sends written chunks and mode="a" duplicates them."""
    from anton.core.tools.generate_artifact.engine import _TRUNCATED_MSG

    assert "did take effect" in _TRUNCATED_MSG.lower()
    assert "do not re-send" in _TRUNCATED_MSG.lower()


async def test_run_loop_writes_when_response_is_not_capped(tmp_path: Path):
    """The same call goes through when the output is not capped — the detection must not be blanket."""
    session = AsyncMock()
    session._llm._max_tokens = 100
    session._llm.plan = AsyncMock(
        return_value=_resp_capped(
            [ToolCall(id="1", name="write_file",
                      input={"path": "index.html", "content": "<html></html>"})],
            output_tokens=42,
        )
    )
    session._llm.code = AsyncMock(
        return_value=_resp([ToolCall(id="2", name="finish", input={"summary": "ok"})])
    )
    result = await _run_loop(
        session=session, system="s", kickoff="k", artifact_path=tmp_path,
        node_label="generate_frontend",
    )
    assert isinstance(result, dict)
    assert result["files_written"] == ["index.html"]
    assert (tmp_path / "index.html").read_text(encoding="utf-8") == "<html></html>"


async def test_run_loop_ignores_unknown_token_cap(tmp_path: Path):
    """session is an AsyncMock: _max_tokens is not an int, output_tokens unknown → no flag.

    This protects every other test in the repo: they drive _run_loop with an
    AsyncMock session and must not suddenly start getting write rejections.
    """
    session = AsyncMock()  # _max_tokens will be a mock, not a number
    session._llm.plan = AsyncMock(
        return_value=_resp([ToolCall(id="1", name="write_file",
                                     input={"path": "index.html", "content": "<html></html>"})])
    )
    session._llm.code = AsyncMock(
        return_value=_resp([ToolCall(id="2", name="finish", input={"summary": "ok"})])
    )
    result = await _run_loop(
        session=session, system="s", kickoff="k", artifact_path=tmp_path,
        node_label="generate_frontend",
    )
    assert isinstance(result, dict)
    assert result["files_written"] == ["index.html"]


async def test_run_loop_passes_append_mode_through(tmp_path: Path):
    """Chunked assembly: two calls in one round build the file."""
    session = AsyncMock()
    session._llm.plan = AsyncMock(
        return_value=_resp([
            ToolCall(id="1", name="write_file",
                     input={"path": "d.html", "content": "<head>", "mode": "w"}),
            ToolCall(id="2", name="write_file",
                     input={"path": "d.html", "content": "<body></body>", "mode": "a"}),
            ToolCall(id="3", name="finish", input={"summary": "chunked"}),
        ])
    )
    result = await _run_loop(
        session=session, system="s", kickoff="k", artifact_path=tmp_path,
        node_label="generate_frontend",
    )
    assert isinstance(result, dict)
    assert result["files_written"] == ["d.html"]
    assert (tmp_path / "d.html").read_text(encoding="utf-8") == "<head><body></body>"


def test_round_budget_leaves_headroom_for_chunked_writes():
    """Chunked assembly costs rounds: head + sections + scripts + closing tags.

    16 was only enough for a monolithic write. Raised to 20 — but only after
    undelivered content is rejected (task 1), or the loop just gets longer.
    """
    from anton.core.tools.generate_artifact.engine import MAX_ROUNDS

    assert MAX_ROUNDS == 20
