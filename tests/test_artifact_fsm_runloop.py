from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, Mock

from anton.core.llm.provider import LLMResponse, StreamComplete, ToolCall, Usage
from anton.core.tools.generate_artifact.engine import _run_loop


def _resp(tool_calls):
    return LLMResponse(content="", tool_calls=tool_calls,
                       usage=Usage(input_tokens=1, output_tokens=1), stop_reason="tool_use")


async def _one_event_stream(response):
    yield StreamComplete(response=response)


def _stream_mock(*responses):
    """`plan_stream`/`code_stream` fake: each call returns a fresh one-event
    stream. A single response repeats on every call (mirrors
    `AsyncMock(return_value=...)`); several are consumed one per call
    (mirrors `AsyncMock(side_effect=[...])`)."""
    if len(responses) == 1:
        response = responses[0]
        return Mock(side_effect=lambda **kw: _one_event_stream(response))
    return Mock(side_effect=[_one_event_stream(r) for r in responses])


async def test_run_loop_allows_no_files_when_not_required(tmp_path: Path):
    session = AsyncMock()
    # Round 0 (plan): call finish immediately, writing nothing.
    session._llm.plan_stream = _stream_mock(
        _resp([ToolCall(id="1", name="finish", input={"summary": "pad `a` cell 2: 100 rows"})])
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
    session._llm.plan_stream = _stream_mock(
        _resp([ToolCall(id="1", name="finish", input={"summary": "done"})])
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
    session._llm.plan_stream = _stream_mock(
        _resp([
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
    session._llm.plan_stream = _stream_mock(
        _resp([ToolCall(id="1", name="write_file", input={"path": "index.html"})])
    )
    session._llm.code_stream = _stream_mock(
        _resp([ToolCall(id="2", name="finish", input={"summary": "gave up"})])
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
    session._llm.plan_stream = _stream_mock(
        _resp([ToolCall(id="1", name="write_file",
                        input={"path": "index.html", "content": ""})])
    )
    session._llm.code_stream = _stream_mock(
        _resp([ToolCall(id="2", name="finish", input={"summary": "gave up"})])
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
    session._llm.max_tokens = 100
    session._llm.plan_stream = _stream_mock(
        _resp_capped(
            [
                ToolCall(id="1", name="write_file",
                         input={"path": "d.html", "content": "<head></head>", "mode": "w"}),
                ToolCall(id="2", name="write_file",
                         input={"path": "d.html", "content": "<body><div", "mode": "a"}),
            ],
            output_tokens=100,
        )
    )
    session._llm.code_stream = _stream_mock(
        _resp([ToolCall(id="3", name="finish", input={"summary": "stopped"})])
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
    session._llm.max_tokens = 100
    session._llm.plan_stream = _stream_mock(
        _resp_capped(
            [ToolCall(id="1", name="write_file",
                      input={"path": "index.html", "content": "<html></html>"})],
            output_tokens=42,
        )
    )
    session._llm.code_stream = _stream_mock(
        _resp([ToolCall(id="2", name="finish", input={"summary": "ok"})])
    )
    result = await _run_loop(
        session=session, system="s", kickoff="k", artifact_path=tmp_path,
        node_label="generate_frontend",
    )
    assert isinstance(result, dict)
    assert result["files_written"] == ["index.html"]
    assert (tmp_path / "index.html").read_text(encoding="utf-8") == "<html></html>"


async def test_run_loop_ignores_unknown_token_cap(tmp_path: Path):
    """session is an AsyncMock: max_tokens is not an int, output_tokens unknown → no flag.

    This protects every other test in the repo: they drive _run_loop with an
    AsyncMock session and must not suddenly start getting write rejections.
    """
    session = AsyncMock()  # max_tokens will be a mock, not a number
    session._llm.plan_stream = _stream_mock(
        _resp([ToolCall(id="1", name="write_file",
                        input={"path": "index.html", "content": "<html></html>"})])
    )
    session._llm.code_stream = _stream_mock(
        _resp([ToolCall(id="2", name="finish", input={"summary": "ok"})])
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
    session._llm.plan_stream = _stream_mock(
        _resp([
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


async def test_run_loop_flags_truncation_by_stop_reason_alone(tmp_path: Path):
    """`stop_reason: "length"` must reject the last call even with no readable cap.

    The gateway reports it correctly since 2026-08-03, and a cut that stopped
    just under the cap is invisible to the token count.
    """
    session = AsyncMock()  # max_tokens is a mock → cap is unknown
    session._llm.plan_stream = _stream_mock(
        LLMResponse(
            content="", tool_calls=[
                ToolCall(id="1", name="write_file",
                         input={"path": "d.html", "content": "<head></head>", "mode": "w"}),
                ToolCall(id="2", name="write_file",
                         input={"path": "d.html", "content": "<body><div", "mode": "a"}),
            ],
            usage=Usage(input_tokens=1, output_tokens=50), stop_reason="length",
        )
    )
    session._llm.code_stream = _stream_mock(
        _resp([ToolCall(id="3", name="finish", input={"summary": "stopped"})])
    )
    result = await _run_loop(
        session=session, system="s", kickoff="k", artifact_path=tmp_path,
        require_files=False, node_label="generate_frontend",
    )
    assert isinstance(result, dict)
    assert (tmp_path / "d.html").read_text(encoding="utf-8") == "<head></head>"


async def test_round_budget_with_files_hands_them_to_the_caller(tmp_path: Path):
    """Budget exhaustion is not evidence the files are bad (live run 2026-08-27).

    A complete page was deleted and regenerated because the loop died counting
    its own slides. With files on disk the loop must return a dict so the
    verifier judges the actual output; `finished: False` records how it ended.
    """
    session = AsyncMock()
    write_resp = _resp([ToolCall(id="1", name="write_file",
                                 input={"path": "d.html", "content": "x", "mode": "a"})])
    session._llm.plan_stream = _stream_mock(write_resp)
    session._llm.code_stream = Mock(side_effect=lambda **kw: _one_event_stream(write_resp))

    result = await _run_loop(
        session=session, system="s", kickoff="k", artifact_path=tmp_path,
        node_label="generate_frontend",
    )
    assert isinstance(result, dict)
    assert result["finished"] is False
    assert result["files_written"] == ["d.html"]
    assert (tmp_path / "d.html").exists()


async def test_round_budget_without_files_is_still_an_error(tmp_path: Path):
    session = AsyncMock()
    view_resp = _resp([ToolCall(id="1", name="scratchpad",
                                input={"action": "view", "name": "pad"})])
    session._llm.plan_stream = _stream_mock(view_resp)
    session._llm.code_stream = Mock(side_effect=lambda **kw: _one_event_stream(view_resp))
    import anton.core.tools.tool_handlers as tool_handlers
    from unittest.mock import patch

    with patch.object(tool_handlers, "handle_scratchpad", AsyncMock(return_value="ok")):
        result = await _run_loop(
            session=session, system="s", kickoff="k", artifact_path=tmp_path,
            node_label="generate_frontend",
        )
    assert isinstance(result, str)
    assert "round budget" in result


async def test_finished_flag_is_true_on_a_clean_finish(tmp_path: Path):
    session = AsyncMock()
    session._llm.plan_stream = _stream_mock(
        _resp([
            ToolCall(id="1", name="write_file",
                     input={"path": "d.html", "content": "<html></html>"}),
            ToolCall(id="2", name="finish", input={"summary": "ok"}),
        ])
    )
    result = await _run_loop(
        session=session, system="s", kickoff="k", artifact_path=tmp_path,
        node_label="generate_frontend",
    )
    assert isinstance(result, dict)
    assert result["finished"] is True


async def test_rounds_left_note_rides_on_every_tool_result_message(tmp_path: Path):
    """The model cannot see the budget any other way; near the end it must be
    told to wrap up instead of spending the tail on self-checks."""
    session = AsyncMock()
    captured: list[list[dict]] = []

    def _capture_stream(**kw):
        captured.append([m for m in kw["messages"]])
        return _one_event_stream(
            _resp([ToolCall(id=str(len(captured)), name="write_file",
                            input={"path": "d.html", "content": "x", "mode": "a"})])
        )

    session._llm.plan_stream = Mock(side_effect=_capture_stream)
    session._llm.code_stream = Mock(side_effect=_capture_stream)

    await _run_loop(
        session=session, system="s", kickoff="k", artifact_path=tmp_path,
        node_label="generate_frontend",
    )
    # The messages of the LAST round contain every earlier round's results.
    final_messages = captured[-1]
    user_results = [m for m in final_messages if m["role"] == "user"][1:]  # skip kickoff
    notes = [
        b["text"]
        for m in user_results
        for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    assert notes, "every tool-result message must carry a rounds-left note"
    assert all("round(s) left" in n for n in notes)
    assert any("wrap up" in n for n in notes), "the tail rounds must tell the model to finish"


async def test_read_file_full_flag_is_passed_through(tmp_path: Path, monkeypatch):
    from anton.core.tools.generate_artifact import sub_tools

    seen: dict = {}

    def fake_read_file(root, rel, *, full=False):
        seen["full"] = full
        return {"ok": True, "message": "content"}

    monkeypatch.setattr(sub_tools, "read_file", fake_read_file)
    session = AsyncMock()
    session._llm.plan_stream = _stream_mock(
        _resp([ToolCall(id="1", name="read_file", input={"path": "d.html", "full": True}),
               ToolCall(id="2", name="finish", input={"summary": "ok"})])
    )
    result = await _run_loop(
        session=session, system="s", kickoff="k", artifact_path=tmp_path,
        require_files=False, node_label="generate_frontend",
    )
    assert isinstance(result, dict)
    assert seen["full"] is True
