from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, Mock

# Must be the SAME module the engine catches against. A plain `httpx` here
# builds its error from a different class tree, the engine's except clause
# never matches, and this file would assert a retry while actually proving
# that a dropped stream is fatal.
import httpx2 as httpx
import pytest

from anton.core.llm.provider import LLMResponse, StreamComplete, ToolCall, Usage
from anton.core.tools.generate_artifact.engine import _run_loop
from anton.core.tools.generate_artifact.state import GEN_WRITE_MAX_TOKENS
from anton.core.tools.generate_artifact.sub_tools import (
    FILE_BEGIN_MARKER,
    FILE_END_MARKER,
)


def _body(text: str, *, closed: bool = True) -> str:
    """A file body the way the model sends it: plain text between the markers.

    `closed=False` reproduces a reply cut off before the end marker — the only
    signal that a body is incomplete, and the reason the protocol can tell a
    truncated file from a finished one at all.
    """
    tail = f"\n{FILE_END_MARKER}" if closed else ""
    return f"{FILE_BEGIN_MARKER}\n{text}{tail}"


def _resp(tool_calls, body: str | None = None):
    return LLMResponse(content=body or "", tool_calls=tool_calls,
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


def _resp_capped(tool_calls, *, output_tokens: int, body: str | None = None):
    """A reply whose output hit the cap — the truncation signal."""
    return LLMResponse(content=body or "", tool_calls=tool_calls,
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


async def test_a_truncated_body_writes_nothing(tmp_path: Path):
    """Replaces the old "reject only the LAST tool call of a truncated reply".

    That rule existed because truncation cut a tool argument and the earlier
    calls of the same reply had still landed. With the body in the text there is
    one body per reply, so there is no "earlier call" to preserve: a reply cut
    off before the end marker carries a half-file and must write nothing at all.
    Writing it would put a truncated file on disk with nothing saying so, which
    is the failure this protocol exists to prevent.
    """
    session = AsyncMock()
    session._llm.max_tokens = 100
    session._llm.plan_stream = _stream_mock(
        _resp_capped(
            [ToolCall(id="1", name="write_file", input={"path": "d.html"})],
            output_tokens=100,
            body=_body("<head></head><body><div", closed=False),
        )
    )
    session._llm.code_stream = _stream_mock(
        _resp([ToolCall(id="2", name="finish", input={"summary": "stopped"})])
    )
    result = await _run_loop(
        session=session, system="s", kickoff="k", artifact_path=tmp_path,
        require_files=False, node_label="generate_frontend",
    )
    assert isinstance(result, dict)
    assert result["files_written"] == []
    assert not (tmp_path / "d.html").exists()


async def test_a_body_without_write_file_still_answers_the_reply_s_other_tool_calls(tmp_path: Path):
    """The reply "final chunk + finish, write_file forgotten" put an
    assistant turn with a `finish` tool_use into the history and answered it
    with a bare user text. The next request was
    rejected by the provider (tool_use without tool_result) and the whole
    run ended as "generator crashed". Every tool_use of that reply gets a
    tool_result — `finish` a refusal, not an acceptance — and the advice
    rides as a text block in the same message."""
    from anton.core.tools.generate_artifact.engine import (
        _BODY_WITHOUT_CALL_MSG, _FINISH_REFUSED_BODY_UNWRITTEN_MSG,
    )

    session = AsyncMock()
    session._llm.max_tokens = None
    session._llm.plan_stream = _stream_mock(
        _resp([ToolCall(id="f1", name="finish", input={"summary": "done"})], body=_body("<h1>Hi</h1>"))
    )
    session._llm.code_stream = _stream_mock(
        _resp([ToolCall(id="w1", name="write_file", input={"path": "index.html", "mode": "w"})]),
        _resp([ToolCall(id="f2", name="finish", input={"summary": "done"})]),
    )
    result = await _run_loop(
        session=session, system="s", kickoff="k", artifact_path=tmp_path,
        node_label="generate_frontend",
    )
    assert isinstance(result, dict) and result["files_written"] == ["index.html"]
    assert (tmp_path / "index.html").read_text() == "<h1>Hi</h1>"

    # The message that answered round 0. The loop passes ONE list object to
    # every call and keeps appending to it, so the recorded argument holds the
    # whole history by now; find the answer by the tool_use id it closes.
    history = session._llm.code_stream.call_args_list[0].kwargs["messages"]
    [reply] = [
        m for m in history
        if m["role"] == "user" and isinstance(m["content"], list)
        and any(b.get("type") == "tool_result" and b.get("tool_use_id") == "f1" for b in m["content"])
    ]
    [answer] = [b for b in reply["content"] if b.get("type") == "tool_result"]
    assert answer["content"] == _FINISH_REFUSED_BODY_UNWRITTEN_MSG
    assert reply["content"][-1] == {"type": "text", "text": _BODY_WITHOUT_CALL_MSG}


async def test_a_cut_off_body_and_a_forgotten_marker_get_different_advice(tmp_path: Path):
    """Same symptom, opposite fixes: re-send it smaller vs. add one line.

    Telling a model that merely forgot the marker that its reply was "cut off"
    makes it regenerate the whole file to recover a single missing line.
    """
    from anton.core.tools.generate_artifact.sub_tools import extract_file_body

    _, cut_off, _ = extract_file_body(_body("x", closed=False), looks_truncated=True)
    _, forgot, _ = extract_file_body(_body("x", closed=False), looks_truncated=False)

    # Cut off: the fix is less content, so the advice is to split — and it must
    # say the shorter part goes in the NEXT REPLY, body included. A live run
    # answered the earlier wording, "emit the first part now, then
    # append the rest on the next turn", by announcing the part and calling
    # write_file with no body at all: it read the sentence as a two-turn plan.
    assert "was cut off" in cut_off
    assert "SHORTER" in cut_off
    assert "NEXT REPLY" in cut_off
    assert "announce" in cut_off

    # Forgot the line: the fix is one line, and the message says so plainly —
    # including that the reply was NOT cut off, so the model does not start
    # regenerating a file that is already complete.
    assert "not cut off" in forgot
    assert "smaller" not in forgot
    assert FILE_END_MARKER in forgot

    # Both must state the outcome: what the model does next depends on whether
    # the file moved, and leaving that to inference is how `mode="a"` doubles.
    assert "nothing was written" in cut_off.lower()
    assert "nothing was written" in forgot.lower()


async def test_run_loop_writes_when_response_is_not_capped(tmp_path: Path):
    """The same call goes through when the output is not capped — the detection must not be blanket."""
    session = AsyncMock()
    session._llm.max_tokens = 100
    session._llm.plan_stream = _stream_mock(
        _resp_capped(
            [ToolCall(id="1", name="write_file", input={"path": "index.html"})],
            output_tokens=42,
            body=_body("<html></html>"),
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
        _resp([ToolCall(id="1", name="write_file", input={"path": "index.html"})],
              body=_body("<html></html>"))
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


async def test_append_mode_builds_a_file_across_rounds(tmp_path: Path):
    """Chunked assembly, one part per round.

    Both parts used to ride in a single reply ("several small calls in one reply
    cost one round together"). A reply now carries one body, so each part is its
    own round — and the measured habit that rule encouraged, splitting a file
    that would have fit, goes away with it.
    """
    session = AsyncMock()
    session._llm.plan_stream = _stream_mock(
        _resp([ToolCall(id="1", name="write_file", input={"path": "d.html", "mode": "w"})],
              body=_body("<head>"))
    )
    session._llm.code_stream = _stream_mock(
        _resp([ToolCall(id="2", name="write_file", input={"path": "d.html", "mode": "a"})],
              body=_body("<body></body>")),
        _resp([ToolCall(id="3", name="finish", input={"summary": "chunked"})]),
    )
    result = await _run_loop(
        session=session, system="s", kickoff="k", artifact_path=tmp_path,
        node_label="generate_frontend",
    )
    assert isinstance(result, dict)
    assert result["files_written"] == ["d.html"]
    assert (tmp_path / "d.html").read_text(encoding="utf-8") == "<head><body></body>"


async def test_a_second_write_in_one_reply_is_refused(tmp_path: Path):
    """One body per reply, so the second call has nothing to write.

    Refuse both rather than pick one: guessing which call the body belonged to
    is guessing what lands on disk.
    """
    session = AsyncMock()
    session._llm.plan_stream = _stream_mock(
        _resp([
            ToolCall(id="1", name="write_file", input={"path": "a.html"}),
            ToolCall(id="2", name="write_file", input={"path": "b.html"}),
        ], body=_body("<html></html>"))
    )
    session._llm.code_stream = _stream_mock(
        _resp([ToolCall(id="3", name="finish", input={"summary": "gave up"})])
    )
    result = await _run_loop(
        session=session, system="s", kickoff="k", artifact_path=tmp_path,
        require_files=False, node_label="generate_frontend",
    )
    assert isinstance(result, dict)
    assert result["files_written"] == []
    assert not (tmp_path / "a.html").exists()
    assert not (tmp_path / "b.html").exists()


def test_round_budget_leaves_headroom_for_chunked_writes():
    """Chunked assembly costs rounds: head + sections + scripts + closing tags.

    16 was only enough for a monolithic write. Raised to 20 — but only after
    undelivered content is rejected (task 1), or the loop just gets longer.
    """
    from anton.core.tools.generate_artifact.engine import MAX_ROUNDS

    assert MAX_ROUNDS == 20


async def test_truncation_is_caught_by_stop_reason_when_no_cap_is_readable(tmp_path: Path):
    """`stop_reason: "length"` with no readable cap still has to be recognised.

    The body is unclosed either way, so nothing is written regardless; what
    `stop_reason` decides is WHICH advice the model gets back — see
    `test_a_cut_off_body_and_a_forgotten_marker_get_different_advice`.
    """
    session = AsyncMock()  # max_tokens is a mock → cap is unknown
    session._llm.plan_stream = _stream_mock(
        LLMResponse(
            content=_body("<head></head><body><div", closed=False),
            tool_calls=[ToolCall(id="1", name="write_file", input={"path": "d.html"})],
            usage=Usage(input_tokens=1, output_tokens=50), stop_reason="length",
        )
    )
    session._llm.code_stream = _stream_mock(
        _resp([ToolCall(id="2", name="finish", input={"summary": "stopped"})])
    )
    result = await _run_loop(
        session=session, system="s", kickoff="k", artifact_path=tmp_path,
        require_files=False, node_label="generate_frontend",
    )
    assert isinstance(result, dict)
    assert not (tmp_path / "d.html").exists()


async def test_round_budget_with_files_hands_them_to_the_caller(tmp_path: Path):
    """Budget exhaustion is not evidence the files are bad.

    A complete page was deleted and regenerated because the loop died counting
    its own slides. With files on disk the loop must return a dict so the
    verifier judges the actual output; `finished: False` records how it ended.
    """
    session = AsyncMock()
    write_resp = _resp([ToolCall(id="1", name="write_file",
                                 input={"path": "d.html", "mode": "a"})],
                       body=_body("x"))
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
            ToolCall(id="1", name="write_file", input={"path": "d.html"}),
            ToolCall(id="2", name="finish", input={"summary": "ok"}),
        ], body=_body("<html></html>"))
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


# ── Output budget per round, and surviving a dropped stream ──────────────────


async def test_every_round_gets_the_raised_budget_including_round_zero(
    tmp_path: Path,
):
    """Round 0 used to be held at the client default, and that is worth
    remembering rather than silently reversing.

    The reason was real while the body rode in a tool argument: round 0 runs on
    the slower planning model, a tool argument is not streamed, and at the
    raised budget such a call stayed silent long enough to be dropped — 4
    failures out of 4, measured 2026-08-28. At 8192 the same write merely
    truncated, which the loop survives.

    The body is streamed text now, so there is no silence to survive, and the
    cap only cost rounds: measured 2026-09-15, round 0 hit exactly 8192 output
    tokens mid-body and bought nothing. If a future change puts bulk content
    back into a tool argument, this lock is the one to revisit first.
    """
    session = AsyncMock()
    session._llm.plan_stream = _stream_mock(
        _resp([ToolCall(id="1", name="write_file", input={"path": "index.html"})],
              body=_body("<html></html>"))
    )
    session._llm.code_stream = _stream_mock(
        _resp([ToolCall(id="2", name="finish", input={"summary": "ok"})])
    )
    await _run_loop(
        session=session, system="s", kickoff="k", artifact_path=tmp_path,
        node_label="generate_frontend",
    )
    assert session._llm.plan_stream.call_args.kwargs["max_tokens"] == GEN_WRITE_MAX_TOKENS
    assert session._llm.code_stream.call_args.kwargs["max_tokens"] == GEN_WRITE_MAX_TOKENS


async def test_truncation_on_a_write_round_is_judged_against_its_own_budget(
    tmp_path: Path,
):
    """A big chunk must not be mistaken for a truncated one.

    The write rounds run on GEN_WRITE_MAX_TOKENS, so a reply well above the
    client's default is normal. Judging it against the default instead would
    reject the last tool call of every large chunk — the exact failure the
    raised budget exists to remove.
    """
    session = AsyncMock()
    session._llm.max_tokens = 8192          # client default, far below the round's
    session._llm.plan_stream = _stream_mock(
        # An inert round-0 call: read_file is handled entirely inside sub_tools,
        # so the round is consumed without dragging a real handler onto the mock.
        _resp([ToolCall(id="0", name="read_file", input={"path": "absent.html"})])
    )
    session._llm.code_stream = _stream_mock(
        _resp_capped(
            [ToolCall(id="1", name="write_file", input={"path": "index.html"})],
            output_tokens=12_000,           # > default, < GEN_WRITE_MAX_TOKENS
            body=_body("<html></html>"),
        ),
        _resp([ToolCall(id="2", name="finish", input={"summary": "ok"})]),
    )
    result = await _run_loop(
        session=session, system="s", kickoff="k", artifact_path=tmp_path,
        node_label="generate_frontend",
    )
    assert isinstance(result, dict)
    assert result["files_written"] == ["index.html"]
    assert (tmp_path / "index.html").read_text(encoding="utf-8") == "<html></html>"


def _dropping_stream():
    """A stream that dies mid-iteration, as a real dropped connection does."""

    async def _gen(**kw):
        if False:  # pragma: no cover - makes this an async generator
            yield None
        raise httpx.RemoteProtocolError("peer closed connection")

    return _gen


async def test_dropped_stream_is_retried_once_with_a_halved_budget(tmp_path: Path):
    """A mid-stream drop must not kill the generation.

    Large tool-call arguments are not streamed incrementally, so the connection
    is silent for the whole generation and can be dropped outright. Retrying on
    the same budget would mostly reproduce it, so the retry halves the budget:
    a shorter call is less likely to be dropped, and if it truncates instead the
    loop already recovers from that.
    """
    session = AsyncMock()
    session._llm.plan_stream = _stream_mock(
        # An inert round-0 call: read_file is handled entirely inside sub_tools,
        # so the round is consumed without dragging a real handler onto the mock.
        _resp([ToolCall(id="0", name="read_file", input={"path": "absent.html"})])
    )
    good = _one_event_stream(
        _resp([ToolCall(id="1", name="write_file", input={"path": "index.html"})],
              body=_body("<html></html>"))
    )
    finish = _one_event_stream(
        _resp([ToolCall(id="2", name="finish", input={"summary": "ok"})])
    )
    session._llm.code_stream = Mock(
        side_effect=[_dropping_stream()(), good, finish]
    )
    result = await _run_loop(
        session=session, system="s", kickoff="k", artifact_path=tmp_path,
        node_label="generate_frontend",
    )
    assert isinstance(result, dict)
    assert result["files_written"] == ["index.html"]
    budgets = [c.kwargs["max_tokens"] for c in session._llm.code_stream.call_args_list]
    assert budgets[0] == GEN_WRITE_MAX_TOKENS
    assert budgets[1] == GEN_WRITE_MAX_TOKENS // 2


async def test_a_second_drop_in_the_same_round_propagates(tmp_path: Path):
    """One retry, not a loop: a persistently dead connection must surface.

    Retrying forever would spend the whole round budget on a link that is not
    coming back, and report the failure as something else entirely.
    """
    session = AsyncMock()
    session._llm.plan_stream = _stream_mock(
        # An inert round-0 call: read_file is handled entirely inside sub_tools,
        # so the round is consumed without dragging a real handler onto the mock.
        _resp([ToolCall(id="0", name="read_file", input={"path": "absent.html"})])
    )
    session._llm.code_stream = Mock(
        side_effect=[_dropping_stream()(), _dropping_stream()()]
    )
    with pytest.raises(httpx.RemoteProtocolError):
        await _run_loop(
            session=session, system="s", kickoff="k", artifact_path=tmp_path,
            node_label="generate_frontend",
        )
