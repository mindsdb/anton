"""Answer lines on the cloud turn's stdin."""

from __future__ import annotations

import asyncio
import io
import json
import logging

import pytest

from anton.cloud_turn.stdin import (
    MAX_ANSWER_LINE_BYTES,
    iter_lines,
    parse_answer_line,
    parse_host_tool_line,
    start_answer_reader,
)


def _line(**over) -> bytes:
    body = {"kind": "answer", "question_id": "q1", "answer_id": "a1",
            "values": ["pg"], "text": "", "skipped": False}
    body.update(over)
    return json.dumps(body).encode()


def test_iter_lines_splits_and_strips_newlines():
    assert list(iter_lines(io.BytesIO(b"one\ntwo\n"), 100)) == [b"one", b"two"]


def test_iter_lines_yields_a_final_line_without_newline():
    assert list(iter_lines(io.BytesIO(b"one\ntwo"), 100)) == [b"one", b"two"]


def test_iter_lines_skips_an_over_long_line_once(caplog):
    data = b"x" * 1000 + b"\n" + b"ok\n"
    with caplog.at_level(logging.WARNING):
        assert list(iter_lines(io.BytesIO(data), 64)) == [b"ok"]
    assert len([r for r in caplog.records if "over-long" in r.getMessage()]) == 1


def test_parse_answer_line_normalizes_defaults():
    raw = json.dumps({"kind": "answer", "question_id": "q1", "answer_id": "a1"}).encode()
    assert parse_answer_line(raw) == {"question_id": "q1", "answer_id": "a1",
                                      "values": [], "text": "", "skipped": False}


def test_parse_answer_line_keeps_a_full_answer():
    assert parse_answer_line(_line(values=["pg", "my"], text="t")) == {
        "question_id": "q1", "answer_id": "a1", "values": ["pg", "my"],
        "text": "t", "skipped": False}


@pytest.mark.parametrize("raw", [
    b"not json",
    b"[1, 2]",
    _line(kind="other"),
    _line(question_id=""),
    _line(answer_id=None),
    _line(values="pg"),
    _line(values=["pg", 3]),
    _line(text=5),
    _line(skipped="yes"),
])
def test_parse_answer_line_rejects_malformed(raw):
    assert parse_answer_line(raw) is None


async def test_reader_delivers_valid_answers_on_the_loop():
    stream = io.BytesIO(_line(answer_id="a1") + b"\n" + b"junk\n" + _line(answer_id="a2") + b"\n")
    got = []
    thread = start_answer_reader(stream, asyncio.get_running_loop(), got.append)
    await asyncio.to_thread(thread.join, 5)
    await asyncio.sleep(0)  # run the call_soon_threadsafe callbacks
    assert [a["answer_id"] for a in got] == ["a1", "a2"]
    assert thread.daemon is True


def test_max_answer_line_is_64_kb():
    assert MAX_ANSWER_LINE_BYTES == 64 * 1024


# ── host tool lines ──────────────────────────────────────────────────

CALL = "htool:" + "0123456789abcdef" * 2


def _host_line(**body) -> bytes:
    return json.dumps(body).encode()


def test_a_progress_line_parses_without_the_entry_id():
    raw = _host_line(kind="host_tool_progress", call_id=CALL, entry_id=CALL + ":0", text="Editing “Intro”")
    assert parse_host_tool_line(raw) == {"kind": "host_tool_progress", "call_id": CALL, "text": "Editing “Intro”"}


def test_a_result_part_parses_without_the_entry_id():
    raw = _host_line(kind="host_tool_result", call_id=CALL, entry_id=CALL + ":2",
                     seq=2, last=True, ok=False, text="")
    assert parse_host_tool_line(raw) == {"kind": "host_tool_result", "call_id": CALL,
                                         "seq": 2, "last": True, "ok": False, "text": ""}


@pytest.mark.parametrize("raw", [
    b"not json",
    b"[1]",
    _host_line(kind="host_tool_call", call_id=CALL, text="x"),
    _host_line(kind="host_tool_progress", call_id="htool:ABC", text="x"),
    _host_line(kind="host_tool_progress", call_id=CALL + "0", text="x"),
    _host_line(kind="host_tool_progress", call_id="acheck:" + "0" * 32, text="x"),
    _host_line(kind="host_tool_progress", call_id=7, text="x"),
    _host_line(kind="host_tool_progress", call_id=CALL, text=""),
    _host_line(kind="host_tool_progress", call_id=CALL, text="x" * 501),
    _host_line(kind="host_tool_progress", call_id=CALL),
    _host_line(kind="host_tool_result", call_id=CALL, seq=-1, last=True, ok=True, text=""),
    _host_line(kind="host_tool_result", call_id=CALL, seq=1_000_000, last=True, ok=True, text=""),
    _host_line(kind="host_tool_result", call_id=CALL, seq=True, last=True, ok=True, text=""),
    _host_line(kind="host_tool_result", call_id=CALL, seq="0", last=True, ok=True, text=""),
    _host_line(kind="host_tool_result", call_id=CALL, seq=0, last=1, ok=True, text=""),
    _host_line(kind="host_tool_result", call_id=CALL, seq=0, last=True, ok="yes", text=""),
    _host_line(kind="host_tool_result", call_id=CALL, seq=0, last=True, ok=True, text=None),
    _host_line(kind="host_tool_result", call_id=CALL, seq=0, last=True, ok=True),
])
def test_malformed_host_tool_lines_are_refused(raw):
    assert parse_host_tool_line(raw) is None


async def test_the_reader_hands_host_tool_lines_to_their_handler():
    got: list[dict] = []
    answers: list[dict] = []
    closed = asyncio.Event()
    stream = io.BytesIO(
        _host_line(kind="host_tool_progress", call_id=CALL, entry_id=CALL + ":0", text="step") + b"\n"
        + _line() + b"\n"
        + _host_line(kind="host_tool_result", call_id=CALL, entry_id=CALL + ":0",
                     seq=0, last=True, ok=True, text="done") + b"\n"
    )
    thread = start_answer_reader(stream, asyncio.get_running_loop(), answers.append,
                                 on_close=closed.set, deliver_host_tool=got.append)
    await asyncio.wait_for(closed.wait(), 5)
    thread.join(timeout=5)
    assert [line["kind"] for line in got] == ["host_tool_progress", "host_tool_result"]
    assert [a["answer_id"] for a in answers] == ["a1"]


async def test_the_reader_skips_host_tool_lines_without_a_handler():
    got: list[dict] = []
    thread = start_answer_reader(
        io.BytesIO(_host_line(kind="host_tool_progress", call_id=CALL, text="step") + b"\n"),
        asyncio.get_running_loop(), got.append, got.append, got.append)
    thread.join(timeout=5)
    await asyncio.sleep(0)
    assert got == []
