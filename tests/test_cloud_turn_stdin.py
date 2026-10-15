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
