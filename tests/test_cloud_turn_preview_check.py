"""The user's open preview, asked from a cloud turn (ENG-4227): event out, stdin line back."""

from __future__ import annotations

import asyncio
import io
import json
import os

import pytest

from anton.cloud_turn.contract import TurnRequestV1
from anton.cloud_turn.preview_check import CloudPreviewChecks
from anton.cloud_turn.stdin import parse_check_result_line, start_answer_reader
from anton.core.artifact_editor.checkers import PreviewChecker
from anton.core.llm.provider import StreamTextDelta

from tests.test_cloud_turn_entrypoint import _FakeSession


def line(**body) -> bytes:
    return json.dumps(body).encode()


PASSED = {"status": "passed", "errors": [], "ms": 812}
FAILED = {"status": "failed", "ms": 900, "errors": [
    {"type": "error", "message": "TypeError: x is undefined", "file": "index.html", "line": 12, "url": None}]}


# ── the stdin line ───────────────────────────────────────────────────


def test_a_check_result_line_parses():
    parsed = parse_check_result_line(line(kind="check_result", check_id="acheck:ab12", body=FAILED))
    assert parsed == {"check_id": "acheck:ab12", "body": FAILED}


def test_a_null_body_means_nobody_answered():
    assert parse_check_result_line(line(kind="check_result", check_id="acheck:ab12", body=None)) == {
        "check_id": "acheck:ab12", "body": None}


def test_a_body_is_cleaned():
    parsed = parse_check_result_line(line(kind="check_result", check_id="acheck:1", body={
        "status": "failed", "errors": [{"type": "error", "message": "m"}, "junk", 3], "ms": True}))
    assert parsed["body"] == {"status": "failed", "errors": [{"type": "error", "message": "m"}], "ms": 0}


@pytest.mark.parametrize("raw", [
    b"not json",
    line(kind="answer", check_id="acheck:1", body=PASSED),
    line(kind="check_result", check_id="nope:1", body=PASSED),
    line(kind="check_result", check_id="acheck:" + "a" * 200, body=PASSED),
    line(kind="check_result", check_id=7, body=PASSED),
    line(kind="check_result", check_id="acheck:1", body={"status": "great"}),
    line(kind="check_result", check_id="acheck:1", body={"status": "passed", "errors": "x"}),
    line(kind="check_result", check_id="acheck:1", body="passed"),
])
def test_malformed_check_result_lines_are_refused(raw):
    assert parse_check_result_line(raw) is None


def test_the_reader_hands_check_lines_on_and_reports_eof():
    async def scenario():
        loop = asyncio.get_running_loop()
        got: list[dict] = []
        closed = asyncio.Event()
        stream = io.BytesIO(line(kind="check_result", check_id="acheck:1", body=PASSED) + b"\n"
                            + line(kind="answer", question_id="q", answer_id="a") + b"\n")
        thread = start_answer_reader(stream, loop, None, None, deliver_check=got.append, on_close=closed.set)
        await asyncio.wait_for(closed.wait(), 5)
        thread.join(timeout=5)
        return got

    assert asyncio.run(scenario()) == [{"check_id": "acheck:1", "body": PASSED}]


def test_the_reader_skips_check_lines_without_a_handler():
    async def scenario():
        got: list[dict] = []
        thread = start_answer_reader(
            io.BytesIO(line(kind="check_result", check_id="acheck:1", body=PASSED) + b"\n"),
            asyncio.get_running_loop(), got.append, got.append)
        thread.join(timeout=5)
        await asyncio.sleep(0)
        return got

    assert asyncio.run(scenario()) == []


# ── CloudPreviewChecks ───────────────────────────────────────────────


async def test_a_check_emits_the_event_and_returns_the_answer():
    events: list[dict] = []
    checks = CloudPreviewChecks(events.append)
    task = asyncio.ensure_future(checks("sales-dashboard", 20))
    await asyncio.sleep(0)
    event = events[0]
    assert event["kind"] == "artifact_check" and event["type"] == "response.artifact_check"
    assert event["check_id"].startswith("acheck:") and event["artifact_slug"] == "sales-dashboard"
    assert event["timeout_s"] == 20
    checks.deliver({"check_id": event["check_id"], "body": FAILED})
    assert await task == FAILED
    assert checks.pending == 0


async def test_no_answer_within_the_timeout_is_none():
    checks = CloudPreviewChecks(lambda e: None)
    started = asyncio.get_running_loop().time()
    assert await checks("x", 1) is None
    assert asyncio.get_running_loop().time() - started < 3
    assert checks.pending == 0


async def test_a_null_answer_is_none():
    events: list[dict] = []
    checks = CloudPreviewChecks(events.append)
    task = asyncio.ensure_future(checks("x", 20))
    await asyncio.sleep(0)
    checks.deliver({"check_id": events[0]["check_id"], "body": None})
    assert await task is None


async def test_answers_for_other_or_finished_checks_are_ignored():
    events: list[dict] = []
    checks = CloudPreviewChecks(events.append)
    task = asyncio.ensure_future(checks("x", 20))
    await asyncio.sleep(0)
    checks.deliver({"check_id": "acheck:someone-else", "body": PASSED})
    assert not task.done()
    check_id = events[0]["check_id"]
    checks.deliver({"check_id": check_id, "body": PASSED})
    checks.deliver({"check_id": check_id, "body": FAILED})  # a repeat changes nothing
    assert await task == PASSED


async def test_closing_resolves_pending_checks_and_later_ones_skip_the_wait():
    events: list[dict] = []
    checks = CloudPreviewChecks(events.append)
    task = asyncio.ensure_future(checks("x", 60))
    await asyncio.sleep(0)
    checks.close()
    assert await task is None
    assert await checks("y", 60) is None
    assert len(events) == 1  # no event for a check asked after the close


async def test_the_editor_reads_the_answer_as_a_preview_check(tmp_path):
    events: list[dict] = []
    checks = CloudPreviewChecks(events.append)
    checker = PreviewChecker(checks, slug="sales", timeout_s=20)

    async def answer():
        while not events:
            await asyncio.sleep(0)
        checks.deliver({"check_id": events[0]["check_id"], "body": FAILED})

    result, _ = await asyncio.gather(checker.check(None, ["index.html"]), answer())
    assert result.status == "failed" and result.channel == "preview"
    assert result.findings[0].message == "TypeError: x is undefined"
    assert result.findings[0].source == "index.html:12"


async def test_an_unavailable_preview_is_unverified():
    events: list[dict] = []
    checks = CloudPreviewChecks(events.append)
    checker = PreviewChecker(checks, slug="sales", timeout_s=20)

    async def answer():
        while not events:
            await asyncio.sleep(0)
        checks.deliver({"check_id": events[0]["check_id"], "body": {"status": "unavailable", "errors": [], "ms": 1}})

    result, _ = await asyncio.gather(checker.check(None, ["index.html"]), answer())
    assert result.status == "unverified"


# ── the turn ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("value,expected", [(True, True), (None, False), ("true", False), (1, False)])
def test_the_request_opts_in_only_with_true(value, expected):
    body = {"protocol_version": 1, "conversation_id": "c", "input": "hi"}
    if value is not None:
        body["preview_checks"] = value
    assert TurnRequestV1.from_json(json.dumps(body)).preview_checks is expected


class _CheckingSession(_FakeSession):
    """Asks the preview checker it was built with, the way edit_artifact's PreviewChecker would."""

    def __init__(self):
        super().__init__()
        self.preview_checker = None
        self.answer = "unset"

    async def turn_stream(self, user_input, **kwargs):
        self.answer = await self.preview_checker("sales", 20)
        yield StreamTextDelta(text="done")


def _run_turn(request: dict, builder, on_event=None):
    from anton.cloud_turn.__main__ import stream_turn

    read_fd, write_fd = os.pipe()
    stdin = os.fdopen(read_fd, "rb")
    events: list[dict] = []

    def emit(event):
        events.append(event)
        if on_event is not None:
            reply = on_event(event)
            if reply is not None:
                os.write(write_fd, (json.dumps(reply) + "\n").encode())

    try:
        asyncio.run(stream_turn(json.dumps(request), emit, session_builder=builder, stdin=stdin))
    finally:
        os.close(write_fd)
        stdin.close()
    return events


def test_a_turn_with_preview_checks_asks_the_host_and_takes_the_stdin_answer():
    session = _CheckingSession()

    def builder(req, elicitor=None, preview_checker=None):
        session.preview_checker = preview_checker
        return session

    def host(event):
        if event["kind"] == "artifact_check":
            return {"kind": "check_result", "check_id": event["check_id"], "body": PASSED}
        return None

    events = _run_turn({"protocol_version": 1, "conversation_id": "c", "input": "hi", "preview_checks": True},
                       builder, host)
    assert session.answer == PASSED
    check = next(e for e in events if e["kind"] == "artifact_check")
    assert check["artifact_slug"] == "sales" and check["type"] == "response.artifact_check"
    assert events[-1]["kind"] == "turn_completed"


def test_without_preview_checks_no_checker_is_built():
    built = {}

    def builder(req, elicitor=None, preview_checker=None):
        built["checker"] = preview_checker
        return _FakeSession()

    _run_turn({"protocol_version": 1, "conversation_id": "c", "input": "hi"}, builder)
    assert built["checker"] is None


def test_a_builder_that_takes_no_checker_still_runs():
    def builder(req, elicitor=None):
        return _FakeSession()

    events = _run_turn({"protocol_version": 1, "conversation_id": "c", "input": "hi", "preview_checks": True},
                       builder)
    assert events[-1]["kind"] == "turn_completed"
