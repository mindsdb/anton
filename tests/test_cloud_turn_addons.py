"""Messages added to a running cloud turn: stdin addon lines in, addon events out (ENG-4194)."""

from __future__ import annotations

import asyncio
import io
import json
import os
import threading
import time

import pytest

import anton.cloud_turn.__main__ as entry_mod
from anton.cloud_turn.__main__ import stream_turn
from anton.cloud_turn.addons import (
    MAX_ADDON_CHARS,
    MAX_ADDON_TURN_BYTES,
    MAX_ADDONS_PER_TURN,
    CloudAddons,
)
from anton.cloud_turn.stdin import parse_addon_line, parse_answer_line, start_answer_reader
from anton.core.llm.provider import StreamTextDelta, StreamTurnEnded

REQ = json.dumps({"protocol_version": 1, "conversation_id": "c", "input": "hi"})


def _line(**body) -> bytes:
    return (json.dumps(body) + "\n").encode()


def _addon(addon_id, text="more", decided_by="jev") -> bytes:
    return _line(kind="addon", addon_id=addon_id, text=text, decided_by=decided_by)


def _withdraw(addon_id, request_id) -> bytes:
    return _line(kind="addon_withdraw", addon_id=addon_id, request_id=request_id)


class _Session:
    """ChatSession's added-input contract, with a turn the test scripts."""

    def __init__(self, script=None):
        self._lock = threading.Lock()
        self._running = False
        self._inputs: list[dict] = []
        self.on_user_input_read = None
        self.on_user_input_ready = None
        self.calls: list[tuple] = []
        self._script = script
        self.closed = False

    def add_user_input(self, addon_id, text):
        self.calls.append(("add", addon_id))
        with self._lock:
            if not self._running:
                return False
            self._inputs.append({"id": addon_id, "text": text})
            return True

    def withdraw_user_input(self, addon_id):
        self.calls.append(("withdraw", addon_id))
        with self._lock:
            for i, entry in enumerate(self._inputs):
                if entry["id"] == addon_id:
                    del self._inputs[i]
                    return True
            return False

    def take_unread_user_inputs(self):
        with self._lock:
            unread, self._inputs = self._inputs, []
        return unread

    def show_to_model(self, addon_id):
        """What the tool loop does at a step boundary, for one id."""
        with self._lock:
            self._inputs = [e for e in self._inputs if e["id"] != addon_id]
        self.on_user_input_read([addon_id])

    async def wait_calls(self, n):
        deadline = time.monotonic() + 5
        while len(self.calls) < n:
            assert time.monotonic() < deadline, self.calls
            await asyncio.sleep(0.01)

    def start_turn(self):
        with self._lock:
            self._running = True
        if self.on_user_input_ready is not None:
            self.on_user_input_ready()

    async def turn_stream(self, user_input, **kwargs):
        self.start_turn()
        try:
            async for event in self._script(self):
                yield event
        finally:
            with self._lock:
                self._running = False

    async def settle_memory_writes(self):
        return None

    def close(self):
        self.closed = True


def _run(session, on_event=None, stdin=None, builder=None):
    """Drive one turn over a pipe stdin; *on_event(event, write)* may write lines."""
    read_fd, write_fd = os.pipe()
    own_pipe = stdin is None
    if own_pipe:
        stdin = os.fdopen(read_fd, "rb")
    else:
        os.close(read_fd)
    events = []
    threads = []
    real_start = entry_mod.start_answer_reader

    def recording_start(*args):
        thread = real_start(*args)
        threads.append(thread)
        return thread

    def emit(event):
        events.append(event)
        if on_event is not None:
            on_event(event, lambda data: os.write(write_fd, data))

    mp = pytest.MonkeyPatch()
    mp.setattr(entry_mod, "start_answer_reader", recording_start)
    try:
        asyncio.run(stream_turn(REQ, emit, session_builder=builder or (lambda r: session),
                                stdin=stdin))
    finally:
        mp.undo()
        os.close(write_fd)
        for thread in threads:
            thread.join(timeout=5)
        stdin.close()
    return events


READY = {"kind": "addon", "type": "response.addon", "state": "ready"}


def _addon_events(events):
    return [{k: v for k, v in e.items() if k not in ("kind", "type")}
            for e in events if e["kind"] == "addon" and e != READY]


def test_added_read_withdrawn_already_read_and_returned_in_one_turn():
    async def script(session):
        yield StreamTextDelta(text="go")
        # a1, a2, a3 added, a3 withdrawn; the unknown and malformed lines skipped.
        await session.wait_calls(4)
        session.show_to_model("a1")
        yield StreamTextDelta(text="next")
        await session.wait_calls(5)  # the late withdraw of a1

    session = _Session(script)

    def on_event(event, write):
        if event.get("text") == "go" and event["kind"] == "delta":
            write(_addon("a1", "first") + _addon("a2", "second", "user") + _addon("a3", "third")
                  + _withdraw("a3", "r1") + _line(kind="mystery", addon_id="a9")
                  + b"not json\n")
        if event["kind"] == "addon" and event["state"] == "read":
            write(_withdraw("a1", "r2"))

    events = _run(session, on_event)

    assert _addon_events(events) == [
        {"addon_id": "a1", "state": "added", "text": "first", "decided_by": "jev"},
        {"addon_id": "a2", "state": "added", "text": "second", "decided_by": "user"},
        {"addon_id": "a3", "state": "added", "text": "third", "decided_by": "jev"},
        {"addon_id": "a3", "state": "withdrawn", "request_id": "r1"},
        {"addon_id": "a1", "state": "read", "text": "first", "decided_by": "jev"},
        {"addon_id": "a1", "state": "already_read", "request_id": "r2"},
        {"addon_id": "a2", "state": "returned", "text": "second", "decided_by": "user"},
    ]
    # Exactly one ready, before anything the turn streamed.
    assert events.count(READY) == 1
    assert events.index(READY) < next(i for i, e in enumerate(events) if e["kind"] == "delta")
    assert all(e["type"] == "response.addon" for e in events if e["kind"] == "addon")
    assert events[-1] == {"kind": "turn_completed"}
    # The returned event goes out before the terminal: cowork stops reading at it.
    kinds = [e["kind"] for e in events]
    assert kinds.index("turn_completed") > max(i for i, k in enumerate(kinds) if k == "addon")
    assert ("add", "a9") not in session.calls


def test_lines_before_the_turn_starts_are_refused():
    """Not buffered: a line landing before the session exists is refused, and
    cowork-server keeps the message as the next task."""
    session = _Session(lambda s: _empty())
    stdin = io.BytesIO(_addon("early", "too soon") + _withdraw("early", "r0"))
    both_answered = threading.Event()
    seen = []

    def on_event(event, write):
        seen.append(event)
        if len(_addon_events(seen)) == 2:
            both_answered.set()

    def builder(req):
        # The build blocks until the loop has answered both lines, so they
        # land before the session exists.
        assert both_answered.wait(5)
        return session

    events = _run(session, on_event, stdin=stdin, builder=builder)
    assert _addon_events(events) == [
        {"addon_id": "early", "state": "refused", "text": "too soon", "decided_by": "jev"},
        {"addon_id": "early", "state": "already_read", "request_id": "r0"},
    ]
    assert session.calls == []  # never reached a session
    assert events[-1] == {"kind": "turn_completed"}


async def _empty():
    return
    yield  # pragma: no cover


def test_unread_messages_return_when_the_turn_fails():
    async def script(session):
        yield StreamTextDelta(text="go")
        await session.wait_calls(1)
        raise RuntimeError("boom")

    def on_event(event, write):
        if event["kind"] == "delta":
            write(_addon("a1", "keep me"))

    events = _run(_Session(script), on_event)
    assert _addon_events(events)[-1] == {
        "addon_id": "a1", "state": "returned", "text": "keep me", "decided_by": "jev"}
    assert events[-1]["kind"] == "turn_failed"
    assert events[-2]["kind"] == "addon"


def test_unread_messages_return_when_the_turn_ends_on_an_unanswered_question():
    async def script(session):
        yield StreamTextDelta(text="go")
        await session.wait_calls(1)
        yield StreamTurnEnded(ended_by="question_unanswered")

    def on_event(event, write):
        if event["kind"] == "delta":
            write(_addon("a1", "later"))

    events = _run(_Session(script), on_event)
    assert [e["kind"] for e in events[-3:]] == ["turn_ended", "addon", "turn_completed"]
    assert events[-2]["state"] == "returned"


def test_cancelled_turn_returns_unread_before_the_interrupted_terminal():
    session = _Session()
    events = []

    async def main():
        async def script(s):
            yield StreamTextDelta(text="go")
            s.add_user_input("a1", "later")
            raise asyncio.CancelledError

        session._script = script
        with pytest.raises(asyncio.CancelledError):
            await stream_turn(REQ, events.append, session_builder=lambda r: session)

    asyncio.run(main())
    assert events[-2] == {"kind": "addon", "type": "response.addon", "addon_id": "a1",
                          "state": "returned", "text": "later", "decided_by": None}
    assert events[-1]["kind"] == "turn_failed"


# ── CloudAddons on its own ──────────────────────────────────────────────────

def test_refused_when_the_session_has_no_running_turn():
    events = []
    addons = CloudAddons(events.append)
    session = _Session()
    addons.attach(session)
    addons.deliver({"kind": "addon", "addon_id": "a1", "text": "x", "decided_by": "user"})
    assert events == [{"kind": "addon", "type": "response.addon", "addon_id": "a1",
                       "state": "refused", "text": "x", "decided_by": "user"}]


def test_refused_when_the_session_predates_added_input():
    events = []
    addons = CloudAddons(events.append)
    addons.attach(object())
    addons.deliver({"kind": "addon", "addon_id": "a1", "text": "x", "decided_by": "user"})
    addons.return_unread()  # no-op, no raise
    assert [e["state"] for e in events] == ["refused"]


def test_a_repeated_addon_line_is_not_added_twice():
    events = []
    addons = CloudAddons(events.append)
    session = _Session()
    session.start_turn()
    addons.attach(session)
    line = {"kind": "addon", "addon_id": "a1", "text": "x", "decided_by": "user"}
    addons.deliver(line)
    addons.deliver(line)
    assert [e["state"] for e in events] == ["added", "added"]
    assert session.take_unread_user_inputs() == [{"id": "a1", "text": "x"}]


# ── stdin parsing ───────────────────────────────────────────────────────────

def test_parse_addon_lines():
    assert parse_addon_line(_addon("a1", "t", "jev")) == {
        "kind": "addon", "addon_id": "a1", "text": "t", "decided_by": "jev"}
    assert parse_addon_line(_withdraw("a1", "r1")) == {
        "kind": "addon_withdraw", "addon_id": "a1", "request_id": "r1"}
    assert parse_answer_line(_addon("a1")) is None


@pytest.mark.parametrize("raw", [
    b"not json",
    b"[1]",
    _line(kind="mystery", addon_id="a1", text="t"),
    _line(kind="addon", addon_id="", text="t"),
    _line(kind="addon", addon_id="a1", text="  "),
    _line(kind="addon", addon_id="a1", text=5),
    _line(kind="addon", addon_id="a1", text="t", decided_by=3),
    _line(kind="addon_withdraw", addon_id=None, request_id="r"),
    _line(kind="addon_withdraw", addon_id="a1", request_id=7),
])
def test_parse_addon_line_rejects_malformed(raw):
    assert parse_addon_line(raw) is None


async def test_reader_routes_answers_and_addons_and_skips_the_rest():
    answer = _line(kind="answer", question_id="q1", answer_id="x1")
    stream = io.BytesIO(answer + _addon("a1") + _line(kind="mystery") + _withdraw("a1", "r"))
    answers, addons = [], []
    thread = start_answer_reader(stream, asyncio.get_running_loop(), answers.append, addons.append)
    await asyncio.to_thread(thread.join, 5)
    await asyncio.sleep(0)
    assert [a["answer_id"] for a in answers] == ["x1"]
    assert [a["kind"] for a in addons] == ["addon", "addon_withdraw"]


async def test_reader_without_an_answer_handler_skips_answers():
    stream = io.BytesIO(_line(kind="answer", question_id="q1", answer_id="x1") + _addon("a1"))
    addons = []
    thread = start_answer_reader(stream, asyncio.get_running_loop(), None, addons.append)
    await asyncio.to_thread(thread.join, 5)
    await asyncio.sleep(0)
    assert [a["addon_id"] for a in addons] == ["a1"]


# ── Self-review fixes ───────────────────────────────────────────────────────

def _line_for(addon_id, text="x"):
    return {"kind": "addon", "addon_id": addon_id, "text": text, "decided_by": "user"}


def _running_addons():
    events = []
    addons = CloudAddons(events.append)
    session = _Session()
    session.start_turn()
    addons.attach(session)
    return addons, session, events


def test_cloud_turn_takes_at_most_twenty_messages():
    addons, session, events = _running_addons()
    for i in range(MAX_ADDONS_PER_TURN + 1):
        addons.deliver(_line_for(f"a{i}"))
    states = [e["state"] for e in events]
    assert states == ["added"] * MAX_ADDONS_PER_TURN + ["refused"]
    assert ("add", f"a{MAX_ADDONS_PER_TURN}") not in session.calls


def test_cloud_turn_refuses_a_message_over_the_character_limit():
    addons, session, events = _running_addons()
    addons.deliver(_line_for("big", "x" * (MAX_ADDON_CHARS + 1)))
    addons.deliver(_line_for("ok", "x" * MAX_ADDON_CHARS))
    assert [e["state"] for e in events] == ["refused", "added"]


def test_cloud_turn_takes_at_most_64_kb_of_added_text():
    addons, session, events = _running_addons()
    for i in range(9):  # 8000 bytes each: the ninth passes 64 KB
        addons.deliver(_line_for(f"a{i}", "é" * MAX_ADDON_CHARS))
    assert [e["state"] for e in events] == ["added"] * 8 + ["refused"]
    assert MAX_ADDON_TURN_BYTES == 64 * 1024


def test_a_refused_id_stays_refused():
    events = []
    addons = CloudAddons(events.append)
    session = _Session()
    addons.attach(session)
    addons.deliver(_line_for("a1"))  # no turn running yet
    session.start_turn()
    addons.deliver(_line_for("a1"))  # the same id again, now that one is
    assert [e["state"] for e in events] == ["refused", "refused"]
    assert session.calls == [("add", "a1")]
    assert session.take_unread_user_inputs() == []


def test_withdraw_after_the_message_was_returned_reports_returned():
    addons, session, events = _running_addons()
    addons.deliver(_line_for("a1"))
    addons.return_unread()
    addons.deliver({"kind": "addon_withdraw", "addon_id": "a1", "request_id": "r1"})
    assert events[-1] == {"kind": "addon", "type": "response.addon", "addon_id": "a1",
                          "state": "returned", "request_id": "r1"}


def test_no_ready_without_the_stdin_reader():
    """`ready` invites lines; with no stdin reader nothing would take them."""
    session = _Session()
    events = []

    async def script(s):
        yield StreamTextDelta(text="go")

    session._script = script
    asyncio.run(stream_turn(REQ, events.append, session_builder=lambda r: session))
    assert READY not in events
    assert events[-1] == {"kind": "turn_completed"}


def test_ready_comes_once_at_turn_start_before_any_model_call():
    order = []

    async def script(s):
        order.append("model call")
        yield StreamTextDelta(text="go")

    session = _Session(script)
    events = _run(session, lambda event, write: order.append(event["kind"]))
    assert events.count(READY) == 1
    assert order.index("addon") < order.index("model call")


async def test_the_session_announces_ready_at_turn_start(tmp_path):
    """ChatSession calls on_user_input_ready once, before its first model call."""
    from unittest.mock import MagicMock, patch
    from tests.conftest import make_mock_llm
    from anton.core.llm.provider import LLMResponse, StreamComplete, Usage
    from anton.core.session import ChatSession, ChatSessionConfig

    order = []
    llm = make_mock_llm()
    llm.usage_listener = None

    def _plan_stream(**kwargs):
        async def _gen():
            order.append("model call")
            yield StreamComplete(response=LLMResponse(
                content="hi", tool_calls=[], usage=Usage(input_tokens=1, output_tokens=1),
                stop_reason="end_turn"))
        return _gen()

    llm.plan_stream = _plan_stream
    session = ChatSession(ChatSessionConfig(llm_client=llm, workspace=MagicMock(base=tmp_path)))
    events = []
    addons = CloudAddons(lambda e: (events.append(e), order.append("ready")))
    addons.attach(session, announce=True)
    with patch("anton.analytics.send_event"):
        async for _ in session.turn_stream("hello"):
            pass
    assert events == [READY]
    assert order[:2] == ["ready", "model call"]
