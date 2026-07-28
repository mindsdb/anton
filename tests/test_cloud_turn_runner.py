"""Tests for the cloud-turn runner + entrypoint.

Fully offline: a fake session (scripted history growth) replaces the real
ChatSession, so no LLM key or scratchpad is needed.
"""

from __future__ import annotations

import asyncio
import json

from anton.cloud_turn.protocol import TurnRequestV1
from anton.cloud_turn.runner import EXIT_FAILED, EXIT_OK, run_turn


class _FakeSession:
    """Simulates Anton's history growth: turn_stream echoes the user input as a
    user message (as the real session does), then appends scripted generated
    messages, yielding a drive event per append."""

    def __init__(self, initial_history=None, generated=None, raise_on_stream=None, print_mid=False):
        self.history = list(initial_history or [])
        self._generated = list(generated or [])
        self._raise = raise_on_stream
        self._print_mid = print_mid
        self.closed = False

    async def turn_stream(self, user_input, **kwargs):
        if self._raise:
            raise self._raise
        self.history.append({"role": "user", "content": user_input})
        yield object()
        for msg in self._generated:
            if self._print_mid:
                print("STRAY STDOUT that must not corrupt the protocol")
            self.history.append(msg)
            yield object()

    def close(self):
        self.closed = True


def _req(**over) -> TurnRequestV1:
    body = dict(run_id="r", attempt_id="a", conversation_id="c", input="hi")
    body.update(over)
    return TurnRequestV1(**body)


def _collect_emit():
    events = []
    return events, events.append


# ── output messages (item 2) ────────────────────────────────────────────────

def test_text_only_turn_returns_final_assistant_message():
    events, emit = _collect_emit()
    session = _FakeSession(generated=[{"role": "assistant", "content": "Hello world"}])
    code = asyncio.run(run_turn(_req(), emit, session_builder=lambda r: session))
    assert code == EXIT_OK
    assert [e.kind for e in events] == ["turn.started", "turn.completed"]
    completed = events[1]
    assert completed.final_text == "Hello world"
    assert len(completed.output_messages) == 1
    assert completed.output_messages[0].role == "assistant"
    assert completed.output_messages[0].content == "Hello world"


def test_tool_using_turn_returns_all_persistable_messages_in_order():
    events, emit = _collect_emit()
    generated = [
        {"role": "assistant", "content": [
            {"type": "text", "text": "let me compute"},
            {"type": "tool_use", "id": "t1", "name": "scratchpad", "input": {"code": "1+1"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "2"},
        ]},
        {"role": "assistant", "content": "The answer is 2"},
    ]
    session = _FakeSession(generated=generated)
    asyncio.run(run_turn(_req(), emit, session_builder=lambda r: session))
    msgs = events[1].output_messages
    # assistant(text+tool_use) → user(tool_result) → final assistant, in order.
    assert [m.role for m in msgs] == ["assistant", "user", "assistant"]
    assert msgs[0].content[1].type == "tool_use"
    assert msgs[1].content[0].type == "tool_result"
    assert msgs[1].content[0].tool_use_id == "t1"
    assert events[1].final_text == "The answer is 2"


def test_input_history_not_duplicated_in_output_messages():
    events, emit = _collect_emit()
    prior = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
    ]
    session = _FakeSession(
        initial_history=prior,
        generated=[{"role": "assistant", "content": "new answer"}],
    )
    asyncio.run(run_turn(_req(), emit, session_builder=lambda r: session))
    msgs = events[1].output_messages
    # Only the newly generated assistant message — not the prior turn, not the
    # echoed current user input.
    assert len(msgs) == 1
    assert msgs[0].content == "new answer"


def test_final_text_matches_final_assistant_text():
    events, emit = _collect_emit()
    generated = [
        {"role": "assistant", "content": "intermediate narration"},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "x"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "FINAL"}]},
    ]
    session = _FakeSession(generated=generated)
    asyncio.run(run_turn(_req(), emit, session_builder=lambda r: session))
    assert events[1].final_text == "FINAL"


# ── terminal-event contract (item 7) ─────────────────────────────────────────

def test_sequence_numbers_started_then_terminal():
    events, emit = _collect_emit()
    session = _FakeSession(generated=[{"role": "assistant", "content": "ok"}])
    asyncio.run(run_turn(_req(), emit, session_builder=lambda r: session))
    assert [e.sequence for e in events] == [1, 2]


def test_builder_error_emits_started_then_failed():
    events, emit = _collect_emit()

    def boom(_req):
        raise RuntimeError("no session")

    code = asyncio.run(run_turn(_req(), emit, session_builder=boom))
    assert code == EXIT_FAILED
    assert [e.kind for e in events] == ["turn.started", "turn.failed"]
    assert events[-1].error.code.value == "internal_turn_failure"


def test_stream_error_emits_exactly_one_terminal():
    events, emit = _collect_emit()
    session = _FakeSession(raise_on_stream=ValueError("mid-turn boom"))
    code = asyncio.run(run_turn(_req(), emit, session_builder=lambda r: session))
    assert code == EXIT_FAILED
    terminals = [e for e in events if e.kind in ("turn.completed", "turn.failed")]
    assert len(terminals) == 1
    assert terminals[0].kind == "turn.failed"
    assert session.closed is True  # closed even on failure


# ── deadline semantics (item 5) ──────────────────────────────────────────────

def test_future_deadline_runs_normally():
    events, emit = _collect_emit()
    session = _FakeSession(generated=[{"role": "assistant", "content": "done"}])
    code = asyncio.run(
        run_turn(_req(deadline_unix_ms=10_000), emit,
                 session_builder=lambda r: session, now_ms=0)
    )
    assert code == EXIT_OK
    assert events[-1].kind == "turn.completed"


def test_already_expired_deadline_fails_before_building_session():
    events, emit = _collect_emit()
    built = {"n": 0}

    def builder(_r):
        built["n"] += 1
        return _FakeSession(generated=[{"role": "assistant", "content": "x"}])

    code = asyncio.run(
        run_turn(_req(deadline_unix_ms=100), emit, session_builder=builder, now_ms=200)
    )
    assert code == EXIT_FAILED
    assert [e.kind for e in events] == ["turn.started", "turn.failed"]
    assert events[-1].error.code.value == "deadline_exceeded"
    assert built["n"] == 0  # failed immediately, before building the session


def test_timeout_during_execution_fails_with_one_terminal():
    events, emit = _collect_emit()

    class _Slow(_FakeSession):
        async def turn_stream(self, user_input, **kwargs):
            await asyncio.sleep(0.5)
            self.history.append({"role": "assistant", "content": "late"})
            yield object()

    code = asyncio.run(
        run_turn(_req(deadline_unix_ms=50), emit,
                 session_builder=lambda r: _Slow(), now_ms=0)
    )
    assert code == EXIT_FAILED
    terminals = [e for e in events if e.kind in ("turn.completed", "turn.failed")]
    assert len(terminals) == 1
    assert terminals[0].error.code.value == "deadline_exceeded"




# ── entrypoint core (_run): request parsing → runner, no FD/stdin handling ────
# FD-level stdout isolation and bounded stdin reads run in a REAL subprocess —
# see tests/test_cloud_turn_process.py. Here we exercise the parse→emit core in
# process with a capturing emit.

def _run_core(stdin_text, session_builder=None):
    from anton.cloud_turn import __main__ as entry

    events = []
    code = entry._run(stdin_text, events.append, session_builder=session_builder)
    return code, events


def test_run_completes_and_emits_started_then_completed():
    session = _FakeSession(generated=[{"role": "assistant", "content": "done"}])
    req = json.dumps({"run_id": "r", "attempt_id": "a", "conversation_id": "c", "input": "hi"})
    code, events = _run_core(req, session_builder=lambda r: session)
    assert code == EXIT_OK
    assert [e.kind for e in events] == ["turn.started", "turn.completed"]
    assert [e.sequence for e in events] == [1, 2]
    assert events[-1].final_text == "done"


def test_run_bad_request_one_failed_no_started():
    code, events = _run_core('{"not":"a valid request"}')
    assert code == 2
    assert len(events) == 1
    assert events[0].kind == "turn.failed"
    assert events[0].sequence == 1
    assert events[0].error.code.value == "invalid_request"


# ── item 1: identifiers on pre-validation failures ───────────────────────────

def _bad_request_event(stdin_text):
    code, events = _run_core(stdin_text)
    assert code == 2
    assert len(events) == 1 and events[0].kind == "turn.failed" and events[0].sequence == 1
    return events[0]


def test_malformed_json_reports_null_ids():
    ev = _bad_request_event("{not json at all")
    assert ev.run_id is None and ev.attempt_id is None
    assert ev.error.code.value == "invalid_request"


def test_empty_stdin_reports_null_ids():
    ev = _bad_request_event("")
    assert ev.run_id is None and ev.attempt_id is None


def test_missing_run_id_keeps_attempt_id():
    ev = _bad_request_event(
        json.dumps({"attempt_id": "a", "conversation_id": "c", "input": "hi"})
    )
    assert ev.run_id is None
    assert ev.attempt_id == "a"  # echo the valid one, invent nothing


def test_missing_attempt_id_keeps_run_id():
    ev = _bad_request_event(
        json.dumps({"run_id": "r", "conversation_id": "c", "input": "hi"})
    )
    assert ev.run_id == "r"
    assert ev.attempt_id is None


def test_wrong_id_types_report_null():
    ev = _bad_request_event(
        json.dumps({"run_id": 123, "attempt_id": ["x"], "conversation_id": "c", "input": "hi"})
    )
    assert ev.run_id is None and ev.attempt_id is None


# ── item 2: the runner closes the session on every terminal path ─────────────

def test_session_closed_on_success():
    session = _FakeSession(generated=[{"role": "assistant", "content": "ok"}])
    asyncio.run(run_turn(_req(), lambda e: None, session_builder=lambda r: session))
    assert session.closed is True


def test_session_closed_on_deadline_timeout():
    class _Slow(_FakeSession):
        async def turn_stream(self, user_input, **kwargs):
            await asyncio.sleep(0.5)
            self.history.append({"role": "assistant", "content": "late"})
            yield object()

    slow = _Slow()
    code = asyncio.run(
        run_turn(_req(deadline_unix_ms=50), lambda e: None,
                 session_builder=lambda r: slow, now_ms=0)
    )
    assert code == EXIT_FAILED
    assert slow.closed is True  # closed even when the soft deadline fires
