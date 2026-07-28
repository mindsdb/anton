"""Entrypoint wire contract: request parsing + streaming event emission.

Offline: a fake session (scripted stream events) replaces ChatSession, so no LLM
key or scratchpad is needed. Mirrors the controller/cowork contract:
`delta` -> ... -> `turn_completed` | `turn_failed`.
"""

from __future__ import annotations

import asyncio
import json

from anton.cloud_turn.contract import TurnRequestV1
from anton.cloud_turn.__main__ import stream_turn
from anton.core.llm.provider import StreamTextDelta


# ── contract parsing ─────────────────────────────────────────────────────────

def test_from_json_parses_full_request():
    req = TurnRequestV1.from_json(json.dumps({
        "protocol_version": 1, "conversation_id": "c", "input": "hi",
        "workspace_path": "/workspace", "model": "m",
        "history": [{"role": "user", "content": "prev"}],
    }))
    assert req.conversation_id == "c"
    assert req.input == "hi"
    assert req.history == [{"role": "user", "content": "prev"}]


def test_from_json_history_defaults_empty():
    req = TurnRequestV1.from_json('{"protocol_version":1,"conversation_id":"c","input":"hi"}')
    assert req.history == []
    assert req.workspace_path is None and req.model is None


# ── streaming event emission ─────────────────────────────────────────────────

class _FakeSession:
    def __init__(self, deltas=(), raise_on_stream=None):
        self._deltas = list(deltas)
        self._raise = raise_on_stream
        self.closed = False

    async def turn_stream(self, user_input, **kwargs):
        if self._raise:
            raise self._raise
        for d in self._deltas:
            yield StreamTextDelta(text=d)

    def close(self):
        self.closed = True


def _drive(session, req_json='{"protocol_version":1,"conversation_id":"c","input":"hi"}'):
    events = []
    asyncio.run(stream_turn(req_json, events.append, session_builder=lambda r: session))
    return events


def test_emits_deltas_then_completed():
    session = _FakeSession(deltas=["he", "llo"])
    events = _drive(session)
    assert events == [
        {"kind": "delta", "text": "he"},
        {"kind": "delta", "text": "llo"},
        {"kind": "turn_completed"},
    ]
    assert session.closed is True  # session closed on success


def test_text_only_no_deltas_still_completes():
    events = _drive(_FakeSession(deltas=[]))
    assert events == [{"kind": "turn_completed"}]


def test_turn_failure_is_terminal_and_scrubbed():
    session = _FakeSession(raise_on_stream=RuntimeError("boom sk-ant-" + "A" * 80))
    events = _drive(session)
    assert len(events) == 1
    assert events[0]["kind"] == "turn_failed"
    assert "boom" in events[0]["error"]
    assert "sk-ant-" + "A" * 80 not in events[0]["error"]  # credential scrubbed
    assert session.closed is True  # closed even on failure


def test_bad_request_is_a_single_turn_failed():
    events = []
    asyncio.run(stream_turn("{not valid json", events.append))
    assert len(events) == 1
    assert events[0]["kind"] == "turn_failed"
    assert events[0]["error"]  # non-empty, scrubbed
