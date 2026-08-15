"""Entrypoint wire contract: request parsing + streaming event emission.

Offline: a fake session (scripted stream events) replaces ChatSession, so no LLM
key or scratchpad is needed. Mirrors the controller/cowork contract:
`delta` -> ... -> `turn_completed` | `turn_failed`.
"""

from __future__ import annotations

import asyncio
import json
import logging

from anton.cloud_turn.contract import TurnRequestV1
from anton.cloud_turn.__main__ import _clip_result_content, stream_turn
from anton.core.llm.provider import (
    LLMResponse,
    StreamComplete,
    StreamContextCompacted,
    StreamTaskProgress,
    StreamTextDelta,
    StreamToolResult,
    StreamToolUseDelta,
    StreamToolUseEnd,
    StreamToolUseStart,
)


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


def test_from_json_passes_through_llm_block():
    """The per-turn MindsHub credential block survives the wire round-trip."""
    req = TurnRequestV1.from_json(json.dumps({
        "protocol_version": 1, "conversation_id": "c", "input": "hi",
        "llm": {"provider": "minds-cloud", "api_key": "mdb_turnkey",
                "base_url": "https://api.mindshub.ai/v1"},
    }))
    assert req.llm == {
        "provider": "minds-cloud", "api_key": "mdb_turnkey",
        "base_url": "https://api.mindshub.ai/v1",
    }


def test_from_json_llm_defaults_none():
    req = TurnRequestV1.from_json('{"protocol_version":1,"conversation_id":"c","input":"hi"}')
    assert req.llm is None


# ── streaming event emission ─────────────────────────────────────────────────

class _FakeSession:
    """Stands in for ChatSession, including its memory-write tracking contract."""

    def __init__(self, deltas=(), raise_on_stream=None):
        self._deltas = list(deltas)
        self._raise = raise_on_stream
        self.closed = False
        self._memory_writes = set()

    async def turn_stream(self, user_input, **kwargs):
        if self._raise:
            raise self._raise
        for d in self._deltas:
            yield StreamTextDelta(text=d)

    def _track_memory_write(self, task):
        self._memory_writes.add(task)
        task.add_done_callback(self._memory_writes.discard)

    async def settle_memory_writes(self):
        if self._memory_writes:
            await asyncio.gather(*tuple(self._memory_writes), return_exceptions=True)

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


def test_step_events_go_on_the_wire():
    """Tool/progress/round events are emitted as structured wire events so the
    cowork step UI can render them; tool args accumulate pod-side and ship once
    on tool_end."""
    class _S(_FakeSession):
        async def turn_stream(self, user_input, **kwargs):
            yield StreamToolUseStart(id="t1", name="scratchpad")
            yield StreamToolUseDelta(id="t1", json_delta='{"code":')
            yield StreamToolUseDelta(id="t1", json_delta='"1+1"}')
            yield StreamToolUseEnd(id="t1")
            yield StreamTaskProgress(phase="scratchpad_done", message="ran",
                                     eta_seconds=1.5, id="t1", ok=True)
            yield StreamToolResult(name="scratchpad", content="2", action="exec", id="t1")
            yield StreamContextCompacted(message="squeezed")
            yield StreamComplete(response=LLMResponse(content="", stop_reason="end_turn"))
            yield StreamTextDelta(text="done")

    events = _drive(_S())
    assert {"kind": "tool_start", "id": "t1", "name": "scratchpad"} in events
    assert {"kind": "tool_end", "id": "t1", "args": '{"code":"1+1"}'} in events
    assert {"kind": "progress", "phase": "scratchpad_done", "message": "ran",
            "eta_seconds": 1.5, "id": "t1", "ok": True} in events
    assert {"kind": "tool_result", "id": "t1", "name": "scratchpad",
            "action": "exec", "content": "2"} in events
    assert {"kind": "compacted", "message": "squeezed"} in events
    assert {"kind": "round_end", "stop_reason": "end_turn",
            "had_tool_calls": False} in events
    assert events[-1] == {"kind": "turn_completed"}


def test_result_clip_preserves_error_verdict():
    """A long failed cell must still classify as failed downstream: the clip
    shrinks bulky fields but keeps the JSON and its error field intact."""
    cell = {"code": "x", "stdout": "s" * 70_000, "error": "boom"}
    clipped = _clip_result_content(json.dumps(cell))
    assert len(clipped) <= 65536
    parsed = json.loads(clipped)
    assert parsed["error"] == "boom"
    assert "truncated" in parsed["stdout"]


def test_result_clip_non_json_keeps_tail():
    text = "a" * 70_000 + "TAIL-ERROR"
    clipped = _clip_result_content(text)
    assert len(clipped) <= 65536
    assert clipped.endswith("TAIL-ERROR")


def test_progress_flood_is_rate_limited_but_step_phases_pass():
    class _S(_FakeSession):
        async def turn_stream(self, user_input, **kwargs):
            yield StreamTaskProgress(phase="scratchpad_start", message="s", id="t1")
            for i in range(50):
                yield StreamTaskProgress(phase="progress", message=f"m{i}")
            yield StreamTaskProgress(phase="scratchpad_done", message="d", id="t1")

    events = _drive(_S())
    progress = [e for e in events if e.get("kind") == "progress"]
    phases = [e["phase"] for e in progress]
    assert "scratchpad_start" in phases and "scratchpad_done" in phases
    assert len(progress) < 10  # the 50-call flood collapses on the wire


def test_tool_args_accumulation_is_bounded():
    class _S(_FakeSession):
        async def turn_stream(self, user_input, **kwargs):
            yield StreamToolUseStart(id="t1", name="scratchpad")
            for _ in range(20):
                yield StreamToolUseDelta(id="t1", json_delta="x" * 10_000)
            yield StreamToolUseEnd(id="t1")

    events = _drive(_S())
    end = next(e for e in events if e.get("kind") == "tool_end")
    assert len(end["args"]) <= 65536


def test_action_events_are_logged(caplog):
    """Discrete turn actions (tool calls, progress, results) are narrated to the
    logs so the pod's activity is observable in the controller, in addition to
    the structured wire events; text deltas are NOT logged (emitted only)."""
    class _S(_FakeSession):
        async def turn_stream(self, user_input, **kwargs):
            yield StreamToolUseStart(id="t1", name="scratchpad")
            yield StreamTaskProgress(phase="scratchpad_start", message="running cell")
            yield StreamToolResult(name="scratchpad", content="x" * 20, action="exec", id="t1")
            yield StreamTextDelta(text="hi")

    with caplog.at_level(logging.INFO):
        events = _drive(_S())

    assert {"kind": "delta", "text": "hi"} in events
    assert {"kind": "turn_completed"} in events
    assert "tool call: scratchpad" in caplog.text
    assert "progress [scratchpad_start]: running cell" in caplog.text
    assert "tool result: scratchpad action=exec (20 chars)" in caplog.text
    assert "cloud turn completed" in caplog.text
    # the answer text is emitted, never narrated into the logs
    assert "hi" not in caplog.text


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


# ── memory events (pod reports what to remember) ─────────────────────────────

def test_from_json_passes_through_memory_block():
    req = TurnRequestV1.from_json(json.dumps({
        "protocol_version": 1, "conversation_id": "c", "input": "hi",
        "memory": {"global": {"rules": "## Always\n- be terse"}},
    }))
    assert req.memory == {"global": {"rules": "## Always\n- be terse"}}


def test_from_json_memory_defaults_none():
    req = TurnRequestV1.from_json('{"protocol_version":1,"conversation_id":"c","input":"hi"}')
    assert req.memory is None


def test_from_json_passes_through_skills_block():
    skills = {"csv-summary": {"files": {"SKILL.md": "---\nname: csv-summary\n---\nbody"}}}
    req = TurnRequestV1.from_json(json.dumps({
        "protocol_version": 1, "conversation_id": "c", "input": "hi",
        "skills": skills,
    }))
    assert req.skills == skills


def test_from_json_skills_defaults_none():
    req = TurnRequestV1.from_json('{"protocol_version":1,"conversation_id":"c","input":"hi"}')
    assert req.skills is None


# ── memory write-back events ─────────────────────────────────────────────────

class _MemorySession(_FakeSession):
    """Session whose cortex has captured engrams, as `memorize` would leave it."""

    def __init__(self, pending, deltas=(), encode_late=False, track=True):
        super().__init__(deltas=deltas)
        self._cortex = type("_C", (), {"pending_memory": list(pending)})()
        self._encode_late = encode_late
        self._track = track

    async def turn_stream(self, user_input, **kwargs):
        if self._raise:
            raise self._raise
        for d in self._deltas:
            yield StreamTextDelta(text=d)
        if self._encode_late:
            # Mirror handle_memorize: fired as a task, registered, never awaited.
            async def _encode():
                for _ in range(5):        # several passes: settling must be awaited,
                    await asyncio.sleep(0)  # not land by scheduling luck
                self._cortex.pending_memory.append(_ENTRY)
            task = asyncio.get_running_loop().create_task(_encode())
            if self._track:
                self._track_memory_write(task)


_ENTRY = {"text": "Reply in Spanish", "kind": "always", "scope": "global",
          "topic": "", "confidence": "high", "source": "user"}


def test_memory_emitted_before_the_terminal_event():
    """cowork stops reading at the terminal reply, so memory must precede it."""
    events = _drive(_MemorySession([_ENTRY], deltas=["ok"]))
    assert events == [
        {"kind": "delta", "text": "ok"},
        {"kind": "memory", "entries": [_ENTRY]},
        {"kind": "turn_completed"},
    ]


def test_no_memory_event_when_nothing_was_remembered():
    events = _drive(_MemorySession([], deltas=["ok"]))
    assert events == [{"kind": "delta", "text": "ok"}, {"kind": "turn_completed"}]


def test_memory_from_a_late_encode_task_is_not_lost():
    """A memorize on the final round encodes via create_task; the process exits
    right after, so the entrypoint must let it settle first."""
    events = _drive(_MemorySession([], deltas=["ok"], encode_late=True))
    assert {"kind": "memory", "entries": [_ENTRY]} in events


def test_failed_turn_emits_no_memory():
    """A turn that raised has no trustworthy result; nothing is persisted."""
    session = _MemorySession([_ENTRY])
    session._raise = RuntimeError("boom")
    kinds = [e["kind"] for e in _drive(session)]
    assert kinds == ["turn_failed"]


def test_untracked_late_write_is_not_awaited():
    """Settling is explicit: only registered writes are awaited, so the entrypoint
    never blocks on unrelated live tasks (scratchpad readers, the heartbeat)."""
    events = _drive(_MemorySession([], deltas=["ok"], encode_late=True, track=False))
    assert events == [{"kind": "delta", "text": "ok"}, {"kind": "turn_completed"}]
