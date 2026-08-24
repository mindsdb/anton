"""Heartbeat ticker in `stream_turn`: emits `{"kind": "heartbeat"}` on a fixed
interval during a slow turn, without disturbing the delta/terminal events."""

from __future__ import annotations

import asyncio

import pytest

import anton.cloud_turn.__main__ as m


@pytest.mark.asyncio
async def test_stream_turn_emits_heartbeat_during_slow_turn(monkeypatch):
    from anton.core.llm.provider import StreamTextDelta

    class SlowSession:
        # `**kwargs` as every other fake in tests/ does: the entrypoint passes
        # trace_metadata (ENG-1459) and may pass more later.
        async def turn_stream(self, user_input, **kwargs):
            await asyncio.sleep(0.25)  # quiet period longer than the heartbeat interval
            yield StreamTextDelta(text="done")

        def close(self): ...

    monkeypatch.setenv("ANTON_CLOUD_TURN_HEARTBEAT_SECONDS", "0.05")
    events = []
    raw = '{"protocol_version":1,"conversation_id":"c","input":"hi"}'
    await m.stream_turn(raw, emit=events.append, session_builder=lambda req: SlowSession())

    kinds = [e["kind"] for e in events]
    assert "heartbeat" in kinds
    assert kinds[-1] == "turn_completed"
    assert {"kind": "delta", "text": "done"} in events
