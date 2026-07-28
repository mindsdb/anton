import pytest
from anton.core.llm.provider import StreamTextDelta
from anton.cloud_turn.__main__ import stream_turn
from anton.cloud_turn.contract import TurnRequestV1


@pytest.mark.asyncio
async def test_stream_turn_emits_deltas_then_completed(monkeypatch):
    class FakeSession:
        async def turn_stream(self, user_input):
            yield StreamTextDelta(text="he")
            yield StreamTextDelta(text="llo")

    async def fake_build(**kwargs):
        assert kwargs["session_id"] == "conv-1"
        return FakeSession()

    monkeypatch.setattr("anton.cloud_turn.__main__.build_chat_session", fake_build)
    req = TurnRequestV1(protocol_version=1, conversation_id="conv-1", input="hi")
    events = [ev async for ev in stream_turn(req)]
    assert events == [
        {"kind": "delta", "text": "he"},
        {"kind": "delta", "text": "llo"},
        {"kind": "turn_completed"},
    ]


@pytest.mark.asyncio
async def test_stream_turn_failure_is_terminal(monkeypatch):
    async def fake_build(**kwargs):
        raise RuntimeError("boom")
    monkeypatch.setattr("anton.cloud_turn.__main__.build_chat_session", fake_build)
    req = TurnRequestV1(protocol_version=1, conversation_id="c", input="hi")
    events = [ev async for ev in stream_turn(req)]
    assert events[-1]["kind"] == "turn_failed"
    assert "boom" in events[-1]["error"]
