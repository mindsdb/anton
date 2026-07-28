import json
import pytest
from anton.cloud_turn.__main__ import run_turn
from anton.cloud_turn.contract import TurnRequestV1


@pytest.mark.asyncio
async def test_run_turn_returns_completed(monkeypatch):
    class FakeSession:
        async def turn(self, user_input):
            return f"echo: {user_input}"

    async def fake_build(**kwargs):
        assert kwargs["session_id"] == "conv-1"
        return FakeSession()

    monkeypatch.setattr("anton.cloud_turn.__main__.build_chat_session", fake_build)
    req = TurnRequestV1(protocol_version=1, conversation_id="conv-1", input="hi")
    result = await run_turn(req)
    assert result.kind == "turn_completed"
    assert result.final_text == "echo: hi"
