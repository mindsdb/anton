from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest


@pytest.mark.asyncio
async def test_anton_agent_run_builds_context_and_streams(monkeypatch, tmp_path):
    from minds.agents.anton_agent.agent import AntonAgent
    from minds.agents.anton_agent.anton.llm.provider import StreamComplete, StreamTextDelta
    from minds.agents.base import AgentRunContext
    from minds.schemas.chat import Message, Role

    class DummyAnton:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False

        async def chat_stream(self, prompt):
            yield StreamTextDelta(text="A")
            yield StreamComplete(
                response=SimpleNamespace(content="A", tool_calls=[], usage=SimpleNamespace(context_pressure=0))
            )

        async def close(self):
            self.closed = True

    monkeypatch.setattr("minds.agents.anton_agent.agent.Anton", DummyAnton)

    monkeypatch.setattr(
        "minds.agents.anton_agent.agent.agent_settings",
        SimpleNamespace(
            root_workspace_dir=str(tmp_path),
            output_dir="out",
            output_file_name="report.html",
            minds_internal_url="http://minds",
            default_anthropic_planning_model="pA",
            default_anthropic_coding_model="cA",
            default_openai_planning_model="pO",
            default_openai_coding_model="cO",
            backend="docker",
            scratchpad_persist_session=True,
            scratchpad_session_path="/anton_scratchpad_session.pkl",
        ),
    )
    monkeypatch.setattr("minds.agents.anton_agent.agent.mind_layer", lambda _mind: "SYS")

    app_settings = SimpleNamespace(
        default_models=SimpleNamespace(
            default_provider="openai",
            default_coding_provider="openai",
            openai_model="pO",
            openai_coding_model="cO",
            anthropic_model="pA",
            anthropic_coding_model="cA",
        ),
        openai=SimpleNamespace(api_key="okey"),
        anthropic=SimpleNamespace(api_key="akey"),
    )
    monkeypatch.setattr("minds.agents.anton_agent.agent.get_app_settings", lambda: app_settings)

    mind = Mock()
    mind.id = "m1"
    mind.name = "mind"
    mind.provider = "openai"
    mind.model_name = None
    mind.parameters = {}
    mind.organization_id = UUID("00000000-0000-0000-0000-000000000002")
    mind.user_id = UUID("00000000-0000-0000-0000-000000000001")

    ds = Mock()
    ds.name = "db"
    ds.engine = "postgres"
    rel = Mock()
    rel.datasource = ds
    mind.mind_datasources = [rel]

    streamer = Mock()
    streamer.push = AsyncMock()

    agent = AntonAgent(mind=mind, mindsdb_client=Mock())
    run_ctx = AgentRunContext(
        metadata=None,
        instrument=True,
        conversation_id=UUID("00000000-0000-0000-0000-0000000000aa"),
        message_id=UUID("00000000-0000-0000-0000-0000000000bb"),
    )
    messages = [Message(role=Role.user, content="hi")]
    resp = await agent.run(messages=messages, streamer=streamer, stream=True, run_context=run_ctx)

    pushed = [c.kwargs["content"] for c in streamer.push.await_args_list]
    assert any(p == "A" for p in pushed)
    assert resp.answer == "A"
