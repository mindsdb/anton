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
    # Keep this test fully offline/deterministic.
    from minds.agents.anton_agent.agent import _DEFAULT_CLASSIFICATION

    monkeypatch.setattr(
        "minds.agents.anton_agent.agent.classify_query",
        AsyncMock(return_value=_DEFAULT_CLASSIFICATION),
    )

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

    mindsdb_client = Mock()
    mindsdb_client.databases.get.return_value.params = {}
    agent = AntonAgent(mind=mind, mindsdb_client=mindsdb_client)
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "db_params",
    [
        {"schema": "public"},
        '{"schema": "public"}',
    ],
)
async def test_anton_agent_includes_schema_in_prompt_and_env(monkeypatch, tmp_path, db_params):
    import json

    from minds.agents.anton_agent.agent import _DEFAULT_CLASSIFICATION, AntonAgent
    from minds.agents.anton_agent.anton.llm.provider import StreamComplete, StreamTextDelta
    from minds.agents.base import AgentRunContext
    from minds.schemas.chat import Message, Role

    class DummyAnton:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def chat_stream(self, _prompt):
            yield StreamTextDelta(text="A")
            yield StreamComplete(
                response=SimpleNamespace(content="A", tool_calls=[], usage=SimpleNamespace(context_pressure=0))
            )

        async def close(self):
            return None

    monkeypatch.setattr("minds.agents.anton_agent.agent.Anton", DummyAnton)
    monkeypatch.setattr(
        "minds.agents.anton_agent.agent.classify_query", AsyncMock(return_value=_DEFAULT_CLASSIFICATION)
    )
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
    monkeypatch.setattr("minds.agents.anton_agent.agent.mind_layer", lambda _mind: "")

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

    mindsdb_client = Mock()
    mindsdb_client.databases.get.return_value.params = db_params

    agent = AntonAgent(mind=mind, mindsdb_client=mindsdb_client)
    run_ctx = AgentRunContext(
        metadata=None,
        instrument=True,
        conversation_id=UUID("00000000-0000-0000-0000-0000000000aa"),
        message_id=UUID("00000000-0000-0000-0000-0000000000bb"),
    )
    messages = [Message(role=Role.user, content="hi")]
    resp = await agent.run(messages=messages, streamer=streamer, stream=True, run_context=run_ctx)
    assert resp.answer == "A"

    anton_kwargs = agent.agent.kwargs

    # Schema should be mentioned in the datasource list and schema restriction rules should be injected.
    runtime_context = anton_kwargs["runtime_context"]
    assert "Name: db, Engine: postgres, Schema: public" in runtime_context
    assert "If the schema for a data source has been provided, you are only allowed" in runtime_context

    # Extra env should include schema in the serialized datasource list.
    extra_env = anton_kwargs["extra_env"]
    assert extra_env["ANTON_MINDS_DATASOURCE"] == "db"
    ds_list = json.loads(extra_env["ANTON_MINDS_DATASOURCES_JSON"])
    assert ds_list == [{"name": "db", "engine": "postgres", "schema": "public"}]
