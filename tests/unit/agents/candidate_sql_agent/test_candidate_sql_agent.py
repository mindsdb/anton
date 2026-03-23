"""
Unit tests for CandidateSQLAgent orchestrator.

These tests mirror the database-agent unit test style:
- pytest + AsyncMock
- patch global pydantic-ai Agents to avoid network/model calls
- validate branch behavior (router feedback vs SQL handoff vs exception fallback)
"""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

# Mock langfuse before importing any modules that may instrument.
if "langfuse" not in sys.modules:
    mock_langfuse = Mock()
    mock_langfuse.observe = lambda f=None, **_: (lambda *a, **k: f(*a, **k)) if f else (lambda x: x)
    sys.modules["langfuse"] = mock_langfuse

import pytest

from minds.agents.base import AgentRunContext
from minds.agents.base_response import AgentResponse
from minds.agents.candidate_sql_agent.agent import CandidateSQLAgent
from minds.schemas.chat import Message, Role


class TestCandidateSQLAgentConvertMessages:
    def test_convert_to_pydantic_ai_messages_user_and_assistant_and_system(self):
        mind = Mock()
        agent = CandidateSQLAgent(mind=mind, mindsdb_client=Mock())

        messages = [
            Message(role=Role.system, content="sys"),
            Message(role=Role.user, content="u1"),
            Message(role=Role.assistant, content="a1"),
            Message(role=Role.user, content="u2"),
        ]

        converted = agent._convert_to_pydantic_ai_messages(messages)

        # We don't assert exact pydantic-ai classes/parts (implementation detail),
        # but we do assert that message count is preserved and no exceptions occur.
        assert len(converted) == 4

    def test_convert_to_pydantic_ai_messages_unsupported_role_raises(self):
        mind = Mock()
        agent = CandidateSQLAgent(mind=mind, mindsdb_client=Mock())

        messages = [Message(role=Role.function, content="not supported here")]
        with pytest.raises(ValueError, match="Unsupported role"):
            agent._convert_to_pydantic_ai_messages(messages)


class TestCandidateSQLAgentRun:
    @pytest.fixture
    def mind(self):
        mind = Mock()
        mind.name = "test-mind"
        return mind

    @pytest.fixture
    def streamer(self):
        s = Mock()
        s.push = AsyncMock()
        return s

    @pytest.mark.asyncio
    async def test_run_router_feedback_path_pushes_feedback_and_returns_response(self, mind, streamer):
        messages = [Message(role=Role.user, content="hello"), Message(role=Role.user, content="last")]

        router_result = SimpleNamespace(output=SimpleNamespace(handoff=False, feedback="Need more details."))

        with (
            patch("minds.agents.candidate_sql_agent.agent.is_native_query_mode_enabled", return_value=False),
            patch("minds.agents.candidate_sql_agent.agent.model_for", return_value=Mock()),
            patch("minds.agents.candidate_sql_agent.agent.lightweight_router_agent") as mock_router_agent,
        ):
            mock_router_agent.run = AsyncMock(return_value=router_result)

            agent = CandidateSQLAgent(mind=mind, mindsdb_client=Mock())
            resp = await agent.run(
                messages=list(messages),
                streamer=streamer,
                stream=False,
                run_context=AgentRunContext(instrument=False),
            )

            streamer.push.assert_awaited_once_with(role=Role.assistant, content="Need more details.")
            assert isinstance(resp, AgentResponse)
            assert resp.sql == ""
            assert resp.answer == "Need more details."

    @pytest.mark.asyncio
    async def test_run_handoff_path_runs_pipeline_streams_feedback_and_appends_execution_result(self, mind, streamer):
        messages = [Message(role=Role.user, content="prior"), Message(role=Role.user, content="Question?")]

        router_result = SimpleNamespace(output=SimpleNamespace(handoff=True, feedback=None))
        tool_result = SimpleNamespace(final_query="SELECT 1", execution_result="| x |\n|---|\n| 1 |")

        # Stream chunks: feedback grows, next_steps grows; agent should push deltas only.
        chunks = [
            SimpleNamespace(feedback="F", next_steps=""),
            SimpleNamespace(feedback="FB", next_steps="N"),
            SimpleNamespace(feedback="FB", next_steps="NS"),
        ]

        class MockAsyncContextManager:
            async def __aenter__(self):
                return SimpleNamespace(
                    stream_output=_stream_output,
                )

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return False

        async def _stream_output():
            for c in chunks:
                yield c

        with (
            patch("minds.agents.candidate_sql_agent.agent.is_native_query_mode_enabled", return_value=False),
            patch("minds.agents.candidate_sql_agent.agent.model_for", return_value=Mock()),
            patch("minds.agents.candidate_sql_agent.agent.lightweight_router_agent") as mock_router_agent,
            patch("minds.agents.candidate_sql_agent.agent.TextToSQLPipeline") as mock_pipeline_cls,
            patch("minds.agents.candidate_sql_agent.agent.answer_feedback_agent") as mock_answer_feedback_agent,
        ):
            mock_router_agent.run = AsyncMock(return_value=router_result)
            mock_pipeline = Mock()
            mock_pipeline.run = AsyncMock(return_value=tool_result)
            mock_pipeline_cls.return_value = mock_pipeline
            mock_answer_feedback_agent.run_stream = Mock(return_value=MockAsyncContextManager())

            agent = CandidateSQLAgent(mind=mind, mindsdb_client=Mock())
            resp = await agent.run(
                messages=list(messages),
                streamer=streamer,
                stream=True,
                run_context=AgentRunContext(instrument=False),
            )

            # Expected pushed deltas:
            # - feedback deltas: "F", then "B" (from "FB")
            # - then newline between feedback and first next step, then next step deltas: "N", then "S"
            # - then "\n\n" and the execution_result at the end
            pushed = [c.kwargs["content"] for c in streamer.push.await_args_list]
            assert "F" in pushed
            assert "B" in pushed
            assert "N" in pushed
            assert "S" in pushed
            assert tool_result.execution_result in pushed
            assert resp.sql == "SELECT 1"
            assert tool_result.execution_result in resp.answer

    @pytest.mark.asyncio
    async def test_run_exception_falls_back_to_feedback_agent(self, mind, streamer):
        messages = [Message(role=Role.user, content="prior"), Message(role=Role.user, content="Question?")]

        router_result = SimpleNamespace(output=SimpleNamespace(handoff=True, feedback=None))
        feedback_result = SimpleNamespace(output="Sorry, something went wrong.")

        with (
            patch("minds.agents.candidate_sql_agent.agent.is_native_query_mode_enabled", return_value=False),
            patch("minds.agents.candidate_sql_agent.agent.model_for", return_value=Mock()),
            patch("minds.agents.candidate_sql_agent.agent.lightweight_router_agent") as mock_router_agent,
            patch("minds.agents.candidate_sql_agent.agent.TextToSQLPipeline") as mock_pipeline_cls,
            patch("minds.agents.candidate_sql_agent.agent.feedback_agent") as mock_feedback_agent,
        ):
            mock_router_agent.run = AsyncMock(return_value=router_result)
            mock_pipeline = Mock()
            mock_pipeline.run = AsyncMock(side_effect=RuntimeError("boom"))
            mock_pipeline_cls.return_value = mock_pipeline
            mock_feedback_agent.run = AsyncMock(return_value=feedback_result)

            agent = CandidateSQLAgent(mind=mind, mindsdb_client=Mock())
            resp = await agent.run(
                messages=list(messages),
                streamer=streamer,
                stream=False,
                run_context=AgentRunContext(instrument=False),
            )

            streamer.push.assert_awaited_with(role=Role.assistant, content="Sorry, something went wrong.")
            assert resp.sql == ""
            assert resp.answer == "Sorry, something went wrong."
