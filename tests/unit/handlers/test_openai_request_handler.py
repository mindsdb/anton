from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

import pytest
from mindsdb_sdk.server import Server
from sqlmodel import Session

from minds.handlers.openai_request_handler import OpenAIRequestHandler
from minds.model.mind_datasource import DataCatalogStatus
from minds.requests.context import Context
from minds.requests.stream import MessageStreamer
from minds.schemas.chat import Message, Role


@pytest.fixture
def mock_session():
    return Mock(spec=Session)


@pytest.fixture
def mock_context():
    return Context(
        user_id=UUID("00000000-0000-0000-0000-000000000001"),
        organization_id=UUID("00000000-0000-0000-0000-000000000002"),
    )


@pytest.fixture
def mock_mindsdb_client():
    return Mock(spec=Server)


@pytest.fixture
def sample_messages():
    return [Message(role=Role.user, content="Hello")]


@pytest.mark.asyncio
async def test_create_sets_agent_and_mind_ready_true(mock_session, mock_context, mock_mindsdb_client, sample_messages):
    with (
        patch("minds.handlers.openai_request_handler.MindsService") as mock_minds_service_cls,
        patch("minds.handlers.openai_request_handler.AgentController") as mock_agent_controller_cls,
    ):
        # Mind fixture returned by MindsService
        mind = Mock()
        mind.parameters = {}
        rel = Mock()
        rel.status = AsyncMock(return_value=DataCatalogStatus.COMPLETED)()
        mind.mind_datasources = [rel]

        mock_minds_service = Mock()
        mock_minds_service.get_mind_model = AsyncMock(return_value=mind)
        mock_minds_service_cls.return_value = mock_minds_service

        # Agent fixture returned by AgentController
        mock_agent = Mock()
        mock_agent_controller = Mock()
        mock_agent_controller.get_agent.return_value = mock_agent
        mock_agent_controller_cls.return_value = mock_agent_controller

        handler = await OpenAIRequestHandler.create(
            session=mock_session,
            context=mock_context,
            mindsdb_client=mock_mindsdb_client,
            messages=sample_messages,
            model="test-model",
            stream=False,
            metadata=None,
            instrument=True,
        )

        assert handler.mind_ready is True
        assert handler.agent == mock_agent


@pytest.mark.asyncio
async def test_create_sets_mind_ready_false_when_catalog_loading(
    mock_session, mock_context, mock_mindsdb_client, sample_messages
):
    with (
        patch("minds.handlers.openai_request_handler.MindsService") as mock_minds_service_cls,
        patch("minds.handlers.openai_request_handler.AgentController") as mock_agent_controller_cls,
    ):
        mind = Mock()
        mind.parameters = {}
        rel = Mock()
        rel.status = AsyncMock(return_value=DataCatalogStatus.LOADING)()
        mind.mind_datasources = [rel]

        mock_minds_service = Mock()
        mock_minds_service.get_mind_model = AsyncMock(return_value=mind)
        mock_minds_service_cls.return_value = mock_minds_service

        mock_agent_controller = Mock()
        mock_agent_controller.get_agent.return_value = Mock()
        mock_agent_controller_cls.return_value = mock_agent_controller

        handler = await OpenAIRequestHandler.create(
            session=mock_session,
            context=mock_context,
            mindsdb_client=mock_mindsdb_client,
            messages=sample_messages,
            model="test-model",
            stream=False,
            metadata=None,
            instrument=True,
        )

        assert handler.mind_ready is False


@pytest.mark.asyncio
async def test_chat_completions_returns_early_when_mind_not_ready(
    mock_session, mock_context, mock_mindsdb_client, sample_messages
):
    handler = OpenAIRequestHandler(
        session=mock_session,
        context=mock_context,
        mindsdb_client=mock_mindsdb_client,
        messages=sample_messages,
        model="test-model",
        stream=False,
        metadata=None,
        instrument=True,
    )
    handler.mind_ready = False
    handler.agent = Mock()

    streamer = Mock(spec=MessageStreamer)
    streamer.push = AsyncMock()

    await handler.chat_completions(streamer)

    streamer.push.assert_awaited_once_with(
        role=Role.assistant,
        content="The Mind is not ready yet. Please try again later.",
    )


@pytest.mark.asyncio
async def test_chat_completions_calls_agent_run_when_ready(
    mock_session, mock_context, mock_mindsdb_client, sample_messages
):
    handler = OpenAIRequestHandler(
        session=mock_session,
        context=mock_context,
        mindsdb_client=mock_mindsdb_client,
        messages=sample_messages,
        model="test-model",
        stream=True,
        metadata=None,
        instrument=True,
    )
    handler.mind_ready = True
    handler.agent = Mock()
    handler.agent.run = AsyncMock(return_value=None)
    handler.agent.get_last_run_usage = AsyncMock(return_value=(11, 7))

    streamer = Mock(spec=MessageStreamer)
    streamer.push = AsyncMock()

    await handler.chat_completions(streamer)

    handler.agent.run.assert_awaited_once_with(messages=sample_messages, streamer=streamer, stream=True)


@pytest.mark.asyncio
async def test_responses_updates_conversation_message(mock_session, mock_context, mock_mindsdb_client, sample_messages):
    handler = OpenAIRequestHandler(
        session=mock_session,
        context=mock_context,
        mindsdb_client=mock_mindsdb_client,
        messages=sample_messages,
        model="test-model",
        stream=False,
        metadata=None,
        instrument=True,
    )
    handler.mind_ready = True

    # Agent returns an object with answer and sql
    agent_response = Mock()
    agent_response.answer = "final answer"
    agent_response.sql = "SELECT 1"
    handler.agent = Mock()
    handler.agent.run = AsyncMock(return_value=agent_response)
    handler.agent.get_last_run_usage = AsyncMock(return_value=(5, 3))

    streamer = Mock(spec=MessageStreamer)
    streamer.push = AsyncMock()

    message = Mock()

    with patch("minds.handlers.openai_request_handler.ConversationsService") as mock_conv_service_cls:
        conv_service = Mock()
        conv_service.update_conversation_message_content = AsyncMock()
        mock_conv_service_cls.return_value = conv_service

        await handler.responses(streamer=streamer, message=message)

        conv_service.update_conversation_message_content.assert_awaited_once()
        kwargs = conv_service.update_conversation_message_content.call_args.kwargs
        assert kwargs["message"] == message
        assert kwargs["content"] == "final answer"
        assert kwargs["sql_query"] == "SELECT 1"
