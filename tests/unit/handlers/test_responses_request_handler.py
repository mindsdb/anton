import importlib
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID, uuid4

import pytest
from sqlmodel import Session
from starlette.responses import JSONResponse, StreamingResponse

from minds.common.settings.app_settings import Agent
from minds.requests.responses_request import ResponsesRequest
from minds.schemas.chat import Message, Role
from minds.schemas.conversations import ConversationMetadata, ConversationResponse
from minds.schemas.messages import MessageContent, MessageContentType, MessageResponse


@pytest.fixture()
def handler_mod(monkeypatch):
    # Neutralize observe decorator before import
    import langfuse as dec

    monkeypatch.setattr(
        dec,
        "observe",
        lambda f=None, **_: (lambda *a, **k: f(*a, **k)) if f else (lambda x: x),
    )

    import minds.handlers.responses_request_handler as mod

    importlib.reload(mod)
    return mod


@pytest.fixture
def mock_session():
    """Mock SQLModel session."""
    sess = Mock(spec=Session)
    # Default: no explicit agent parameter stored in DB.
    row = Mock()
    row.parameters = {"agent": None}
    sess.exec.return_value.first.return_value = row
    return sess


@pytest.fixture
def mock_mindsdb_client():
    """Mock MindsDB client."""
    return Mock(spec=["connect"])


@pytest.fixture
def mock_context():
    """Mock Context."""
    from minds.requests.context import Context

    return Context(
        user_id=UUID("00000000-0000-0000-0000-000000000001"),
        organization_id=UUID("00000000-0000-0000-0000-000000000002"),
    )


@pytest.fixture
def mock_conversation_service():
    """Mock ConversationsService."""
    service = Mock()
    service.user_id = "test-user-123"
    service.organization_id = "test-organization-456"
    return service


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        Message(role=Role.user, content="Hello, how are you?"),
        Message(role=Role.assistant, content="I'm doing well, thank you!"),
    ]


@pytest.fixture
def sample_responses_request(sample_messages):
    """Sample ResponsesRequest for testing."""
    return ResponsesRequest(
        model="gpt-3.5-turbo",
        input="What is the weather?",
        conversation=None,
        stream=False,
    )


@pytest.fixture
def sample_streaming_responses_request(sample_messages):
    """Sample streaming ResponsesRequest for testing."""
    return ResponsesRequest(
        model="gpt-3.5-turbo",
        input="What is the weather?",
        conversation=None,
        stream=True,
    )


@pytest.fixture
def sample_conversation_response():
    """Sample ConversationResponse for testing."""
    return ConversationResponse(
        id=uuid4(),
        metadata=ConversationMetadata(topic="Test Conversation", model_name="gpt-3.5-turbo"),
        created_at="2023-01-01T12:00:00",
        modified_at="2023-01-01T12:00:00",
    )


@pytest.fixture
def sample_message_responses():
    """Sample MessageResponse list for testing."""
    return [
        MessageResponse(
            id=uuid4(),
            role=Role.user,
            content=MessageContent(type=MessageContentType.input_text, text="Hello"),
            created_at="2023-01-01T12:00:00",
            modified_at="2023-01-01T12:00:00",
        ),
        MessageResponse(
            id=uuid4(),
            role=Role.assistant,
            content=MessageContent(type=MessageContentType.output_text, text="Hi there!"),
            created_at="2023-01-01T12:00:01",
            modified_at="2023-01-01T12:00:01",
        ),
    ]


class TestResponsesRequestHandler:
    @pytest.fixture(autouse=True)
    def _stable_langfuse_capture(self, handler_mod):
        """Force capture_langfuse_generation_context to return None across this
        class so the trace context the request handler forwards to
        OpenAIRequestHandler.create is deterministic and not contaminated by
        whether the conftest-level Langfuse stub or the real (auth-disabled)
        Langfuse client got loaded first. Tests that specifically need to
        assert a non-None captured context should patch this themselves.
        """
        with patch.object(handler_mod, "capture_langfuse_generation_context", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_responses_request_handler_streaming(
        self,
        handler_mod,
        mock_session,
        mock_mindsdb_client,
        sample_streaming_responses_request,
        mock_context,
        mock_conversation_service,
        sample_conversation_response,
        sample_message_responses,
    ):
        """Test streaming responses request handling."""
        mock_streaming_response = Mock(spec=StreamingResponse)
        message_id = uuid4()

        with (
            patch.object(handler_mod, "OpenAIRequestHandler") as mock_handler_class,
            patch.object(handler_mod, "process_streaming_producer", new_callable=AsyncMock) as mock_process_streaming,
            patch.object(handler_mod.MindsService, "get_mind_model", new_callable=AsyncMock) as mock_get_mind_model,
        ):
            # Setup mocks
            mock_handler_instance = Mock()
            mock_handler_class.create = AsyncMock(return_value=mock_handler_instance)
            mock_handler_instance.responses = AsyncMock()
            mock_process_streaming.return_value = mock_streaming_response

            fake_mind = Mock()
            # Ensure streaming path does not switch to Anton-specific DB session/service.
            fake_mind.parameters = {"agent_name": "candidate_sql_agent"}
            mock_get_mind_model.return_value = fake_mind

            # Mock conversation service methods
            mock_conversation_service.create_conversation = AsyncMock(return_value=sample_conversation_response)
            mock_conversation_service.get_conversation_messages = AsyncMock(return_value=sample_message_responses)
            mock_message = Mock()
            mock_message.id = message_id
            mock_conversation_service.create_conversation_message_placeholder = AsyncMock(return_value=mock_message)
            mock_conversation_service.update_conversation_message_content = AsyncMock()

            # Call the handler
            result = await handler_mod.responses_request_handler(
                session=mock_session,
                context=mock_context,
                mindsdb_client=mock_mindsdb_client,
                responses_request=sample_streaming_responses_request,
                conversation_service=mock_conversation_service,
            )

            # Verify conversation was created
            mock_conversation_service.create_conversation.assert_called_once()
            create_call = mock_conversation_service.create_conversation.call_args[0][0]
            assert len(create_call.items) == 1
            assert create_call.items[0].role == Role.user
            assert create_call.items[0].content == "What is the weather?"

            # Verify messages were retrieved
            mock_conversation_service.get_conversation_messages.assert_called_once_with(sample_conversation_response.id)

            # Verify OpenAIRequestHandler was created with correct parameters
            mock_handler_class.create.assert_awaited_once_with(
                session=mock_session,
                context=mock_context,
                mindsdb_client=mock_mindsdb_client,
                messages=[Message(role=Role.user, content="Hello"), Message(role=Role.assistant, content="Hi there!")],
                model=sample_streaming_responses_request.model,
                stream=True,
                metadata=None,
                instrument=True,
                langfuse_trace_context=None,
                limits_service=None,
            )

            # Verify message placeholder was created
            mock_conversation_service.create_conversation_message_placeholder.assert_called_once_with(
                conversation_id=sample_conversation_response.id, role=Role.assistant
            )

            # Verify process_streaming_producer was called
            mock_process_streaming.assert_called_once()
            call_args = mock_process_streaming.call_args
            assert call_args[1]["model"] == sample_streaming_responses_request.model
            assert call_args[1]["message_id"] == message_id

            # Verify the producer function works
            producer_func = call_args[1]["producer"]
            mock_streamer = Mock()
            await producer_func(mock_streamer)
            mock_handler_instance.responses.assert_awaited_once_with(streamer=mock_streamer, message=mock_message)

            # Verify return value
            assert result == mock_streaming_response

    @pytest.mark.asyncio
    async def test_responses_request_handler_streaming_anton_stores_events_and_closes_owned_session(
        self,
        handler_mod,
        mock_session,
        mock_mindsdb_client,
        sample_streaming_responses_request,
        mock_context,
        sample_conversation_response,
        sample_message_responses,
    ):
        """Streaming + Anton should pass event_callback and use owned session."""
        mock_streaming_response = Mock(spec=StreamingResponse)
        message_id = uuid4()

        owned_session = Mock()
        owned_session.close = Mock()

        anton_service = Mock()
        anton_service.create_conversation = AsyncMock(return_value=sample_conversation_response)
        anton_service.get_conversation_messages = AsyncMock(return_value=sample_message_responses)
        mock_message = Mock()
        mock_message.id = message_id
        anton_service.create_conversation_message_placeholder = AsyncMock(return_value=mock_message)
        anton_service.update_conversation_message_content = AsyncMock()
        anton_service.create_conversation_message_event = AsyncMock()

        with (
            patch.object(handler_mod, "OpenAIRequestHandler") as mock_handler_class,
            patch.object(handler_mod, "process_streaming_producer", new_callable=AsyncMock) as mock_process_streaming,
            patch.object(handler_mod.MindsService, "get_mind_model", new_callable=AsyncMock) as mock_get_mind_model,
            patch.object(handler_mod, "get_open_session", return_value=owned_session) as mock_get_open_session,
            patch.object(handler_mod, "ConversationsService", return_value=anton_service) as mock_cs_ctor,
        ):
            mock_handler_instance = Mock()
            mock_handler_class.create = AsyncMock(return_value=mock_handler_instance)
            mock_handler_instance.responses = AsyncMock()
            mock_process_streaming.return_value = mock_streaming_response

            fake_mind = Mock()
            fake_mind.parameters = {"agent_name": Agent.ANTON.value}
            mock_get_mind_model.return_value = fake_mind

            result = await handler_mod.responses_request_handler(
                session=mock_session,
                context=mock_context,
                mindsdb_client=mock_mindsdb_client,
                responses_request=sample_streaming_responses_request,
                conversation_service=Mock(),
            )

            # Owned session created and used to build a new ConversationsService.
            mock_get_open_session.assert_called_once()
            mock_cs_ctor.assert_called_once_with(
                session=owned_session,
                mindsdb_client=mock_mindsdb_client,
                user_id=mock_context.user_id,
                organization_id=mock_context.organization_id,
            )

            # process_streaming_producer should receive event_callback for persistence.
            mock_process_streaming.assert_called_once()
            call_kwargs = mock_process_streaming.call_args.kwargs
            assert call_kwargs["event_callback"] == anton_service.create_conversation_message_event

            # Response should close owned session in background.
            assert result == mock_streaming_response
            assert result.background is not None
            assert result.background.func == owned_session.close

    @pytest.mark.asyncio
    async def test_responses_request_handler_non_streaming(
        self,
        handler_mod,
        mock_session,
        mock_mindsdb_client,
        sample_responses_request,
        mock_context,
        mock_conversation_service,
        sample_conversation_response,
        sample_message_responses,
    ):
        """Test non-streaming responses request handling."""
        mock_json_response = Mock(spec=JSONResponse)
        message_id = uuid4()

        with (
            patch.object(handler_mod, "OpenAIRequestHandler") as mock_handler_class,
            patch.object(
                handler_mod, "process_non_streaming_producer", new_callable=AsyncMock
            ) as mock_process_non_streaming,
        ):
            # Setup mocks
            mock_handler_instance = Mock()
            mock_handler_class.create = AsyncMock(return_value=mock_handler_instance)
            mock_handler_instance.responses = AsyncMock()
            mock_process_non_streaming.return_value = mock_json_response

            # Mock conversation service methods
            mock_conversation_service.create_conversation = AsyncMock(return_value=sample_conversation_response)
            mock_conversation_service.get_conversation_messages = AsyncMock(return_value=sample_message_responses)
            mock_message = Mock()
            mock_message.id = message_id
            mock_conversation_service.create_conversation_message_placeholder = AsyncMock(return_value=mock_message)
            mock_conversation_service.update_conversation_message_content = AsyncMock()

            # Call the handler
            result = await handler_mod.responses_request_handler(
                session=mock_session,
                context=mock_context,
                mindsdb_client=mock_mindsdb_client,
                responses_request=sample_responses_request,
                conversation_service=mock_conversation_service,
            )

            # Verify conversation was created
            mock_conversation_service.create_conversation.assert_called_once()

            # Verify messages were retrieved
            mock_conversation_service.get_conversation_messages.assert_called_once_with(sample_conversation_response.id)

            # Verify OpenAIRequestHandler was created with correct parameters
            mock_handler_class.create.assert_awaited_once_with(
                session=mock_session,
                context=mock_context,
                mindsdb_client=mock_mindsdb_client,
                messages=[Message(role=Role.user, content="Hello"), Message(role=Role.assistant, content="Hi there!")],
                model=sample_responses_request.model,
                stream=False,
                metadata=None,
                instrument=True,
                langfuse_trace_context=None,
                limits_service=None,
            )

            # Verify message placeholder was created
            mock_conversation_service.create_conversation_message_placeholder.assert_called_once_with(
                conversation_id=sample_conversation_response.id, role=Role.assistant
            )

            # Verify process_non_streaming_producer was called
            mock_process_non_streaming.assert_called_once()
            call_args = mock_process_non_streaming.call_args
            assert call_args[1]["model"] == sample_responses_request.model
            assert call_args[1]["message_id"] == message_id

            # Verify the producer function works
            producer_func = call_args[1]["producer"]
            mock_streamer = Mock()
            await producer_func(mock_streamer)
            mock_handler_instance.responses.assert_awaited_once_with(streamer=mock_streamer, message=mock_message)

            # Verify return value
            assert result == mock_json_response

    @pytest.mark.asyncio
    async def test_responses_request_handler_stream_none(
        self,
        handler_mod,
        mock_session,
        mock_mindsdb_client,
        mock_context,
        mock_conversation_service,
        sample_conversation_response,
        sample_message_responses,
    ):
        """Test responses request when stream parameter is None (should default to False)."""
        mock_json_response = Mock(spec=JSONResponse)
        message_id = uuid4()

        # Create request with stream=None
        responses_request = ResponsesRequest(
            model="gpt-4o",
            input="Test input",
            conversation=None,
            stream=None,
        )

        with (
            patch.object(handler_mod, "OpenAIRequestHandler") as mock_handler_class,
            patch.object(
                handler_mod, "process_non_streaming_producer", new_callable=AsyncMock
            ) as mock_process_non_streaming,
        ):
            # Setup mocks
            mock_handler_instance = Mock()
            mock_handler_class.create = AsyncMock(return_value=mock_handler_instance)
            mock_process_non_streaming.return_value = mock_json_response

            # Mock conversation service methods
            mock_conversation_service.create_conversation = AsyncMock(return_value=sample_conversation_response)
            mock_conversation_service.get_conversation_messages = AsyncMock(return_value=sample_message_responses)
            mock_message = Mock()
            mock_message.id = message_id
            mock_conversation_service.create_conversation_message_placeholder = AsyncMock(return_value=mock_message)
            mock_conversation_service.update_conversation_message_content = AsyncMock()

            # Call the handler
            result = await handler_mod.responses_request_handler(
                session=mock_session,
                context=mock_context,
                mindsdb_client=mock_mindsdb_client,
                responses_request=responses_request,
                conversation_service=mock_conversation_service,
            )

            # Verify OpenAIRequestHandler was created with stream=False (default)
            mock_handler_class.create.assert_awaited_once_with(
                session=mock_session,
                context=mock_context,
                mindsdb_client=mock_mindsdb_client,
                messages=[Message(role=Role.user, content="Hello"), Message(role=Role.assistant, content="Hi there!")],
                model="gpt-4o",
                stream=False,  # Should default to False when None
                metadata=None,
                instrument=True,
                langfuse_trace_context=None,
                limits_service=None,
            )

            # Verify non-streaming producer was called (not streaming)
            mock_process_non_streaming.assert_called_once()

            # Verify return value
            assert result == mock_json_response

    @pytest.mark.asyncio
    async def test_responses_request_handler_with_existing_conversation(
        self,
        handler_mod,
        mock_session,
        mock_mindsdb_client,
        mock_context,
        mock_conversation_service,
        sample_conversation_response,
        sample_message_responses,
    ):
        """Test responses request with existing conversation ID."""
        mock_json_response = Mock(spec=JSONResponse)
        conversation_id = uuid4()
        message_id = uuid4()

        responses_request = ResponsesRequest(
            model="gpt-3.5-turbo",
            input="New message",
            conversation=str(conversation_id),
            stream=False,
        )

        with (
            patch.object(handler_mod, "OpenAIRequestHandler") as mock_handler_class,
            patch.object(
                handler_mod, "process_non_streaming_producer", new_callable=AsyncMock
            ) as mock_process_non_streaming,
        ):
            # Setup mocks
            mock_handler_instance = Mock()
            mock_handler_class.create = AsyncMock(return_value=mock_handler_instance)
            mock_process_non_streaming.return_value = mock_json_response

            # Mock conversation service methods
            mock_conversation_service.get_conversation = AsyncMock(return_value=sample_conversation_response)
            mock_conversation_service.create_conversation_message = AsyncMock()
            mock_conversation_service.get_conversation_messages = AsyncMock(return_value=sample_message_responses)
            mock_message = Mock()
            mock_message.id = message_id
            mock_conversation_service.create_conversation_message_placeholder = AsyncMock(return_value=mock_message)
            mock_conversation_service.update_conversation_message_content = AsyncMock()

            # Call the handler
            result = await handler_mod.responses_request_handler(
                session=mock_session,
                context=mock_context,
                mindsdb_client=mock_mindsdb_client,
                responses_request=responses_request,
                conversation_service=mock_conversation_service,
            )

            # Verify conversation was NOT created (using existing one)
            mock_conversation_service.create_conversation.assert_not_called()

            # Verify message was added to existing conversation
            # The handler passes conversation_id as string from the request
            mock_conversation_service.create_conversation_message.assert_called_once()
            call_args = mock_conversation_service.create_conversation_message.call_args
            # The handler passes the string UUID, so compare as strings
            assert str(call_args[1]["conversation_id"]) == str(conversation_id)
            assert call_args[1]["role"] == Role.user
            assert call_args[1]["content"] == "New message"

            # Verify messages were retrieved
            # The handler passes conversation_id as string from the request
            mock_conversation_service.get_conversation_messages.assert_called_once()
            get_messages_call = mock_conversation_service.get_conversation_messages.call_args
            assert str(get_messages_call[0][0]) == str(conversation_id)

            # Verify return value
            assert result == mock_json_response

    @pytest.mark.asyncio
    async def test_responses_request_handler_with_message_list(
        self,
        handler_mod,
        mock_session,
        mock_mindsdb_client,
        mock_context,
        mock_conversation_service,
        sample_conversation_response,
        sample_message_responses,
    ):
        """Test responses request with list of messages as input."""
        mock_json_response = Mock(spec=JSONResponse)
        message_id = uuid4()
        sample_messages = [
            Message(role=Role.user, content="First message"),
            Message(role=Role.user, content="Second message"),
        ]

        responses_request = ResponsesRequest(
            model="gpt-3.5-turbo",
            input=sample_messages,
            conversation=None,
            stream=False,
        )

        with (
            patch.object(handler_mod, "OpenAIRequestHandler") as mock_handler_class,
            patch.object(
                handler_mod, "process_non_streaming_producer", new_callable=AsyncMock
            ) as mock_process_non_streaming,
        ):
            # Setup mocks
            mock_handler_instance = Mock()
            mock_handler_class.create = AsyncMock(return_value=mock_handler_instance)
            mock_process_non_streaming.return_value = mock_json_response

            # Mock conversation service methods
            mock_conversation_service.create_conversation = AsyncMock(return_value=sample_conversation_response)
            mock_conversation_service.get_conversation_messages = AsyncMock(return_value=sample_message_responses)
            mock_message = Mock()
            mock_message.id = message_id
            mock_conversation_service.create_conversation_message_placeholder = AsyncMock(return_value=mock_message)
            mock_conversation_service.update_conversation_message_content = AsyncMock()

            # Call the handler
            result = await handler_mod.responses_request_handler(
                session=mock_session,
                context=mock_context,
                mindsdb_client=mock_mindsdb_client,
                responses_request=responses_request,
                conversation_service=mock_conversation_service,
            )

            # Verify conversation was created with multiple items
            mock_conversation_service.create_conversation.assert_called_once()
            create_call = mock_conversation_service.create_conversation.call_args[0][0]
            assert len(create_call.items) == 2
            assert create_call.items[0].role == Role.user
            assert create_call.items[0].content == "First message"
            assert create_call.items[1].role == Role.user
            assert create_call.items[1].content == "Second message"

            # Verify return value
            assert result == mock_json_response

    @pytest.mark.asyncio
    async def test_responses_request_handler_with_existing_conversation_and_message_list(
        self,
        handler_mod,
        mock_session,
        mock_mindsdb_client,
        mock_context,
        mock_conversation_service,
        sample_conversation_response,
        sample_message_responses,
    ):
        """Test responses request with existing conversation and list of messages."""
        mock_json_response = Mock(spec=JSONResponse)
        conversation_id = uuid4()
        message_id = uuid4()
        sample_messages = [
            Message(role=Role.user, content="First message"),
            Message(role=Role.assistant, content="Response"),
        ]

        responses_request = ResponsesRequest(
            model="gpt-3.5-turbo",
            input=sample_messages,
            conversation=str(conversation_id),
            stream=False,
        )

        with (
            patch.object(handler_mod, "OpenAIRequestHandler") as mock_handler_class,
            patch.object(
                handler_mod, "process_non_streaming_producer", new_callable=AsyncMock
            ) as mock_process_non_streaming,
        ):
            # Setup mocks
            mock_handler_instance = Mock()
            mock_handler_class.create = AsyncMock(return_value=mock_handler_instance)
            mock_process_non_streaming.return_value = mock_json_response

            # Mock conversation service methods
            mock_conversation_service.get_conversation = AsyncMock(return_value=sample_conversation_response)
            mock_conversation_service.create_conversation_message = AsyncMock()
            mock_conversation_service.get_conversation_messages = AsyncMock(return_value=sample_message_responses)
            mock_message = Mock()
            mock_message.id = message_id
            mock_conversation_service.create_conversation_message_placeholder = AsyncMock(return_value=mock_message)
            mock_conversation_service.update_conversation_message_content = AsyncMock()

            # Call the handler
            result = await handler_mod.responses_request_handler(
                session=mock_session,
                context=mock_context,
                mindsdb_client=mock_mindsdb_client,
                responses_request=responses_request,
                conversation_service=mock_conversation_service,
            )

            # Verify conversation was NOT created
            mock_conversation_service.create_conversation.assert_not_called()

            # Verify messages were added to existing conversation
            assert mock_conversation_service.create_conversation_message.call_count == 2
            calls = mock_conversation_service.create_conversation_message.call_args_list
            # Convert UUID to string for comparison
            assert str(calls[0][1]["conversation_id"]) == str(conversation_id)
            assert calls[0][1]["role"] == Role.user
            assert calls[0][1]["content"] == "First message"
            assert str(calls[1][1]["conversation_id"]) == str(conversation_id)
            assert calls[1][1]["role"] == Role.assistant
            assert calls[1][1]["content"] == "Response"

            # Verify return value
            assert result == mock_json_response

    @pytest.mark.asyncio
    async def test_responses_request_handler_logging(
        self,
        handler_mod,
        mock_session,
        mock_mindsdb_client,
        sample_responses_request,
        mock_context,
        mock_conversation_service,
        sample_conversation_response,
        sample_message_responses,
    ):
        """Test that logging calls are made correctly."""
        request_id = str(mock_context.request_id)
        mock_json_response = Mock(spec=JSONResponse)
        message_id = uuid4()

        with (
            patch.object(handler_mod, "OpenAIRequestHandler") as mock_handler_class,
            patch.object(
                handler_mod, "process_non_streaming_producer", new_callable=AsyncMock
            ) as mock_process_non_streaming,
            patch.object(handler_mod, "get_langfuse_trace_id", return_value=None),
            patch.object(handler_mod, "logger") as mock_logger,
        ):
            # Setup mocks
            mock_handler_instance = Mock()
            mock_handler_class.create = AsyncMock(return_value=mock_handler_instance)
            mock_process_non_streaming.return_value = mock_json_response

            # Mock conversation service methods
            mock_conversation_service.create_conversation = AsyncMock(return_value=sample_conversation_response)
            mock_conversation_service.get_conversation_messages = AsyncMock(return_value=sample_message_responses)
            mock_message = Mock()
            mock_message.id = message_id
            mock_conversation_service.create_conversation_message_placeholder = AsyncMock(return_value=mock_message)
            mock_conversation_service.update_conversation_message_content = AsyncMock()

            # Call the handler
            await handler_mod.responses_request_handler(
                mindsdb_client=mock_mindsdb_client,
                session=mock_session,
                context=mock_context,
                responses_request=sample_responses_request,
                conversation_service=mock_conversation_service,
            )

            # Verify logging calls were made
            assert mock_logger.debug.call_count >= 5  # At least 5 debug calls expected

            # Verify specific log messages
            debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]

            # Check that request details are logged
            assert any(f"🔄[{request_id}] Responses Request:" in call for call in debug_calls)
            assert any(f"🔄[{request_id}] Stream:" in call for call in debug_calls)
            assert any(f"🔄[{request_id}] Conversation:" in call for call in debug_calls)
            assert any(f"🔄[{request_id}] Input:" in call for call in debug_calls)
            assert any(f"🔄[{request_id}] Model:" in call for call in debug_calls)
            assert any(f"🔄[{request_id}] Responses API request is non-streaming." in call for call in debug_calls)

    @pytest.mark.asyncio
    async def test_responses_request_handler_parameter_extraction(
        self,
        handler_mod,
        mock_session,
        mock_mindsdb_client,
        mock_context,
        mock_conversation_service,
        sample_conversation_response,
        sample_message_responses,
    ):
        """Test that parameters are correctly extracted from the request."""
        mock_json_response = Mock(spec=JSONResponse)
        message_id = uuid4()

        # Create request with specific parameters
        responses_request = ResponsesRequest(
            model="custom-model-v1",
            input="Custom input",
            conversation=None,
            stream=False,
        )

        with (
            patch.object(handler_mod, "OpenAIRequestHandler") as mock_handler_class,
            patch.object(
                handler_mod, "process_non_streaming_producer", new_callable=AsyncMock
            ) as mock_process_non_streaming,
        ):
            # Setup mocks
            mock_handler_instance = Mock()
            mock_handler_class.create = AsyncMock(return_value=mock_handler_instance)
            mock_process_non_streaming.return_value = mock_json_response

            # Mock conversation service methods
            mock_conversation_service.create_conversation = AsyncMock(return_value=sample_conversation_response)
            mock_conversation_service.get_conversation_messages = AsyncMock(return_value=sample_message_responses)
            mock_message = Mock()
            mock_message.id = message_id
            mock_conversation_service.create_conversation_message_placeholder = AsyncMock(return_value=mock_message)
            mock_conversation_service.update_conversation_message_content = AsyncMock()

            # Call the handler
            await handler_mod.responses_request_handler(
                session=mock_session,
                context=mock_context,
                mindsdb_client=mock_mindsdb_client,
                responses_request=responses_request,
                conversation_service=mock_conversation_service,
            )

            # Verify parameters were extracted and passed correctly
            mock_handler_class.create.assert_awaited_once_with(
                session=mock_session,
                context=mock_context,
                mindsdb_client=mock_mindsdb_client,
                messages=[Message(role=Role.user, content="Hello"), Message(role=Role.assistant, content="Hi there!")],
                model="custom-model-v1",  # Custom model
                stream=False,  # Explicit stream value
                metadata=None,
                instrument=True,
                langfuse_trace_context=None,
                limits_service=None,
            )

            # Verify process_non_streaming_producer received correct parameters
            call_args = mock_process_non_streaming.call_args
            assert call_args[1]["model"] == "custom-model-v1"
            assert call_args[1]["message_id"] == message_id

    @pytest.mark.asyncio
    async def test_responses_request_handler_no_input(
        self,
        handler_mod,
        mock_session,
        mock_mindsdb_client,
        mock_context,
        mock_conversation_service,
        sample_conversation_response,
        sample_message_responses,
    ):
        """Test responses request with no input provided."""
        mock_json_response = Mock(spec=JSONResponse)
        message_id = uuid4()

        responses_request = ResponsesRequest(
            model="gpt-3.5-turbo",
            input=None,
            conversation=None,
            stream=False,
        )

        with (
            patch.object(handler_mod, "OpenAIRequestHandler") as mock_handler_class,
            patch.object(
                handler_mod, "process_non_streaming_producer", new_callable=AsyncMock
            ) as mock_process_non_streaming,
        ):
            # Setup mocks
            mock_handler_instance = Mock()
            mock_handler_class.create = AsyncMock(return_value=mock_handler_instance)
            mock_process_non_streaming.return_value = mock_json_response

            # Mock conversation service methods
            mock_conversation_service.create_conversation = AsyncMock(return_value=sample_conversation_response)
            mock_conversation_service.get_conversation_messages = AsyncMock(return_value=sample_message_responses)
            mock_message = Mock()
            mock_message.id = message_id
            mock_conversation_service.create_conversation_message_placeholder = AsyncMock(return_value=mock_message)
            mock_conversation_service.update_conversation_message_content = AsyncMock()

            # Call the handler
            result = await handler_mod.responses_request_handler(
                session=mock_session,
                context=mock_context,
                mindsdb_client=mock_mindsdb_client,
                responses_request=responses_request,
                conversation_service=mock_conversation_service,
            )

            # Verify conversation was created with no items
            mock_conversation_service.create_conversation.assert_called_once()
            create_call = mock_conversation_service.create_conversation.call_args[0][0]
            assert create_call.items is None or len(create_call.items) == 0

            # Verify return value
            assert result == mock_json_response

    @pytest.mark.asyncio
    async def test_responses_request_handler_passes_captured_trace_context_to_factory(
        self,
        handler_mod,
        mock_session,
        mock_mindsdb_client,
        sample_streaming_responses_request,
        mock_context,
        mock_conversation_service,
        sample_conversation_response,
        sample_message_responses,
    ):
        """The Responses handler must capture the @observe trace context and
        forward it to OpenAIRequestHandler.create so streaming code paths can
        attach a child generation with token usage after the @observe scope
        closes.
        """
        mock_streaming_response = Mock(spec=StreamingResponse)
        message_id = uuid4()
        captured_ctx = {"trace_id": "trace-z", "parent_span_id": "obs-w"}

        with (
            patch.object(handler_mod, "OpenAIRequestHandler") as mock_handler_class,
            patch.object(handler_mod, "process_streaming_producer", new_callable=AsyncMock) as mock_process_streaming,
            patch.object(handler_mod.MindsService, "get_mind_model", new_callable=AsyncMock) as mock_get_mind_model,
            patch.object(handler_mod, "capture_langfuse_generation_context", return_value=captured_ctx),
        ):
            mock_handler_instance = Mock()
            mock_handler_class.create = AsyncMock(return_value=mock_handler_instance)
            mock_handler_instance.responses = AsyncMock()
            mock_process_streaming.return_value = mock_streaming_response

            fake_mind = Mock()
            fake_mind.parameters = {"agent_name": "candidate_sql_agent"}
            mock_get_mind_model.return_value = fake_mind

            mock_conversation_service.create_conversation = AsyncMock(return_value=sample_conversation_response)
            mock_conversation_service.get_conversation_messages = AsyncMock(return_value=sample_message_responses)
            placeholder = Mock()
            placeholder.id = message_id
            mock_conversation_service.create_conversation_message_placeholder = AsyncMock(return_value=placeholder)
            mock_conversation_service.update_conversation_message_content = AsyncMock()

            await handler_mod.responses_request_handler(
                session=mock_session,
                context=mock_context,
                mindsdb_client=mock_mindsdb_client,
                responses_request=sample_streaming_responses_request,
                conversation_service=mock_conversation_service,
            )

            mock_handler_class.create.assert_awaited_once()
            kwargs = mock_handler_class.create.call_args.kwargs
            assert kwargs["langfuse_trace_context"] == captured_ctx
