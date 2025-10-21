import importlib
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

import pytest
from sqlmodel import Session
from starlette.responses import JSONResponse, StreamingResponse

from minds.requests.chat_completions_request import (
    ChatCompletionRequestMetadata,
    ChatCompletionsRequest,
)
from minds.requests.context import Context
from minds.schemas.chat import Message, Role


@pytest.fixture()
def handler_mod(monkeypatch):
    # Neutralize observe decorator before import
    import langfuse as dec

    monkeypatch.setattr(
        dec,
        "observe",
        lambda f=None, **_: (lambda *a, **k: f(*a, **k)) if f else (lambda x: x),
    )

    import minds.handlers.chat_completions_request_handler as mod

    importlib.reload(mod)
    return mod


@pytest.fixture
def mock_session():
    """Mock SQLModel session."""
    return Mock(spec=Session)


@pytest.fixture
def mock_mindsdb_client():
    """Mock MindsDB client."""
    return Mock(spec=["connect"])


@pytest.fixture
def mock_context():
    """Mock Context."""
    return Context(
        user_id=UUID("00000000-0000-0000-0000-000000000001"), tenant_id=UUID("00000000-0000-0000-0000-000000000002")
    )


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        Message(role=Role.user, content="Hello, how are you?"),
        Message(role=Role.assistant, content="I'm doing well, thank you!"),
    ]


@pytest.fixture
def sample_chat_request(sample_messages):
    """Sample ChatCompletionsRequest for testing."""
    return ChatCompletionsRequest(
        model="gpt-3.5-turbo",
        messages=sample_messages,
        stream=False,
        metadata=ChatCompletionRequestMetadata(mdb_completions_session_id="test-session"),
    )


@pytest.fixture
def sample_streaming_chat_request(sample_messages):
    """Sample streaming ChatCompletionsRequest for testing."""
    return ChatCompletionsRequest(
        model="gpt-3.5-turbo",
        messages=sample_messages,
        stream=True,
        metadata=ChatCompletionRequestMetadata(mdb_completions_session_id="test-session"),
    )


class TestChatCompletionsRequestHandler:
    @pytest.mark.asyncio
    async def test_chat_completions_request_handler_streaming(
        self,
        handler_mod,
        mock_session,
        mock_mindsdb_client,
        sample_streaming_chat_request,
        mock_context,
    ):
        """Test streaming chat completions request handling."""
        request_id = "test-request-123"
        mock_streaming_response = Mock(spec=StreamingResponse)

        with (
            patch.object(handler_mod, "ChatCompletionsHandler") as mock_handler_class,
            patch.object(handler_mod, "process_streaming_producer", new_callable=AsyncMock) as mock_process_streaming,
        ):
            # Setup mocks
            mock_handler_instance = Mock()
            mock_handler_class.return_value = mock_handler_instance
            mock_process_streaming.return_value = mock_streaming_response

            # Call the handler
            result = await handler_mod.chat_completions_request_handler(
                request_id=request_id,
                session=mock_session,
                context=mock_context,
                mindsdb_client=mock_mindsdb_client,
                chat_completions_request=sample_streaming_chat_request,
            )

            # Verify ChatCompletionsHandler was created with correct parameters
            mock_handler_class.assert_called_once_with(
                session=mock_session,
                context=mock_context,
                mindsdb_client=mock_mindsdb_client,
                messages=sample_streaming_chat_request.messages,
                model=sample_streaming_chat_request.model,
                stream=True,
            )

            # Verify process_streaming_producer was called
            mock_process_streaming.assert_called_once()
            call_args = mock_process_streaming.call_args
            assert call_args[1]["request_id"] == request_id
            assert call_args[1]["model"] == sample_streaming_chat_request.model

            # Verify the producer function works
            producer_func = call_args[1]["producer"]
            mock_streamer = Mock()
            producer_func(mock_streamer)
            mock_handler_instance.chat_completions.assert_called_once_with(streamer=mock_streamer)

            # Verify return value
            assert result == mock_streaming_response

    @pytest.mark.asyncio
    async def test_chat_completions_request_handler_non_streaming(
        self, handler_mod, mock_session, mock_mindsdb_client, sample_chat_request, mock_context
    ):
        """Test non-streaming chat completions request handling."""
        request_id = "test-request-456"
        mock_json_response = Mock(spec=JSONResponse)

        with (
            patch.object(handler_mod, "ChatCompletionsHandler") as mock_handler_class,
            patch.object(
                handler_mod, "process_non_streaming_producer", new_callable=AsyncMock
            ) as mock_process_non_streaming,
        ):
            # Setup mocks
            mock_handler_instance = Mock()
            mock_handler_class.return_value = mock_handler_instance
            mock_process_non_streaming.return_value = mock_json_response

            # Call the handler
            result = await handler_mod.chat_completions_request_handler(
                request_id=request_id,
                session=mock_session,
                context=mock_context,
                mindsdb_client=mock_mindsdb_client,
                chat_completions_request=sample_chat_request,
            )

            # Verify ChatCompletionsHandler was created with correct parameters
            mock_handler_class.assert_called_once_with(
                session=mock_session,
                context=mock_context,
                mindsdb_client=mock_mindsdb_client,
                messages=sample_chat_request.messages,
                model=sample_chat_request.model,
                stream=False,
            )

            # Verify process_non_streaming_producer was called
            mock_process_non_streaming.assert_called_once()
            call_args = mock_process_non_streaming.call_args
            assert call_args[1]["request_id"] == request_id
            assert call_args[1]["model"] == sample_chat_request.model

            # Verify the producer function works
            producer_func = call_args[1]["producer"]
            mock_streamer = Mock()
            producer_func(mock_streamer)
            mock_handler_instance.chat_completions.assert_called_once_with(streamer=mock_streamer)

            # Verify return value
            assert result == mock_json_response

    @pytest.mark.asyncio
    async def test_chat_completions_request_handler_stream_none(
        self, handler_mod, mock_session, mock_mindsdb_client, sample_messages, mock_context
    ):
        """Test chat completions request when stream parameter is None (should default to False)."""
        request_id = "test-request-789"
        mock_json_response = Mock(spec=JSONResponse)

        # Create request with stream=None
        chat_request = ChatCompletionsRequest(
            model="gpt-4",
            messages=sample_messages,
            stream=None,
            metadata=ChatCompletionRequestMetadata(mdb_completions_session_id="test-session"),
        )

        with (
            patch.object(handler_mod, "ChatCompletionsHandler") as mock_handler_class,
            patch.object(
                handler_mod, "process_non_streaming_producer", new_callable=AsyncMock
            ) as mock_process_non_streaming,
        ):
            # Setup mocks
            mock_handler_instance = Mock()
            mock_handler_class.return_value = mock_handler_instance
            mock_process_non_streaming.return_value = mock_json_response

            # Call the handler
            result = await handler_mod.chat_completions_request_handler(
                request_id=request_id,
                session=mock_session,
                context=mock_context,
                mindsdb_client=mock_mindsdb_client,
                chat_completions_request=chat_request,
            )

            # Verify ChatCompletionsHandler was created with stream=False (default)
            mock_handler_class.assert_called_once_with(
                session=mock_session,
                context=mock_context,
                mindsdb_client=mock_mindsdb_client,
                messages=chat_request.messages,
                model=chat_request.model,
                stream=False,  # Should default to False when None
            )

            # Verify non-streaming producer was called (not streaming)
            mock_process_non_streaming.assert_called_once()

            # Verify return value
            assert result == mock_json_response

    @pytest.mark.asyncio
    async def test_chat_completions_request_handler_logging(
        self, handler_mod, mock_session, mock_mindsdb_client, sample_chat_request, mock_context
    ):
        """Test that logging calls are made correctly."""
        request_id = "test-request-logging"
        mock_json_response = Mock(spec=JSONResponse)

        with (
            patch.object(handler_mod, "ChatCompletionsHandler") as mock_handler_class,
            patch.object(
                handler_mod, "process_non_streaming_producer", new_callable=AsyncMock
            ) as mock_process_non_streaming,
            patch.object(handler_mod, "logger") as mock_logger,
        ):
            # Setup mocks
            mock_handler_instance = Mock()
            mock_handler_class.return_value = mock_handler_instance
            mock_process_non_streaming.return_value = mock_json_response

            # Call the handler
            await handler_mod.chat_completions_request_handler(
                request_id=request_id,
                mindsdb_client=mock_mindsdb_client,
                session=mock_session,
                context=mock_context,
                chat_completions_request=sample_chat_request,
            )

            # Verify logging calls were made
            assert mock_logger.debug.call_count >= 4  # At least 4 debug calls expected

            # Verify specific log messages
            debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]

            # Check that request details are logged
            assert any(f"🔄[{request_id}] Chat Completion Request:" in call for call in debug_calls)
            assert any(f"🔄[{request_id}] Stream:" in call for call in debug_calls)
            assert any(f"🔄[{request_id}] Messages:" in call for call in debug_calls)
            assert any(f"🔄[{request_id}] Model:" in call for call in debug_calls)
            assert any(f"🔄[{request_id}] Chat completions request is non-streaming." in call for call in debug_calls)

    @pytest.mark.asyncio
    async def test_chat_completions_request_handler_parameter_extraction(
        self, handler_mod, mock_session, mock_mindsdb_client, sample_messages, mock_context
    ):
        """Test that parameters are correctly extracted from the request."""
        request_id = "test-request-params"
        mock_json_response = Mock(spec=JSONResponse)

        # Create request with specific parameters
        chat_request = ChatCompletionsRequest(
            model="custom-model-v1",
            messages=sample_messages,
            stream=False,
            metadata=ChatCompletionRequestMetadata(mdb_completions_session_id="custom-session"),
        )

        with (
            patch.object(handler_mod, "ChatCompletionsHandler") as mock_handler_class,
            patch.object(
                handler_mod, "process_non_streaming_producer", new_callable=AsyncMock
            ) as mock_process_non_streaming,
        ):
            # Setup mocks
            mock_handler_instance = Mock()
            mock_handler_class.return_value = mock_handler_instance
            mock_process_non_streaming.return_value = mock_json_response

            # Call the handler
            await handler_mod.chat_completions_request_handler(
                request_id=request_id,
                session=mock_session,
                context=mock_context,
                mindsdb_client=mock_mindsdb_client,
                chat_completions_request=chat_request,
            )

            # Verify parameters were extracted and passed correctly
            mock_handler_class.assert_called_once_with(
                session=mock_session,
                context=mock_context,
                mindsdb_client=mock_mindsdb_client,
                messages=sample_messages,  # Original messages
                model="custom-model-v1",  # Custom model
                stream=False,  # Explicit stream value
            )

            # Verify process_non_streaming_producer received correct parameters
            call_args = mock_process_non_streaming.call_args
            assert call_args[1]["request_id"] == request_id
            assert call_args[1]["model"] == "custom-model-v1"
