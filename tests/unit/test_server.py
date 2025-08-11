import pytest
from unittest.mock import AsyncMock, Mock, patch
from fastapi import Request
from starlette.responses import StreamingResponse, JSONResponse

from minds.server import chat_completions
from minds.requests.chat_completions_request import ChatCompletionsRequest, ChatCompletionRequestMetadata
from minds.requests.schemas import Message, Role
from minds.requests.context import Context


class TestChatCompletions:
    """Test suite for the chat_completions endpoint"""

    @pytest.fixture
    def mock_request(self):
        """Create a mock FastAPI Request object"""
        request = Mock(spec=Request)
        # Create a mock headers object with get method
        mock_headers = Mock()
        mock_headers.get = Mock(side_effect=lambda key, default=None: {
            "x-user-id": "123",
            "x-user-email": "test@example.com", 
            "x-company-id": "456"
        }.get(key, default))
        request.headers = mock_headers
        return request

    @pytest.fixture
    def sample_chat_request(self):
        """Create a sample ChatCompletionsRequest"""
        return ChatCompletionsRequest(
            model="minds",
            messages=[
                Message(role=Role.user, content="Hello, how are you?")
            ],
            metadata=ChatCompletionRequestMetadata(mdb_completions_session_id=1748503096170),
            stream=False
        )

    @pytest.fixture
    def sample_streaming_chat_request(self):
        """Create a sample streaming ChatCompletionsRequest"""
        return ChatCompletionsRequest(
            model="minds",
            messages=[
                Message(role=Role.user, content="Hello, how are you?")
            ],
            metadata=ChatCompletionRequestMetadata(mdb_completions_session_id=1748503096170),
            stream=True
        )

    @patch('minds.server.setup_langfuse_observation')
    @patch('minds.server.extract_context_from_request')
    @patch('minds.server.chat_completions_request_handler')
    async def test_chat_completions_non_streaming_success(
        self,
        mock_handler,
        mock_extract_context,
        mock_setup_langfuse,
        mock_request,
        sample_chat_request
    ):
        """Test successful non-streaming chat completions"""
        # Setup mocks
        mock_context = Context(
            user_id=123,
            user_email="test@example.com",
            company_id=456
        )
        mock_extract_context.return_value = mock_context
        mock_setup_langfuse.return_value = "test-request-id"
        
        mock_response = JSONResponse({"choices": [{"message": {"role": "assistant", "content": "Hello!"}}]})
        mock_handler.return_value = mock_response

        # Execute
        result = await chat_completions(sample_chat_request, mock_request)

        # Verify
        assert result == mock_response
        mock_extract_context.assert_called_once_with(request=mock_request)
        mock_setup_langfuse.assert_called_once_with(context=mock_context)
        mock_handler.assert_called_once_with(
            request_id="test-request-id",
            chat_completions_request=sample_chat_request
        )

    @patch('minds.server.setup_langfuse_observation')
    @patch('minds.server.extract_context_from_request')
    @patch('minds.server.chat_completions_request_handler')
    async def test_chat_completions_streaming_success(
        self,
        mock_handler,
        mock_extract_context,
        mock_setup_langfuse,
        mock_request,
        sample_streaming_chat_request
    ):
        """Test successful streaming chat completions"""
        # Setup mocks
        mock_context = Context(
            user_id=123,
            user_email="test@example.com",
            company_id=456
        )
        mock_extract_context.return_value = mock_context
        mock_setup_langfuse.return_value = "test-request-id"
        
        mock_response = StreamingResponse(
            iter([b"data: test\n\n"]),
            media_type="text/event-stream"
        )
        mock_handler.return_value = mock_response

        # Execute
        result = await chat_completions(sample_streaming_chat_request, mock_request)

        # Verify
        assert result == mock_response
        mock_extract_context.assert_called_once_with(request=mock_request)
        mock_setup_langfuse.assert_called_once_with(context=mock_context)
        mock_handler.assert_called_once_with(
            request_id="test-request-id",
            chat_completions_request=sample_streaming_chat_request
        )

    @patch('minds.server.setup_langfuse_observation')
    @patch('minds.server.extract_context_from_request')
    @patch('minds.server.chat_completions_request_handler')
    async def test_chat_completions_handler_exception(
        self,
        mock_handler,
        mock_extract_context,
        mock_setup_langfuse,
        mock_request,
        sample_chat_request
    ):
        """Test exception handling in chat completions"""
        # Setup mocks
        mock_context = Context(
            user_id=123,
            user_email="test@example.com",
            company_id=456
        )
        mock_extract_context.return_value = mock_context
        mock_setup_langfuse.return_value = "test-request-id"
        
        # Make handler raise an exception
        mock_handler.side_effect = Exception("Test error")

        # Execute and verify exception
        with pytest.raises(Exception) as exc_info:
            await chat_completions(sample_chat_request, mock_request)
        
        assert "Test error" in str(exc_info.value)
        mock_handler.assert_called_once_with(
            request_id="test-request-id",
            chat_completions_request=sample_chat_request
        )

    @patch('minds.server.setup_langfuse_observation')
    @patch('minds.server.extract_context_from_request')
    @patch('minds.server.chat_completions_request_handler')
    async def test_chat_completions_context_extraction(
        self,
        mock_handler,
        mock_extract_context,
        mock_setup_langfuse,
        mock_request,
        sample_chat_request
    ):
        """Test context extraction from request headers"""
        # Setup mocks with specific context values
        mock_context = Context(
            user_id=999,
            user_email="specific@test.com",
            company_id=777
        )
        mock_extract_context.return_value = mock_context
        mock_setup_langfuse.return_value = "context-test-id"
        
        mock_response = JSONResponse({"test": "response"})
        mock_handler.return_value = mock_response

        # Execute
        result = await chat_completions(sample_chat_request, mock_request)

        # Verify context was properly extracted and used
        mock_extract_context.assert_called_once_with(request=mock_request)
        mock_setup_langfuse.assert_called_once_with(context=mock_context)
        mock_handler.assert_called_once_with(
            request_id="context-test-id",
            chat_completions_request=sample_chat_request
        )

    @patch('minds.server.setup_langfuse_observation')
    @patch('minds.server.extract_context_from_request')
    @patch('minds.server.chat_completions_request_handler')
    async def test_chat_completions_with_multiple_messages(
        self,
        mock_handler,
        mock_extract_context,
        mock_setup_langfuse,
        mock_request
    ):
        """Test chat completions with multiple messages"""
        # Create request with multiple messages
        multi_message_request = ChatCompletionsRequest(
            model="minds",
            messages=[
                Message(role=Role.system, content="You are a helpful assistant."),
                Message(role=Role.user, content="What is the weather like?"),
                Message(role=Role.assistant, content="I don't have access to weather data."),
                Message(role=Role.user, content="Can you help with math?")
            ],
            metadata=ChatCompletionRequestMetadata(mdb_completions_session_id=1748503096170),
            stream=False
        )
        
        # Setup mocks
        mock_context = Context(user_id=123, user_email="test@example.com", company_id=456)
        mock_extract_context.return_value = mock_context
        mock_setup_langfuse.return_value = "multi-msg-id"
        
        mock_response = JSONResponse({"choices": [{"message": {"role": "assistant", "content": "Yes, I can help!"}}]})
        mock_handler.return_value = mock_response

        # Execute
        result = await chat_completions(multi_message_request, mock_request)

        # Verify
        assert result == mock_response
        mock_handler.assert_called_once_with(
            request_id="multi-msg-id",
            chat_completions_request=multi_message_request
        )

    @patch('minds.server.setup_langfuse_observation')
    @patch('minds.server.extract_context_from_request')
    @patch('minds.server.chat_completions_request_handler')
    async def test_chat_completions_langfuse_observation_failure(
        self,
        mock_handler,
        mock_extract_context,
        mock_setup_langfuse,
        mock_request,
        sample_chat_request
    ):
        """Test behavior when Langfuse observation setup fails"""
        # Setup mocks
        mock_context = Context(user_id=123, user_email="test@example.com", company_id=456)
        mock_extract_context.return_value = mock_context
        
        # Make Langfuse setup fail and raise exception (this should propagate)
        mock_setup_langfuse.side_effect = Exception("Langfuse error")
        
        # Execute and verify exception is propagated
        with pytest.raises(Exception) as exc_info:
            await chat_completions(sample_chat_request, mock_request)
        
        assert "Langfuse error" in str(exc_info.value)
        mock_extract_context.assert_called_once_with(request=mock_request)
        mock_setup_langfuse.assert_called_once_with(context=mock_context)
