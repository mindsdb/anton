import pytest
from unittest.mock import AsyncMock, Mock, patch
from starlette.responses import StreamingResponse, JSONResponse

from minds.handlers.chat_completions_request_handler import chat_completions_request_handler
from minds.requests.chat_completions_request import ChatCompletionsRequest, ChatCompletionRequestMetadata
from minds.requests.schemas import Message, Role


class TestChatCompletionsRequestHandler:
    """Test suite for the chat_completions_request_handler function"""

    @pytest.fixture
    def sample_non_streaming_request(self):
        """Create a sample non-streaming ChatCompletionsRequest"""
        return ChatCompletionsRequest(
            model="minds",
            messages=[
                Message(role=Role.user, content="Hello, how are you?")
            ],
            metadata=ChatCompletionRequestMetadata(mdb_completions_session_id=1748503096170),
            stream=False
        )

    @pytest.fixture
    def sample_streaming_request(self):
        """Create a sample streaming ChatCompletionsRequest"""
        return ChatCompletionsRequest(
            model="minds",
            messages=[
                Message(role=Role.user, content="Hello, how are you?")
            ],
            metadata=ChatCompletionRequestMetadata(mdb_completions_session_id=1748503096170),
            stream=True
        )

    @pytest.fixture
    def sample_request_no_stream_flag(self):
        """Create a sample request without explicit stream flag (should default to False)"""
        return ChatCompletionsRequest(
            model="minds",
            messages=[
                Message(role=Role.user, content="Hello, how are you?")
            ],
            metadata=ChatCompletionRequestMetadata(mdb_completions_session_id=1748503096170)
            # stream is None by default
        )

    @patch('minds.handlers.chat_completions_request_handler.process_non_streaming_producer')
    @patch('minds.handlers.chat_completions_request_handler.ChatCompletionsHandler')
    async def test_non_streaming_request_handler(
        self,
        mock_handler_class,
        mock_process_non_streaming,
        sample_non_streaming_request
    ):
        """Test non-streaming request handling"""
        # Setup mocks
        mock_handler_instance = Mock()
        mock_handler_class.return_value = mock_handler_instance
        
        mock_response = JSONResponse({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! I'm doing well, thank you for asking."
                    },
                    "finish_reason": "stop"
                }
            ]
        })
        mock_process_non_streaming.return_value = mock_response

        # Execute
        result = await chat_completions_request_handler(
            request_id="test-request-123",
            chat_completions_request=sample_non_streaming_request
        )

        # Verify
        assert result == mock_response
        
        # Verify ChatCompletionsHandler was created with correct parameters
        mock_handler_class.assert_called_once_with(
            messages=sample_non_streaming_request.messages,
            model=sample_non_streaming_request.model,
            stream=False
        )
        
        # Verify non-streaming processor was called
        mock_process_non_streaming.assert_called_once()
        call_args = mock_process_non_streaming.call_args
        assert call_args.kwargs['request_id'] == "test-request-123"
        assert call_args.kwargs['model'] == "minds"
        
        # Verify the producer lambda function
        producer = call_args.kwargs['producer']
        assert callable(producer)

    @patch('minds.handlers.chat_completions_request_handler.process_streaming_producer')
    @patch('minds.handlers.chat_completions_request_handler.ChatCompletionsHandler')
    async def test_streaming_request_handler(
        self,
        mock_handler_class,
        mock_process_streaming,
        sample_streaming_request
    ):
        """Test streaming request handling"""
        # Setup mocks
        mock_handler_instance = Mock()
        mock_handler_class.return_value = mock_handler_instance
        
        mock_response = StreamingResponse(
            iter([b"data: {'content': 'Hello!'}\n\n"]),
            media_type="text/event-stream"
        )
        mock_process_streaming.return_value = mock_response

        # Execute
        result = await chat_completions_request_handler(
            request_id="test-stream-456",
            chat_completions_request=sample_streaming_request
        )

        # Verify
        assert result == mock_response
        
        # Verify ChatCompletionsHandler was created with correct parameters
        mock_handler_class.assert_called_once_with(
            messages=sample_streaming_request.messages,
            model=sample_streaming_request.model,
            stream=True
        )
        
        # Verify streaming processor was called
        mock_process_streaming.assert_called_once()
        call_args = mock_process_streaming.call_args
        assert call_args.kwargs['request_id'] == "test-stream-456"
        assert call_args.kwargs['model'] == "minds"
        
        # Verify the producer lambda function
        producer = call_args.kwargs['producer']
        assert callable(producer)

    @patch('minds.handlers.chat_completions_request_handler.process_non_streaming_producer')
    @patch('minds.handlers.chat_completions_request_handler.ChatCompletionsHandler')
    async def test_request_handler_default_stream_false(
        self,
        mock_handler_class,
        mock_process_non_streaming,
        sample_request_no_stream_flag
    ):
        """Test that stream defaults to False when not specified"""
        # Setup mocks
        mock_handler_instance = Mock()
        mock_handler_class.return_value = mock_handler_instance
        
        mock_response = JSONResponse({"test": "response"})
        mock_process_non_streaming.return_value = mock_response

        # Execute
        result = await chat_completions_request_handler(
            request_id="test-default-stream",
            chat_completions_request=sample_request_no_stream_flag
        )

        # Verify
        assert result == mock_response
        
        # Verify ChatCompletionsHandler was created with stream=False
        mock_handler_class.assert_called_once_with(
            messages=sample_request_no_stream_flag.messages,
            model=sample_request_no_stream_flag.model,
            stream=False  # Should default to False
        )
        
        # Verify non-streaming processor was called (not streaming)
        mock_process_non_streaming.assert_called_once()

    @patch('minds.handlers.chat_completions_request_handler.process_non_streaming_producer')
    @patch('minds.handlers.chat_completions_request_handler.ChatCompletionsHandler')
    async def test_request_handler_with_multiple_messages(
        self,
        mock_handler_class,
        mock_process_non_streaming
    ):
        """Test request handler with multiple messages"""
        # Create request with multiple messages
        multi_message_request = ChatCompletionsRequest(
            model="gpt-4",
            messages=[
                Message(role=Role.system, content="You are a helpful assistant."),
                Message(role=Role.user, content="What is 2+2?"),
                Message(role=Role.assistant, content="2+2 equals 4."),
                Message(role=Role.user, content="What about 3+3?")
            ],
            metadata=ChatCompletionRequestMetadata(mdb_completions_session_id=9876543210),
            stream=False
        )
        
        # Setup mocks
        mock_handler_instance = Mock()
        mock_handler_class.return_value = mock_handler_instance
        
        mock_response = JSONResponse({"choices": [{"message": {"content": "3+3 equals 6."}}]})
        mock_process_non_streaming.return_value = mock_response

        # Execute
        result = await chat_completions_request_handler(
            request_id="multi-msg-test",
            chat_completions_request=multi_message_request
        )

        # Verify
        assert result == mock_response
        
        # Verify ChatCompletionsHandler was created with all messages
        mock_handler_class.assert_called_once_with(
            messages=multi_message_request.messages,
            model="gpt-4",
            stream=False
        )
        
        # Verify the messages were passed correctly
        call_args = mock_handler_class.call_args
        passed_messages = call_args.kwargs['messages']
        assert len(passed_messages) == 4
        assert passed_messages[0].role == Role.system
        assert passed_messages[1].role == Role.user
        assert passed_messages[2].role == Role.assistant
        assert passed_messages[3].role == Role.user

    @patch('minds.handlers.chat_completions_request_handler.process_streaming_producer')
    @patch('minds.handlers.chat_completions_request_handler.ChatCompletionsHandler')
    async def test_producer_lambda_function_execution(
        self,
        mock_handler_class,
        mock_process_streaming,
        sample_streaming_request
    ):
        """Test that the producer lambda function correctly calls the handler"""
        # Setup mocks
        mock_handler_instance = Mock()
        mock_handler_instance.chat_completions = AsyncMock(return_value="test_result")
        mock_handler_class.return_value = mock_handler_instance
        
        mock_response = StreamingResponse(
            iter([b"data: test\n\n"]),
            media_type="text/event-stream"
        )
        mock_process_streaming.return_value = mock_response

        # Execute
        result = await chat_completions_request_handler(
            request_id="lambda-test",
            chat_completions_request=sample_streaming_request
        )

        # Verify
        assert result == mock_response
        
        # Get the producer lambda that was passed to process_streaming_producer
        call_args = mock_process_streaming.call_args
        producer = call_args.kwargs['producer']
        
        # Test the producer lambda by calling it with a mock streamer
        mock_streamer = Mock()
        await producer(mock_streamer)
        
        # Verify the handler's chat_completions method was called with the streamer
        mock_handler_instance.chat_completions.assert_called_once_with(streamer=mock_streamer)

    @patch('minds.handlers.chat_completions_request_handler.process_non_streaming_producer')
    @patch('minds.handlers.chat_completions_request_handler.ChatCompletionsHandler')
    async def test_request_handler_with_different_models(
        self,
        mock_handler_class,
        mock_process_non_streaming
    ):
        """Test request handler with different model names"""
        models_to_test = ["minds", "gpt-4", "gpt-3.5-turbo", "claude-3"]
        
        for model_name in models_to_test:
            # Reset mocks for each iteration
            mock_handler_class.reset_mock()
            mock_process_non_streaming.reset_mock()
            
            # Create request with specific model
            request = ChatCompletionsRequest(
                model=model_name,
                messages=[Message(role=Role.user, content="Test message")],
                stream=False
            )
            
            # Setup mocks
            mock_handler_instance = Mock()
            mock_handler_class.return_value = mock_handler_instance
            mock_response = JSONResponse({"model": model_name})
            mock_process_non_streaming.return_value = mock_response

            # Execute
            result = await chat_completions_request_handler(
                request_id=f"test-{model_name}",
                chat_completions_request=request
            )

            # Verify
            assert result == mock_response
            mock_handler_class.assert_called_once_with(
                messages=request.messages,
                model=model_name,
                stream=False
            )
            
            # Verify model was passed to processor
            call_args = mock_process_non_streaming.call_args
            assert call_args.kwargs['model'] == model_name

    @patch('minds.handlers.chat_completions_request_handler.process_streaming_producer')
    @patch('minds.handlers.chat_completions_request_handler.ChatCompletionsHandler')
    async def test_request_handler_exception_handling(
        self,
        mock_handler_class,
        mock_process_streaming,
        sample_streaming_request
    ):
        """Test exception handling in request handler"""
        # Setup mocks to raise an exception
        mock_handler_class.side_effect = Exception("Handler creation failed")

        # Execute and verify exception is propagated
        with pytest.raises(Exception) as exc_info:
            await chat_completions_request_handler(
                request_id="exception-test",
                chat_completions_request=sample_streaming_request
            )
        
        assert "Handler creation failed" in str(exc_info.value)
        mock_handler_class.assert_called_once()

    @patch('minds.handlers.chat_completions_request_handler.process_non_streaming_producer')
    @patch('minds.handlers.chat_completions_request_handler.ChatCompletionsHandler')
    async def test_request_handler_with_metadata(
        self,
        mock_handler_class,
        mock_process_non_streaming
    ):
        """Test request handler properly handles metadata"""
        # Create request with metadata
        request_with_metadata = ChatCompletionsRequest(
            model="minds",
            messages=[Message(role=Role.user, content="Test with metadata")],
            metadata=ChatCompletionRequestMetadata(mdb_completions_session_id=1234567890),
            stream=False
        )
        
        # Setup mocks
        mock_handler_instance = Mock()
        mock_handler_class.return_value = mock_handler_instance
        mock_response = JSONResponse({"status": "success"})
        mock_process_non_streaming.return_value = mock_response

        # Execute
        result = await chat_completions_request_handler(
            request_id="metadata-test",
            chat_completions_request=request_with_metadata
        )

        # Verify
        assert result == mock_response
        
        # Verify that the handler was created correctly (metadata doesn't affect handler creation)
        mock_handler_class.assert_called_once_with(
            messages=request_with_metadata.messages,
            model="minds",
            stream=False
        )
        
        # The metadata should be preserved in the original request object
        assert request_with_metadata.metadata.mdb_completions_session_id == 1234567890
