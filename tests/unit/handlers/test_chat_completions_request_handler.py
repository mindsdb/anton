"""Tests for chat completions request handler."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlmodel import Session
from starlette.responses import JSONResponse

from minds.handlers.chat_completions_request_handler import chat_completions_request_handler
from minds.requests.chat_completions_request import ChatCompletionsRequest
from minds.requests.context import Context
from minds.schemas.chat import Message


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    return MagicMock(spec=Session)


@pytest.fixture
def mock_context():
    """Create a mock request context."""
    return Context(
        user_id=uuid4(),
        organization_id=uuid4(),
        user_email="test@example.com",
        user_roles=["user"],
    )


@pytest.fixture
def chat_completions_request():
    """Create a sample chat completions request."""
    return ChatCompletionsRequest(
        model="latest:sonnet",
        messages=[
            Message(role="user", content="Hello"),
        ],
        stream=False,
    )


class TestChatCompletionsRequestHandler:
    """Tests for the chat completions request handler."""

    @pytest.mark.asyncio
    async def test_handler_creates_openai_request_handler(self, mock_session, mock_context, chat_completions_request):
        """Test that handler creates OpenAIRequestHandler correctly."""
        mock_response = JSONResponse(
            content={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "Hi!"}}],
            }
        )

        with patch("minds.handlers.chat_completions_request_handler.OpenAIRequestHandler") as mock_handler_class, patch(
            "minds.handlers.chat_completions_request_handler.setup_langfuse_observation"
        ), patch(
            "minds.handlers.chat_completions_request_handler.get_langfuse_trace_id", return_value=None
        ), patch("minds.handlers.chat_completions_request_handler.capture_langfuse_generation_context"):
            mock_handler_instance = AsyncMock()
            mock_handler_instance.proxy_chat_completions = AsyncMock(return_value=mock_response)
            mock_handler_class.create = AsyncMock(return_value=mock_handler_instance)

            response = await chat_completions_request_handler(
                session=mock_session,
                context=mock_context,
                chat_completions_request=chat_completions_request,
            )

            assert response is not None
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_handler_passes_model_to_handler(self, mock_session, mock_context, chat_completions_request):
        """Test that model is passed correctly to OpenAIRequestHandler."""
        mock_response = JSONResponse(content={"id": "chatcmpl-123"})

        with patch("minds.handlers.chat_completions_request_handler.OpenAIRequestHandler") as mock_handler_class:
            mock_handler_instance = AsyncMock()
            mock_handler_instance.proxy_chat_completions = AsyncMock(return_value=mock_response)
            mock_handler_class.create = AsyncMock(return_value=mock_handler_instance)

            with patch("minds.handlers.chat_completions_request_handler.setup_langfuse_observation"):
                with patch("minds.handlers.chat_completions_request_handler.get_langfuse_trace_id", return_value=None):
                    with patch("minds.handlers.chat_completions_request_handler.capture_langfuse_generation_context"):
                        await chat_completions_request_handler(
                            session=mock_session,
                            context=mock_context,
                            chat_completions_request=chat_completions_request,
                        )

                        # Verify the handler was created with the correct model
                        call_kwargs = mock_handler_class.create.call_args[1]
                        assert call_kwargs["model"] == "latest:sonnet"

    @pytest.mark.asyncio
    async def test_handler_passes_messages_to_handler(self, mock_session, mock_context, chat_completions_request):
        """Test that messages are passed correctly to OpenAIRequestHandler."""
        mock_response = JSONResponse(content={"id": "chatcmpl-123"})

        with patch("minds.handlers.chat_completions_request_handler.OpenAIRequestHandler") as mock_handler_class:
            mock_handler_instance = AsyncMock()
            mock_handler_instance.proxy_chat_completions = AsyncMock(return_value=mock_response)
            mock_handler_class.create = AsyncMock(return_value=mock_handler_instance)

            with patch("minds.handlers.chat_completions_request_handler.setup_langfuse_observation"):
                with patch("minds.handlers.chat_completions_request_handler.get_langfuse_trace_id", return_value=None):
                    with patch("minds.handlers.chat_completions_request_handler.capture_langfuse_generation_context"):
                        await chat_completions_request_handler(
                            session=mock_session,
                            context=mock_context,
                            chat_completions_request=chat_completions_request,
                        )

                        call_kwargs = mock_handler_class.create.call_args[1]
                        assert call_kwargs["messages"] == chat_completions_request.messages

    @pytest.mark.asyncio
    async def test_handler_passes_streaming_flag(self, mock_session, mock_context):
        """Test that streaming flag is passed to handler."""
        streaming_request = ChatCompletionsRequest(
            model="latest:gpt-4o",
            messages=[Message(role="user", content="Hello")],
            stream=True,
        )

        mock_response = JSONResponse(content={"id": "chatcmpl-123"})

        with patch("minds.handlers.chat_completions_request_handler.OpenAIRequestHandler") as mock_handler_class:
            mock_handler_instance = AsyncMock()
            mock_handler_instance.proxy_chat_completions = AsyncMock(return_value=mock_response)
            mock_handler_class.create = AsyncMock(return_value=mock_handler_instance)

            with patch("minds.handlers.chat_completions_request_handler.setup_langfuse_observation"):
                with patch("minds.handlers.chat_completions_request_handler.get_langfuse_trace_id", return_value=None):
                    with patch("minds.handlers.chat_completions_request_handler.capture_langfuse_generation_context"):
                        await chat_completions_request_handler(
                            session=mock_session,
                            context=mock_context,
                            chat_completions_request=streaming_request,
                        )

                        call_kwargs = mock_handler_class.create.call_args[1]
                        assert call_kwargs["stream"] is True

    @pytest.mark.asyncio
    async def test_handler_passes_tools_and_tool_choice(self, mock_session, mock_context):
        """Test that tools and tool_choice are passed to handler."""
        request_with_tools = ChatCompletionsRequest(
            model="latest:sonnet",
            messages=[Message(role="user", content="Search for something")],
            stream=False,
            tools=[{"type": "function", "function": {"name": "search"}}],
            tool_choice="auto",
        )

        mock_response = JSONResponse(content={"id": "chatcmpl-123"})

        with patch("minds.handlers.chat_completions_request_handler.OpenAIRequestHandler") as mock_handler_class:
            mock_handler_instance = AsyncMock()
            mock_handler_instance.proxy_chat_completions = AsyncMock(return_value=mock_response)
            mock_handler_class.create = AsyncMock(return_value=mock_handler_instance)

            with patch("minds.handlers.chat_completions_request_handler.setup_langfuse_observation"):
                with patch("minds.handlers.chat_completions_request_handler.get_langfuse_trace_id", return_value=None):
                    with patch("minds.handlers.chat_completions_request_handler.capture_langfuse_generation_context"):
                        await chat_completions_request_handler(
                            session=mock_session,
                            context=mock_context,
                            chat_completions_request=request_with_tools,
                        )

                        call_kwargs = mock_handler_class.create.call_args[1]
                        assert call_kwargs["tools"] is not None
                        assert call_kwargs["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_handler_passes_temperature_and_max_tokens(self, mock_session, mock_context):
        """Test that sampling parameters are passed to handler."""
        request_with_params = ChatCompletionsRequest(
            model="latest:sonnet",
            messages=[Message(role="user", content="Hello")],
            stream=False,
            temperature=0.8,
            max_tokens=500,
        )

        mock_response = JSONResponse(content={"id": "chatcmpl-123"})

        with patch("minds.handlers.chat_completions_request_handler.OpenAIRequestHandler") as mock_handler_class:
            mock_handler_instance = AsyncMock()
            mock_handler_instance.proxy_chat_completions = AsyncMock(return_value=mock_response)
            mock_handler_class.create = AsyncMock(return_value=mock_handler_instance)

            with patch("minds.handlers.chat_completions_request_handler.setup_langfuse_observation"):
                with patch("minds.handlers.chat_completions_request_handler.get_langfuse_trace_id", return_value=None):
                    with patch("minds.handlers.chat_completions_request_handler.capture_langfuse_generation_context"):
                        await chat_completions_request_handler(
                            session=mock_session,
                            context=mock_context,
                            chat_completions_request=request_with_params,
                        )

                        call_kwargs = mock_handler_class.create.call_args[1]
                        assert call_kwargs["temperature"] == 0.8
                        assert call_kwargs["max_tokens"] == 500

    @pytest.mark.asyncio
    async def test_handler_returns_handler_response(self, mock_session, mock_context, chat_completions_request):
        """Test that handler returns the response from OpenAIRequestHandler."""
        expected_response = JSONResponse(
            content={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
            }
        )

        with patch("minds.handlers.chat_completions_request_handler.OpenAIRequestHandler") as mock_handler_class:
            mock_handler_instance = AsyncMock()
            mock_handler_instance.proxy_chat_completions = AsyncMock(return_value=expected_response)
            mock_handler_class.create = AsyncMock(return_value=mock_handler_instance)

            with patch("minds.handlers.chat_completions_request_handler.setup_langfuse_observation"):
                with patch("minds.handlers.chat_completions_request_handler.get_langfuse_trace_id", return_value=None):
                    with patch("minds.handlers.chat_completions_request_handler.capture_langfuse_generation_context"):
                        response = await chat_completions_request_handler(
                            session=mock_session,
                            context=mock_context,
                            chat_completions_request=chat_completions_request,
                        )

                        assert response == expected_response
