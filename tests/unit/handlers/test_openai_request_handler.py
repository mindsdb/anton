"""Tests for OpenAI request handler."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlmodel import Session
from starlette.responses import JSONResponse

from minds.handlers.openai_request_handler import OpenAIRequestHandler
from minds.inference.service import InferenceResult
from minds.inference.types import UsageBox
from minds.requests.context import Context
from minds.schemas.chat import Message


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    session = MagicMock(spec=Session)
    session.add = MagicMock()
    session.commit = MagicMock()
    return session


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
def sample_messages():
    """Create sample messages for testing."""
    return [
        Message(role="user", content="Hello, world!"),
    ]


@pytest.fixture
def mock_inference_result():
    """Create a mock InferenceResult with usage."""
    usage_box = UsageBox()
    usage_box.value = (100, 50)
    usage_box.output_payload = {"role": "assistant", "content": "Hello!"}

    return InferenceResult(
        config=MagicMock(
            model_name="claude-sonnet-4",
            alias="sonnet",
            label="anthropic",
            reasoning_effort=None,
            to_observability_metadata=MagicMock(return_value=MagicMock(to_metadata=MagicMock(return_value={}))),
        ),
        usage=(100, 50),
        output={"role": "assistant", "content": "Hello!"},
        artifacts=[],
        usage_box=usage_box,
    )


class TestOpenAIRequestHandlerInitialization:
    """Tests for OpenAIRequestHandler initialization."""

    def test_handler_initialization(self, mock_session, mock_context, sample_messages):
        """Test basic handler initialization."""
        handler = OpenAIRequestHandler(
            session=mock_session,
            context=mock_context,
            messages=sample_messages,
            model="latest:sonnet",
            stream=False,
        )

        assert handler.session == mock_session
        assert handler.context == mock_context
        assert handler.messages == sample_messages
        assert handler.model == "latest:sonnet"
        assert handler.stream is False
        assert handler.is_passthrough is True

    def test_handler_initialization_with_tools(self, mock_session, mock_context, sample_messages):
        """Test handler initialization with tools."""
        tools = [{"type": "function", "function": {"name": "search"}}]
        handler = OpenAIRequestHandler(
            session=mock_session,
            context=mock_context,
            messages=sample_messages,
            model="latest:sonnet",
            stream=False,
            tools=tools,
            tool_choice="auto",
        )

        assert handler.tools == tools
        assert handler.tool_choice == "auto"

    def test_handler_initialization_with_sampling_params(self, mock_session, mock_context, sample_messages):
        """Test handler initialization with sampling parameters."""
        handler = OpenAIRequestHandler(
            session=mock_session,
            context=mock_context,
            messages=sample_messages,
            model="latest:sonnet",
            stream=False,
            temperature=0.7,
            max_tokens=200,
        )

        assert handler.temperature == 0.7
        assert handler.max_tokens == 200

    @pytest.mark.asyncio
    async def test_handler_create_factory(self, mock_session, mock_context, sample_messages):
        """Test async factory method for handler creation."""
        with patch("minds.handlers.openai_request_handler.InferenceService"):
            handler = await OpenAIRequestHandler.create(
                session=mock_session,
                context=mock_context,
                messages=sample_messages,
                model="latest:sonnet",
                stream=False,
                request_id="test-req-id",
            )

            assert handler is not None
            assert handler.request_id == "test-req-id"
            assert handler.inference_service is not None


class TestOpenAIRequestHandlerUsageTracking:
    """Tests for usage tracking and database persistence."""

    @pytest.mark.asyncio
    async def test_save_usage_creates_chat_completion(self, mock_session, mock_context, sample_messages):
        """Test that _save_usage persists ChatCompletion to database."""
        handler = OpenAIRequestHandler(
            session=mock_session,
            context=mock_context,
            messages=sample_messages,
            model="latest:sonnet",
            stream=False,
            request_id="test-req-id",
            langfuse_trace_id="trace-id",
        )

        # Mock the update_generation_usage function
        with patch("minds.handlers.openai_request_handler.update_generation_usage"):
            handler._save_usage(usage=(100, 50))

        # Verify ChatCompletion was added to session
        mock_session.add.assert_called()
        mock_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_save_usage_with_zero_tokens(self, mock_session, mock_context, sample_messages):
        """Test save_usage with zero token usage."""
        handler = OpenAIRequestHandler(
            session=mock_session,
            context=mock_context,
            messages=sample_messages,
            model="latest:sonnet",
            stream=False,
            request_id="test-req-id",
        )

        with patch("minds.handlers.openai_request_handler.update_generation_usage"):
            handler._save_usage(usage=(0, 0))

        mock_session.add.assert_called()
        mock_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_save_usage_with_none_usage(self, mock_session, mock_context, sample_messages):
        """Test save_usage with None usage (error case)."""
        handler = OpenAIRequestHandler(
            session=mock_session,
            context=mock_context,
            messages=sample_messages,
            model="latest:sonnet",
            stream=False,
            request_id="test-req-id",
        )

        with patch("minds.handlers.openai_request_handler.update_generation_usage"):
            handler._save_usage(usage=None)

        # Should still persist with 0 tokens
        mock_session.add.assert_called()
        mock_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_save_usage_with_metadata(self, mock_session, mock_context, sample_messages):
        """Test save_usage with extra metadata (server artifacts)."""
        handler = OpenAIRequestHandler(
            session=mock_session,
            context=mock_context,
            messages=sample_messages,
            model="latest:sonnet",
            stream=False,
            request_id="test-req-id",
        )

        mock_result = MagicMock()
        mock_result.config = MagicMock(
            model_name="claude-sonnet-4",
            alias="sonnet",
            label="anthropic",
            reasoning_effort=None,
            to_observability_metadata=MagicMock(return_value=MagicMock(to_metadata=MagicMock(return_value={}))),
        )

        with patch("minds.handlers.openai_request_handler.update_generation_usage"):
            handler._save_usage(
                usage=(100, 50),
                extra_metadata={"server_artifacts": [{"type": "web_search_call", "query": "test"}]},
                result=mock_result,
            )

        mock_session.add.assert_called()
        mock_session.commit.assert_called()


class TestOpenAIRequestHandlerNonStreamingPath:
    """Tests for non-streaming chat completions proxy."""

    @pytest.mark.asyncio
    async def test_proxy_chat_completions_non_streaming_success(
        self, mock_session, mock_context, sample_messages, mock_inference_result
    ):
        """Test successful non-streaming chat completion."""
        mock_response = JSONResponse(
            content={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "claude-sonnet-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            }
        )

        handler = OpenAIRequestHandler(
            session=mock_session,
            context=mock_context,
            messages=sample_messages,
            model="latest:sonnet",
            stream=False,
            request_id="test-req-id",
        )

        with patch.object(handler, "inference_service") as mock_inference_service:
            mock_inference_service.inference = AsyncMock(return_value=(mock_response, mock_inference_result))

            with patch.object(handler, "_save_usage"):
                response = await handler.proxy_chat_completions()

                assert response is not None
                assert isinstance(response, JSONResponse)

    @pytest.mark.asyncio
    async def test_proxy_chat_completions_with_tool_calls(self, mock_session, mock_context, sample_messages):
        """Test non-streaming response with tool calls."""
        mock_response = JSONResponse(
            content={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "claude-sonnet-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "search", "arguments": '{"query": "test"}'},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            }
        )

        tool_call_output = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"query": "test"}'},
                }
            ],
        }

        mock_inference_result = InferenceResult(
            config=MagicMock(
                model_name="claude-sonnet-4",
                alias="sonnet",
                label="anthropic",
                reasoning_effort=None,
                to_observability_metadata=MagicMock(return_value=MagicMock(to_metadata=MagicMock(return_value={}))),
            ),
            usage=(100, 50),
            output=tool_call_output,
            artifacts=[],
            usage_box=None,
        )

        handler = OpenAIRequestHandler(
            session=mock_session,
            context=mock_context,
            messages=sample_messages,
            model="latest:sonnet",
            stream=False,
            request_id="test-req-id",
            tools=[{"type": "function", "function": {"name": "search"}}],
        )

        with patch.object(handler, "inference_service") as mock_inference_service:
            mock_inference_service.inference = AsyncMock(return_value=(mock_response, mock_inference_result))

            with patch.object(handler, "_save_usage"):
                response = await handler.proxy_chat_completions()

                assert response is not None
                assert isinstance(response, JSONResponse)


class TestOpenAIRequestHandlerErrorHandling:
    """Tests for error handling in proxy."""

    @pytest.mark.asyncio
    async def test_proxy_handles_provider_error_response(self, mock_session, mock_context, sample_messages):
        """Test handling of 5xx provider error response."""
        error_response = JSONResponse(
            content={
                "error": {
                    "message": "Provider service unavailable",
                    "type": "server_error",
                    "code": None,
                }
            },
            status_code=502,
        )

        mock_inference_result = InferenceResult(
            config=MagicMock(),
            usage=None,
            output=None,
            artifacts=[],
            usage_box=None,
        )

        handler = OpenAIRequestHandler(
            session=mock_session,
            context=mock_context,
            messages=sample_messages,
            model="latest:sonnet",
            stream=False,
            request_id="test-req-id",
        )

        with patch.object(handler, "inference_service") as mock_inference_service:
            mock_inference_service.inference = AsyncMock(return_value=(error_response, mock_inference_result))

            with patch.object(handler, "_save_usage"):
                response = await handler.proxy_chat_completions()

                # Error response should be passed through
                assert response.status_code == 502

    @pytest.mark.asyncio
    async def test_proxy_handles_authentication_error(self, mock_session, mock_context, sample_messages):
        """Test handling of authentication error (401)."""
        error_response = JSONResponse(
            content={"error": {"message": "Invalid API key", "type": "authentication_error"}},
            status_code=401,
        )

        mock_inference_result = InferenceResult(
            config=MagicMock(),
            usage=None,
            output=None,
            artifacts=[],
            usage_box=None,
        )

        handler = OpenAIRequestHandler(
            session=mock_session,
            context=mock_context,
            messages=sample_messages,
            model="latest:sonnet",
            stream=False,
            request_id="test-req-id",
        )

        with patch.object(handler, "inference_service") as mock_inference_service:
            mock_inference_service.inference = AsyncMock(return_value=(error_response, mock_inference_result))

            with patch.object(handler, "_save_usage"):
                response = await handler.proxy_chat_completions()

                assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_db_error_during_usage_save(self, mock_session, mock_context, sample_messages):
        """Test handling of database errors during usage save."""
        mock_session.commit.side_effect = Exception("Database connection lost")

        mock_response = JSONResponse(content={"id": "chatcmpl-123"})
        mock_inference_result = InferenceResult(
            config=MagicMock(),
            usage=(100, 50),
            output={"role": "assistant", "content": "Hello!"},
            artifacts=[],
            usage_box=None,
        )

        handler = OpenAIRequestHandler(
            session=mock_session,
            context=mock_context,
            messages=sample_messages,
            model="latest:sonnet",
            stream=False,
            request_id="test-req-id",
        )

        with patch.object(handler, "inference_service") as mock_inference_service:
            mock_inference_service.inference = AsyncMock(return_value=(mock_response, mock_inference_result))

            # DB error during save_usage should propagate
            with patch.object(
                handler, "_save_usage", side_effect=Exception("Database connection lost")
            ), pytest.raises(Exception, match="Database connection lost"):
                await handler.proxy_chat_completions()


class TestBuildPassthroughInputPayload:
    """Tests for input payload construction."""

    def test_build_input_payload(self, mock_session, mock_context, sample_messages):
        """Test building input payload for Langfuse."""
        handler = OpenAIRequestHandler(
            session=mock_session,
            context=mock_context,
            messages=sample_messages,
            model="latest:sonnet",
            stream=False,
            temperature=0.8,
            max_tokens=500,
            tools=[{"type": "function", "function": {"name": "search"}}],
            tool_choice="auto",
        )

        payload = handler._build_passthrough_input_payload()

        assert payload["model"] == "latest:sonnet"
        assert payload["stream"] is False
        assert len(payload["messages"]) == 1
        assert payload["temperature"] == 0.8
        assert payload["max_tokens"] == 500
        assert payload["tools"] is not None
        assert payload["tool_choice"] == "auto"

    def test_build_input_payload_without_optional_params(self, mock_session, mock_context, sample_messages):
        """Test building input payload without optional parameters."""
        handler = OpenAIRequestHandler(
            session=mock_session,
            context=mock_context,
            messages=sample_messages,
            model="latest:sonnet",
            stream=False,
        )

        payload = handler._build_passthrough_input_payload()

        assert payload["model"] == "latest:sonnet"
        assert payload["stream"] is False
        assert payload["temperature"] is None
        assert payload["max_tokens"] is None
        assert payload["tools"] is None
        assert payload["tool_choice"] is None
