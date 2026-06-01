"""Tests for InferenceService implementation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.responses import JSONResponse, StreamingResponse

from minds.common.passthrough_config import ApiKind
from minds.inference.model_resolver import ModelResolver
from minds.inference.providers.anthropic_adapter import AnthropicAdapter
from minds.inference.providers.gemini_adapter import GeminiAdapter
from minds.inference.providers.openai_adapter import OpenAIAdapter
from minds.inference.service import InferenceResult, InferenceService
from minds.schemas.chat import Message, Role


@pytest.fixture
def mock_settings():
    """Create a mock AppSettings with all providers configured."""
    settings = MagicMock()

    # Anthropic
    settings.anthropic.api_key = "test-anthropic-key"
    settings.anthropic.passthrough_sonnet_model = "claude-3-5-sonnet-20241022"
    settings.anthropic.passthrough_opus_model = "claude-3-opus-20240229"
    settings.anthropic.passthrough_haiku_model = "claude-3-haiku-20240307"

    # OpenAI
    settings.openai.api_key = "test-openai-key"
    settings.openai.api_url = None
    settings.openai.passthrough_gpt_model = "gpt-4o"
    settings.openai.passthrough_gpt_codex_model = "gpt-4o"
    settings.openai.passthrough_gpt_mini_model = "gpt-4o-mini"
    settings.openai.passthrough_gpt_nano_model = "gpt-4o-mini"

    # Gemini
    settings.gemini.api_key = "test-gemini-key"
    settings.gemini.passthrough_gemini_model = "gemini-2.0-flash"
    settings.gemini.passthrough_gemini_flash_model = "gemini-2.0-flash"

    # Fireworks
    settings.fireworks.api_key = "test-fireworks-key"
    settings.fireworks.anthropic_base_url = "https://api.fireworks.ai/account/v1/completions"
    settings.fireworks.passthrough_kimi_model = "accounts/fireworks/models/kimi-k2.5"
    settings.fireworks.passthrough_deepseek_model = "accounts/fireworks/models/deepseek-r1"
    settings.fireworks.passthrough_qwen_model = "accounts/fireworks/models/qwen-qwq"

    return settings


@pytest.fixture
def model_resolver(mock_settings):
    """Create a ModelResolver with mock settings."""
    return ModelResolver(mock_settings)


@pytest.fixture
def inference_service(model_resolver):
    """Create an InferenceService with a model resolver."""
    return InferenceService(model_resolver)


@pytest.fixture
def test_messages():
    """Create test messages."""
    return [Message(role=Role.user, content="Hello")]


class TestInferenceServiceResolveModel:
    """Test model resolution within inference."""

    @pytest.mark.asyncio
    async def test_inference_resolves_model_alias(self, inference_service, test_messages):
        """Resolve latest:sonnet to correct config."""
        with patch.object(inference_service, "_create_adapter") as mock_create_adapter:
            mock_adapter = AsyncMock()
            mock_adapter.complete = AsyncMock(return_value=JSONResponse(content={}))
            mock_adapter.get_last_usage.return_value = None
            mock_adapter.get_last_output.return_value = None
            mock_adapter.get_last_artifacts.return_value = []
            mock_create_adapter.return_value = mock_adapter

            response, result = await inference_service.inference(
                model_name="latest:sonnet",
                messages=test_messages,
                stream=False,
                request_id="req-123",
            )

            assert result.config.alias == "sonnet"
            assert result.config.model_name == "claude-3-5-sonnet-20241022"

    @pytest.mark.asyncio
    async def test_inference_model_resolver_error_propagates(self, inference_service, test_messages):
        """HTTPException from resolver is propagated."""
        with pytest.raises(HTTPException) as exc_info:
            await inference_service.inference(
                model_name="latest:unknown-alias",
                messages=test_messages,
                stream=False,
                request_id="req-123",
            )

        assert exc_info.value.status_code == 400
        assert "Unknown passthrough alias" in exc_info.value.detail


class TestInferenceServiceAdapterSelection:
    """Test adapter selection based on api_kind."""

    @pytest.mark.asyncio
    async def test_inference_calls_correct_adapter_openai(self, inference_service, test_messages):
        """OPENAI_RESPONSES routes to OpenAI adapter."""
        with patch.object(inference_service, "_create_adapter") as mock_create_adapter:
            mock_adapter = AsyncMock()
            mock_adapter.complete = AsyncMock(return_value=JSONResponse(content={}))
            mock_adapter.get_last_usage.return_value = (100, 50)
            mock_adapter.get_last_output.return_value = None
            mock_adapter.get_last_artifacts.return_value = []
            mock_create_adapter.return_value = mock_adapter

            await inference_service.inference(
                model_name="latest:gpt",
                messages=test_messages,
                stream=False,
                request_id="req-123",
            )

            mock_create_adapter.assert_called_once_with(ApiKind.OPENAI_RESPONSES)
            mock_adapter.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_inference_calls_correct_adapter_anthropic(self, inference_service, test_messages):
        """ANTHROPIC_MESSAGES routes to Anthropic adapter."""
        with patch.object(inference_service, "_create_adapter") as mock_create_adapter:
            mock_adapter = AsyncMock()
            mock_adapter.complete = AsyncMock(return_value=JSONResponse(content={}))
            mock_adapter.get_last_usage.return_value = (50, 25)
            mock_adapter.get_last_output.return_value = None
            mock_adapter.get_last_artifacts.return_value = []
            mock_create_adapter.return_value = mock_adapter

            await inference_service.inference(
                model_name="latest:sonnet",
                messages=test_messages,
                stream=False,
                request_id="req-123",
            )

            mock_create_adapter.assert_called_once_with(ApiKind.ANTHROPIC_MESSAGES)

    @pytest.mark.asyncio
    async def test_inference_calls_correct_adapter_gemini(self, inference_service, test_messages):
        """GEMINI_NATIVE routes to Gemini adapter."""
        with patch.object(inference_service, "_create_adapter") as mock_create_adapter:
            mock_adapter = AsyncMock()
            mock_adapter.complete = AsyncMock(return_value=JSONResponse(content={}))
            mock_adapter.get_last_usage.return_value = (75, 35)
            mock_adapter.get_last_output.return_value = None
            mock_adapter.get_last_artifacts.return_value = []
            mock_create_adapter.return_value = mock_adapter

            await inference_service.inference(
                model_name="latest:gemini",
                messages=test_messages,
                stream=False,
                request_id="req-123",
            )

            mock_create_adapter.assert_called_once_with(ApiKind.GEMINI_NATIVE)


class TestInferenceServiceResultCapture:
    """Test result creation and state capture."""

    @pytest.mark.asyncio
    async def test_inference_returns_response_and_result(self, inference_service, test_messages):
        """Return value is tuple of (response, InferenceResult)."""
        with patch.object(inference_service, "_create_adapter") as mock_create_adapter:
            mock_adapter = AsyncMock()
            expected_response = JSONResponse(content={"test": "data"})
            mock_adapter.complete = AsyncMock(return_value=expected_response)
            mock_adapter.get_last_usage.return_value = None
            mock_adapter.get_last_output.return_value = None
            mock_adapter.get_last_artifacts.return_value = []
            mock_create_adapter.return_value = mock_adapter

            response, result = await inference_service.inference(
                model_name="latest:sonnet",
                messages=test_messages,
                stream=False,
                request_id="req-123",
            )

            assert response is expected_response
            assert isinstance(result, InferenceResult)
            assert result.config.alias == "sonnet"

    @pytest.mark.asyncio
    async def test_inference_captures_usage(self, inference_service, test_messages):
        """Result captures usage from adapter."""
        with patch.object(inference_service, "_create_adapter") as mock_create_adapter:
            mock_adapter = MagicMock()
            mock_adapter.complete = AsyncMock(return_value=JSONResponse(content={}))
            mock_adapter.get_last_usage = MagicMock(return_value=(100, 50))
            mock_adapter.get_last_output = MagicMock(return_value=None)
            mock_adapter.get_last_artifacts = MagicMock(return_value=[])
            mock_create_adapter.return_value = mock_adapter

            response, result = await inference_service.inference(
                model_name="latest:sonnet",
                messages=test_messages,
                stream=False,
                request_id="req-123",
            )

            assert result.usage == (100, 50)

    @pytest.mark.asyncio
    async def test_inference_captures_output(self, inference_service, test_messages):
        """Result captures output message from adapter."""
        with patch.object(inference_service, "_create_adapter") as mock_create_adapter:
            expected_output = {"role": "assistant", "content": "Hello!"}
            mock_adapter = MagicMock()
            mock_adapter.complete = AsyncMock(return_value=JSONResponse(content={}))
            mock_adapter.get_last_usage = MagicMock(return_value=None)
            mock_adapter.get_last_output = MagicMock(return_value=expected_output)
            mock_adapter.get_last_artifacts = MagicMock(return_value=[])
            mock_create_adapter.return_value = mock_adapter

            response, result = await inference_service.inference(
                model_name="latest:sonnet",
                messages=test_messages,
                stream=False,
                request_id="req-123",
            )

            assert result.output == expected_output

    @pytest.mark.asyncio
    async def test_inference_captures_artifacts(self, inference_service, test_messages):
        """Result captures artifacts from adapter."""
        with patch.object(inference_service, "_create_adapter") as mock_create_adapter:
            artifacts = [{"type": "search_result", "query": "test"}]
            mock_adapter = MagicMock()
            mock_adapter.complete = AsyncMock(return_value=JSONResponse(content={}))
            mock_adapter.get_last_usage = MagicMock(return_value=None)
            mock_adapter.get_last_output = MagicMock(return_value=None)
            mock_adapter.get_last_artifacts = MagicMock(return_value=artifacts)
            mock_create_adapter.return_value = mock_adapter

            response, result = await inference_service.inference(
                model_name="latest:sonnet",
                messages=test_messages,
                stream=False,
                request_id="req-123",
            )

            assert result.artifacts == artifacts


class TestInferenceServiceStreaming:
    """Test streaming response handling."""

    @pytest.mark.asyncio
    async def test_inference_streaming_response(self, inference_service, test_messages):
        """Returns StreamingResponse when stream=True."""
        with patch.object(inference_service, "_create_adapter") as mock_create_adapter:

            async def mock_stream():
                yield "data: {}\n\n"

            mock_adapter = AsyncMock()
            expected_response = StreamingResponse(mock_stream(), media_type="text/event-stream")
            mock_adapter.complete = AsyncMock(return_value=expected_response)
            mock_adapter.get_last_usage.return_value = None
            mock_adapter.get_last_output.return_value = None
            mock_adapter.get_last_artifacts.return_value = []
            mock_create_adapter.return_value = mock_adapter

            response, result = await inference_service.inference(
                model_name="latest:sonnet",
                messages=test_messages,
                stream=True,
                request_id="req-123",
            )

            assert isinstance(response, StreamingResponse)

    @pytest.mark.asyncio
    async def test_inference_non_streaming_response(self, inference_service, test_messages):
        """Returns JSONResponse when stream=False."""
        with patch.object(inference_service, "_create_adapter") as mock_create_adapter:
            mock_adapter = AsyncMock()
            expected_response = JSONResponse(content={"test": "data"})
            mock_adapter.complete = AsyncMock(return_value=expected_response)
            mock_adapter.get_last_usage.return_value = None
            mock_adapter.get_last_output.return_value = None
            mock_adapter.get_last_artifacts.return_value = []
            mock_create_adapter.return_value = mock_adapter

            response, result = await inference_service.inference(
                model_name="latest:sonnet",
                messages=test_messages,
                stream=False,
                request_id="req-123",
            )

            assert isinstance(response, JSONResponse)


class TestInferenceServiceParameters:
    """Test parameter passing to adapters."""

    @pytest.mark.asyncio
    async def test_inference_passes_all_parameters(self, inference_service, test_messages):
        """All inference parameters are passed to adapter.complete()."""
        with patch.object(inference_service, "_create_adapter") as mock_create_adapter:
            mock_adapter = AsyncMock()
            mock_adapter.complete = AsyncMock(return_value=JSONResponse(content={}))
            mock_adapter.get_last_usage.return_value = None
            mock_adapter.get_last_output.return_value = None
            mock_adapter.get_last_artifacts.return_value = []
            mock_create_adapter.return_value = mock_adapter

            tools = [{"type": "web_search"}]
            tool_choice = "auto"
            temperature = 0.7
            max_tokens = 1000

            await inference_service.inference(
                model_name="latest:sonnet",
                messages=test_messages,
                stream=False,
                request_id="req-123",
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            call_args = mock_adapter.complete.call_args
            assert call_args.kwargs["tools"] == tools
            assert call_args.kwargs["tool_choice"] == tool_choice
            assert call_args.kwargs["temperature"] == temperature
            assert call_args.kwargs["max_tokens"] == max_tokens


class TestInferenceServiceAdapterFactory:
    """Test the _create_adapter factory method."""

    def test_create_adapter_openai(self, inference_service):
        """_create_adapter creates OpenAIAdapter for OPENAI_RESPONSES."""
        adapter = inference_service._create_adapter(ApiKind.OPENAI_RESPONSES)
        assert isinstance(adapter, OpenAIAdapter)

    def test_create_adapter_anthropic(self, inference_service):
        """_create_adapter creates AnthropicAdapter for ANTHROPIC_MESSAGES."""
        adapter = inference_service._create_adapter(ApiKind.ANTHROPIC_MESSAGES)
        assert isinstance(adapter, AnthropicAdapter)

    def test_create_adapter_gemini(self, inference_service):
        """_create_adapter creates GeminiAdapter for GEMINI_NATIVE."""
        adapter = inference_service._create_adapter(ApiKind.GEMINI_NATIVE)
        assert isinstance(adapter, GeminiAdapter)

    def test_create_adapter_unknown_raises(self, inference_service):
        """_create_adapter raises ValueError for unknown api_kind."""
        with pytest.raises(ValueError, match="Unknown api_kind"):
            # Create a mock ApiKind value that doesn't exist
            inference_service._create_adapter("invalid_kind")  # type: ignore
