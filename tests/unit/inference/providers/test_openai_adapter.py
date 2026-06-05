"""Tests for OpenAIAdapter implementation."""

from unittest.mock import patch

import pytest
from starlette.responses import JSONResponse, StreamingResponse

from minds.inference.providers.openai_adapter import OpenAIAdapter
from minds.inference.types import ApiKind, PassthroughModelConfig
from minds.schemas.chat import Message, Role


@pytest.fixture
def openai_config():
    """Create a test OpenAI config."""

    return PassthroughModelConfig(
        api_kind=ApiKind.OPENAI_RESPONSES,
        model_name="gpt-4",
        api_key="test-key",
        label="openai",
        alias="gpt-4",
    )


@pytest.fixture
def test_messages():
    """Create test messages."""
    return [Message(role=Role.user, content="Hello")]


@pytest.mark.asyncio
async def test_openai_adapter_complete_calls_proxy():
    """OpenAIAdapter.complete() delegates to proxy_openai."""
    adapter = OpenAIAdapter()
    config = PassthroughModelConfig(
        api_kind=ApiKind.OPENAI_RESPONSES,
        model_name="gpt-4",
        api_key="test-key",
        label="openai",
        alias="gpt-4",
    )
    messages = [Message(role=Role.user, content="test")]

    with patch("minds.inference.providers.openai_adapter.openai_module.proxy_openai") as mock_proxy:
        mock_response = JSONResponse(content={"choices": [{"message": {"role": "assistant", "content": "hi"}}]})
        mock_proxy.return_value = mock_response
        response = await adapter.complete(
            config=config,
            messages=messages,
            stream=False,
            request_id="req-123",
        )

        assert response == mock_response
        mock_proxy.assert_called_once()


@pytest.mark.asyncio
async def test_openai_adapter_captures_usage():
    """OpenAIAdapter captures usage from UsageBox."""
    adapter = OpenAIAdapter()
    config = PassthroughModelConfig(
        api_kind=ApiKind.OPENAI_RESPONSES,
        model_name="gpt-4",
        api_key="test-key",
        label="openai",
        alias="gpt-4",
    )
    messages = [Message(role=Role.user, content="test")]
    with patch("minds.inference.providers.openai_adapter.openai_module.proxy_openai") as mock_proxy:

        def set_usage_box(**kwargs):
            usage_box = kwargs.get("usage_box")
            if usage_box:
                usage_box.value = (100, 50)
            return JSONResponse(content={})

        mock_proxy.side_effect = set_usage_box

        await adapter.complete(
            config=config,
            messages=messages,
            stream=False,
            request_id="req-123",
        )
        usage = adapter.get_last_usage()
        assert usage == (100, 50)


@pytest.mark.asyncio
async def test_openai_adapter_captures_output():
    """OpenAIAdapter captures output message from UsageBox."""
    adapter = OpenAIAdapter()
    config = PassthroughModelConfig(
        api_kind=ApiKind.OPENAI_RESPONSES,
        model_name="gpt-4",
        api_key="test-key",
        label="openai",
        alias="gpt-4",
    )
    messages = [Message(role=Role.user, content="test")]

    with patch("minds.inference.providers.openai_adapter.openai_module.proxy_openai") as mock_proxy:
        assistant_msg = {"role": "assistant", "content": "hello"}

        def set_output(**kwargs):
            usage_box = kwargs.get("usage_box")
            if usage_box:
                usage_box.output_payload = assistant_msg
            return JSONResponse(content={})

        mock_proxy.side_effect = set_output
        await adapter.complete(
            config=config,
            messages=messages,
            stream=False,
            request_id="req-123",
        )

        output = adapter.get_last_output()
        assert output == assistant_msg


@pytest.mark.asyncio
async def test_openai_adapter_captures_artifacts():
    """OpenAIAdapter captures server artifacts from UsageBox."""
    adapter = OpenAIAdapter()
    config = PassthroughModelConfig(
        api_kind=ApiKind.OPENAI_RESPONSES,
        model_name="gpt-4",
        api_key="test-key",
        label="openai",
        alias="gpt-4",
    )
    messages = [Message(role=Role.user, content="test")]
    artifacts = [{"type": "web_search", "query": "test query"}]

    with patch("minds.inference.providers.openai_adapter.openai_module.proxy_openai") as mock_proxy:

        def set_artifacts(**kwargs):
            usage_box = kwargs.get("usage_box")
            if usage_box:
                usage_box.server_artifacts = artifacts
            return JSONResponse(content={})

        mock_proxy.side_effect = set_artifacts
        await adapter.complete(
            config=config,
            messages=messages,
            stream=False,
            request_id="req-123",
        )

        captured = adapter.get_last_artifacts()
        assert captured == artifacts


@pytest.mark.asyncio
async def test_openai_adapter_returns_streaming_response():
    """OpenAIAdapter returns StreamingResponse when stream=True."""
    adapter = OpenAIAdapter()
    config = PassthroughModelConfig(
        api_kind=ApiKind.OPENAI_RESPONSES,
        model_name="gpt-4",
        api_key="test-key",
        label="openai",
        alias="gpt-4",
    )
    messages = [Message(role=Role.user, content="test")]

    async def mock_stream_body():
        yield "data: {}\n\n"

    with patch("minds.inference.providers.openai_adapter.openai_module.proxy_openai") as mock_proxy:
        mock_response = StreamingResponse(mock_stream_body(), media_type="text/event-stream")
        mock_proxy.return_value = mock_response
        response = await adapter.complete(
            config=config,
            messages=messages,
            stream=True,
            request_id="req-123",
        )

        assert isinstance(response, StreamingResponse)


def test_openai_adapter_get_last_usage_default():
    """get_last_usage() returns None initially."""
    adapter = OpenAIAdapter()
    assert adapter.get_last_usage() is None


def test_openai_adapter_get_last_output_default():
    """get_last_output() returns None initially."""
    adapter = OpenAIAdapter()
    assert adapter.get_last_output() is None


def test_openai_adapter_get_last_artifacts_default():
    """get_last_artifacts() returns empty list initially."""
    adapter = OpenAIAdapter()
    assert adapter.get_last_artifacts() == []
