"""Tests for GeminiAdapter implementation."""

from unittest.mock import patch

import pytest

from minds.common.passthrough_config import PassthroughModelConfig, ApiKind
from minds.inference.providers.gemini_adapter import GeminiAdapter
from minds.schemas.chat import Message, Role
from starlette.responses import JSONResponse, StreamingResponse


@pytest.mark.asyncio
async def test_gemini_adapter_complete_calls_proxy():
    """GeminiAdapter.complete() delegates to proxy_gemini."""
    adapter = GeminiAdapter()
    config = PassthroughModelConfig(
        api_kind=ApiKind.GEMINI_NATIVE,
        model_name="gemini-2.0-flash",
        api_key="test-key",
        label="gemini",
        alias="gemini",
    )
    messages = [Message(role=Role.user, content="test")]

    with patch("minds.inference.providers.gemini_adapter.gemini_module.proxy_gemini") as mock_proxy:
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
async def test_gemini_adapter_captures_usage():
    """GeminiAdapter captures usage from UsageBox."""
    adapter = GeminiAdapter()
    config = PassthroughModelConfig(
        api_kind=ApiKind.GEMINI_NATIVE,
        model_name="gemini-2.0-flash",
        api_key="test-key",
        label="gemini",
        alias="gemini",
    )
    messages = [Message(role=Role.user, content="test")]

    with patch("minds.inference.providers.gemini_adapter.gemini_module.proxy_gemini") as mock_proxy:
        def set_usage_box(**kwargs):
            usage_box = kwargs.get("usage_box")
            if usage_box:
                usage_box.value = (75, 35)
            return JSONResponse(content={})

        mock_proxy.side_effect = set_usage_box

        await adapter.complete(
            config=config,
            messages=messages,
            stream=False,
            request_id="req-123",
        )

        usage = adapter.get_last_usage()
        assert usage == (75, 35)


@pytest.mark.asyncio
async def test_gemini_adapter_captures_output():
    """GeminiAdapter captures output message from UsageBox."""
    adapter = GeminiAdapter()
    config = PassthroughModelConfig(
        api_kind=ApiKind.GEMINI_NATIVE,
        model_name="gemini-2.0-flash",
        api_key="test-key",
        label="gemini",
        alias="gemini",
    )
    messages = [Message(role=Role.user, content="test")]

    with patch("minds.inference.providers.gemini_adapter.gemini_module.proxy_gemini") as mock_proxy:
        assistant_msg = {"role": "assistant", "content": "hello from Gemini"}

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
async def test_gemini_adapter_captures_artifacts():
    """GeminiAdapter captures server artifacts from UsageBox."""
    adapter = GeminiAdapter()
    config = PassthroughModelConfig(
        api_kind=ApiKind.GEMINI_NATIVE,
        model_name="gemini-2.0-flash",
        api_key="test-key",
        label="gemini",
        alias="gemini",
    )
    messages = [Message(role=Role.user, content="test")]

    artifacts = [{"type": "search_result", "query": "test"}]

    with patch("minds.inference.providers.gemini_adapter.gemini_module.proxy_gemini") as mock_proxy:
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
async def test_gemini_adapter_returns_streaming_response():
    """GeminiAdapter returns StreamingResponse when stream=True."""
    adapter = GeminiAdapter()
    config = PassthroughModelConfig(
        api_kind=ApiKind.GEMINI_NATIVE,
        model_name="gemini-2.0-flash",
        api_key="test-key",
        label="gemini",
        alias="gemini",
    )
    messages = [Message(role=Role.user, content="test")]

    async def mock_stream_body():
        yield "data: {}\n\n"

    with patch("minds.inference.providers.gemini_adapter.gemini_module.proxy_gemini") as mock_proxy:
        mock_response = StreamingResponse(mock_stream_body(), media_type="text/event-stream")
        mock_proxy.return_value = mock_response

        response = await adapter.complete(
            config=config,
            messages=messages,
            stream=True,
            request_id="req-123",
        )

        assert isinstance(response, StreamingResponse)


def test_gemini_adapter_get_last_usage_default():
    """get_last_usage() returns None initially."""
    adapter = GeminiAdapter()
    assert adapter.get_last_usage() is None


def test_gemini_adapter_get_last_output_default():
    """get_last_output() returns None initially."""
    adapter = GeminiAdapter()
    assert adapter.get_last_output() is None


def test_gemini_adapter_get_last_artifacts_default():
    """get_last_artifacts() returns empty list initially."""
    adapter = GeminiAdapter()
    assert adapter.get_last_artifacts() == []
