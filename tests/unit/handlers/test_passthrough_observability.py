"""Tests for the passthrough Langfuse observability surface.

Covers the three additions to ``OpenAIRequestHandler.proxy_chat_completions``:

1. The inbound request shape is built into an ``input_payload`` and passed
   to ``update_generation_usage`` so traces become eval-replayable.
2. Upstream provider errors (``response.status_code >= 400``) still produce
   a Langfuse generation — the JSON error blob is captured as ``output``.
3. Each ``tool_call`` returned by the model produces one Langfuse child
   span via ``record_tool_call_spans``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

import pytest
from sqlmodel import Session
from starlette.responses import JSONResponse, StreamingResponse

from minds.agents.passthrough_agent.agent import PassthroughAgent
from minds.common.passthrough_config import ApiKind, PassthroughModelConfig, WebSearchMode
from minds.handlers.openai_request_handler import (
    OpenAIRequestHandler,
    _extract_jsonresponse_content,
)
from minds.requests.context import Context
from minds.schemas.chat import Message, Role

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def passthrough_config() -> PassthroughModelConfig:
    return PassthroughModelConfig(
        api_kind=ApiKind.ANTHROPIC_MESSAGES,
        model_name="claude-sonnet-4-6",
        api_key="key",
        web_search_mode=WebSearchMode.ANTHROPIC_NATIVE,
        label="anthropic",
        alias="sonnet",
    )


@pytest.fixture
def context() -> Context:
    return Context(
        user_id=UUID("00000000-0000-0000-0000-000000000001"),
        organization_id=UUID("00000000-0000-0000-0000-000000000002"),
    )


@pytest.fixture
def messages() -> list[Message]:
    return [Message(role=Role.user, content="hello")]


def _build_handler(
    *,
    context: Context,
    messages: list[Message],
    config: PassthroughModelConfig,
    agent_output: dict | None,
    agent_usage: tuple[int, int] | None = (5, 7),
    server_artifacts: list[dict] | None = None,
    response: JSONResponse | StreamingResponse,
) -> tuple[OpenAIRequestHandler, PassthroughAgent]:
    """Build an OpenAIRequestHandler wired to a PassthroughAgent stub."""
    agent = PassthroughAgent(config=config, instrument=False)
    agent.proxy = AsyncMock(return_value=response)
    # Stamp the box so the handler reads our expected values.
    agent._usage_box.value = agent_usage
    agent._usage_box.output_payload = agent_output
    if server_artifacts:
        agent._usage_box.server_artifacts.extend(server_artifacts)

    handler = OpenAIRequestHandler(
        session=Mock(spec=Session),
        context=context,
        mindsdb_client=Mock(),
        messages=messages,
        model="latest:sonnet",
        stream=False,
        metadata=None,
        instrument=True,
        tools=[{"type": "function", "function": {"name": "search", "parameters": {}}}],
        tool_choice="auto",
        temperature=0.2,
        max_tokens=128,
    )
    handler.agent = agent
    handler.is_passthrough = True
    return handler, agent


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_streaming_success_captures_input_and_output_on_langfuse(context, messages, passthrough_config):
    agent_output = {"role": "assistant", "content": "hi back"}
    response = JSONResponse(content={"choices": [{"message": agent_output}]}, status_code=200)
    handler, _ = _build_handler(
        context=context,
        messages=messages,
        config=passthrough_config,
        agent_output=agent_output,
        response=response,
    )

    with patch("minds.handlers.openai_request_handler.update_generation_usage") as mock_update:
        out = await handler.proxy_chat_completions()

    assert out is response
    mock_update.assert_called_once()
    kwargs = mock_update.call_args.kwargs
    # Input payload mirrors the request the handler saw.
    assert kwargs["input"]["model"] == "latest:sonnet"
    assert kwargs["input"]["messages"][0]["content"] == "hello"
    assert kwargs["input"]["tools"][0]["function"]["name"] == "search"
    # Output payload is the assistant message from the agent.
    assert kwargs["output"] == agent_output
    # Concrete upstream model goes to Langfuse (so cost rollup hits the registry).
    assert kwargs["model"] == "claude-sonnet-4-6"
    # Alias + provider land in metadata.
    assert kwargs["metadata"]["passthrough_alias"] == "sonnet"
    assert kwargs["metadata"]["provider"] == "anthropic"


@pytest.mark.asyncio
async def test_upstream_error_response_still_recorded_on_langfuse(context, messages, passthrough_config):
    error_body = {"error": {"message": "boom", "type": "api_error"}}
    response = JSONResponse(content=error_body, status_code=502)
    handler, _ = _build_handler(
        context=context,
        messages=messages,
        config=passthrough_config,
        agent_output=None,  # error path: no successful output
        agent_usage=None,
        response=response,
    )

    with patch("minds.handlers.openai_request_handler.update_generation_usage") as mock_update:
        out = await handler.proxy_chat_completions()

    assert out is response
    mock_update.assert_called_once()
    kwargs = mock_update.call_args.kwargs
    # Usage zeroed but call still made — the trace doesn't disappear.
    assert kwargs["usage"] == (0, 0)
    # The decoded error blob lands as output.
    assert kwargs["output"]["error"]["message"] == "boom"
    # Error metadata for filtering.
    assert kwargs["metadata"]["status_code"] == 502
    assert kwargs["metadata"]["level"] == "ERROR"


@pytest.mark.asyncio
async def test_tool_calls_emit_child_spans(context, messages, passthrough_config):
    agent_output = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"q": "milk"}'},
            }
        ],
    }
    response = JSONResponse(content={"choices": [{"message": agent_output}]}, status_code=200)
    handler, _ = _build_handler(
        context=context,
        messages=messages,
        config=passthrough_config,
        agent_output=agent_output,
        response=response,
    )

    with (
        patch("minds.handlers.openai_request_handler.update_generation_usage"),
        patch("minds.handlers.openai_request_handler.record_tool_call_spans") as mock_spans,
    ):
        await handler.proxy_chat_completions()

    mock_spans.assert_called_once()
    kwargs = mock_spans.call_args.kwargs
    assert kwargs["tool_calls"] == agent_output["tool_calls"]
    # Metadata flows through so spans inherit alias/provider.
    assert kwargs["metadata"]["passthrough_alias"] == "sonnet"


@pytest.mark.asyncio
async def test_server_artifacts_attached_to_metadata(context, messages, passthrough_config):
    agent_output = {"role": "assistant", "content": "with citations"}
    artifacts = [{"type": "web_search_tool_result", "tool_use_id": "tu_1"}]
    response = JSONResponse(content={"choices": [{"message": agent_output}]}, status_code=200)
    handler, _ = _build_handler(
        context=context,
        messages=messages,
        config=passthrough_config,
        agent_output=agent_output,
        server_artifacts=artifacts,
        response=response,
    )

    with patch("minds.handlers.openai_request_handler.update_generation_usage") as mock_update:
        await handler.proxy_chat_completions()

    kwargs = mock_update.call_args.kwargs
    assert kwargs["metadata"]["server_artifacts"] == artifacts


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestExtractJsonResponseContent:
    def test_decodes_dict_payload(self):
        resp = JSONResponse(content={"hello": "world"})
        assert _extract_jsonresponse_content(resp) == {"hello": "world"}

    def test_handles_undecodable_body(self):
        resp = JSONResponse(content={"a": 1})
        resp.body = b"\xff\xfe not json"  # type: ignore[attr-defined]
        out = _extract_jsonresponse_content(resp)
        assert "raw" in out
