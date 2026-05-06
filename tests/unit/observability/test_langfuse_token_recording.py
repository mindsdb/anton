"""End-to-end Langfuse token-recording tests.

These tests exercise the real Langfuse v3 SDK and the real production code
paths (no patching of update_generation_usage / update_current_generation /
start_observation) and assert on the actual OpenTelemetry spans that
Langfuse emits. They are the strongest evidence that token usage is being
recorded correctly: a regression in any of the wiring — from
chat_completions_request_handler down through OpenAIRequestHandler._save_usage
into update_generation_usage and through to the Langfuse SDK — will surface
here as a missing or wrong span attribute.

Mocking surface:
- The agent layer (PassthroughAgent.proxy / agent.run / get_last_run_usage)
  is mocked because we don't want to make real upstream LLM calls.
- The DB session is mocked because we don't want to write to Postgres.
- Langfuse's OTLP HTTP exporter is replaced with a no-op so spans don't try
  to leave the test process.
- Everything else — @observe, contextvars, span creation, attribute encoding,
  TraceContext propagation, parent/child relationships — is real.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

import pytest
from sqlmodel import Session
from starlette.responses import JSONResponse, StreamingResponse

from minds.handlers.openai_request_handler import OpenAIRequestHandler
from minds.requests.context import Context
from minds.requests.langfuse_tracing import (
    capture_langfuse_generation_context,
    setup_langfuse_observation,
)
from minds.requests.stream import MessageStreamer
from minds.schemas.chat import Message, Role


# NOTE: ``from langfuse import observe`` is INTENTIONALLY done lazily here.
# Other test modules' conftests (e.g.
# ``tests/unit/agents/candidate_sql_agent/conftest.py``) stuff a fake
# ``langfuse`` module into ``sys.modules`` at COLLECTION time. The
# ``langfuse_capture`` session fixture restores the real package and re-binds
# ``get_client`` on every already-loaded ``minds.*`` module, but a top-level
# ``from langfuse import observe`` here would have been bound to the fake
# no-op decorator BEFORE the fixture had a chance to run.
def _real_observe(**kwargs):
    """Resolve ``langfuse.observe`` from the current ``sys.modules`` so it
    points at the real Langfuse decorator after the fixture has restored it."""
    from langfuse import observe as _observe

    return _observe(**kwargs)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_session():
    return Mock(spec=Session)


@pytest.fixture
def mock_context():
    return Context(
        user_id=UUID("00000000-0000-0000-0000-000000000001"),
        organization_id=UUID("00000000-0000-0000-0000-000000000002"),
    )


@pytest.fixture
def sample_messages():
    return [Message(role=Role.user, content="Hello")]


def _build_handler(
    mock_session,
    mock_context,
    sample_messages,
    *,
    stream: bool,
    model: str = "gpt-4o-mini",
    is_passthrough: bool = False,
    langfuse_trace_context: dict | None = None,
) -> OpenAIRequestHandler:
    handler = OpenAIRequestHandler(
        session=mock_session,
        context=mock_context,
        mindsdb_client=Mock(),
        messages=sample_messages,
        model=model,
        stream=stream,
        metadata=None,
        instrument=True,
        langfuse_trace_context=langfuse_trace_context,
    )
    handler.is_passthrough = is_passthrough
    return handler


# ---------------------------------------------------------------------------
# Passthrough chat-completions (the path the user originally noticed broken)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_passthrough_non_streaming_records_usage_on_parent_generation(
    langfuse_capture, mock_session, mock_context, sample_messages
):
    """Non-streaming path: the @observe-created generation itself carries
    the token usage attached via update_current_generation."""

    handler = _build_handler(mock_session, mock_context, sample_messages, stream=False, is_passthrough=True)
    handler.agent = Mock()
    handler.agent.proxy = AsyncMock(return_value=Mock(spec=JSONResponse))
    handler.agent.get_last_run_usage = AsyncMock(return_value=(11, 7))

    @_real_observe(name="Chat Completions Handler v1", as_type="generation")
    async def run_through():
        # Mirror what chat_completions_request_handler does.
        setup_langfuse_observation(context=mock_context)
        handler.langfuse_trace_context = capture_langfuse_generation_context()
        return await handler.proxy_chat_completions()

    await run_through()

    spans = langfuse_capture.get_spans()
    parents = [s for s in spans if s.name == "Chat Completions Handler v1"]
    assert len(parents) == 1, f"Expected 1 parent generation, got {[s.name for s in spans]}"
    parent = parents[0]

    assert langfuse_capture.observation_type(parent) == "generation"
    assert langfuse_capture.model_name(parent) == "gpt-4o-mini"
    assert langfuse_capture.usage_details(parent) == {"input": 11, "output": 7, "total": 18}

    # No detached child generation in non-streaming mode.
    children = [s for s in spans if s.name == "llm-usage"]
    assert children == []


@pytest.mark.asyncio
async def test_passthrough_streaming_attaches_child_generation_with_usage(
    langfuse_capture, mock_session, mock_context, sample_messages
):
    """Streaming path: the @observe scope closes BEFORE token counts are known
    (the body iterator drains after the route handler returns). Verify a real
    child generation observation, attached to the SAME trace via TraceContext,
    carries the usage_details and the right model name."""

    async def _upstream_body():
        yield b"chunk-1"
        yield b"chunk-2"

    upstream_response = StreamingResponse(_upstream_body(), media_type="text/event-stream")

    handler = _build_handler(mock_session, mock_context, sample_messages, stream=True, is_passthrough=True)
    handler.agent = Mock()
    handler.agent.proxy = AsyncMock(return_value=upstream_response)
    handler.agent.get_last_run_usage = AsyncMock(return_value=(7, 11))

    @_real_observe(name="Chat Completions Handler v1", as_type="generation")
    async def run_through():
        setup_langfuse_observation(context=mock_context)
        handler.langfuse_trace_context = capture_langfuse_generation_context()
        return await handler.proxy_chat_completions()

    wrapped = await run_through()

    # Drain the streaming body — this is what FastAPI/Starlette would do.
    chunks = []
    async for c in wrapped.body_iterator:
        chunks.append(c)
    assert chunks == [b"chunk-1", b"chunk-2"]

    spans = langfuse_capture.get_spans()

    parents = [s for s in spans if s.name == "Chat Completions Handler v1"]
    children = [s for s in spans if s.name == "llm-usage"]
    assert len(parents) == 1, f"Expected 1 parent, got {[s.name for s in spans]}"
    assert len(children) == 1, f"Expected 1 'llm-usage' child, got {[s.name for s in spans]}"

    parent = parents[0]
    child = children[0]

    # Same trace; child is parented at the @observe span.
    assert child.context.trace_id == parent.context.trace_id, (
        "child generation must live in the same Langfuse trace as the parent @observe span"
    )
    assert child.parent is not None
    assert child.parent.span_id == parent.context.span_id, (
        "child generation must declare the @observe span as its parent"
    )

    # The child carries the token usage and model.
    assert langfuse_capture.observation_type(child) == "generation"
    assert langfuse_capture.model_name(child) == "gpt-4o-mini"
    assert langfuse_capture.usage_details(child) == {"input": 7, "output": 11, "total": 18}

    # The parent itself does NOT carry usage — that's the streaming trade-off
    # the team explicitly chose during planning. The child rolls up to the trace.
    assert langfuse_capture.usage_details(parent) is None


# ---------------------------------------------------------------------------
# Responses API
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_responses_non_streaming_records_usage_on_parent_generation(
    langfuse_capture, mock_session, mock_context, sample_messages
):
    """Responses API non-streaming: parent @observe generation carries usage."""

    handler = _build_handler(mock_session, mock_context, sample_messages, stream=False, model="custom-mind")

    agent_response = Mock()
    agent_response.answer = "the answer"
    agent_response.sql = "SELECT 1"
    handler.agent = Mock()
    handler.agent.run = AsyncMock(return_value=agent_response)
    handler.agent.get_last_run_usage = AsyncMock(return_value=(20, 5))

    streamer = Mock(spec=MessageStreamer)
    streamer.push = AsyncMock()

    message = Mock()
    message.conversation_id = UUID("00000000-0000-0000-0000-0000000000aa")
    message.id = UUID("00000000-0000-0000-0000-0000000000bb")

    with patch("minds.handlers.openai_request_handler.ConversationsService") as mock_conv_service_cls:
        conv_service = Mock()
        conv_service.update_conversation_message_content = AsyncMock()
        mock_conv_service_cls.return_value = conv_service

        @_real_observe(name="Responses Handler v1", as_type="generation")
        async def run_through():
            setup_langfuse_observation(context=mock_context)
            handler.langfuse_trace_context = capture_langfuse_generation_context()
            await handler.responses(streamer=streamer, message=message)

        await run_through()

    spans = langfuse_capture.get_spans()
    parents = [s for s in spans if s.name == "Responses Handler v1"]
    assert len(parents) == 1
    parent = parents[0]

    assert langfuse_capture.observation_type(parent) == "generation"
    assert langfuse_capture.model_name(parent) == "custom-mind"
    assert langfuse_capture.usage_details(parent) == {"input": 20, "output": 5, "total": 25}

    children = [s for s in spans if s.name == "llm-usage"]
    assert children == []


@pytest.mark.asyncio
async def test_responses_streaming_attaches_child_generation_with_usage(
    langfuse_capture, mock_session, mock_context, sample_messages
):
    """Responses API streaming: the work happens inside a producer task that
    runs after the @observe scope closes (process_streaming_producer). The
    captured trace_context routes the usage update to a child generation
    bound to the same trace.

    We simulate the producer-task ordering by capturing the trace_context
    inside @observe, exiting the scope, then awaiting handler.responses with
    handler.langfuse_trace_context already populated and stream=True.
    """

    handler = _build_handler(mock_session, mock_context, sample_messages, stream=True, model="custom-mind")

    agent_response = Mock()
    agent_response.answer = "streamed answer"
    agent_response.sql = None
    handler.agent = Mock()
    handler.agent.run = AsyncMock(return_value=agent_response)
    handler.agent.get_last_run_usage = AsyncMock(return_value=(50, 25))

    streamer = Mock(spec=MessageStreamer)
    streamer.push = AsyncMock()

    message = Mock()
    message.conversation_id = UUID("00000000-0000-0000-0000-0000000000aa")
    message.id = UUID("00000000-0000-0000-0000-0000000000bb")

    @_real_observe(name="Responses Handler v1", as_type="generation")
    async def capture_only():
        # Mirror the request handler: capture context inside @observe scope,
        # then return WITHOUT doing any work that would attach usage to the
        # parent. This emulates how the real handler hands a producer to
        # process_streaming_producer and returns the StreamingResponse.
        setup_langfuse_observation(context=mock_context)
        handler.langfuse_trace_context = capture_langfuse_generation_context()

    await capture_only()
    # @observe scope is now CLOSED. Producer task fires here:
    with patch("minds.handlers.openai_request_handler.ConversationsService") as mock_conv_service_cls:
        conv_service = Mock()
        conv_service.update_conversation_message_content = AsyncMock()
        mock_conv_service_cls.return_value = conv_service
        await handler.responses(streamer=streamer, message=message)

    spans = langfuse_capture.get_spans()
    parents = [s for s in spans if s.name == "Responses Handler v1"]
    children = [s for s in spans if s.name == "llm-usage"]
    assert len(parents) == 1
    assert len(children) == 1

    parent = parents[0]
    child = children[0]

    assert child.context.trace_id == parent.context.trace_id
    assert child.parent is not None
    assert child.parent.span_id == parent.context.span_id

    assert langfuse_capture.model_name(child) == "custom-mind"
    assert langfuse_capture.usage_details(child) == {"input": 50, "output": 25, "total": 75}

    # Parent generation has no usage in the streaming path.
    assert langfuse_capture.usage_details(parent) is None


# ---------------------------------------------------------------------------
# Anchor: prove the fixture itself is correctly wired (would fail loudly if
# Langfuse stopped being OTel-based or if our OTLP no-op patch broke).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_capture_fixture_actually_records_real_langfuse_spans(langfuse_capture):
    """Smoke test: a bare @observe generation with usage_details must produce
    exactly one OTel span carrying the right Langfuse attributes. If this
    fails, every other test in this file is meaningless."""
    from langfuse import get_client

    @_real_observe(name="anchor", as_type="generation")
    def f():
        get_client().update_current_generation(model="m", usage_details={"input": 1, "output": 2, "total": 3})

    f()

    spans = langfuse_capture.get_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "anchor"
    assert langfuse_capture.observation_type(span) == "generation"
    assert langfuse_capture.usage_details(span) == {"input": 1, "output": 2, "total": 3}
