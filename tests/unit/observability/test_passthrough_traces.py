"""End-to-end passthrough Langfuse trace tests.

These tests run the entire passthrough chain end-to-end — from the
production `chat_completions_request_handler` entry point through the
``@observe`` decorator, `OpenAIRequestHandler.create` + alias resolution,
`PassthroughAgent.proxy`, the per-provider translator + response decoder,
`_save_usage`, `update_generation_usage`, and `record_tool_call_spans` —
against the **real** Langfuse v3 SDK. Assertions inspect the **actual**
OpenTelemetry spans Langfuse emits. The only seam patched is the upstream
provider SDK client (we don't make real Anthropic API calls); every line
of translation / capture / Langfuse instrumentation runs unmodified.

The DB session and limits service are also stubbed because the test
process has no Postgres / Statsig. ``resolve_passthrough_model`` is
patched to skip env-var-dependent settings lookup so the test runs in any
environment.

The result is concrete proof — not call-shape mocks — that a request
through ``latest:sonnet`` produces a trace that is eval-replayable and
properly grouped into Langfuse sessions:

- The parent generation carries the input request (model / messages /
  tools / tool_choice / temperature / max_tokens) and the output assistant
  message (content + tool_calls), plus token usage and provider metadata.
- Each ``tool_call`` produces a ``tool:<name>`` child span with parsed
  JSON arguments visible in the UI and a ``call_id`` for correlation.
- Upstream errors still produce a generation — with ``level=ERROR`` and
  the error body as ``output`` — so failed requests don't silently
  disappear from observability.
- Server-side intermediates (Anthropic ``server_tool_use`` /
  ``web_search_tool_result``) land on metadata even though they are
  deliberately not in the client response.
- The ``Langfuse-Session-Id`` / ``Langfuse-Tags`` / ``Langfuse-Metadata``
  headers materialize as ``session.id``, ``langfuse.trace.tags``,
  ``langfuse.trace.name``, and trace-level metadata.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

import httpx
import pytest
from anthropic import APIStatusError as AnthropicAPIStatusError
from sqlmodel import Session

from minds.common.passthrough_config import ApiKind, PassthroughModelConfig, WebSearchMode
from minds.requests.chat_completions_request import ChatCompletionsRequest
from minds.requests.context import Context
from minds.schemas.chat import Message, Role

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
def sonnet_config() -> PassthroughModelConfig:
    """A realistic ``latest:sonnet`` resolved config used by all tests.

    We patch ``resolve_passthrough_model`` to return this rather than
    reading API keys from settings, so tests run identically on any
    machine (CI included).
    """
    return PassthroughModelConfig(
        api_kind=ApiKind.ANTHROPIC_MESSAGES,
        model_name="claude-sonnet-4-6",
        api_key="key-for-tests",
        web_search_mode=WebSearchMode.ANTHROPIC_NATIVE,
        label="anthropic",
        alias="sonnet",
    )


def _request_with_tools(
    *,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
    stream: bool = False,
    content: str = "What's the weather in Paris?",
) -> ChatCompletionsRequest:
    """Build a realistic inbound ChatCompletionsRequest aimed at latest:sonnet."""
    return ChatCompletionsRequest(
        model="latest:sonnet",
        messages=[Message(role=Role.user, content=content)],
        stream=stream,
        tools=tools,
        tool_choice=tool_choice,
        temperature=0.0,
        max_tokens=256,
    )


# ---------------------------------------------------------------------------
# Anthropic SDK response builders
#
# The provider proxy reads response objects via attribute access only
# (``response.content``, ``block.type``, ``block.text``, ``response.usage``,
# etc.), so plain SimpleNamespace stand-ins suffice — no need to build
# real SDK pydantic models, which keeps the fixtures readable.
# ---------------------------------------------------------------------------


def _text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(*, id: str, name: str, input: dict):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input)


def _server_tool_use_block(*, id: str, name: str, input: dict):
    return SimpleNamespace(type="server_tool_use", id=id, name=name, input=input)


def _web_search_result_block(*, tool_use_id: str, content):
    return SimpleNamespace(type="web_search_tool_result", tool_use_id=tool_use_id, content=content)


def _anthropic_message(*, content: list, stop_reason: str, input_tokens: int, output_tokens: int):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _anthropic_status_error(status_code: int, message: str):
    """Build a real ``AnthropicAPIStatusError`` so the proxy's except branch runs unmodified."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code=status_code, request=request, json={"error": {"message": message}})
    return AnthropicAPIStatusError(message=message, response=response, body={"error": {"message": message}})


class _AsyncEventIter:
    """Async iterator yielding fake Anthropic streaming events.

    Constructed via ``SimpleNamespace`` so the converter's attribute access
    (``event.type``, ``event.delta.text_delta``, …) works against plain
    Python objects — no need to import or mimic SDK pydantic types.
    """

    def __init__(self, events):
        self._events = list(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)


def _stream_text_then_tool_use():
    """Realistic Anthropic stream: text deltas + a tool_use with partial-JSON args."""
    return _AsyncEventIter(
        [
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(usage=SimpleNamespace(input_tokens=17)),
            ),
            SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="text")),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="hello "),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="world"),
            ),
            SimpleNamespace(type="content_block_stop"),
            SimpleNamespace(
                type="content_block_start",
                content_block=SimpleNamespace(type="tool_use", id="toolu_s_1", name="lookup"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="input_json_delta", partial_json='{"q":"'),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="input_json_delta", partial_json='milk"}'),
            ),
            SimpleNamespace(type="content_block_stop"),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="tool_use"),
                usage=SimpleNamespace(output_tokens=9),
            ),
            SimpleNamespace(type="message_stop"),
        ]
    )


def _patch_anthropic(*, create_return_value=None, create_side_effect=None):
    """Patch the Anthropic client creation in the provider module.

    ``side_effect`` lets each construction build a fresh client object so
    a streaming + non-streaming call within the same test don't interfere.
    """

    def _factory(**_kwargs):
        client = Mock()
        client.messages = Mock()
        if create_side_effect is not None:
            client.messages.create = AsyncMock(side_effect=create_side_effect)
        else:
            client.messages.create = AsyncMock(return_value=create_return_value)
        return client

    return patch(
        "minds.inference.providers.anthropic._get_anthropic_client",
        side_effect=lambda config: _factory(),
    )


def _patch_alias_resolution(config: PassthroughModelConfig):
    """Patch the ModelResolver so the test doesn't depend on env-var API keys."""

    def _resolver_factory(*args, **kwargs):
        resolver = Mock()
        resolver.resolve = Mock(return_value=config)
        return resolver

    return patch(
        "minds.handlers.openai_request_handler.ModelResolver",
        side_effect=_resolver_factory,
    )


async def _drive_handler(
    *,
    context: Context,
    session: Session,
    chat_request: ChatCompletionsRequest,
):
    """Call the real ``chat_completions_request_handler`` exactly like the endpoint does.

    Uses the actual ``@observe``-decorated handler from
    ``minds/handlers/chat_completions_request_handler.py`` so the
    production decorator configuration (``capture_input=False`` /
    ``capture_output=False``) is what's exercised — not an inline
    re-creation that could drift.

    The reload dance: other test conftests stuff a fake ``langfuse`` into
    ``sys.modules`` at collection time. The ``langfuse_capture`` fixture
    rebinds the ``observe`` symbol on every ``minds.*`` module, but a
    decorator that was already applied at import time still wraps the
    no-op ``observe``. Reloading the handler module re-executes its
    top-level ``@observe(...)`` against the now-real decorator so the
    function we call is the production-shaped one.
    """
    import importlib

    from minds.handlers import chat_completions_request_handler as h_mod

    h_mod = importlib.reload(h_mod)

    return await h_mod.chat_completions_request_handler(
        session=session,
        context=context,
        mindsdb_client=Mock(),
        chat_completions_request=chat_request,
        instrument=True,
        limits_service=None,
    )


# ---------------------------------------------------------------------------
# Span filtering helpers
# ---------------------------------------------------------------------------


def _parent_generation(spans, name: str = "Chat Completions Handler v1"):
    parents = [s for s in spans if s.name == name]
    assert len(parents) == 1, f"Expected exactly 1 parent named {name!r}, got {[s.name for s in spans]}"
    return parents[0]


def _tool_spans(spans):
    return [s for s in spans if s.name.startswith("tool:")]


# ---------------------------------------------------------------------------
# Scenario A — non-streaming, text + tool_calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_streaming_text_plus_tool_calls_produces_complete_trace(
    langfuse_capture, mock_session, mock_context, sonnet_config
):
    """The full happy-path trace shape: input, output, usage, metadata, child spans.

    Drives a non-streaming request through the real handler with a stubbed
    Anthropic SDK that returns one text block plus one tool_use block.
    Asserts the resulting Langfuse parent generation carries the inbound
    request as ``input``, the assistant message (text + tool_calls) as
    ``output``, token usage, the concrete upstream model, and provider
    metadata. Asserts a ``tool:<name>`` child span exists carrying parsed
    JSON arguments and the ``call_id``.
    """
    fake_response = _anthropic_message(
        content=[
            _text_block("I'll look that up."),
            _tool_use_block(id="toolu_abc123", name="get_weather", input={"city": "Paris"}),
        ],
        stop_reason="tool_use",
        input_tokens=42,
        output_tokens=11,
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Look up current weather for a city.",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]

    chat_request = _request_with_tools(tools=tools, tool_choice="auto")

    with (
        _patch_alias_resolution(sonnet_config),
        _patch_anthropic(create_return_value=fake_response),
    ):
        response = await _drive_handler(context=mock_context, session=mock_session, chat_request=chat_request)

    # Client surface looks right before we assert on Langfuse.
    assert response.status_code == 200
    body = json.loads(response.body)
    assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"

    spans = langfuse_capture.get_spans()
    parent = _parent_generation(spans)

    # --- core generation attributes ----------------------------------------
    assert langfuse_capture.observation_type(parent) == "generation"
    assert langfuse_capture.model_name(parent) == "claude-sonnet-4-6"
    assert langfuse_capture.usage_details(parent) == {"input": 42, "output": 11, "total": 53}

    # --- input is the inbound request -------------------------------------
    input_payload = langfuse_capture.input(parent)
    assert input_payload["model"] == "latest:sonnet"
    assert input_payload["temperature"] == 0.0
    assert input_payload["max_tokens"] == 256
    assert input_payload["tool_choice"] == "auto"
    assert input_payload["messages"][0]["content"] == "What's the weather in Paris?"
    assert input_payload["tools"][0]["function"]["name"] == "get_weather"

    # --- output is the assistant message that the client saw ---------------
    output_payload = langfuse_capture.output(parent)
    assert output_payload["role"] == "assistant"
    assert output_payload["content"] == "I'll look that up."
    assert output_payload["tool_calls"][0]["function"]["name"] == "get_weather"
    # Tool args are JSON-encoded in the OpenAI response shape; assert the
    # structured value is recoverable so eval replays don't re-parse twice.
    args = json.loads(output_payload["tool_calls"][0]["function"]["arguments"])
    assert args == {"city": "Paris"}

    # --- metadata carries alias / provider / api_kind ----------------------
    meta = langfuse_capture.metadata(parent) or {}
    assert meta.get("passthrough_alias") == "sonnet"
    assert meta.get("provider") == "anthropic"
    assert meta.get("api_kind") == "anthropic_messages"

    # --- tool_call child span ---------------------------------------------
    tool_spans = _tool_spans(spans)
    assert len(tool_spans) == 1, f"Expected one tool:* span, got {[s.name for s in spans]}"
    tool_span = tool_spans[0]
    assert tool_span.name == "tool:get_weather"
    assert langfuse_capture.observation_type(tool_span) == "span"
    assert langfuse_capture.input(tool_span) == {"city": "Paris"}
    span_meta = langfuse_capture.metadata(tool_span) or {}
    assert span_meta.get("call_id") == "toolu_abc123"
    # Spans inherit alias / provider so a "show me every get_weather call
    # we made via latest:sonnet" query is one filter away.
    assert span_meta.get("passthrough_alias") == "sonnet"


# ---------------------------------------------------------------------------
# Scenario B — streaming end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_reconstructs_output_and_attaches_child_generation(
    langfuse_capture, mock_session, mock_context, sonnet_config
):
    """Streaming: the @observe scope closes before tokens are known.

    A detached ``llm-usage`` child generation must (a) live in the SAME
    trace as the parent, (b) carry the full reconstructed assistant
    message (text deltas + tool_call args) as ``output``, (c) carry token
    usage, and (d) carry the concrete upstream model. The tool-call child
    span must also be emitted, so multi-step replays show tool selection
    + arguments per turn.
    """
    chat_request = _request_with_tools(
        tools=[{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
        stream=True,
    )

    with (
        _patch_alias_resolution(sonnet_config),
        _patch_anthropic(create_return_value=_stream_text_then_tool_use()),
    ):
        wrapped = await _drive_handler(context=mock_context, session=mock_session, chat_request=chat_request)

        # Drain the SSE body iterator — what Starlette does in production.
        # The converter yields ``str`` chunks; Starlette encodes them on the
        # wire. We just concatenate to inspect the stream as a string.
        body_chunks: list = []
        async for chunk in wrapped.body_iterator:
            body_chunks.append(chunk if isinstance(chunk, str) else chunk.decode())

    # The client-facing SSE stream is well-formed: role chunk first, [DONE] last.
    body = "".join(body_chunks)
    assert body.startswith('data: {"id":'), body[:100]
    assert body.endswith("data: [DONE]\n\n"), body[-40:]

    spans = langfuse_capture.get_spans()
    parent = _parent_generation(spans)
    children = [s for s in spans if s.name == "llm-usage"]
    assert len(children) == 1, f"Expected 1 'llm-usage' child, got {[s.name for s in spans]}"
    child = children[0]

    # Same trace; child parented at the @observe span.
    assert child.context.trace_id == parent.context.trace_id
    assert child.parent is not None and child.parent.span_id == parent.context.span_id

    # Child carries usage + concrete model.
    assert langfuse_capture.model_name(child) == "claude-sonnet-4-6"
    assert langfuse_capture.usage_details(child) == {"input": 17, "output": 9, "total": 26}

    # Child carries the full reconstructed assistant message.
    output_payload = langfuse_capture.output(child)
    assert output_payload["role"] == "assistant"
    assert output_payload["content"] == "hello world"
    tool_calls = output_payload["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "lookup"
    # Args were streamed as two partial_json deltas — they must be
    # concatenated, not last-wins.
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"q": "milk"}

    # Child carries input payload too — eval replay needs both sides.
    input_payload = langfuse_capture.input(child)
    assert input_payload["model"] == "latest:sonnet"
    assert input_payload["messages"][0]["content"] == "What's the weather in Paris?"

    # The tool_call child span is emitted at the trace level.
    tool_spans = _tool_spans(spans)
    assert len(tool_spans) == 1
    assert tool_spans[0].name == "tool:lookup"
    assert langfuse_capture.input(tool_spans[0]) == {"q": "milk"}


# ---------------------------------------------------------------------------
# Scenario C — upstream error becomes an error generation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upstream_error_is_recorded_as_error_generation(
    langfuse_capture, mock_session, mock_context, sonnet_config
):
    """A 5xx from Anthropic must still produce a Langfuse generation.

    The failed request would otherwise be invisible in observability; the
    parent generation must instead carry ``status_code`` + ``level=ERROR``
    metadata, the decoded error body as ``output``, and zero usage. The
    HTTP response surfaced to the client is the upstream status code.
    """
    chat_request = _request_with_tools()
    err = _anthropic_status_error(502, "upstream is sad")

    with (
        _patch_alias_resolution(sonnet_config),
        _patch_anthropic(create_side_effect=err),
    ):
        # The handler's outer try/except in the endpoint converts unhandled
        # exceptions to HTTP 500 — but proxy_anthropic catches the SDK
        # error and returns a JSONResponse, so no exception escapes here.
        response = await _drive_handler(context=mock_context, session=mock_session, chat_request=chat_request)

    assert response.status_code == 502

    spans = langfuse_capture.get_spans()
    parent = _parent_generation(spans)

    # Usage zeroed but the trace exists — no observability black hole.
    assert langfuse_capture.usage_details(parent) == {"input": 0, "output": 0, "total": 0}
    assert langfuse_capture.model_name(parent) == "claude-sonnet-4-6"

    output_payload = langfuse_capture.output(parent)
    assert output_payload is not None
    assert output_payload["error"]["message"] == "upstream is sad"
    assert output_payload["error"]["type"] == "api_error"

    meta = langfuse_capture.metadata(parent) or {}
    assert meta.get("status_code") == 502
    assert meta.get("level") == "ERROR"
    # Provider context still attached so error rows are filterable by alias.
    assert meta.get("passthrough_alias") == "sonnet"


# ---------------------------------------------------------------------------
# Scenario D — Langfuse-* headers materialize on the trace
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_langfuse_headers_materialize_session_tags_and_trace_name(langfuse_capture, mock_session, sonnet_config):
    """When the client sends the three Langfuse-proxy convention headers, their values
    reach the captured span as ``session.id`` / ``langfuse.trace.tags`` /
    ``langfuse.trace.name`` / trace-level metadata. This is what lets cowork's
    multi-turn loops appear as a single Session in the Langfuse UI."""
    context_with_headers = Context(
        user_id=UUID("00000000-0000-0000-0000-000000000001"),
        organization_id=UUID("00000000-0000-0000-0000-000000000002"),
        langfuse_session_id="conv-20260518-xyz",
        langfuse_tags=["cowork", "agent-loop"],
        langfuse_metadata={"turn_id": 3, "harness": "cowork", "experiment": "A"},
    )
    fake_response = _anthropic_message(
        content=[_text_block("ok")],
        stop_reason="end_turn",
        input_tokens=5,
        output_tokens=2,
    )

    with (
        _patch_alias_resolution(sonnet_config),
        _patch_anthropic(create_return_value=fake_response),
    ):
        await _drive_handler(
            context=context_with_headers,
            session=mock_session,
            chat_request=_request_with_tools(),
        )

    spans = langfuse_capture.get_spans()
    parent = _parent_generation(spans)

    # Session grouping: the Sessions view collects this trace under conv-…-xyz.
    assert langfuse_capture.session_id(parent) == "conv-20260518-xyz"

    # Trace name comes from harness + turn_id — dashboards can scan by name.
    assert langfuse_capture.trace_name(parent) == "cowork:turn-3"

    # Client tags merged into identity tags without clobbering them.
    tags = langfuse_capture.trace_tags(parent) or ()
    assert "cowork" in tags
    assert "agent-loop" in tags
    assert any(t.startswith("user_id:") for t in tags)

    # Free-form client metadata flows into trace metadata, alongside identity keys.
    trace_meta = langfuse_capture.trace_metadata(parent) or {}
    assert trace_meta.get("experiment") == "A"
    # turn_id is a JSON value; Langfuse may serialize as int or str — accept both.
    assert trace_meta.get("turn_id") in (3, "3")
    assert trace_meta.get("harness") == "cowork"
    # Identity metadata (user_id / org / request_id) is still present —
    # client-supplied keys don't clobber it.
    assert trace_meta.get("user_id") == "00000000-0000-0000-0000-000000000001"


# ---------------------------------------------------------------------------
# Scenario E — server-side artifacts attach to metadata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_server_tool_artifacts_attached_to_generation_metadata(
    langfuse_capture, mock_session, mock_context, sonnet_config
):
    """Anthropic-native web_search / web_fetch intermediates land on metadata.

    The client-facing response intentionally strips ``server_tool_use`` and
    ``web_search_tool_result`` blocks — but for evals and troubleshooting
    they are the highest-signal artifacts. They must appear on the Langfuse
    generation's metadata under ``server_artifacts``.
    """
    fake_response = _anthropic_message(
        content=[
            _server_tool_use_block(id="srv_1", name="web_search", input={"q": "weather paris"}),
            _web_search_result_block(
                tool_use_id="srv_1",
                content=[{"title": "Weather in Paris", "url": "https://example.com"}],
            ),
            _text_block("It's sunny."),
        ],
        stop_reason="end_turn",
        input_tokens=20,
        output_tokens=5,
    )
    chat_request = _request_with_tools(tools=[{"type": "web_search"}])

    with (
        _patch_alias_resolution(sonnet_config),
        _patch_anthropic(create_return_value=fake_response),
    ):
        await _drive_handler(context=mock_context, session=mock_session, chat_request=chat_request)

    spans = langfuse_capture.get_spans()
    parent = _parent_generation(spans)
    meta = langfuse_capture.metadata(parent) or {}
    artifacts = meta.get("server_artifacts")
    assert artifacts, f"Expected server_artifacts on metadata, got {meta!r}"
    types = {a["type"] for a in artifacts}
    assert "server_tool_use" in types
    assert "web_search_tool_result" in types

    # The web_search query is preserved so evals can grade groundedness.
    web_search_call = next(a for a in artifacts if a["type"] == "server_tool_use")
    assert web_search_call["input"] == {"q": "weather paris"}

    # The model's final answer is still on output so clients still get text.
    output_payload = langfuse_capture.output(parent)
    assert output_payload["content"] == "It's sunny."
