"""Unit tests for native web_search/fetch tool translation in passthrough_agent.

Most of this file is hermetic — it exercises the translation helpers and the
proxy methods with mocked upstream clients. The bottom section contains
end-to-end *live* tests that actually call OpenAI and Anthropic; those are
auto-skipped when the corresponding API key is not configured in the
environment / ``.env``.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from minds.agents.passthrough_agent.agent import (
    _ANTHROPIC_WEB_FETCH_BETA_HEADER,
    PassthroughAgent,
    _chat_messages_to_responses_input,
    _chat_tool_choice_to_responses,
    _is_generic_web_tool,
    _only_web_tools,
    _responses_response_to_chat_completion,
    _translate_tools_for_anthropic,
    _translate_tools_for_openai,
)
from minds.common.passthrough_config import PassthroughModelConfig
from minds.common.settings.app_settings import get_app_settings
from minds.schemas.chat import Message, Role

# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def test_is_generic_web_tool_recognizes_both_types():
    assert _is_generic_web_tool({"type": "web_search"}) is True
    assert _is_generic_web_tool({"type": "fetch"}) is True


def test_is_generic_web_tool_rejects_other_types():
    assert _is_generic_web_tool({"type": "function", "function": {"name": "foo"}}) is False
    assert _is_generic_web_tool({"type": "web_search_20250305"}) is False
    assert _is_generic_web_tool({}) is False
    # Non-dicts shouldn't crash.
    assert _is_generic_web_tool("web_search") is False  # type: ignore[arg-type]


def test_only_web_tools_all_web_returns_true():
    assert _only_web_tools([{"type": "web_search"}]) is True
    assert _only_web_tools([{"type": "fetch"}, {"type": "web_search"}]) is True


def test_only_web_tools_mixed_returns_false():
    assert _only_web_tools([{"type": "web_search"}, {"type": "function", "function": {"name": "f"}}]) is False


def test_only_web_tools_no_web_returns_false():
    assert _only_web_tools([{"type": "function", "function": {"name": "f"}}]) is False


def test_only_web_tools_empty_or_none_returns_false():
    assert _only_web_tools(None) is False
    assert _only_web_tools([]) is False


# ---------------------------------------------------------------------------
# _translate_tools_for_anthropic
# ---------------------------------------------------------------------------


def test_translate_anthropic_web_search_only():
    tools, beta = _translate_tools_for_anthropic([{"type": "web_search"}])
    assert tools == [{"type": "web_search_20250305", "name": "web_search"}]
    assert beta is False


def test_translate_anthropic_fetch_only_sets_beta_flag():
    tools, beta = _translate_tools_for_anthropic([{"type": "fetch"}])
    assert tools == [{"type": "web_fetch_20250910", "name": "web_fetch"}]
    assert beta is True


def test_translate_anthropic_both_web_tools():
    tools, beta = _translate_tools_for_anthropic([{"type": "web_search"}, {"type": "fetch"}])
    assert tools == [
        {"type": "web_search_20250305", "name": "web_search"},
        {"type": "web_fetch_20250910", "name": "web_fetch"},
    ]
    assert beta is True


def test_translate_anthropic_mixed_with_function_tool():
    function_tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look something up",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }
    tools, beta = _translate_tools_for_anthropic([{"type": "web_search"}, function_tool])

    assert tools == [
        {"type": "web_search_20250305", "name": "web_search"},
        {
            "name": "lookup",
            "description": "Look something up",
            "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    ]
    assert beta is False


def test_translate_anthropic_empty_or_none():
    assert _translate_tools_for_anthropic(None) == ([], False)
    assert _translate_tools_for_anthropic([]) == ([], False)


def test_translate_anthropic_unknown_type_silently_skipped():
    # Preserves existing _openai_tools_to_anthropic behavior for unknown types.
    tools, beta = _translate_tools_for_anthropic([{"type": "mystery"}])
    assert tools == []
    assert beta is False


# ---------------------------------------------------------------------------
# _translate_tools_for_openai
# ---------------------------------------------------------------------------


def test_translate_openai_web_search_emits_native_tool():
    # On the Responses API, {"type":"web_search"} is a valid tools[] entry.
    assert _translate_tools_for_openai([{"type": "web_search"}]) == [{"type": "web_search"}]


def test_translate_openai_fetch_collapses_to_web_search():
    # OpenAI's web_search bundles fetching; fetch alone still emits a single
    # web_search tool (no separate fetch tool exists).
    assert _translate_tools_for_openai([{"type": "fetch"}]) == [{"type": "web_search"}]


def test_translate_openai_both_web_tools_dedupe():
    assert _translate_tools_for_openai([{"type": "web_search"}, {"type": "fetch"}]) == [{"type": "web_search"}]
    # Order swapped — same result.
    assert _translate_tools_for_openai([{"type": "fetch"}, {"type": "web_search"}]) == [{"type": "web_search"}]


def test_translate_openai_function_tool_flattened_to_responses_shape():
    # Chat-completions function tool: {"type":"function","function":{name,description,parameters}}
    # → Responses shape: {"type":"function","name":...,"description":...,"parameters":...}
    out = _translate_tools_for_openai(
        [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look it up",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            }
        ]
    )
    assert out == [
        {
            "type": "function",
            "name": "lookup",
            "description": "Look it up",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        }
    ]


def test_translate_openai_function_tool_without_description():
    out = _translate_tools_for_openai([{"type": "function", "function": {"name": "lookup", "parameters": {}}}])
    assert out == [{"type": "function", "name": "lookup", "parameters": {}}]


def test_translate_openai_mixed_keeps_both():
    out = _translate_tools_for_openai(
        [
            {"type": "web_search"},
            {"type": "function", "function": {"name": "f", "parameters": {}}},
        ]
    )
    assert out == [
        {"type": "web_search"},
        {"type": "function", "name": "f", "parameters": {}},
    ]


def test_translate_openai_empty_or_none():
    assert _translate_tools_for_openai(None) == []
    assert _translate_tools_for_openai([]) == []


# ---------------------------------------------------------------------------
# _chat_messages_to_responses_input
# ---------------------------------------------------------------------------


def test_chat_to_responses_extracts_system_as_instructions():
    instructions, items = _chat_messages_to_responses_input(
        [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ]
    )
    assert instructions == "be brief"
    assert items == [{"role": "user", "content": "hi"}]


def test_chat_to_responses_concatenates_multiple_system_messages():
    instructions, _ = _chat_messages_to_responses_input(
        [
            {"role": "system", "content": "first"},
            {"role": "system", "content": "second"},
            {"role": "user", "content": "hi"},
        ]
    )
    assert instructions == "first\n\nsecond"


def test_chat_to_responses_no_system_returns_none_instructions():
    instructions, items = _chat_messages_to_responses_input([{"role": "user", "content": "hi"}])
    assert instructions is None
    assert items == [{"role": "user", "content": "hi"}]


def test_chat_to_responses_assistant_tool_calls_become_function_call_items():
    _, items = _chat_messages_to_responses_input(
        [
            {"role": "user", "content": "what's the weather?"},
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc", "content": "72F sunny"},
        ]
    )

    assert items == [
        {"role": "user", "content": "what's the weather?"},
        {"role": "assistant", "content": "Let me check."},
        {
            "type": "function_call",
            "call_id": "call_abc",
            "name": "get_weather",
            "arguments": '{"city":"NYC"}',
        },
        {"type": "function_call_output", "call_id": "call_abc", "output": "72F sunny"},
    ]


def test_chat_to_responses_assistant_with_only_tool_calls_no_text():
    _, items = _chat_messages_to_responses_input(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "x", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
            },
        ]
    )
    # No empty assistant text item — only the function_call.
    assert items == [
        {"type": "function_call", "call_id": "x", "name": "f", "arguments": "{}"},
    ]


# ---------------------------------------------------------------------------
# _chat_tool_choice_to_responses
# ---------------------------------------------------------------------------


def test_chat_tool_choice_strings_pass_through():
    assert _chat_tool_choice_to_responses("auto") == "auto"
    assert _chat_tool_choice_to_responses("required") == "required"
    assert _chat_tool_choice_to_responses("none") == "none"


def test_chat_tool_choice_none_returns_none():
    assert _chat_tool_choice_to_responses(None) is None


def test_chat_tool_choice_function_specific_flattened():
    out = _chat_tool_choice_to_responses({"type": "function", "function": {"name": "lookup"}})
    assert out == {"type": "function", "name": "lookup"}


# ---------------------------------------------------------------------------
# _responses_response_to_chat_completion
# ---------------------------------------------------------------------------


def test_responses_to_chat_extracts_text_and_usage():
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hello world.")],
            ),
        ],
        usage=SimpleNamespace(input_tokens=5, output_tokens=3),
    )
    payload = _responses_response_to_chat_completion(response, "gpt-x")
    choice = payload["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "Hello world."
    assert choice["finish_reason"] == "stop"
    assert payload["usage"] == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    assert payload["model"] == "gpt-x"


def test_responses_to_chat_function_call_becomes_tool_call_and_finish_reason():
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                call_id="call_1",
                name="lookup",
                arguments='{"q":"x"}',
            ),
        ],
        usage=SimpleNamespace(input_tokens=4, output_tokens=2),
    )
    payload = _responses_response_to_chat_completion(response, "gpt-x")
    choice = payload["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"q":"x"}'},
        }
    ]
    # No text content was produced.
    assert choice["message"]["content"] is None


def test_responses_to_chat_ignores_web_search_call_and_reasoning_items():
    # Server-side intermediates should not surface to the chat completion shape.
    response = SimpleNamespace(
        output=[
            SimpleNamespace(type="web_search_call", id="ws_1"),
            SimpleNamespace(type="reasoning", id="r_1"),
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Final answer.")],
            ),
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=4),
    )
    payload = _responses_response_to_chat_completion(response, "gpt-x")
    assert payload["choices"][0]["message"]["content"] == "Final answer."
    assert "tool_calls" not in payload["choices"][0]["message"]
    assert payload["choices"][0]["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# Proxy methods — tool_choice drop logic and Anthropic beta header
# ---------------------------------------------------------------------------


def _make_anthropic_response() -> SimpleNamespace:
    """Minimal Anthropic non-streaming response stub."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hello")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def _make_openai_responses_response() -> SimpleNamespace:
    """Minimal OpenAI Responses API non-streaming response stub."""
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                role="assistant",
                content=[SimpleNamespace(type="output_text", text="hello")],
            ),
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def _fake_openai_client(create_mock: AsyncMock) -> SimpleNamespace:
    """Stand-in for an AsyncOpenAI client exposing only ``responses.create``."""
    return SimpleNamespace(responses=SimpleNamespace(create=create_mock))


@pytest.mark.asyncio
async def test_proxy_anthropic_drops_tool_choice_when_only_web_tools(monkeypatch):
    agent = PassthroughAgent(
        config=PassthroughModelConfig(provider="anthropic", model_name="claude-x"),
    )

    create_mock = AsyncMock(return_value=_make_anthropic_response())
    fake_client = SimpleNamespace(messages=SimpleNamespace(create=create_mock))
    monkeypatch.setattr(agent, "_get_anthropic_client", lambda: fake_client)

    await agent._proxy_anthropic(
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        request_id="req-1",
        tools=[{"type": "web_search"}],
        tool_choice="required",
    )

    create_mock.assert_awaited_once()
    kwargs = create_mock.await_args.kwargs
    assert "tool_choice" not in kwargs
    assert kwargs["tools"] == [{"type": "web_search_20250305", "name": "web_search"}]


@pytest.mark.asyncio
async def test_proxy_anthropic_forwards_tool_choice_when_mixed(monkeypatch):
    agent = PassthroughAgent(
        config=PassthroughModelConfig(provider="anthropic", model_name="claude-x"),
    )

    create_mock = AsyncMock(return_value=_make_anthropic_response())
    fake_client = SimpleNamespace(messages=SimpleNamespace(create=create_mock))
    monkeypatch.setattr(agent, "_get_anthropic_client", lambda: fake_client)

    function_tool = {
        "type": "function",
        "function": {"name": "f", "parameters": {}},
    }
    await agent._proxy_anthropic(
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        request_id="req-2",
        tools=[{"type": "web_search"}, function_tool],
        tool_choice="required",
    )

    kwargs = create_mock.await_args.kwargs
    # "required" maps to {"type": "any"} on Anthropic.
    assert kwargs["tool_choice"] == {"type": "any"}


@pytest.mark.asyncio
async def test_proxy_anthropic_sets_beta_header_for_fetch(monkeypatch):
    agent = PassthroughAgent(
        config=PassthroughModelConfig(provider="anthropic", model_name="claude-x"),
    )

    create_mock = AsyncMock(return_value=_make_anthropic_response())
    fake_client = SimpleNamespace(messages=SimpleNamespace(create=create_mock))
    monkeypatch.setattr(agent, "_get_anthropic_client", lambda: fake_client)

    await agent._proxy_anthropic(
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        request_id="req-3",
        tools=[{"type": "fetch"}],
    )

    kwargs = create_mock.await_args.kwargs
    assert kwargs["extra_headers"] == {"anthropic-beta": _ANTHROPIC_WEB_FETCH_BETA_HEADER}


@pytest.mark.asyncio
async def test_proxy_anthropic_no_beta_header_when_no_fetch(monkeypatch):
    agent = PassthroughAgent(
        config=PassthroughModelConfig(provider="anthropic", model_name="claude-x"),
    )

    create_mock = AsyncMock(return_value=_make_anthropic_response())
    fake_client = SimpleNamespace(messages=SimpleNamespace(create=create_mock))
    monkeypatch.setattr(agent, "_get_anthropic_client", lambda: fake_client)

    await agent._proxy_anthropic(
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        request_id="req-4",
        tools=[{"type": "web_search"}],
    )

    kwargs = create_mock.await_args.kwargs
    assert "extra_headers" not in kwargs


@pytest.mark.asyncio
async def test_proxy_openai_uses_responses_api_with_web_search(monkeypatch):
    agent = PassthroughAgent(
        config=PassthroughModelConfig(provider="openai", model_name="gpt-x"),
    )

    create_mock = AsyncMock(return_value=_make_openai_responses_response())
    monkeypatch.setattr(agent, "_get_openai_client", lambda: _fake_openai_client(create_mock))

    await agent._proxy_openai(
        messages=[{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}],
        stream=False,
        request_id="req-5",
        tools=[{"type": "web_search"}],
        tool_choice="required",
    )

    create_mock.assert_awaited_once()
    kwargs = create_mock.await_args.kwargs
    # System message extracted as instructions; user message in input list.
    assert kwargs["instructions"] == "be brief"
    assert kwargs["input"] == [{"role": "user", "content": "hi"}]
    # Web search appears as a tools[] entry on the Responses API.
    assert kwargs["tools"] == [{"type": "web_search"}]
    # No web_search_options on Responses API (that was the old chat-completions path).
    assert "web_search_options" not in kwargs
    # tool_choice dropped because only web tools were provided.
    assert "tool_choice" not in kwargs


@pytest.mark.asyncio
async def test_proxy_openai_forwards_tool_choice_when_mixed(monkeypatch):
    agent = PassthroughAgent(
        config=PassthroughModelConfig(provider="openai", model_name="gpt-x"),
    )

    create_mock = AsyncMock(return_value=_make_openai_responses_response())
    monkeypatch.setattr(agent, "_get_openai_client", lambda: _fake_openai_client(create_mock))

    function_tool = {
        "type": "function",
        "function": {"name": "f", "parameters": {}},
    }
    await agent._proxy_openai(
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        request_id="req-6",
        tools=[{"type": "web_search"}, function_tool],
        tool_choice="required",
    )

    kwargs = create_mock.await_args.kwargs
    assert kwargs["tool_choice"] == "required"
    # Function tool is flattened to Responses API shape; web_search is included.
    assert kwargs["tools"] == [
        {"type": "web_search"},
        {"type": "function", "name": "f", "parameters": {}},
    ]


@pytest.mark.asyncio
async def test_proxy_openai_fetch_alone_emits_single_web_search(monkeypatch):
    agent = PassthroughAgent(
        config=PassthroughModelConfig(provider="openai", model_name="gpt-x"),
    )

    create_mock = AsyncMock(return_value=_make_openai_responses_response())
    monkeypatch.setattr(agent, "_get_openai_client", lambda: _fake_openai_client(create_mock))

    await agent._proxy_openai(
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        request_id="req-7",
        tools=[{"type": "fetch"}],
    )

    kwargs = create_mock.await_args.kwargs
    # fetch alone collapses to a single web_search tool on Responses.
    assert kwargs["tools"] == [{"type": "web_search"}]


@pytest.mark.asyncio
async def test_proxy_openai_max_tokens_renamed_to_max_output_tokens(monkeypatch):
    agent = PassthroughAgent(
        config=PassthroughModelConfig(provider="openai", model_name="gpt-x"),
    )

    create_mock = AsyncMock(return_value=_make_openai_responses_response())
    monkeypatch.setattr(agent, "_get_openai_client", lambda: _fake_openai_client(create_mock))

    await agent._proxy_openai(
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        request_id="req-9",
        max_tokens=512,
    )

    kwargs = create_mock.await_args.kwargs
    assert kwargs["max_output_tokens"] == 512
    # Chat-completions ``max_tokens`` should not leak through.
    assert "max_tokens" not in kwargs


@pytest.mark.asyncio
async def test_proxy_openai_specific_function_tool_choice_translated(monkeypatch):
    agent = PassthroughAgent(
        config=PassthroughModelConfig(provider="openai", model_name="gpt-x"),
    )

    create_mock = AsyncMock(return_value=_make_openai_responses_response())
    monkeypatch.setattr(agent, "_get_openai_client", lambda: _fake_openai_client(create_mock))

    function_tool = {"type": "function", "function": {"name": "f", "parameters": {}}}
    await agent._proxy_openai(
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        request_id="req-10",
        tools=[function_tool],
        tool_choice={"type": "function", "function": {"name": "f"}},
    )

    kwargs = create_mock.await_args.kwargs
    # Chat-completions specific-function shape → Responses flat shape.
    assert kwargs["tool_choice"] == {"type": "function", "name": "f"}


# ---------------------------------------------------------------------------
# Live provider tests
# ---------------------------------------------------------------------------
#
# These tests issue real network calls against OpenAI / Anthropic using the
# API keys configured in the environment (typically loaded from ``.env`` by
# pydantic-settings via ``get_app_settings()``). They demonstrate end-to-end
# that the generic ``{"type": "web_search"}`` / ``{"type": "fetch"}`` shape
# is correctly translated into something each provider understands.
#
# Each test auto-skips when the corresponding key is not configured, so the
# default ``make test/unit`` run on a developer machine without keys still
# passes. To run them locally, set ``OPENAI__API_KEY`` and/or
# ``ANTHROPIC__API_KEY`` (in ``.env`` or env vars).
#
# Models can be overridden via ``LIVE_TEST_OPENAI_MODEL`` /
# ``LIVE_TEST_ANTHROPIC_MODEL`` env vars in case the defaults don't match
# what the test account has access to.


_settings = get_app_settings()

_anthropic_key_set = bool(_settings.anthropic.api_key)
_openai_key_set = bool(_settings.openai.api_key) and _settings.openai.api_key != "not set"

# Defaults chosen for breadth of native-tool support on the OpenAI Responses
# API (gpt-5-mini is fast and supports {"type":"web_search"}); override via
# env vars if your account doesn't have access to these specific models.
_LIVE_ANTHROPIC_MODEL = os.getenv("LIVE_TEST_ANTHROPIC_MODEL", "claude-sonnet-4-6")
_LIVE_OPENAI_MODEL = os.getenv("LIVE_TEST_OPENAI_MODEL", "gpt-5-mini")

requires_anthropic = pytest.mark.skipif(not _anthropic_key_set, reason="ANTHROPIC__API_KEY not configured")
requires_openai = pytest.mark.skipif(not _openai_key_set, reason="OPENAI__API_KEY not configured")


def _decode_response_body(response) -> dict:
    """Decode a Starlette JSONResponse body to a dict, with diagnostics on error."""
    raw = response.body
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _assert_non_empty_assistant_text(payload: dict) -> str:
    """Verify the OpenAI-shaped payload has a non-empty assistant text reply."""
    assert "choices" in payload, f"missing 'choices' in response: {payload}"
    assert payload["choices"], f"empty 'choices' in response: {payload}"
    msg = payload["choices"][0].get("message", {})
    content = msg.get("content")
    assert content, f"empty assistant content: {payload}"
    return content


@requires_anthropic
@pytest.mark.asyncio
async def test_live_anthropic_web_search():
    """Anthropic + generic web_search → translated to web_search_20250305."""
    agent = PassthroughAgent(
        config=PassthroughModelConfig(provider="anthropic", model_name=_LIVE_ANTHROPIC_MODEL),
    )
    response = await agent.proxy(
        messages=[Message(role=Role.user, content="Give me 200 words on what happened today.")],
        stream=False,
        request_id="live-anthropic-search",
        tools=[{"type": "web_search"}],
        max_tokens=2048,
    )

    assert response.status_code == 200, f"non-200 response: {response.body!r}"
    payload = _decode_response_body(response)
    text = _assert_non_empty_assistant_text(payload)
    # Non-trivial reply; web-grounded output should be more than a few words.
    assert len(text.split()) >= 30, f"unexpectedly short reply: {text!r}"


@requires_anthropic
@pytest.mark.asyncio
async def test_live_anthropic_fetch():
    """Anthropic + generic fetch → translated to web_fetch_20250910 + beta header."""
    agent = PassthroughAgent(
        config=PassthroughModelConfig(provider="anthropic", model_name=_LIVE_ANTHROPIC_MODEL),
    )
    response = await agent.proxy(
        messages=[
            Message(
                role=Role.user,
                content="Fetch https://example.com and summarize what's on the page in 100 words.",
            ),
        ],
        stream=False,
        request_id="live-anthropic-fetch",
        tools=[{"type": "fetch"}],
        max_tokens=2048,
    )

    assert response.status_code == 200, f"non-200 response: {response.body!r}"
    payload = _decode_response_body(response)
    _assert_non_empty_assistant_text(payload)


@requires_anthropic
@pytest.mark.asyncio
async def test_live_anthropic_web_search_and_fetch_combined():
    """Anthropic with both generic web tools at once."""
    agent = PassthroughAgent(
        config=PassthroughModelConfig(provider="anthropic", model_name=_LIVE_ANTHROPIC_MODEL),
    )
    response = await agent.proxy(
        messages=[
            Message(role=Role.user, content="Give me 200 words on what happened today."),
        ],
        stream=False,
        request_id="live-anthropic-both",
        tools=[{"type": "web_search"}, {"type": "fetch"}],
        max_tokens=2048,
    )

    assert response.status_code == 200, f"non-200 response: {response.body!r}"
    payload = _decode_response_body(response)
    _assert_non_empty_assistant_text(payload)


@requires_openai
@pytest.mark.asyncio
async def test_live_openai_web_search():
    """OpenAI + generic web_search → translated to native {"type": "web_search"}."""
    agent = PassthroughAgent(
        config=PassthroughModelConfig(provider="openai", model_name=_LIVE_OPENAI_MODEL),
    )
    response = await agent.proxy(
        messages=[Message(role=Role.user, content="Give me 200 words on what happened today.")],
        stream=False,
        request_id="live-openai-search",
        tools=[{"type": "web_search"}],
    )

    assert response.status_code == 200, f"non-200 response: {response.body!r}"
    payload = _decode_response_body(response)
    text = _assert_non_empty_assistant_text(payload)
    assert len(text.split()) >= 30, f"unexpectedly short reply: {text!r}"


@requires_openai
@pytest.mark.asyncio
async def test_live_openai_fetch_synthesizes_web_search():
    """OpenAI + generic fetch → translated to native web_search (which bundles fetch)."""
    agent = PassthroughAgent(
        config=PassthroughModelConfig(provider="openai", model_name=_LIVE_OPENAI_MODEL),
    )
    response = await agent.proxy(
        messages=[
            Message(
                role=Role.user,
                content="Fetch https://example.com and summarize what's on the page in 100 words.",
            ),
        ],
        stream=False,
        request_id="live-openai-fetch",
        tools=[{"type": "fetch"}],
    )

    assert response.status_code == 200, f"non-200 response: {response.body!r}"
    payload = _decode_response_body(response)
    _assert_non_empty_assistant_text(payload)
