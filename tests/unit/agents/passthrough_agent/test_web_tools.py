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
from fastapi import HTTPException

from minds.agents.passthrough_agent.agent import (
    AnthropicToolsTranslation,
    ChatCompletionsFunctionTool,
    GenericFetchTool,
    GenericWebSearchTool,
    PassthroughAgent,
    _chat_messages_to_gemini,
    _chat_messages_to_responses_input,
    _chat_tool_choice_to_gemini,
    _chat_tool_choice_to_responses,
    _classify_tool,
    _gemini_response_to_openai,
    _is_generic_web_tool,
    _only_web_tools,
    _responses_response_to_chat_completion,
    _translate_tools_for_anthropic,
    _translate_tools_for_gemini,
    _translate_tools_for_openai,
)
from minds.common.passthrough_config import (
    PassthroughModelConfig,
    resolve_passthrough_model,
)
from minds.common.settings.app_settings import get_app_settings
from minds.schemas.chat import Message, Role


def test_public_api_reexports_from_agent_module():
    """File-split smoke test: every name historically imported from
    ``passthrough_agent.agent`` must still resolve after the per-provider
    refactor (each provider module star-exports into agent.py)."""
    from minds.agents.passthrough_agent.agent import (  # noqa: F401
        AnthropicToolsTranslation,
        ChatCompletionsFunctionTool,
        GenericFetchTool,
        GenericWebSearchTool,
        PassthroughAgent,
        _chat_messages_to_gemini,
        _chat_messages_to_responses_input,
        _chat_tool_choice_to_gemini,
        _chat_tool_choice_to_responses,
        _classify_tool,
        _gemini_response_to_openai,
        _is_generic_web_tool,
        _only_web_tools,
        _responses_response_to_chat_completion,
        _translate_tools_for_anthropic,
        _translate_tools_for_gemini,
        _translate_tools_for_openai,
    )

    assert callable(PassthroughAgent)


# ---------------------------------------------------------------------------
# Test config helpers
# ---------------------------------------------------------------------------
#
# PassthroughModelConfig is keyed on `api_kind` rather than a free-text
# `provider` string; these helpers spell out the boilerplate once so each
# test only has to override the fields it actually exercises.


def _anthropic_config(
    model_name: str = "claude-x",
    web_search_mode: str = "anthropic_native",
) -> PassthroughModelConfig:
    return PassthroughModelConfig(
        api_kind="anthropic_messages",
        model_name=model_name,
        api_key="test-key",
        web_search_mode=web_search_mode,
        label="anthropic",
    )


def _openai_config(model_name: str = "gpt-x", web_search_mode: str = "openai_native") -> PassthroughModelConfig:
    return PassthroughModelConfig(
        api_kind="openai_responses",
        model_name=model_name,
        api_key="test-key",
        web_search_mode=web_search_mode,
        label="openai",
    )


def _fireworks_config(model_name: str = "accounts/fireworks/models/kimi-k2p6") -> PassthroughModelConfig:
    return PassthroughModelConfig(
        api_kind="anthropic_messages",
        model_name=model_name,
        api_key="test-key",
        base_url="https://api.fireworks.ai/inference",
        web_search_mode="drop",
        label="fireworks",
    )


def _gemini_config(model_name: str = "gemini-3.1-pro-preview") -> PassthroughModelConfig:
    return PassthroughModelConfig(
        api_kind="gemini_native",
        model_name=model_name,
        api_key="test-key",
        web_search_mode="gemini_google_search",
        label="gemini",
    )


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
    result = _translate_tools_for_anthropic([{"type": "web_search"}])
    assert isinstance(result, AnthropicToolsTranslation)
    assert result.tools == [
        {"type": get_app_settings().anthropic.web_search_tool_type, "name": "web_search"},
    ]
    assert result.needs_web_fetch_beta is False


def test_translate_anthropic_fetch_only_sets_beta_flag():
    result = _translate_tools_for_anthropic([{"type": "fetch"}])
    assert result.tools == [
        {"type": get_app_settings().anthropic.web_fetch_tool_type, "name": "web_fetch"},
    ]
    assert result.needs_web_fetch_beta is True


def test_translate_anthropic_both_web_tools():
    s = get_app_settings().anthropic
    result = _translate_tools_for_anthropic([{"type": "web_search"}, {"type": "fetch"}])
    assert result.tools == [
        {"type": s.web_search_tool_type, "name": "web_search"},
        {"type": s.web_fetch_tool_type, "name": "web_fetch"},
    ]
    assert result.needs_web_fetch_beta is True


def test_translate_anthropic_mixed_with_function_tool():
    function_tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look something up",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }
    result = _translate_tools_for_anthropic([{"type": "web_search"}, function_tool])

    assert result.tools == [
        {"type": get_app_settings().anthropic.web_search_tool_type, "name": "web_search"},
        {
            "name": "lookup",
            "description": "Look something up",
            "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    ]
    assert result.needs_web_fetch_beta is False


def test_translate_anthropic_empty_or_none():
    empty_none = _translate_tools_for_anthropic(None)
    assert empty_none.tools == [] and empty_none.needs_web_fetch_beta is False
    empty_list = _translate_tools_for_anthropic([])
    assert empty_list.tools == [] and empty_list.needs_web_fetch_beta is False


def test_translate_anthropic_unknown_type_silently_skipped():
    # Unknown tool types don't crash; classifier logs at debug, translator drops.
    result = _translate_tools_for_anthropic([{"type": "mystery"}])
    assert result.tools == []
    assert result.needs_web_fetch_beta is False


def test_translate_anthropic_malformed_function_tool_skipped():
    # A function tool missing the inner "function" payload fails Pydantic
    # validation; the classifier logs a warning and the translator skips it.
    result = _translate_tools_for_anthropic([{"type": "function"}])
    assert result.tools == []


# ---------------------------------------------------------------------------
# Pydantic tool models / _classify_tool
# ---------------------------------------------------------------------------


def test_classify_tool_recognizes_generic_web_search():
    parsed = _classify_tool({"type": "web_search"})
    assert isinstance(parsed, GenericWebSearchTool)


def test_classify_tool_recognizes_generic_fetch():
    parsed = _classify_tool({"type": "fetch"})
    assert isinstance(parsed, GenericFetchTool)


def test_classify_tool_recognizes_function_tool_and_parses_fields():
    parsed = _classify_tool(
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look it up",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    )
    assert isinstance(parsed, ChatCompletionsFunctionTool)
    assert parsed.function.name == "lookup"
    assert parsed.function.description == "Look it up"


def test_classify_tool_returns_none_for_non_dict():
    assert _classify_tool("web_search") is None  # type: ignore[arg-type]
    assert _classify_tool(None) is None  # type: ignore[arg-type]


def test_classify_tool_returns_none_for_malformed_function_tool():
    # Missing the nested "function" payload entirely.
    assert _classify_tool({"type": "function"}) is None
    # function payload present but missing required "name".
    assert _classify_tool({"type": "function", "function": {}}) is None


def test_classify_tool_returns_none_for_unknown_type():
    assert _classify_tool({"type": "mystery"}) is None


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
    agent = PassthroughAgent(config=_anthropic_config())

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
    agent = PassthroughAgent(config=_anthropic_config())

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
    agent = PassthroughAgent(config=_anthropic_config())

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
    assert kwargs["extra_headers"] == {"anthropic-beta": get_app_settings().anthropic.web_fetch_beta_header}


@pytest.mark.asyncio
async def test_proxy_anthropic_no_beta_header_when_no_fetch(monkeypatch):
    agent = PassthroughAgent(config=_anthropic_config())

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
    agent = PassthroughAgent(config=_openai_config())

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
    agent = PassthroughAgent(config=_openai_config())

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
    agent = PassthroughAgent(config=_openai_config())

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
    agent = PassthroughAgent(config=_openai_config())

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
    agent = PassthroughAgent(config=_openai_config())

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


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
@pytest.mark.asyncio
async def test_proxy_openai_passes_reasoning_effort_when_set(monkeypatch, effort: str):
    """When the resolved config carries ``reasoning_effort``, the OpenAI proxy
    forwards it as ``reasoning={"effort": ...}`` on ``responses.create``."""
    cfg = PassthroughModelConfig(
        api_kind="openai_responses",
        model_name="gpt-5.5",
        api_key="test-key",
        web_search_mode="openai_native",
        label="openai",
        alias=f"gpt-{effort}",
        reasoning_effort=effort,
    )
    agent = PassthroughAgent(config=cfg)

    create_mock = AsyncMock(return_value=_make_openai_responses_response())
    monkeypatch.setattr(agent, "_get_openai_client", lambda: _fake_openai_client(create_mock))

    await agent._proxy_openai(
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        request_id="req-reason",
    )

    kwargs = create_mock.await_args.kwargs
    assert kwargs["reasoning"] == {"effort": effort}


@pytest.mark.asyncio
async def test_proxy_openai_omits_reasoning_when_unset(monkeypatch):
    """When ``reasoning_effort`` is None, no ``reasoning`` kwarg is sent."""
    agent = PassthroughAgent(config=_openai_config())  # no reasoning_effort

    create_mock = AsyncMock(return_value=_make_openai_responses_response())
    monkeypatch.setattr(agent, "_get_openai_client", lambda: _fake_openai_client(create_mock))

    await agent._proxy_openai(
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        request_id="req-no-reason",
    )

    assert "reasoning" not in create_mock.await_args.kwargs


# ---------------------------------------------------------------------------
# web_search_mode="drop" — Fireworks parity
# ---------------------------------------------------------------------------


def test_translate_anthropic_drop_mode_strips_web_search():
    # Used for Fireworks: their Anthropic-shape API has no hosted search.
    result = _translate_tools_for_anthropic([{"type": "web_search"}], web_search_mode="drop")
    assert result.tools == []
    assert result.needs_web_fetch_beta is False


def test_translate_anthropic_drop_mode_strips_fetch_and_keeps_function():
    function_tool = {"type": "function", "function": {"name": "f", "parameters": {}}}
    result = _translate_tools_for_anthropic([{"type": "fetch"}, function_tool], web_search_mode="drop")
    # fetch dropped; function tool still translated to Anthropic shape.
    assert result.tools == [{"name": "f", "description": "", "input_schema": {}}]
    assert result.needs_web_fetch_beta is False


def test_translate_openai_drop_mode_strips_web_search_and_fetch():
    function_tool = {"type": "function", "function": {"name": "f", "parameters": {}}}
    out = _translate_tools_for_openai(
        [{"type": "web_search"}, {"type": "fetch"}, function_tool], web_search_mode="drop"
    )
    # Both web tools dropped; function tool flattened.
    assert out == [{"type": "function", "name": "f", "parameters": {}}]


@pytest.mark.asyncio
async def test_proxy_anthropic_with_fireworks_config_drops_web_search(monkeypatch):
    """A Fireworks config (Anthropic-shape transport, web_search_mode='drop')
    must strip a generic web_search tool before forwarding upstream."""
    agent = PassthroughAgent(config=_fireworks_config())

    create_mock = AsyncMock(return_value=_make_anthropic_response())
    fake_client = SimpleNamespace(messages=SimpleNamespace(create=create_mock))
    monkeypatch.setattr(agent, "_get_anthropic_client", lambda: fake_client)

    function_tool = {"type": "function", "function": {"name": "f", "parameters": {}}}
    await agent._proxy_anthropic(
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        request_id="req-fw-1",
        tools=[{"type": "web_search"}, function_tool],
    )

    kwargs = create_mock.await_args.kwargs
    # web_search dropped; function tool survives.
    assert kwargs["tools"] == [{"name": "f", "description": "", "input_schema": {}}]
    assert "extra_headers" not in kwargs


# ---------------------------------------------------------------------------
# Anthropic SDK base_url plumbing — Fireworks via Anthropic shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_client_threads_base_url_from_config(monkeypatch):
    """A Fireworks-style config with base_url must reach AsyncAnthropic(base_url=...)."""
    captured: dict = {}

    class _FakeAsyncAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.messages = SimpleNamespace(create=AsyncMock(return_value=_make_anthropic_response()))

    # Imports are at module-top in agent.py now; patch the symbol where it's used.
    monkeypatch.setattr("minds.agents.passthrough_agent.agent.AsyncAnthropic", _FakeAsyncAnthropic)

    agent = PassthroughAgent(config=_fireworks_config())
    client = agent._get_anthropic_client()

    assert isinstance(client, _FakeAsyncAnthropic)
    assert captured["api_key"] == "test-key"
    assert captured["base_url"] == "https://api.fireworks.ai/inference"


@pytest.mark.asyncio
async def test_anthropic_client_omits_base_url_when_unset(monkeypatch):
    """Direct Anthropic config (no base_url) must NOT pass base_url to the SDK
    so the SDK's default endpoint is used."""
    captured: dict = {}

    class _FakeAsyncAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("minds.agents.passthrough_agent.agent.AsyncAnthropic", _FakeAsyncAnthropic)

    agent = PassthroughAgent(config=_anthropic_config())
    agent._get_anthropic_client()

    assert captured["api_key"] == "test-key"
    assert "base_url" not in captured


# ---------------------------------------------------------------------------
# resolve_passthrough_model — alias table
# ---------------------------------------------------------------------------


def _fake_settings(
    *,
    openai_key: str = "",
    anthropic_key: str = "",
    fireworks_key: str = "",
    gemini_key: str = "",
    openai_gpt_model: str = "gpt-5.5",
    openai_gpt_codex_model: str = "gpt-5.3-codex",
    openai_gpt_mini_model: str = "gpt-5.4-mini",
    openai_gpt_nano_model: str = "gpt-5.4-nano",
    anthropic_sonnet_model: str = "claude-sonnet-4-6",
    anthropic_opus_model: str = "claude-opus-4-7",
    anthropic_haiku_model: str = "claude-haiku-4-5-20251001",
    fireworks_kimi_model: str = "accounts/fireworks/models/kimi-k2p6",
    fireworks_deepseek_model: str = "accounts/fireworks/models/deepseek-v4-pro",
    fireworks_qwen_model: str = "accounts/fireworks/models/qwen3p6-plus",
    gemini_model: str = "gemini-3.1-pro-preview",
) -> SimpleNamespace:
    """Build a stand-in for AppSettings exposing only the fields the resolver reads.

    Model names are explicit args (with the same defaults the real settings
    have) so tests can override them when verifying that settings drive
    the alias table instead of hardcoded literals.
    """
    return SimpleNamespace(
        openai=SimpleNamespace(
            api_key=openai_key,
            api_url="https://api.openai.com/v1",
            passthrough_gpt_model=openai_gpt_model,
            passthrough_gpt_codex_model=openai_gpt_codex_model,
            passthrough_gpt_mini_model=openai_gpt_mini_model,
            passthrough_gpt_nano_model=openai_gpt_nano_model,
        ),
        anthropic=SimpleNamespace(
            api_key=anthropic_key,
            passthrough_sonnet_model=anthropic_sonnet_model,
            passthrough_opus_model=anthropic_opus_model,
            passthrough_haiku_model=anthropic_haiku_model,
        ),
        fireworks=SimpleNamespace(
            api_key=fireworks_key,
            anthropic_base_url="https://api.fireworks.ai/inference",
            passthrough_kimi_model=fireworks_kimi_model,
            passthrough_deepseek_model=fireworks_deepseek_model,
            passthrough_qwen_model=fireworks_qwen_model,
        ),
        gemini=SimpleNamespace(
            api_key=gemini_key,
            passthrough_gemini_model=gemini_model,
        ),
    )


def _patch_settings(monkeypatch, settings) -> None:
    monkeypatch.setattr("minds.common.passthrough_config.get_app_settings", lambda: settings)


# ---------------------------------------------------------------------------
# Anthropic explicit-model aliases
# ---------------------------------------------------------------------------


def test_resolve_sonnet_uses_anthropic(monkeypatch):
    _patch_settings(monkeypatch, _fake_settings(anthropic_key="an-key"))
    cfg = resolve_passthrough_model("latest:sonnet")
    assert cfg.api_kind == "anthropic_messages"
    assert cfg.model_name == "claude-sonnet-4-6"
    assert cfg.api_key == "an-key"
    assert cfg.web_search_mode == "anthropic_native"
    assert cfg.label == "anthropic"
    assert cfg.alias == "sonnet"


def test_resolve_opus_uses_anthropic(monkeypatch):
    _patch_settings(monkeypatch, _fake_settings(anthropic_key="an-key"))
    cfg = resolve_passthrough_model("latest:opus")
    assert cfg.api_kind == "anthropic_messages"
    assert cfg.model_name == "claude-opus-4-7"
    assert cfg.alias == "opus"


def test_resolve_haiku_uses_anthropic(monkeypatch):
    _patch_settings(monkeypatch, _fake_settings(anthropic_key="an-key"))
    cfg = resolve_passthrough_model("latest:haiku")
    assert cfg.api_kind == "anthropic_messages"
    assert cfg.model_name == "claude-haiku-4-5-20251001"
    assert cfg.alias == "haiku"


def test_resolve_sonnet_without_anthropic_key_raises_400(monkeypatch):
    # Anthropic explicit-model aliases must not silently fall back to OpenAI.
    _patch_settings(monkeypatch, _fake_settings(openai_key="ok-key"))
    with pytest.raises(HTTPException) as exc_info:
        resolve_passthrough_model("latest:sonnet")
    assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# OpenAI GPT family — reasoning levels + specialized variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("alias", "expected_effort"),
    [
        # `latest:gpt` is the one-word default — same routing as `latest:gpt-low`.
        ("latest:gpt", "low"),
        ("latest:gpt-low", "low"),
        ("latest:gpt-medium", "medium"),
        ("latest:gpt-high", "high"),
    ],
)
def test_resolve_gpt_aliases_set_reasoning_effort(monkeypatch, alias: str, expected_effort: str):
    _patch_settings(monkeypatch, _fake_settings(openai_key="ok-key"))
    cfg = resolve_passthrough_model(alias)
    assert cfg.api_kind == "openai_responses"
    assert cfg.model_name == "gpt-5.5"
    assert cfg.reasoning_effort == expected_effort
    # alias is the bare body without the `latest:` prefix — what observability records.
    assert cfg.alias == alias.removeprefix("latest:")


def test_resolve_gpt_without_openai_key_raises_400(monkeypatch):
    _patch_settings(monkeypatch, _fake_settings(anthropic_key="an-key"))
    with pytest.raises(HTTPException):
        resolve_passthrough_model("latest:gpt")


def test_resolve_gpt_codex_uses_dedicated_model_no_reasoning_effort(monkeypatch):
    _patch_settings(monkeypatch, _fake_settings(openai_key="ok-key"))
    cfg = resolve_passthrough_model("latest:gpt-codex")
    assert cfg.api_kind == "openai_responses"
    assert cfg.model_name == "gpt-5.3-codex"
    # Codex picks its own internal reasoning_effort; we don't bake one in.
    assert cfg.reasoning_effort is None
    assert cfg.alias == "gpt-codex"


def test_resolve_gpt_mini_uses_mini_model(monkeypatch):
    _patch_settings(monkeypatch, _fake_settings(openai_key="ok-key"))
    cfg = resolve_passthrough_model("latest:gpt-mini")
    assert cfg.api_kind == "openai_responses"
    assert cfg.model_name == "gpt-5.4-mini"
    assert cfg.reasoning_effort is None
    assert cfg.alias == "gpt-mini"


def test_resolve_gpt_nano_uses_nano_model(monkeypatch):
    _patch_settings(monkeypatch, _fake_settings(openai_key="ok-key"))
    cfg = resolve_passthrough_model("latest:gpt-nano")
    assert cfg.api_kind == "openai_responses"
    assert cfg.model_name == "gpt-5.4-nano"
    assert cfg.reasoning_effort is None
    assert cfg.alias == "gpt-nano"


# ---------------------------------------------------------------------------
# Gemini + Fireworks-hosted open models
# ---------------------------------------------------------------------------


def test_resolve_gemini_uses_native_api_with_google_search(monkeypatch):
    _patch_settings(monkeypatch, _fake_settings(gemini_key="gm-key"))
    cfg = resolve_passthrough_model("latest:gemini")
    assert cfg.api_kind == "gemini_native"
    assert cfg.model_name == "gemini-3.1-pro-preview"
    assert cfg.api_key == "gm-key"
    assert cfg.web_search_mode == "gemini_google_search"
    assert cfg.label == "gemini"
    assert cfg.alias == "gemini"


def test_resolve_gemini_without_key_raises_400(monkeypatch):
    _patch_settings(monkeypatch, _fake_settings(openai_key="ok-key", anthropic_key="an-key"))
    with pytest.raises(HTTPException) as exc_info:
        resolve_passthrough_model("latest:gemini")
    assert exc_info.value.status_code == 400


def test_resolve_kimi_uses_fireworks_with_anthropic_shape(monkeypatch):
    _patch_settings(monkeypatch, _fake_settings(fireworks_key="fw-key"))
    cfg = resolve_passthrough_model("latest:kimi")
    assert cfg.api_kind == "anthropic_messages"
    assert cfg.model_name == "accounts/fireworks/models/kimi-k2p6"
    assert cfg.api_key == "fw-key"
    assert cfg.base_url == "https://api.fireworks.ai/inference"
    assert cfg.web_search_mode == "drop"
    assert cfg.label == "fireworks"
    assert cfg.alias == "kimi"


def test_resolve_kimi_without_fireworks_key_raises_400_no_silent_fallback(monkeypatch):
    # Critical: even with Anthropic + OpenAI keys present, latest:kimi must NOT
    # silently fall back to Claude or GPT. Callers asked for Kimi.
    _patch_settings(monkeypatch, _fake_settings(openai_key="ok-key", anthropic_key="an-key"))
    with pytest.raises(HTTPException) as exc_info:
        resolve_passthrough_model("latest:kimi")
    assert exc_info.value.status_code == 400
    assert "latest:kimi" in exc_info.value.detail


def test_resolve_deepseek_uses_fireworks(monkeypatch):
    _patch_settings(monkeypatch, _fake_settings(fireworks_key="fw-key"))
    cfg = resolve_passthrough_model("latest:deepseek")
    assert cfg.api_kind == "anthropic_messages"
    assert cfg.model_name == "accounts/fireworks/models/deepseek-v4-pro"
    assert cfg.label == "fireworks"
    assert cfg.alias == "deepseek"


def test_resolve_qwen_uses_fireworks(monkeypatch):
    _patch_settings(monkeypatch, _fake_settings(fireworks_key="fw-key"))
    cfg = resolve_passthrough_model("latest:qwen")
    assert cfg.api_kind == "anthropic_messages"
    assert cfg.model_name == "accounts/fireworks/models/qwen3p6-plus"
    assert cfg.label == "fireworks"
    assert cfg.alias == "qwen"


def test_resolve_three_fireworks_aliases_route_to_distinct_models(monkeypatch):
    # Sanity: kimi/deepseek/qwen all share the Fireworks key but must resolve
    # to distinct upstream models. Catches a copy-paste bug in the alias table.
    _patch_settings(monkeypatch, _fake_settings(fireworks_key="fw-key"))
    kimi = resolve_passthrough_model("latest:kimi").model_name
    deepseek = resolve_passthrough_model("latest:deepseek").model_name
    qwen = resolve_passthrough_model("latest:qwen").model_name
    assert len({kimi, deepseek, qwen}) == 3


# ---------------------------------------------------------------------------
# Settings-driven model names + regex shape + unknown aliases
# ---------------------------------------------------------------------------


def test_resolve_reads_model_names_from_settings_not_hardcoded(monkeypatch):
    """Override a passthrough_*_model setting and confirm the resolver picks
    it up — proves the alias table is settings-driven, not hardcoded."""
    _patch_settings(
        monkeypatch,
        _fake_settings(
            anthropic_key="an-key",
            anthropic_sonnet_model="claude-future-1",
            fireworks_key="fw-key",
            fireworks_qwen_model="accounts/fireworks/models/qwen-future",
            gemini_key="gm-key",
            gemini_model="gemini-future-pro",
        ),
    )
    assert resolve_passthrough_model("latest:sonnet").model_name == "claude-future-1"
    assert resolve_passthrough_model("latest:qwen").model_name == "accounts/fireworks/models/qwen-future"
    assert resolve_passthrough_model("latest:gemini").model_name == "gemini-future-pro"


def test_passthrough_pattern_accepts_latest_prefix():
    """Regex shape: ``latest:`` prefix required, hyphens allowed in the body."""
    from minds.common.passthrough_config import is_passthrough_model

    assert is_passthrough_model("latest:sonnet") is True
    assert is_passthrough_model("latest:gpt-high") is True
    assert is_passthrough_model("latest:gpt-codex") is True
    assert is_passthrough_model("latest:qwen") is True
    # Old underscore form no longer recognized — hard cutover.
    assert is_passthrough_model("_sonnet_") is False
    assert is_passthrough_model("_gpt-5.5-high_") is False
    # Bare model names still don't match (real model strings stay untouched).
    assert is_passthrough_model("gpt-5.5") is False
    assert is_passthrough_model("claude-sonnet-4-6") is False
    # Prefix is mandatory.
    assert is_passthrough_model("sonnet") is False


def test_resolve_unknown_alias_raises_400(monkeypatch):
    _patch_settings(monkeypatch, _fake_settings(anthropic_key="an-key"))
    with pytest.raises(HTTPException) as exc_info:
        resolve_passthrough_model("latest:mystery")
    assert exc_info.value.status_code == 400
    assert "latest:mystery" in exc_info.value.detail


# ---------------------------------------------------------------------------
# Gemini translation — message + tool round-trips
# ---------------------------------------------------------------------------


def test_chat_messages_to_gemini_extracts_system_instruction():
    system_instruction, contents = _chat_messages_to_gemini(
        [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ]
    )
    assert system_instruction == "be brief"
    # One Content with a single text part for the user turn.
    assert len(contents) == 1
    assert contents[0].role == "user"
    parts = contents[0].parts or []
    assert len(parts) == 1
    assert parts[0].text == "hi"


def test_chat_messages_to_gemini_concatenates_multiple_system_messages():
    system_instruction, _ = _chat_messages_to_gemini(
        [
            {"role": "system", "content": "first"},
            {"role": "system", "content": "second"},
            {"role": "user", "content": "hi"},
        ]
    )
    assert system_instruction == "first\n\nsecond"


def test_chat_messages_to_gemini_assistant_tool_calls_become_function_call_parts():
    _, contents = _chat_messages_to_gemini(
        [
            {"role": "user", "content": "weather?"},
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

    # 3 turns: user, assistant (with text + function_call), tool-result-as-user.
    assert len(contents) == 3

    user_turn = contents[0]
    assert user_turn.role == "user"
    assert (user_turn.parts or [])[0].text == "weather?"

    assistant_turn = contents[1]
    assert assistant_turn.role == "model"
    assistant_parts = assistant_turn.parts or []
    assert len(assistant_parts) == 2
    # First part: leading text.
    assert assistant_parts[0].text == "Let me check."
    # Second part: function call carrying the parsed arguments.
    fn_call = assistant_parts[1].function_call
    assert fn_call.name == "get_weather"
    assert fn_call.args == {"city": "NYC"}

    tool_turn = contents[2]
    assert tool_turn.role == "user"  # Gemini puts tool results on user role
    fn_response = (tool_turn.parts or [])[0].function_response
    assert fn_response.name == "get_weather"
    # Raw text result wrapped under {"result": ...} since Gemini expects a dict.
    assert fn_response.response == {"result": "72F sunny"}


def test_translate_tools_for_gemini_function_declaration():
    out = _translate_tools_for_gemini(
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
    assert len(out) == 1
    decls = out[0].function_declarations or []
    assert len(decls) == 1
    assert decls[0].name == "lookup"
    assert decls[0].description == "Look it up"


def test_translate_tools_for_gemini_web_search_emits_google_search_tool():
    out = _translate_tools_for_gemini([{"type": "web_search"}])
    assert len(out) == 1
    assert out[0].google_search is not None


def test_translate_tools_for_gemini_drops_fetch_silently():
    function_tool = {"type": "function", "function": {"name": "f", "parameters": {}}}
    out = _translate_tools_for_gemini([{"type": "fetch"}, function_tool])
    # fetch dropped (no Gemini equivalent); function tool kept.
    assert len(out) == 1
    assert (out[0].function_declarations or [])[0].name == "f"


def test_translate_tools_for_gemini_drop_mode_strips_web_search():
    out = _translate_tools_for_gemini([{"type": "web_search"}], web_search_mode="drop")
    assert out == []


def test_chat_tool_choice_to_gemini_required_maps_to_any_mode():
    cfg = _chat_tool_choice_to_gemini("required")
    assert cfg is not None
    assert cfg.function_calling_config.mode == "ANY"


def test_chat_tool_choice_to_gemini_specific_function_pins_allowed_names():
    cfg = _chat_tool_choice_to_gemini({"type": "function", "function": {"name": "lookup"}})
    assert cfg is not None
    assert cfg.function_calling_config.mode == "ANY"
    assert cfg.function_calling_config.allowed_function_names == ["lookup"]


def test_chat_tool_choice_to_gemini_auto_returns_none():
    # AUTO is Gemini's default; no config emitted.
    assert _chat_tool_choice_to_gemini("auto") is None
    assert _chat_tool_choice_to_gemini(None) is None


def test_gemini_response_to_chat_extracts_text_and_usage():
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text="Hello world.", function_call=None)]),
                finish_reason="STOP",
            )
        ],
        usage_metadata=SimpleNamespace(prompt_token_count=5, candidates_token_count=3),
    )
    payload = _gemini_response_to_openai(response, "gemini-x")
    choice = payload["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "Hello world."
    assert choice["finish_reason"] == "stop"
    assert payload["usage"] == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    assert payload["model"] == "gemini-x"


def test_gemini_response_to_chat_function_call_becomes_tool_call():
    fn_call = SimpleNamespace(name="lookup", args={"q": "x"})
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text=None, function_call=fn_call)]),
                finish_reason="STOP",
            )
        ],
        usage_metadata=SimpleNamespace(prompt_token_count=4, candidates_token_count=2),
    )
    payload = _gemini_response_to_openai(response, "gemini-x")
    choice = payload["choices"][0]
    # tool_calls finish-reason takes precedence even when Gemini reports STOP.
    assert choice["finish_reason"] == "tool_calls"
    tool_calls = choice["message"]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "lookup"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"q": "x"}
    # Synthesized id pattern.
    assert tool_calls[0]["id"].startswith("call_")
    # No text content in the message.
    assert choice["message"]["content"] is None


def test_gemini_response_to_chat_max_tokens_finish_reason_maps_to_length():
    # Real google.genai FinishReason enum — _gemini_finish_reason_to_openai
    # now compares against the typed enum, not a getattr fallback.
    from google.genai import types as genai_types

    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text="cut off", function_call=None)]),
                finish_reason=genai_types.FinishReason.MAX_TOKENS,
            )
        ],
        usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1),
    )
    payload = _gemini_response_to_openai(response, "gemini-x")
    assert payload["choices"][0]["finish_reason"] == "length"


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
_fireworks_key_set = bool(_settings.fireworks.api_key)
_gemini_key_set = bool(_settings.gemini.api_key)

# Defaults chosen for breadth of native-tool support on the OpenAI Responses
# API (gpt-5-mini is fast and supports {"type":"web_search"}); override via
# env vars if your account doesn't have access to these specific models.
_LIVE_ANTHROPIC_MODEL = os.getenv("LIVE_TEST_ANTHROPIC_MODEL", "claude-sonnet-4-6")
_LIVE_OPENAI_MODEL = os.getenv("LIVE_TEST_OPENAI_MODEL", "gpt-5-mini")
_LIVE_FIREWORKS_MODEL = os.getenv("LIVE_TEST_FIREWORKS_MODEL", "accounts/fireworks/models/kimi-k2p6")
_LIVE_GEMINI_MODEL = os.getenv("LIVE_TEST_GEMINI_MODEL", "gemini-3.1-pro-preview")

requires_anthropic = pytest.mark.skipif(not _anthropic_key_set, reason="ANTHROPIC__API_KEY not configured")
requires_openai = pytest.mark.skipif(not _openai_key_set, reason="OPENAI__API_KEY not configured")
requires_fireworks = pytest.mark.skipif(not _fireworks_key_set, reason="FIREWORKS__API_KEY not configured")
requires_gemini = pytest.mark.skipif(not _gemini_key_set, reason="GEMINI__API_KEY not configured")


# Live-test config helpers read keys from real settings so calls go upstream.
def _live_anthropic_config(model_name: str = "") -> PassthroughModelConfig:
    return PassthroughModelConfig(
        api_kind="anthropic_messages",
        model_name=model_name or _LIVE_ANTHROPIC_MODEL,
        api_key=_settings.anthropic.api_key,
        web_search_mode="anthropic_native",
        label="anthropic",
    )


def _live_openai_config(model_name: str = "") -> PassthroughModelConfig:
    return PassthroughModelConfig(
        api_kind="openai_responses",
        model_name=model_name or _LIVE_OPENAI_MODEL,
        api_key=_settings.openai.api_key,
        base_url=_settings.openai.api_url,
        web_search_mode="openai_native",
        label="openai",
    )


def _live_fireworks_config(model_name: str = "") -> PassthroughModelConfig:
    return PassthroughModelConfig(
        api_kind="anthropic_messages",
        model_name=model_name or _LIVE_FIREWORKS_MODEL,
        api_key=_settings.fireworks.api_key,
        base_url=_settings.fireworks.anthropic_base_url,
        web_search_mode="drop",
        label="fireworks",
    )


def _live_gemini_config(model_name: str = "") -> PassthroughModelConfig:
    return PassthroughModelConfig(
        api_kind="gemini_native",
        model_name=model_name or _LIVE_GEMINI_MODEL,
        api_key=_settings.gemini.api_key,
        web_search_mode="gemini_google_search",
        label="gemini",
    )


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
    agent = PassthroughAgent(config=_live_anthropic_config())
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
    agent = PassthroughAgent(config=_live_anthropic_config())
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
    agent = PassthroughAgent(config=_live_anthropic_config())
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
    agent = PassthroughAgent(config=_live_openai_config())
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
    agent = PassthroughAgent(config=_live_openai_config())
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


@requires_fireworks
@pytest.mark.asyncio
async def test_live_fireworks_kimi_basic_completion():
    """Fireworks Kimi K2.6 via the Anthropic-compatible API at fireworks.ai.

    This validates §6.1 of the design doc — that the Anthropic SDK pointed
    at Fireworks's base_url round-trips a basic prompt for ``kimi-k2p6``.
    """
    agent = PassthroughAgent(config=_live_fireworks_config())
    response = await agent.proxy(
        messages=[Message(role=Role.user, content="In one sentence, what is 2 + 2?")],
        stream=False,
        request_id="live-fireworks-kimi",
        max_tokens=200,
    )

    assert response.status_code == 200, f"non-200 response: {response.body!r}"
    payload = _decode_response_body(response)
    _assert_non_empty_assistant_text(payload)


@requires_fireworks
@pytest.mark.asyncio
async def test_live_fireworks_kimi_drops_web_search_silently():
    """A Fireworks call carrying generic web_search must complete normally;
    the tool is dropped server-side (no Fireworks-hosted search index)."""
    agent = PassthroughAgent(config=_live_fireworks_config())
    response = await agent.proxy(
        messages=[Message(role=Role.user, content="Reply with the single word: ok")],
        stream=False,
        request_id="live-fireworks-no-search",
        tools=[{"type": "web_search"}],
        max_tokens=200,
    )

    # Drop is silent — the request succeeds, no search happens.
    assert response.status_code == 200, f"non-200 response: {response.body!r}"
    payload = _decode_response_body(response)
    _assert_non_empty_assistant_text(payload)


@requires_gemini
@pytest.mark.asyncio
async def test_live_gemini_basic_completion():
    """Gemini native generateContent + text response."""
    agent = PassthroughAgent(config=_live_gemini_config())
    response = await agent.proxy(
        messages=[Message(role=Role.user, content="In one sentence, what is 2 + 2?")],
        stream=False,
        request_id="live-gemini-basic",
        max_tokens=200,
    )

    assert response.status_code == 200, f"non-200 response: {response.body!r}"
    payload = _decode_response_body(response)
    _assert_non_empty_assistant_text(payload)


@requires_gemini
@pytest.mark.asyncio
async def test_live_gemini_web_search_via_google_search():
    """Gemini + generic web_search → Tool(google_search=GoogleSearch())."""
    agent = PassthroughAgent(config=_live_gemini_config())
    response = await agent.proxy(
        messages=[Message(role=Role.user, content="Give me 200 words on what happened today.")],
        stream=False,
        request_id="live-gemini-search",
        tools=[{"type": "web_search"}],
        max_tokens=2048,
    )

    assert response.status_code == 200, f"non-200 response: {response.body!r}"
    payload = _decode_response_body(response)
    text = _assert_non_empty_assistant_text(payload)
    assert len(text.split()) >= 30, f"unexpectedly short reply: {text!r}"


# ---------------------------------------------------------------------------
# Live end-to-end tests per `latest:X` alias
# ---------------------------------------------------------------------------
#
# These exercise the full resolver → SDK → upstream → translated-response
# path for every alias in the table. They auto-skip when the provider's
# API key is unset (so CI without keys still passes), and run as real
# upstream calls when keys are configured. Each test uses a minimal prompt
# and a small ``max_tokens`` budget to keep cost-per-run bounded.


async def _exercise_alias(alias: str) -> str:
    """Resolve ``alias`` end-to-end, call upstream, return the assistant text.

    Goes through ``resolve_passthrough_model`` (rather than building a
    ``PassthroughModelConfig`` by hand) so a broken alias-table entry
    surfaces here as a real upstream failure, not a silently-correct test.

    ``max_tokens`` is generous (2048) because some aliases route to
    reasoning models (Qwen 3.6 Plus, gpt-5.x with effort, etc.) that burn
    most of their output budget on internal chain-of-thought before
    emitting the user-visible text block. A tight budget would cause those
    models to stop mid-thinking with empty text content — a passing-but-
    misleading failure mode.
    """
    cfg = resolve_passthrough_model(alias)
    agent = PassthroughAgent(config=cfg)
    response = await agent.proxy(
        messages=[Message(role=Role.user, content="Reply with the single word: ok.")],
        stream=False,
        request_id=f"live-{alias.replace(':', '-')}",
        max_tokens=2048,
    )
    assert response.status_code == 200, f"non-200 response: {response.body!r}"
    payload = _decode_response_body(response)
    return _assert_non_empty_assistant_text(payload)


@requires_anthropic
@pytest.mark.parametrize("alias", ["latest:sonnet", "latest:opus", "latest:haiku"])
@pytest.mark.asyncio
async def test_live_latest_anthropic_aliases(alias: str):
    """End-to-end: each Anthropic alias resolves and returns a real response."""
    text = await _exercise_alias(alias)
    assert text.strip(), f"empty text for {alias!r}"


@requires_openai
@pytest.mark.parametrize(
    "alias",
    [
        "latest:gpt",
        "latest:gpt-low",
        "latest:gpt-medium",
        "latest:gpt-high",
        "latest:gpt-codex",
        "latest:gpt-mini",
        "latest:gpt-nano",
    ],
)
@pytest.mark.asyncio
async def test_live_latest_openai_aliases(alias: str):
    """End-to-end: each OpenAI alias (reasoning-tier + codex/mini/nano) returns a real response."""
    text = await _exercise_alias(alias)
    assert text.strip(), f"empty text for {alias!r}"


@requires_gemini
@pytest.mark.asyncio
async def test_live_latest_gemini_alias():
    text = await _exercise_alias("latest:gemini")
    assert text.strip()


@requires_fireworks
@pytest.mark.parametrize("alias", ["latest:kimi", "latest:deepseek", "latest:qwen"])
@pytest.mark.asyncio
async def test_live_latest_fireworks_aliases(alias: str):
    """End-to-end: each Fireworks-hosted open model alias returns a real response."""
    text = await _exercise_alias(alias)
    assert text.strip(), f"empty text for {alias!r}"
