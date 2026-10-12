"""One `llm_usage` log line per model call, when ANTON_USAGE_LOG is on.

An eval reads these lines from the host's log instead of Langfuse, so each one
must carry the call's conversation, turn and role, and must be written even
when the reply then fails a check.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anton.core.llm.anthropic import AnthropicProvider
from anton.core.llm.openai import OpenAIProvider
from anton.core.llm.provider import LLMProvider, TransientProviderError, Usage
from anton.core.llm.tracing import TraceContext, call_role, reset_trace_context, set_trace_context

MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.fixture
def usage_log(monkeypatch, caplog):
    monkeypatch.setenv("ANTON_USAGE_LOG", "1")
    caplog.set_level("INFO", logger="anton.usage")

    def lines():
        return [json.loads(r.getMessage().removeprefix("llm_usage "))
                for r in caplog.records if r.name == "anton.usage"]

    return lines


@pytest.fixture
def eval_turn():
    token = set_trace_context(TraceContext(
        session_id="conv-1", tags=("eval_turn:2", "cowork-gate"), metadata={"turn": "2"},
    ))
    yield
    reset_trace_context(token)


def _chat_response(*, content, finish_reason="stop", prompt_tokens=120, cached=100, completion_tokens=16):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None),
                                 finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                              prompt_tokens_details=SimpleNamespace(cached_tokens=cached)),
        model="served-model",
    )


def _openai(create):
    patcher = patch("anton.core.llm.openai.openai")
    mock_openai = patcher.start()
    client = AsyncMock()
    client.chat.completions.create = create
    mock_openai.AsyncOpenAI.return_value = client
    return patcher, OpenAIProvider(api_key="k")


async def _aiter(items):
    for item in items:
        yield item


class _AnthropicStream:
    def __init__(self, events):
        self._events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    def __aiter__(self):
        return _aiter(self._events)


def _anthropic(**client_attrs):
    patcher = patch("anton.core.llm.anthropic.anthropic")
    mock_anthropic = patcher.start()
    client = AsyncMock()
    for name, value in client_attrs.items():
        setattr(client.messages, name, value)
    mock_anthropic.AsyncAnthropic.return_value = client
    return patcher, AnthropicProvider(api_key="k")


def test_nothing_is_logged_when_the_switch_is_off(monkeypatch, caplog):
    monkeypatch.delenv("ANTON_USAGE_LOG", raising=False)
    caplog.set_level("INFO", logger="anton.usage")

    class _Provider(LLMProvider):
        async def complete(self, **kwargs):
            raise NotImplementedError

    _Provider().log_usage(model="m", usage=Usage(input_tokens=1), served_model=None)
    assert not [r for r in caplog.records if r.name == "anton.usage"]


async def test_a_completed_chat_call_logs_its_conversation_turn_role_and_tokens(usage_log, eval_turn):
    patcher, provider = _openai(AsyncMock(return_value=_chat_response(content="Hello")))
    try:
        with call_role("coding"):
            await provider.complete(model="alias-model", system="s", messages=MESSAGES)
    finally:
        patcher.stop()

    assert usage_log() == [{
        "conversation": "conv-1", "turn": "2", "tags": ["eval_turn:2", "cowork-gate"], "role": "coding",
        "model": "alias-model", "served_model": "served-model",
        "input": 20, "output": 16, "cache_read": 100, "cache_creation": 0,
    }]


async def test_a_billed_chat_call_is_logged_even_when_its_empty_reply_raises(usage_log):
    patcher, provider = _openai(AsyncMock(return_value=_chat_response(content="", finish_reason=None)))
    try:
        with pytest.raises(TransientProviderError):
            await provider.complete(model="m", system="s", messages=MESSAGES)
    finally:
        patcher.stop()

    (line,) = usage_log()
    assert (line["input"], line["output"], line["conversation"]) == (20, 16, None)


async def test_a_streamed_chat_call_logs_the_usage_of_its_last_chunk(usage_log, eval_turn):
    def chunk(content=None, finish=None, usage=None):
        choices = [] if usage else [SimpleNamespace(
            delta=SimpleNamespace(content=content, tool_calls=None), finish_reason=finish)]
        return SimpleNamespace(model="served-model", choices=choices, usage=usage)

    chunks = [chunk("Hel"), chunk("lo", finish="stop"),
              chunk(usage=SimpleNamespace(prompt_tokens=50, completion_tokens=3,
                                          prompt_tokens_details=SimpleNamespace(cached_tokens=0)))]
    patcher, provider = _openai(AsyncMock(return_value=_aiter(chunks)))
    provider.trace_role = "router"
    try:
        async for _ in provider.stream(model="gate-model", system="s", messages=MESSAGES):
            pass
    finally:
        patcher.stop()

    (line,) = usage_log()
    assert (line["role"], line["model"], line["input"], line["output"]) == ("router", "gate-model", 50, 3)


async def test_a_completed_anthropic_call_logs_its_cache_tokens(usage_log):
    response = MagicMock()
    response.content = [SimpleNamespace(type="text", text="Hello")]
    response.usage = SimpleNamespace(input_tokens=5, output_tokens=7, cache_read_input_tokens=900,
                                     cache_creation_input_tokens=40)
    response.stop_reason = "end_turn"
    response.model = "claude-served"
    patcher, provider = _anthropic(create=AsyncMock(return_value=response))
    try:
        await provider.complete(model="claude-alias", system="s", messages=MESSAGES)
    finally:
        patcher.stop()

    (line,) = usage_log()
    assert (line["served_model"], line["input"], line["output"], line["cache_read"], line["cache_creation"]) == (
        "claude-served", 5, 7, 900, 40)


async def test_a_streamed_anthropic_call_is_logged_even_when_it_ends_empty(usage_log):
    events = [
        SimpleNamespace(type="message_start", message=SimpleNamespace(
            model="claude-served", usage=SimpleNamespace(input_tokens=30, output_tokens=0))),
    ]
    patcher, provider = _anthropic(stream=MagicMock(return_value=_AnthropicStream(events)))
    try:
        with pytest.raises(TransientProviderError):
            async for _ in provider.stream(model="claude-alias", system="s", messages=MESSAGES):
                pass
    finally:
        patcher.stop()

    (line,) = usage_log()
    assert (line["model"], line["input"]) == ("claude-alias", 30)
