"""One `llm_usage` log line per model call, when ANTON_USAGE_LOG is on.

An eval reads these lines from the host's log instead of Langfuse, so each one
must carry the call's conversation, turn and role, and must be written even
when the reply then fails a check.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import httpx2 as httpx
import pytest

from anton.core.llm.anthropic import AnthropicProvider
from anton.core.llm.openai import OpenAIProvider
from anton.core.llm.provider import LLMProvider, LLMResponse, TransientProviderError, Usage
from anton.core.llm.tracing import (
    TraceContext,
    call_role,
    get_trace_context,
    reset_trace_context,
    set_trace_context,
)
from anton.core.session import ChatSession, ChatSessionConfig
from tests.conftest import make_mock_llm

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


async def _aiter(items, then_raise=None):
    for item in items:
        yield item
    if then_raise is not None:
        raise then_raise


class _AnthropicStream:
    def __init__(self, events, then_raise=None):
        self._events = events
        self._then_raise = then_raise

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    def __aiter__(self):
        return _aiter(self._events, self._then_raise)


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


def _chunk(content=None, finish=None, usage=None):
    choices = [] if usage else [SimpleNamespace(
        delta=SimpleNamespace(content=content, tool_calls=None), finish_reason=finish)]
    return SimpleNamespace(model="served-model", choices=choices, usage=usage)


async def test_a_streamed_chat_call_logs_the_usage_of_its_last_chunk(usage_log, eval_turn):
    chunks = [_chunk("Hel"), _chunk("lo", finish="stop"),
              _chunk(usage=SimpleNamespace(prompt_tokens=50, completion_tokens=3,
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


async def test_a_streamed_chat_call_that_reports_no_usage_logs_nothing(usage_log):
    """A line of zeros would read as a call that cost nothing."""
    patcher, provider = _openai(AsyncMock(return_value=_aiter([_chunk("Hel"), _chunk("lo", finish="stop")])))
    try:
        async for _ in provider.stream(model="m", system="s", messages=MESSAGES):
            pass
    finally:
        patcher.stop()

    assert usage_log() == []


async def test_a_chat_call_that_reports_no_usage_logs_nothing(usage_log):
    response = _chat_response(content="Hello")
    response.usage = None
    patcher, provider = _openai(AsyncMock(return_value=response))
    try:
        await provider.complete(model="m", system="s", messages=MESSAGES)
    finally:
        patcher.stop()

    assert usage_log() == []


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


_MESSAGE_START = SimpleNamespace(type="message_start", message=SimpleNamespace(
    model="claude-served", usage=SimpleNamespace(input_tokens=50000, output_tokens=1)))
_TEXT_DELTA = SimpleNamespace(type="content_block_delta", index=0,
                              delta=SimpleNamespace(type="text_delta", text="Hel"))


def _streaming_anthropic(monkeypatch, stream):
    """A real provider, so its `except` clauses match the SDK's own error types."""
    provider = AnthropicProvider(api_key="k")
    monkeypatch.setattr(provider._client.messages, "stream", MagicMock(return_value=stream))
    return provider


def _connection_lost():
    return anthropic.APIConnectionError(request=httpx.Request("POST", "http://mock/v1/messages"))


async def test_a_streamed_anthropic_call_that_fails_midway_is_still_logged(usage_log, monkeypatch):
    stream = _AnthropicStream([_MESSAGE_START, _TEXT_DELTA], then_raise=_connection_lost())
    provider = _streaming_anthropic(monkeypatch, stream)
    with pytest.raises(TransientProviderError):
        async for _ in provider.stream(model="claude-alias", system="s", messages=MESSAGES):
            pass

    (line,) = usage_log()
    assert (line["served_model"], line["input"]) == ("claude-served", 50000)


async def test_a_streamed_anthropic_call_its_consumer_closes_is_still_logged(usage_log, monkeypatch):
    provider = _streaming_anthropic(monkeypatch, _AnthropicStream([_MESSAGE_START, _TEXT_DELTA, _TEXT_DELTA]))
    events = provider.stream(model="claude-alias", system="s", messages=MESSAGES)
    await anext(events)
    await events.aclose()

    (line,) = usage_log()
    assert line["input"] == 50000


async def test_a_streamed_anthropic_call_that_fails_before_message_start_logs_nothing(usage_log, monkeypatch):
    provider = _streaming_anthropic(monkeypatch, _AnthropicStream([], then_raise=_connection_lost()))
    with pytest.raises(TransientProviderError):
        async for _ in provider.stream(model="claude-alias", system="s", messages=MESSAGES):
            pass

    assert usage_log() == []


async def _trace_seen_by_end_of_turn_work(monkeypatch):
    """Run one turn and return the trace a task scheduled by the end-of-turn
    memory passes sees."""
    llm = make_mock_llm()
    llm.plan.return_value = LLMResponse(content="hi", usage=Usage(input_tokens=1))
    session = ChatSession(ChatSessionConfig(llm_client=llm))
    seen = []

    async def _record():
        seen.append(get_trace_context())

    tasks = []
    monkeypatch.setattr(session, "_schedule_cerebellum_flush", lambda: tasks.append(asyncio.create_task(_record())))
    async for _ in session.turn_stream("one", trace_metadata={"turn": "0"}):
        pass
    await asyncio.gather(*tasks)
    (trace,) = seen
    return trace


async def test_end_of_turn_work_keeps_the_turns_trace_when_the_usage_log_is_on(monkeypatch):
    monkeypatch.setenv("ANTON_USAGE_LOG", "1")
    trace = await _trace_seen_by_end_of_turn_work(monkeypatch)
    assert trace is not None and trace.metadata == {"turn": "0"}


async def test_end_of_turn_work_runs_without_a_trace_when_the_usage_log_is_off(monkeypatch):
    monkeypatch.delenv("ANTON_USAGE_LOG", raising=False)
    assert await _trace_seen_by_end_of_turn_work(monkeypatch) is None
