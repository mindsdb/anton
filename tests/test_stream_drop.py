"""A connection that dies while a stream body is read is a provider blip."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import json

import httpx2
import pytest

from anton.chat import ChatSession
from anton.core.llm.anthropic import AnthropicProvider
from anton.core.llm.openai import OpenAIProvider
from anton.core.llm.provider import (
    ProviderOverloadedError,
    StreamTextDelta,
    TransientProviderError,
    is_stream_drop,
)
from anton.core.session import ChatSessionConfig
from tests.conftest import make_mock_llm

MSGS = [{"role": "user", "content": "hi"}]
DROPS = [
    httpx2.RemoteProtocolError("peer closed connection"),
    httpx2.ReadError("connection reset"),
    httpx2.ReadTimeout("read timed out"),
]


async def _dies_after(items, exc):
    for item in items:
        yield item
    raise exc


def _chunk(text):
    return SimpleNamespace(
        usage=None, model="m",
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text, tool_calls=None), finish_reason=None)],
    )


def _assert_blip(info, error, target):
    assert info.value.code == "connection_error"
    assert info.value.session_backoff is True
    assert info.value.__cause__ is error
    assert f"Lost the connection to {target} mid-response" in str(info.value)


@pytest.mark.parametrize("error", DROPS)
async def test_openai_chat_stream_drop(error):
    with patch("anton.core.llm.openai.openai.AsyncOpenAI") as client_cls:
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=_dies_after([_chunk("par")], error))
        client_cls.return_value = client
        provider = OpenAIProvider(api_key="k", flavor=OpenAIProvider.FLAVOR_MINDS_PASSTHROUGH)
        with pytest.raises(TransientProviderError) as info:
            async for _ in provider.stream(model="m", system="s", messages=MSGS):
                pass
    _assert_blip(info, error, "the model provider")


@pytest.mark.parametrize("error", DROPS)
async def test_openai_responses_stream_drop(error):
    with patch("anton.core.llm.openai.openai.AsyncOpenAI") as client_cls:
        client = MagicMock()
        client.responses.create = AsyncMock(return_value=_dies_after([], error))
        client_cls.return_value = client
        provider = OpenAIProvider(api_key="k", flavor=OpenAIProvider.FLAVOR_OPENAI)
        with pytest.raises(TransientProviderError) as info:
            async for _ in provider.stream(model="gpt-5", system="s", messages=MSGS):
                pass
    _assert_blip(info, error, "the model provider")


async def test_openai_drop_before_the_stream_is_not_a_drop():
    """The SDK already retried a failure to connect, so the session must not."""
    with patch("anton.core.llm.openai.openai.AsyncOpenAI") as client_cls:
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=httpx2.ConnectError("refused"))
        client_cls.return_value = client
        provider = OpenAIProvider(api_key="k", flavor=OpenAIProvider.FLAVOR_MINDS_PASSTHROUGH)
        with pytest.raises(TransientProviderError) as info:
            async for _ in provider.stream(model="m", system="s", messages=MSGS):
                pass
    assert info.value.code == "connection_error"
    assert info.value.session_backoff is False
    assert not is_stream_drop(info.value)


class _DyingAnthropicStream:
    def __init__(self, exc):
        self._exc = exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return _dies_after([], self._exc)


@pytest.mark.parametrize("error", DROPS)
async def test_anthropic_stream_drop(error):
    with patch("anton.core.llm.anthropic.anthropic.AsyncAnthropic") as client_cls:
        client = MagicMock()
        client.messages.stream = MagicMock(return_value=_DyingAnthropicStream(error))
        client_cls.return_value = client
        provider = AnthropicProvider(api_key="k")
        with pytest.raises(TransientProviderError) as info:
            async for _ in provider.stream(model="m", system="s", messages=MSGS):
                pass
    _assert_blip(info, error, "Anthropic")


@pytest.mark.parametrize(
    "exc, expected",
    [
        (httpx2.RemoteProtocolError("x"), True),
        (TransientProviderError("x", code="connection_error", session_backoff=True), True),
        (TransientProviderError("x", code="connection_error", session_backoff=False), False),
        (TransientProviderError("x", code="stream_error", session_backoff=True), False),
        (ValueError("x"), False),
    ],
)
def test_is_stream_drop(exc, expected):
    assert is_stream_drop(exc) is expected


def _session(*, drops: int, calls: dict) -> ChatSession:
    s = ChatSession(ChatSessionConfig(llm_client=make_mock_llm()))
    s._llm.planning_model = "latest:sonnet"
    # Spent before the first retry: drops must not depend on the incident window.
    s._transient_budget_s = 0.0
    s._backoff_sleep = AsyncMock(return_value=False)

    async def _gen(user_msg):
        calls["n"] += 1
        if calls["n"] <= drops:
            raise TransientProviderError(
                "Lost the connection to Anthropic mid-response — try again in a moment.",
                provider="Anthropic", code="connection_error", session_backoff=True,
            )
        yield StreamTextDelta(text="done")

    s._stream_and_handle_tools = _gen
    return s


async def test_turn_retries_a_drop_twice_without_a_recovery_note():
    calls = {"n": 0}
    s = _session(drops=2, calls=calls)

    events = [e async for e in s.turn_stream("build it")]

    assert "done" in "".join(e.text for e in events if isinstance(e, StreamTextDelta))
    assert calls["n"] == 3
    assert s._backoff_sleep.await_count == 2
    assert "An error interrupted execution" not in json.dumps(s._history)


async def test_turn_ends_after_the_third_drop():
    calls = {"n": 0}
    s = _session(drops=3, calls=calls)

    with pytest.raises(ProviderOverloadedError) as info:
        _ = [e async for e in s.turn_stream("build it")]

    assert calls["n"] == 3
    assert "did not recover after 3 attempts" in str(info.value)
    assert "incident" not in str(info.value)
    assert info.value.provider == "Anthropic"
    assert info.value.model == "latest:sonnet"
