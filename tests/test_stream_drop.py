"""A connection that dies while a stream body is read is a provider blip."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import pytest

from anton.core.llm.anthropic import AnthropicProvider
from anton.core.llm.openai import OpenAIProvider
from anton.core.llm.provider import TransientProviderError, is_stream_drop

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
