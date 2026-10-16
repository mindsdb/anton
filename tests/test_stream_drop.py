"""A connection that dies while a stream body is read is a provider blip."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import pytest

from anton.core.llm.anthropic import AnthropicProvider
from anton.core.llm.openai import OpenAIProvider
from anton.core.llm.provider import (
    ProviderOverloadedError,
    StreamTextDelta,
    TransientProviderError,
    is_stream_drop,
)
from tests.test_provider import _FakeAnthropicStream
from tests.test_transient_retry import (
    _chunk,
    _drain_openai_with_create,
    _session,
    _stream_that_fails_then_succeeds,
)

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


def _assert_blip(info, error, target):
    assert info.value.code == "connection_error"
    assert info.value.session_backoff is True
    assert info.value.__cause__ is error
    assert f"Lost the connection to {target} mid-response" in str(info.value)


@pytest.mark.parametrize("error", DROPS)
@pytest.mark.parametrize(
    "flavor, create, model, items",
    [
        (OpenAIProvider.FLAVOR_MINDS_PASSTHROUGH, "chat.completions", "m", [_chunk(content="par")]),
        (OpenAIProvider.FLAVOR_OPENAI, "responses", "gpt-5", []),
    ],
)
async def test_openai_stream_drop(flavor, create, model, items, error):
    with patch("anton.core.llm.openai.openai.AsyncOpenAI") as client_cls:
        client = MagicMock()
        api = client
        for name in create.split("."):
            api = getattr(api, name)
        api.create = AsyncMock(return_value=_dies_after(items, error))
        client_cls.return_value = client
        provider = OpenAIProvider(api_key="k", flavor=flavor)
        with pytest.raises(TransientProviderError) as info:
            async for _ in provider.stream(model=model, system="s", messages=MSGS):
                pass
    _assert_blip(info, error, "the model provider")


async def test_openai_drop_before_the_stream_is_not_a_drop():
    """The SDK already retried a failure to connect, so the session must not."""
    exc = await _drain_openai_with_create(AsyncMock(side_effect=httpx2.ConnectError("refused")))
    assert exc.code == "connection_error"
    assert exc.session_backoff is False
    assert not is_stream_drop(exc)


class _DyingAnthropicStream(_FakeAnthropicStream):
    def __init__(self, exc):
        super().__init__([])
        self._exc = exc

    async def _iter(self):
        raise self._exc
        yield  # pragma: no cover  (makes this an async generator)


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
        (TransientProviderError("x", code="connection_error", session_backoff=True), True),
        (TransientProviderError("x", code="connection_error", session_backoff=False), False),
        (TransientProviderError("x", code="stream_error", session_backoff=True), False),
        # Providers wrap the raw error, so one reaching a caller is not theirs.
        (httpx2.RemoteProtocolError("x"), False),
        (ValueError("x"), False),
    ],
)
def test_is_stream_drop(exc, expected):
    assert is_stream_drop(exc) is expected


def _dropping_session(drops: int, calls: dict):
    s = _session()
    # Spent before the first retry: drops must not depend on the incident window.
    s._transient_budget_s = 0.0
    s._backoff_sleep = AsyncMock(return_value=False)
    drop = TransientProviderError(
        "Lost the connection to Anthropic mid-response — try again in a moment.",
        provider="Anthropic", code="connection_error", session_backoff=True,
    )
    s._stream_and_handle_tools = _stream_that_fails_then_succeeds(drops, calls, drop)
    return s


async def test_turn_retries_a_drop_twice_without_a_recovery_note():
    calls = {"n": 0}
    s = _dropping_session(2, calls)

    events = [e async for e in s.turn_stream("build it")]

    text = "".join(e.text for e in events if isinstance(e, StreamTextDelta))
    assert "recovered-and-done" in text
    assert calls["n"] == 3
    assert s._backoff_sleep.await_count == 2
    assert "An error interrupted execution" not in json.dumps(s._history)


async def test_turn_ends_after_the_third_drop():
    calls = {"n": 0}
    s = _dropping_session(3, calls)

    with pytest.raises(ProviderOverloadedError) as info:
        _ = [e async for e in s.turn_stream("build it")]

    assert calls["n"] == 3
    assert "did not recover after 3 attempts" in str(info.value)
    assert "incident" not in str(info.value)
    assert info.value.provider == "Anthropic"
    assert info.value.model == "latest:sonnet"
