"""A per-call reasoning_effort overrides the provider's own for that call only."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anton.core.llm.anthropic import AnthropicProvider
from anton.core.llm.openai import OpenAIProvider

MSGS = [{"role": "user", "content": "hi"}]


class _Stop(Exception):
    """Ends the call right after the SDK was asked, so kwargs can be read."""


def _openai(flavor, effort="max"):
    patcher = patch("anton.core.llm.openai.openai.AsyncOpenAI")
    client_cls = patcher.start()
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=_Stop())
    client.responses.create = AsyncMock(side_effect=_Stop())
    client_cls.return_value = client
    provider = OpenAIProvider(api_key="k", flavor=flavor, reasoning_effort=effort)
    return provider, client, patcher


def _anthropic(effort="max"):
    patcher = patch("anton.core.llm.anthropic.anthropic.AsyncAnthropic")
    client_cls = patcher.start()
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=_Stop())
    client.messages.stream = MagicMock(side_effect=_Stop())
    client_cls.return_value = client
    return AnthropicProvider(api_key="k", reasoning_effort=effort), client, patcher


def test_property_reports_the_instance_effort():
    provider, _, patcher = _anthropic("xhigh")
    try:
        assert provider.reasoning_effort == "xhigh"
    finally:
        patcher.stop()


@pytest.mark.parametrize(
    "flavor, expected",
    [
        (OpenAIProvider.FLAVOR_OPENAI, True),
        (OpenAIProvider.FLAVOR_MINDS_PASSTHROUGH, True),
        (OpenAIProvider.FLAVOR_OPENAI_COMPATIBLE_GENERIC, False),
    ],
)
def test_large_output_only_for_known_endpoints(flavor, expected):
    provider, _, patcher = _openai(flavor)
    try:
        assert provider.accepts_large_output is expected
    finally:
        patcher.stop()
    anthropic_provider, _, patcher = _anthropic()
    try:
        assert anthropic_provider.accepts_large_output is True
    finally:
        patcher.stop()


@pytest.mark.parametrize("override, expected", [("high", "high"), (None, "max")])
async def test_openai_chat_complete(override, expected):
    provider, client, patcher = _openai(OpenAIProvider.FLAVOR_MINDS_PASSTHROUGH)
    try:
        with pytest.raises(_Stop):
            await provider.complete(model="m", system="s", messages=MSGS, reasoning_effort=override)
        assert client.chat.completions.create.call_args.kwargs["reasoning_effort"] == expected
    finally:
        patcher.stop()


@pytest.mark.parametrize("override, expected", [("high", "high"), (None, "max")])
async def test_openai_chat_stream(override, expected):
    provider, client, patcher = _openai(OpenAIProvider.FLAVOR_MINDS_PASSTHROUGH)
    try:
        with pytest.raises(_Stop):
            async for _ in provider.stream(model="m", system="s", messages=MSGS, reasoning_effort=override):
                pass
        assert client.chat.completions.create.call_args.kwargs["reasoning_effort"] == expected
    finally:
        patcher.stop()


@pytest.mark.parametrize("override, expected", [("high", "high"), (None, "max")])
async def test_openai_responses_complete_and_stream(override, expected):
    provider, client, patcher = _openai(OpenAIProvider.FLAVOR_OPENAI)
    try:
        with pytest.raises(_Stop):
            await provider.complete(model="gpt-5", system="s", messages=MSGS, reasoning_effort=override)
        assert client.responses.create.call_args.kwargs["reasoning"]["effort"] == expected
        with pytest.raises(_Stop):
            async for _ in provider.stream(model="gpt-5", system="s", messages=MSGS, reasoning_effort=override):
                pass
        assert client.responses.create.call_args.kwargs["reasoning"]["effort"] == expected
    finally:
        patcher.stop()


@pytest.mark.parametrize("override, expected", [("high", "high"), (None, "max")])
async def test_anthropic_complete_and_stream(override, expected):
    provider, client, patcher = _anthropic()
    try:
        with pytest.raises(_Stop):
            await provider.complete(model="m", system="s", messages=MSGS, reasoning_effort=override)
        assert client.messages.create.call_args.kwargs["extra_body"] == {"output_config": {"effort": expected}}
        with pytest.raises(_Stop):
            async for _ in provider.stream(model="m", system="s", messages=MSGS, reasoning_effort=override):
                pass
        assert client.messages.stream.call_args.kwargs["extra_body"] == {"output_config": {"effort": expected}}
    finally:
        patcher.stop()
