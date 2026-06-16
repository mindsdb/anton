"""Tests for reasoning-effort translation in the Anthropic and Fireworks proxies.

The level string on ``config.reasoning_effort`` (validated upstream by the
resolver) must reach the wire in each provider's shape: ``output_config`` for
direct Anthropic, ``extra_body.reasoning_effort`` for Fireworks' Anthropic-
compatible endpoint.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from minds.inference.providers.anthropic import proxy_anthropic
from minds.inference.providers.fireworks import proxy_fireworks
from minds.inference.types import ApiKind, PassthroughModelConfig, UsageBox, WebSearchMode


def _final_response(*, text="answer", in_tok=7, out_tok=3):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok),
    )


def _fake_client(create_mock):
    return SimpleNamespace(messages=SimpleNamespace(create=create_mock))


def _anthropic_config(reasoning_effort=None):
    return PassthroughModelConfig(
        api_kind=ApiKind.ANTHROPIC_MESSAGES,
        model_name="claude-opus-4-8",
        api_key="test-key",
        web_search_mode=WebSearchMode.ANTHROPIC_NATIVE,
        label="anthropic",
        alias="opus",
        reasoning_effort=reasoning_effort,
    )


def _fireworks_config(reasoning_effort=None):
    return PassthroughModelConfig(
        api_kind=ApiKind.FIREWORKS,
        model_name="accounts/fireworks/models/deepseek-v4-pro",
        api_key="test-key",
        base_url="https://api.fireworks.ai/inference",
        web_search_mode=WebSearchMode.EXTERNAL_TOOL,
        label="fireworks",
        alias="deepseek",
        reasoning_effort=reasoning_effort,
    )


async def test_anthropic_proxy_sends_output_config_effort():
    create_mock = AsyncMock(return_value=_final_response())
    await proxy_anthropic(
        client=_fake_client(create_mock),
        config=_anthropic_config(reasoning_effort="xhigh"),
        usage_box=UsageBox(),
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        request_id="req",
    )
    assert create_mock.await_args.kwargs["output_config"] == {"effort": "xhigh"}


async def test_anthropic_proxy_omits_output_config_without_effort():
    create_mock = AsyncMock(return_value=_final_response())
    await proxy_anthropic(
        client=_fake_client(create_mock),
        config=_anthropic_config(),
        usage_box=UsageBox(),
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        request_id="req",
    )
    assert "output_config" not in create_mock.await_args.kwargs


async def test_fireworks_proxy_sends_reasoning_effort_in_extra_body():
    create_mock = AsyncMock(return_value=_final_response())
    await proxy_fireworks(
        client=_fake_client(create_mock),
        config=_fireworks_config(reasoning_effort="high"),
        usage_box=UsageBox(),
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        request_id="req",
    )
    assert create_mock.await_args.kwargs["extra_body"] == {"reasoning_effort": "high"}


async def test_fireworks_proxy_omits_extra_body_without_effort():
    create_mock = AsyncMock(return_value=_final_response())
    await proxy_fireworks(
        client=_fake_client(create_mock),
        config=_fireworks_config(),
        usage_box=UsageBox(),
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        request_id="req",
    )
    assert "extra_body" not in create_mock.await_args.kwargs


async def test_fireworks_search_loop_carries_effort_every_turn():
    tool_use = SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", id="tu_1", name="web_search", input={"query": "q"})],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )
    create_mock = AsyncMock(side_effect=[tool_use, _final_response()])
    provider = SimpleNamespace(
        search=AsyncMock(return_value=[]),
        fetch=AsyncMock(return_value=None),
    )
    await proxy_fireworks(
        client=_fake_client(create_mock),
        config=_fireworks_config(reasoning_effort="low"),
        usage_box=UsageBox(),
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        request_id="req",
        tools=[{"type": "web_search"}],
        search_provider=provider,
    )
    assert create_mock.await_count == 2
    for call in create_mock.await_args_list:
        assert call.kwargs["extra_body"] == {"reasoning_effort": "low"}
