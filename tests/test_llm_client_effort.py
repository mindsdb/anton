"""LLMClient: every call leaves the provider's own effort alone, and the stream budget follows it."""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from anton.core.llm.anthropic import AnthropicProvider
from anton.core.llm.client import LLMClient
from anton.core.llm.effort import LONGEST_SILENCE_S, WRITE_BUDGET_BY_EFFORT
from anton.core.llm.openai import OpenAIProvider
from anton.core.llm.provider import LLMProvider, LLMResponse, StreamComplete, ToolCall, Usage


class _Answer(BaseModel):
    ok: bool


class _RecordingProvider(LLMProvider):
    name = "fake"
    accepts_large_output = True

    def __init__(self, effort):
        self._reasoning_effort = effort
        self.calls: list[dict] = []

    async def complete(self, **kw):
        self.calls.append(kw)
        if kw.get("tool_choice"):
            return LLMResponse(
                content="", usage=Usage(output_tokens=1),
                tool_calls=[ToolCall(id="t", name=kw["tools"][0]["name"], input={"ok": True})],
            )
        return LLMResponse(content="ok", usage=Usage(output_tokens=1))

    async def stream(self, **kw):
        self.calls.append(kw)
        yield StreamComplete(response=LLMResponse(content="ok", usage=Usage(output_tokens=1)))


class _LegacyProvider(LLMProvider):
    """Accepts no reasoning_effort at all, like a provider written before it."""

    name = "legacy"

    async def complete(self, *, model, system, messages, tools=None, tool_choice=None,
                       max_tokens=4096, native_web_tools=None):
        return LLMResponse(content="ok", usage=Usage(output_tokens=1))


def _client(planning="max", coding="max", max_tokens=8192):
    planning_p, coding_p = _RecordingProvider(planning), _RecordingProvider(coding)
    llm = LLMClient(
        planning_provider=planning_p, planning_model="p",
        coding_provider=coding_p, coding_model="c", max_tokens=max_tokens,
    )
    return llm, planning_p, coding_p


async def _drain(events):
    return [e async for e in events]


async def _every_call(llm):
    await llm.plan(system="s", messages=[])
    await llm.code(system="s", messages=[])
    await _drain(llm.plan_stream(system="s", messages=[]))
    await _drain(llm.code_stream(system="s", messages=[]))
    await llm.generate_object(_Answer, system="s", messages=[])
    await llm.generate_object_code(_Answer, system="s", messages=[])


@pytest.mark.parametrize("effort", ["max", "xhigh", "high", "low", None])
async def test_no_call_overrides_the_providers_effort(effort):
    llm, planning, coding = _client(planning=effort, coding=effort)
    await _every_call(llm)
    assert len(planning.calls + coding.calls) == 6
    assert all("reasoning_effort" not in c for c in planning.calls + coding.calls)


async def test_legacy_provider_without_the_property_still_works():
    legacy = _LegacyProvider()
    llm = LLMClient(planning_provider=legacy, planning_model="p", coding_provider=legacy, coding_model="c")
    assert (await llm.plan(system="s", messages=[])).content == "ok"
    assert await _drain(llm.plan_stream(system="s", messages=[]))


@pytest.mark.parametrize(
    "effort, expected", [("max", 65536), ("xhigh", 32768), ("high", 16384), ("medium", 8192), (None, 8192)]
)
async def test_stream_budget_follows_the_effort(effort, expected):
    llm, planning, coding = _client(planning=effort, coding=effort)
    assert llm.stream_budget("planning") == expected
    await _drain(llm.plan_stream(system="s", messages=[]))
    await _drain(llm.code_stream(system="s", messages=[]))
    assert planning.calls[0]["max_tokens"] == expected
    assert coding.calls[0]["max_tokens"] == expected


async def test_unknown_endpoint_keeps_the_client_default():
    class _Generic(_RecordingProvider):
        accepts_large_output = False

    generic = _Generic("max")
    llm = LLMClient(planning_provider=generic, planning_model="p", coding_provider=generic, coding_model="c")
    assert llm.stream_budget("planning") == 8192
    await _drain(llm.plan_stream(system="s", messages=[]))
    assert generic.calls[0]["max_tokens"] == 8192


async def test_stream_budget_never_goes_below_a_large_default():
    big, _, _ = _client(planning="high", max_tokens=100000)
    assert big.stream_budget("planning") == 100000


async def test_explicit_max_tokens_wins_and_non_streamed_calls_keep_the_default():
    llm, planning, coding = _client(planning="max", coding="max")
    await _drain(llm.plan_stream(system="s", messages=[], max_tokens=20480))
    await llm.plan(system="s", messages=[])
    await llm.code(system="s", messages=[])
    await llm.generate_object(_Answer, system="s", messages=[])
    assert [c["max_tokens"] for c in planning.calls] == [20480, 8192, 8192]
    assert coding.calls[0]["max_tokens"] == 8192
    assert llm.max_tokens == 8192


@pytest.mark.parametrize("effort, expected", [("max", 128000), ("xhigh", 32768), ("high", 16384), (None, 8192)])
async def test_a_write_call_asks_for_the_write_tables_budget(effort, expected):
    llm, planning, _ = _client(planning=effort)
    assert llm.stream_budget("planning", by_effort=WRITE_BUDGET_BY_EFFORT) == expected
    await _drain(llm.plan_stream(system="s", messages=[], max_tokens=llm.stream_budget(
        "planning", by_effort=WRITE_BUDGET_BY_EFFORT)))
    assert planning.calls[0]["max_tokens"] == expected


async def test_an_unknown_endpoint_keeps_the_client_default_for_a_write_call_too():
    class _Generic(_RecordingProvider):
        accepts_large_output = False

    generic = _Generic("max")
    llm = LLMClient(planning_provider=generic, planning_model="p", coding_provider=generic, coding_model="c")
    assert llm.stream_budget("planning", by_effort=WRITE_BUDGET_BY_EFFORT) == 8192


@pytest.mark.parametrize("budget, expected", [(2048, 600.0), (8192, 600.0), (65536, 655.36), (128000, 1280.0)])
def test_the_silence_deadline_grows_with_a_budget_that_can_outlast_it(budget, expected):
    llm, _, _ = _client()
    assert llm._idle_timeout_for(max_tokens=budget) == expected


@pytest.mark.parametrize("configured", [0, -1, float("nan")])
def test_a_deadline_turned_off_stays_off_at_any_budget(configured):
    llm, _, _ = _client()
    llm.model_call_idle_timeout_s = configured
    assert llm._idle_timeout_for(max_tokens=128000) is None


@pytest.mark.parametrize("budget, expected", [(8192, 0.05), (60000, 0.05), (65536, 655.36)])
def test_a_short_configured_deadline_stays_until_the_budget_outgrows_what_600_s_covers(budget, expected):
    llm, _, _ = _client()
    llm.model_call_idle_timeout_s = 0.05
    assert llm._idle_timeout_for(max_tokens=budget) == expected


def test_a_configured_deadline_above_the_budgets_silence_still_wins():
    llm, _, _ = _client()
    llm.model_call_idle_timeout_s = 3000.0
    assert llm._idle_timeout_for(max_tokens=128000) == 3000.0


@pytest.mark.parametrize("make", [
    lambda: OpenAIProvider(api_key="k", base_url="https://gateway.invalid/v1",
                           flavor=OpenAIProvider.FLAVOR_MINDS_PASSTHROUGH),
    lambda: OpenAIProvider(api_key="k", base_url="https://gateway.invalid/v1"),
    lambda: OpenAIProvider(api_key="k", base_url="https://azure.invalid", api_version="2025-04-01-preview"),
    lambda: AnthropicProvider(api_key="k"),
])
async def test_the_sdk_read_timeout_never_cuts_a_silence_the_deadline_allows(make):
    provider = make()
    try:
        assert provider._client.timeout.read == LONGEST_SILENCE_S
        assert provider._client.timeout.connect == 5.0
    finally:
        await provider.aclose()
