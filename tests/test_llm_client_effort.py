"""LLMClient: the effort ceiling and the effort-derived stream budget."""
from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from anton.core.llm import client as client_module
from anton.core.llm.client import LLMClient
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


async def test_without_a_ceiling_no_override_is_sent():
    llm, planning, coding = _client()
    await _every_call(llm)
    assert all("reasoning_effort" not in c for c in planning.calls + coding.calls)


async def test_ceiling_lowers_every_planning_and_coding_call():
    llm, planning, coding = _client()
    with llm.effort_ceiling("high"):
        await _every_call(llm)
    assert [c["reasoning_effort"] for c in planning.calls + coding.calls] == ["high"] * 6


@pytest.mark.parametrize("effort", ["low", None, "turbo", "high"])
async def test_ceiling_leaves_lower_and_unknown_efforts_alone(effort):
    llm, planning, _ = _client(planning=effort)
    with llm.effort_ceiling("high"):
        await llm.plan(system="s", messages=[])
    assert "reasoning_effort" not in planning.calls[0]


async def test_legacy_provider_without_the_property_still_works():
    legacy = _LegacyProvider()
    llm = LLMClient(planning_provider=legacy, planning_model="p", coding_provider=legacy, coding_model="c")
    with llm.effort_ceiling("high"):
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


async def test_stream_budget_respects_the_ceiling_and_a_large_default():
    llm, _, _ = _client(planning="max")
    with llm.effort_ceiling("high"):
        assert llm.stream_budget("planning") == 16384
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


async def test_ceiling_is_reset_on_exit_in_the_same_task():
    llm, _, _ = _client()
    with llm.effort_ceiling("high"):
        assert client_module._EFFORT_CEILING.get() == "high"
    assert client_module._EFFORT_CEILING.get() is None


async def test_nested_ceiling_can_only_lower():
    llm, planning, _ = _client(planning="max")
    with llm.effort_ceiling("low"):
        with llm.effort_ceiling("high"):
            await llm.plan(system="s", messages=[])
        with llm.effort_ceiling("minimal"):
            await llm.plan(system="s", messages=[])
    assert [c["reasoning_effort"] for c in planning.calls] == ["low", "minimal"]


async def test_nested_unknown_level_keeps_the_outer_cap():
    llm, planning, _ = _client(planning="max")
    with llm.effort_ceiling("low"):
        with llm.effort_ceiling("turbo"):
            await llm.plan(system="s", messages=[])
    assert planning.calls[0]["reasoning_effort"] == "low"


async def test_ceiling_reaches_gathered_tasks():
    llm, _, _ = _client()
    seen: list[str | None] = []

    async def probe():
        seen.append(client_module._EFFORT_CEILING.get())

    with llm.effort_ceiling("high"):
        await asyncio.gather(probe(), probe())
    assert seen == ["high", "high"]
