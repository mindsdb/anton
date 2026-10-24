"""The effort an operator sets is the effort the completion check and the page builder send.

These run the real completion check and the real generate_artifact pipeline over a provider set to
each effort, and read what each call asks the provider for: the effort it sends and its output
budget. The budget follows the effort, so the same reads show that it still does.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from anton.core.artifacts import ArtifactStore
from anton.core.llm.client import LLMClient
from anton.core.llm.provider import LLMProvider, LLMResponse, StreamComplete, StreamTextDelta, ToolCall, Usage
from anton.core.session import _VERIFIER_TOKEN_BUDGETS, ChatSession, ChatSessionConfig
from anton.core.tools.tool_defs import ToolDef

PROBE_TOOL = "probe"
GUARD_S = 5.0


@dataclass(frozen=True)
class SentCall:
    kind: str
    effort: str | None
    max_tokens: int


class _EffortProvider(LLMProvider):
    """Plays one scripted stream per `stream` call and one scripted answer per `complete` call.

    Like the real providers, a call that names no effort sends the provider's own, so each
    recorded call holds the effort it would send.
    """

    name = "effort-probe"
    accepts_large_output = True

    def __init__(self, effort: str | None, *, streams=(), completes=()) -> None:
        self._reasoning_effort = effort
        self.streams = list(streams)
        self.completes = list(completes)
        self.sent: list[SentCall] = []

    def _record(self, kind: str, kwargs: dict) -> None:
        effort = kwargs.get("reasoning_effort") or self.reasoning_effort
        self.sent.append(SentCall(kind=kind, effort=effort, max_tokens=kwargs["max_tokens"]))

    async def stream(self, **kwargs):
        self._record("stream", kwargs)
        async for event in self.streams.pop(0)():
            yield event

    async def complete(self, **kwargs) -> LLMResponse:
        self._record("complete", kwargs)
        return await self.completes.pop(0)()


def _usage() -> Usage:
    return Usage(input_tokens=10, output_tokens=5)


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(content=text, tool_calls=[], usage=_usage(), stop_reason="end_turn")


def _tool_response(call: ToolCall) -> LLMResponse:
    return LLMResponse(content="", tool_calls=[call], usage=_usage(), stop_reason="tool_use")


def _verdict_response() -> LLMResponse:
    call = ToolCall(id="v1", name="_VerifierVerdict", input={"status": "COMPLETE", "reason": "done"})
    return _tool_response(call)


def _stream_of(response: LLMResponse):
    async def _events():
        if response.content:
            yield StreamTextDelta(text=response.content)
        yield StreamComplete(response=response)

    return _events


def _answer(response: LLMResponse):
    async def _call():
        return response

    return _call


async def _drain(session: ChatSession, message: str) -> list:
    return [event async for event in session.turn_stream(message)]


@pytest.mark.parametrize("effort", ["max", "xhigh", "medium", None])
async def test_the_completion_check_sends_the_configured_effort(effort, monkeypatch):
    # One tool round is enough to reach the completion check, and the LLM check decides alone.
    monkeypatch.setenv("ANTON_VERIFY_MIN_TOOL_ROUNDS", "1")
    monkeypatch.setenv("ANTON_VERIFIER_JEV", "off")
    provider = _EffortProvider(
        effort,
        streams=[
            _stream_of(_tool_response(ToolCall(id="call_1", name=PROBE_TOOL, input={}))),
            _stream_of(_text_response("done")),
        ],
        completes=[_answer(_verdict_response())],
    )
    client = LLMClient(
        planning_provider=provider, planning_model="planner",
        coding_provider=provider, coding_model="coder",
    )
    base = Path(__file__).resolve().parents[1] / ".pytest-workspace"
    base.mkdir(parents=True, exist_ok=True)
    session = ChatSession(ChatSessionConfig(llm_client=client, workspace=MagicMock(base=base)))

    async def _quick_tool(_session, _input):
        return "ok"

    session.tool_registry.register_tool(ToolDef(
        name=PROBE_TOOL, description="test-only tool",
        input_schema={"type": "object", "properties": {}}, handler=_quick_tool,
    ))
    try:
        await _drain(session, "do it")
    finally:
        await session.close()

    checks = [call for call in provider.sent if call.kind == "complete"]
    assert [call.max_tokens for call in checks] == [_VERIFIER_TOKEN_BUDGETS[0]]
    assert {call.effort for call in provider.sent} == {effort}


@pytest.mark.parametrize(
    ("effort", "budget"), [("max", 65536), ("xhigh", 32768), ("high", 16384), ("medium", 8192)]
)
async def test_the_builders_first_call_sends_the_configured_effort_and_its_budget(effort, budget, tmp_path):
    artifacts = tmp_path / "artifacts"
    slug = ArtifactStore(artifacts).create(name="Clock", description="d", type="html-app").slug
    build = _tool_response(ToolCall(
        id="ga1", name="generate_artifact",
        input={"slug": slug, "user_request": "build a clock", "agent_understanding": "an analog clock",
               "user_language": "English"},
    ))
    first_pipeline_call_sent = asyncio.Event()

    async def _held_pipeline_call():
        first_pipeline_call_sent.set()
        await asyncio.Event().wait()
        yield StreamComplete(response=_text_response("unreachable"))

    provider = _EffortProvider(effort, streams=[_stream_of(build), _held_pipeline_call])
    client = LLMClient(
        planning_provider=provider, planning_model="planner",
        coding_provider=provider, coding_model="coder",
    )
    session = ChatSession(ChatSessionConfig(
        llm_client=client, workspace=MagicMock(base=tmp_path, artifacts_dir=artifacts),
    ))
    turn = asyncio.ensure_future(_drain(session, "make me a clock artifact"))
    try:
        await asyncio.wait_for(first_pipeline_call_sent.wait(), timeout=GUARD_S)
    finally:
        turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)
        await session.close()

    # The agent's own call, then the builder's first: the same effort and budget.
    assert provider.sent == [SentCall(kind="stream", effort=effort, max_tokens=budget)] * 2
