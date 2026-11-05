"""generate_artifact on a model and effort of its own: `ANTON_ARTIFACT_MODEL` and
`ANTON_ARTIFACT_REASONING_EFFORT`.

Every planning, coding and structured call the pipeline makes asks for the
builder's model, on the agent's own provider and so its endpoint, with the
builder's effort. The agent's calls in the same turn keep their own. The tests drive a real `LLMClient` over a provider that
records each call, because the route is in what each call asks for.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anthropic
import httpx2 as httpx
import openai
import pytest
from pydantic import BaseModel

import anton.core.tools.generate_artifact as gen_pkg
from anton.core.artifacts import ArtifactStore
from anton.core.llm.anthropic import AnthropicProvider
from anton.core.llm.client import LLMClient, ModelRoute, model_route
from anton.core.llm.openai import OpenAIProvider
from anton.core.llm.provider import LLMProvider, LLMResponse, StreamComplete, ToolCall, Usage
from anton.core.tools import tool_handlers
from anton.core.tools.generate_artifact import orchestrator
from anton.core.tools.generate_artifact.debug_trace import GenTrace
from anton.core.tools.generate_artifact.state import (
    ARTIFACT_EFFORT_ENV_VAR, ARTIFACT_MODEL_ENV_VAR, FetchVerdict, GenState, RequiredData, artifact_model_route,
)
from anton.core.tools.progress import ToolProgress
from anton.core.tools.tool_handlers import handle_generate_artifact

BUILDER = "builder-model"


class _Answer(BaseModel):
    ok: bool


class _RecordingProvider(LLMProvider):
    """Records each call's model and effort, and answers every structured call
    with the arguments the data loop's two decisions and `_Answer` take."""

    name = "fake"
    accepts_large_output = True

    def __init__(self, effort: str | None = "max") -> None:
        self._reasoning_effort = effort
        self.calls: list[tuple[str, str | None]] = []

    @property
    def reasoning_effort(self) -> str | None:
        return self._reasoning_effort

    def _record(self, kw: dict) -> None:
        # The key's absence means the provider's own effort; "" means none at all.
        self.calls.append((kw["model"], kw.get("reasoning_effort", "own")))

    async def complete(self, **kw):
        self._record(kw)
        if kw.get("tools"):
            name = kw["tools"][0]["name"]
            arguments = {
                "RequiredData": {"items": [], "reasoning": "the parts table"},
                "FetchVerdict": {"possible": True, "reasoning": "it can be read"},
            }.get(name, {"ok": True})
            return LLMResponse(content="", usage=Usage(output_tokens=1),
                               tool_calls=[ToolCall(id="t", name=name, input=arguments)])
        return LLMResponse(content="ok", usage=Usage(output_tokens=1), model=kw["model"])

    async def stream(self, **kw):
        self._record(kw)
        yield StreamComplete(response=LLMResponse(content="ok", usage=Usage(output_tokens=1), model=kw["model"]))


def _client(effort: str | None = "max") -> tuple[LLMClient, _RecordingProvider]:
    provider = _RecordingProvider(effort)
    llm = LLMClient(planning_provider=provider, planning_model="agent-model",
                    coding_provider=provider, coding_model="coding-model")
    return llm, provider


async def _drain(events) -> None:
    async for _ in events:
        pass


async def _every_call(llm: LLMClient) -> None:
    await llm.plan(system="s", messages=[])
    await llm.code(system="s", messages=[])
    await _drain(llm.plan_stream(system="s", messages=[]))
    await _drain(llm.code_stream(system="s", messages=[]))
    await llm.generate_object(_Answer, system="s", messages=[])
    await llm.generate_object_code(_Answer, system="s", messages=[])


# ── the route on every call ───────────────────────────────────────────────────


async def test_outside_a_route_every_call_keeps_its_roles_model_and_effort():
    llm, provider = _client()
    await _every_call(llm)
    assert provider.calls == [("agent-model", "own"), ("coding-model", "own")] * 3


async def test_inside_a_route_every_call_asks_for_its_model_and_effort():
    llm, provider = _client()
    with model_route(ModelRoute(BUILDER, "medium")):
        await _every_call(llm)
    assert provider.calls == [(BUILDER, "medium")] * 6


async def test_a_route_above_high_sends_its_effort_uncapped():
    llm, provider = _client()
    with model_route(ModelRoute(BUILDER, "xhigh")):
        await _every_call(llm)
    assert provider.calls == [(BUILDER, "xhigh")] * 6


async def test_a_route_with_no_effort_clears_the_providers_own():
    llm, provider = _client(effort="max")
    with model_route(ModelRoute(BUILDER)):
        await _every_call(llm)
    assert provider.calls == [(BUILDER, "")] * 6


async def test_a_route_with_no_effort_over_a_provider_with_none_passes_nothing():
    """A host's provider that takes no effort argument still works."""
    llm, provider = _client(effort=None)
    with model_route(ModelRoute(BUILDER)):
        await _every_call(llm)
    assert provider.calls == [(BUILDER, "own")] * 6


@pytest.mark.parametrize("api", ["chat", "responses", "messages"])
async def test_an_empty_effort_sends_no_effort_on_the_wire(api):
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        if api == "chat":
            return httpx.Response(200, json={
                "id": "c", "object": "chat.completion", "created": 1, "model": BUILDER,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })
        if api == "responses":
            return httpx.Response(200, json={
                "id": "r", "object": "response", "created_at": 1, "model": BUILDER, "status": "completed",
                "error": None, "incomplete_details": None,
                "output": [{"type": "message", "id": "m", "status": "completed", "role": "assistant",
                            "content": [{"type": "output_text", "text": "ok", "annotations": []}]}],
                "usage": {"input_tokens": 1, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 1,
                          "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 2},
            })
        return httpx.Response(200, json={
            "id": "msg", "type": "message", "role": "assistant", "model": BUILDER,
            "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    if api == "messages":
        provider = AnthropicProvider(api_key="k", reasoning_effort="high")
        provider._client = anthropic.AsyncAnthropic(api_key="k", max_retries=0, http_client=transport)
    else:
        flavor = OpenAIProvider.FLAVOR_OPENAI if api == "responses" else OpenAIProvider.FLAVOR_OPENAI_COMPATIBLE_GENERIC
        provider = OpenAIProvider(api_key="k", base_url="https://models.example.test/v1", flavor=flavor,
                                  reasoning_effort="high")
        provider._client = openai.AsyncOpenAI(api_key="k", base_url="https://models.example.test/v1", max_retries=0,
                                              http_client=transport)
    llm = LLMClient(planning_provider=provider, planning_model="agent-model",
                    coding_provider=provider, coding_model="agent-model")

    await llm.plan(system="s", messages=[{"role": "user", "content": "q"}])
    with model_route(ModelRoute(BUILDER)):
        await llm.plan(system="s", messages=[{"role": "user", "content": "q"}])

    agents, builders = sent
    assert builders["model"] == BUILDER
    effort_keys = ("reasoning_effort", "reasoning", "output_config")
    assert any(key in agents for key in effort_keys), agents
    assert not any(key in builders for key in effort_keys), builders


async def test_usage_names_the_routed_model_and_the_identity_line_keeps_the_agents():
    llm, _ = _client()
    reported: list[tuple[str, str]] = []
    llm.usage_listener = lambda role, model, _usage: reported.append((role, model))

    await llm.plan(system="s", messages=[])
    with model_route(ModelRoute(BUILDER)):
        await llm.plan(system="s", messages=[])
        await _drain(llm.plan_stream(system="s", messages=[]))

    assert reported == [("planning", "agent-model"), ("planning", BUILDER), ("planning", BUILDER)]
    assert llm.last_served_model == "agent-model"


@pytest.mark.parametrize(("effort", "budget"), [("max", 65536), ("high", 16384), (None, 8192)])
async def test_the_stream_budget_follows_the_routes_effort(effort, budget):
    llm, _ = _client(effort="low")
    with model_route(ModelRoute(BUILDER, effort)):
        assert llm.stream_budget("planning") == budget


async def test_the_route_reaches_gathered_tasks_and_ends_with_its_block():
    llm, provider = _client()
    with model_route(ModelRoute(BUILDER, "medium")):
        await asyncio.gather(_drain(llm.plan_stream(system="s", messages=[])),
                             _drain(llm.code_stream(system="s", messages=[])))
    await llm.plan(system="s", messages=[])
    assert provider.calls == [(BUILDER, "medium"), (BUILDER, "medium"), ("agent-model", "own")]


# ── where the route comes from ────────────────────────────────────────────────


@pytest.mark.parametrize(("model", "effort", "route"), [
    (None, None, None),
    ("  ", "medium", None),
    (BUILDER, None, ModelRoute(BUILDER)),
    (BUILDER, " ", ModelRoute(BUILDER)),
    (f" {BUILDER} ", "medium", ModelRoute(BUILDER, "medium")),
], ids=["unset", "blank-model", "no-effort", "blank-effort", "both"])
def test_the_builders_route_comes_from_the_process_environment(monkeypatch, model, effort, route):
    for name, value in ((ARTIFACT_MODEL_ENV_VAR, model), (ARTIFACT_EFFORT_ENV_VAR, effort)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert artifact_model_route() == route


def _make_artifact(tmp_path: Path) -> str:
    return ArtifactStore(tmp_path / "artifacts").create(name="Clock", description="d", type="fullstack-stateless-app").slug


async def _generate(session, slug: str) -> None:
    tc_input = {"slug": slug, "user_request": "build it", "agent_understanding": "an app", "user_language": "English"}
    async for item in handle_generate_artifact(session, tc_input):
        assert isinstance(item, ToolProgress) or item is not None


@pytest.mark.parametrize(("effort", "sent"), [("medium", "medium"), ("xhigh", "xhigh"), (None, "")])
async def test_generation_sends_every_call_to_the_builders_model_and_the_agent_keeps_its_own(
    tmp_path: Path, monkeypatch, effort, sent,
):
    """The fullstack build's two write loops run under `asyncio.gather`, and
    the data loop's decisions are structured calls: each asks for the
    builder's model. The route is entered inside the task, so the agent's call
    after it keeps the agent's model and effort."""
    monkeypatch.setenv(ARTIFACT_MODEL_ENV_VAR, BUILDER)
    if effort is None:
        monkeypatch.delenv(ARTIFACT_EFFORT_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(ARTIFACT_EFFORT_ENV_VAR, effort)
    slug = _make_artifact(tmp_path)
    llm, provider = _client(effort="max")
    session = SimpleNamespace(_workspace=SimpleNamespace(artifacts_dir=tmp_path / "artifacts"), _llm=llm)

    async def fake_generate(**kw):
        await llm.generate_object(RequiredData, system="s", messages=[])
        await llm.generate_object(FetchVerdict, system="s", messages=[])
        await asyncio.gather(_drain(llm.code_stream(system="s", messages=[])),
                             _drain(llm.plan_stream(system="s", messages=[])))
        return "Backend verification failed after retry: stop here"

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    await _generate(session, slug)
    await llm.plan(system="s", messages=[])

    assert provider.calls == [(BUILDER, sent)] * 4 + [("agent-model", "own")]


async def test_without_a_builder_model_generation_keeps_the_agents_models_and_efforts(tmp_path: Path, monkeypatch):
    monkeypatch.delenv(ARTIFACT_MODEL_ENV_VAR, raising=False)
    monkeypatch.setenv(ARTIFACT_EFFORT_ENV_VAR, "medium")
    slug = _make_artifact(tmp_path)
    llm, provider = _client(effort="max")
    session = SimpleNamespace(_workspace=SimpleNamespace(artifacts_dir=tmp_path / "artifacts"), _llm=llm)

    async def fake_generate(**kw):
        await llm.plan(system="s", messages=[])
        await _drain(llm.code_stream(system="s", messages=[]))
        return "Backend verification failed after retry: stop here"

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    await _generate(session, slug)

    assert provider.calls == [("agent-model", "own"), ("coding-model", "own")]


# ── the step trace ────────────────────────────────────────────────────────────


async def test_the_data_loops_llm_call_events_name_the_model_each_call_asked_for(tmp_path: Path, monkeypatch):
    trace = tmp_path / "trace.jsonl"
    llm, provider = _client()
    session = AsyncMock()
    session._scratchpads.cell_sandbox = None
    session._llm = llm
    state = GenState(
        session=session, artifact_type="html-app", artifact_path=tmp_path, slug="a",
        brief="Parts that run short", is_fullstack=False, trace_log=GenTrace(str(trace)),
    )
    state.gathering_complete = True
    state.unverified_sources = ["parts table"]

    async def fetched(_state, _required_text=""):
        return "pulled a sample", True

    monkeypatch.setattr(orchestrator, "_fetch_data_sample", fetched)
    with model_route(ModelRoute(BUILDER, "medium")):
        assert await orchestrator._data_phase(state) is None

    events = [json.loads(line) for line in trace.read_text().splitlines()]
    calls = [(event["node"], event["model"]) for event in events if event["event"] == "llm_call"]
    assert calls == [("define_required_data", BUILDER), ("is_possible_to_fetch", BUILDER)]
    assert provider.calls == [(BUILDER, "medium")] * 2


# ── screenshot refusals, per model ────────────────────────────────────────────


def _session_on(llm: LLMClient) -> SimpleNamespace:
    return SimpleNamespace(_llm=llm, _render_screenshots_refused=False, _builder_screenshots_refused=set())


def test_the_builder_models_screenshot_refusal_leaves_the_agents_screenshots_on():
    llm, _ = _client()
    session = _session_on(llm)
    with model_route(ModelRoute(BUILDER)):
        tool_handlers.record_render_screenshots_refused(session)
        assert tool_handlers.render_screenshots_refused(session) is True
    assert tool_handlers.render_screenshots_refused(session) is False
    assert session._render_screenshots_refused is False


def test_the_agents_screenshot_refusal_leaves_the_builders_screenshots_on():
    llm, _ = _client()
    session = _session_on(llm)
    tool_handlers.record_render_screenshots_refused(session)
    with model_route(ModelRoute(BUILDER)):
        assert tool_handlers.render_screenshots_refused(session) is False
    assert tool_handlers.render_screenshots_refused(session) is True


def test_a_builder_on_the_agents_own_model_shares_the_agents_refusal():
    llm, _ = _client()
    session = _session_on(llm)
    with model_route(ModelRoute("agent-model")):
        tool_handlers.record_render_screenshots_refused(session)
    assert tool_handlers.render_screenshots_refused(session) is True
