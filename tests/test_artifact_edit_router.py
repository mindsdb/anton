"""route_request: the quick-edit bar's router (ENG-4227), against a fake Jev."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx2 as httpx
import pytest

from anton.core.artifact_editor.router import (
    ACCEPT_P,
    CRITERIA,
    MAX_MESSAGE_CHARS,
    RouteDecision,
    describe_artifact,
    route_request,
)
from anton.core.llm.jev import POLICY_UNAVAILABLE

FIXTURES = Path(__file__).parent / "fixtures" / "artifact_edits"
BASE = "https://hub.example/v1"


def answer(choice: str, p: float, *, model: str = "jev-1.13.0") -> dict:
    other = "agent" if choice == "edit" else "edit"
    return {"model": model, "answers": {"route": {"type": "choice", "choice": choice, "confidence": p,
                                                  "probabilities": {choice: p, other: round(1 - p, 6)}}},
            "usage": {"input_tokens": 400, "output_tokens": 30}}


class FakeJev:
    """Answers each POST with the next scripted reply and keeps the requests."""

    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.requests: list[httpx.Request] = []

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, BaseException):
            raise reply
        if callable(reply):
            return await reply(request)
        if isinstance(reply, httpx.Response):
            return reply
        return httpx.Response(200, json=reply)

    def body(self, index: int = 0) -> dict:
        return json.loads(self.requests[index].content)


async def route(jev: FakeJev, message: str = "make the revenue line purple", **kw) -> RouteDecision:
    params = {"files": ["index.html"], "kind": "html-app", "base_url": BASE, "api_key": "k-test", "timeout_s": 1.0}
    params.update(kw)
    async with httpx.AsyncClient(transport=httpx.MockTransport(jev.handler)) as client:
        return await route_request(message, http=client, **params)


# ── accepted routes ──────────────────────────────────────────────────


async def test_a_confident_edit_is_routed_to_the_editor():
    jev = FakeJev(answer("edit", 0.97))
    decision = await route(jev)

    assert decision.route == "edit" and decision.reason == "jev"
    assert decision.confidence == pytest.approx(0.97)
    assert decision.ms >= 0
    request = jev.requests[0]
    assert str(request.url) == f"{BASE}/decisions"
    assert request.headers["Authorization"] == "Bearer k-test"
    body = jev.body()
    assert body["model"] == "jev-1.13.0"
    assert body["state"]["request"] == "make the revenue line purple"
    assert body["state"]["artifact"] == {"kind": "html-app", "files": ["index.html"]}
    question = body["questions"]["route"]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == {"edit", "agent"} == set(CRITERIA)
    assert "uncertain" in question["instructions"]


async def test_a_confident_agent_answer_goes_to_the_agent():
    decision = await route(FakeJev(answer("agent", 0.99)), "add last quarter's numbers")
    assert (decision.route, decision.reason) == ("agent", "jev")
    assert decision.confidence == pytest.approx(0.99)


async def test_the_threshold_is_inclusive():
    decision = await route(FakeJev(answer("edit", ACCEPT_P)))
    assert decision.route == "edit"


# ── uncertainty and failures all go to the agent ─────────────────────


@pytest.mark.parametrize("choice,p,p_agent", [("edit", 0.79, 0.21), ("agent", 0.6, 0.6), ("edit", 0.5, 0.5)])
async def test_an_uncertain_answer_goes_to_the_agent(choice, p, p_agent):
    decision = await route(FakeJev(answer(choice, p)))
    assert (decision.route, decision.reason) == ("agent", "uncertain")
    assert decision.confidence == pytest.approx(p_agent)


async def test_a_timeout_goes_to_the_agent_within_the_deadline():
    async def slow(request):
        await asyncio.sleep(5)
        return httpx.Response(200, json=answer("edit", 1.0))

    decision = await route(FakeJev(slow), timeout_s=0.05)
    assert (decision.route, decision.reason, decision.confidence) == ("agent", "router_timeout", 0.0)
    assert decision.ms < 2000


@pytest.mark.parametrize("reply", [
    httpx.Response(500, text="boom"),
    httpx.Response(403, json={"error": {"message": "not allowed"}}),
    httpx.Response(200, text="not json"),
    {"model": "jev-1.13.0", "answers": {}},
    {"model": "jev-1.13.0", "answers": {"route": {"type": "choice", "choice": "maybe",
                                                  "probabilities": {"maybe": 1.0}}}},
    {"model": "jev-1.13.0", "answers": {"route": {"type": "choice", "choice": "edit",
                                                  "probabilities": {"edit": 1.7}}}},
    {"model": "jev-1.13.0", "answers": {"route": {"type": "choice", "choice": "edit",
                                                  "probabilities": {"edit": True}}}},
    {"model": "jev-1.13.0", "answers": {"route": {"type": "noul", "choice": "edit",
                                                  "probabilities": {"edit": 1.0}}}},
    httpx.ConnectError("refused"),
])
async def test_errors_go_to_the_agent(reply):
    decision = await route(FakeJev(reply))
    assert (decision.route, decision.reason, decision.confidence) == ("agent", "router_error", 0.0)


async def test_a_policy_unavailable_503_is_retried_once():
    jev = FakeJev(httpx.Response(503, json={"error": {"message": POLICY_UNAVAILABLE}}), answer("edit", 0.95))
    decision = await route(jev)
    assert (decision.route, decision.reason) == ("edit", "jev")
    assert len(jev.requests) == 2
    assert jev.body(0) == jev.body(1)


async def test_a_plain_text_policy_503_is_retried_too():
    jev = FakeJev(httpx.Response(503, text=POLICY_UNAVAILABLE), answer("edit", 0.95))
    assert (await route(jev)).route == "edit"
    assert len(jev.requests) == 2


async def test_a_second_policy_503_is_not_retried_again():
    refusal = httpx.Response(503, json={"error": {"message": POLICY_UNAVAILABLE}})
    jev = FakeJev(refusal, refusal, answer("edit", 0.95))
    decision = await route(jev)
    assert (decision.route, decision.reason) == ("agent", "router_error")
    assert len(jev.requests) == 2


async def test_any_other_503_is_not_retried():
    jev = FakeJev(httpx.Response(503, text="upstream overloaded"), answer("edit", 0.95))
    decision = await route(jev)
    assert (decision.route, decision.reason) == ("agent", "router_error")
    assert len(jev.requests) == 1


async def test_the_retry_stays_inside_the_deadline():
    async def refuse_slowly(request):
        await asyncio.sleep(0.03)
        return httpx.Response(503, json={"error": {"message": POLICY_UNAVAILABLE}})

    async def answer_late(request):
        await asyncio.sleep(5)
        return httpx.Response(200, json=answer("edit", 1.0))

    jev = FakeJev(refuse_slowly, answer_late)
    decision = await route(jev, timeout_s=0.1)
    assert decision.reason == "router_timeout" and decision.ms < 2000


# ── no call at all ───────────────────────────────────────────────────


@pytest.mark.parametrize("kw,message,reason", [
    ({"files": []}, "make it blue", "empty_artifact"),
    ({"api_key": ""}, "make it blue", "router_disabled"),
    ({"base_url": ""}, "make it blue", "router_disabled"),
    ({"model": None}, "make it blue", "no_decisions_model"),
    ({"model": ""}, "make it blue", "no_decisions_model"),
    ({}, "   ", "empty_message"),
    ({}, "x" * (MAX_MESSAGE_CHARS + 1), "message_too_long"),
])
async def test_nothing_to_ask_goes_to_the_agent_without_a_call(kw, message, reason):
    jev = FakeJev(answer("edit", 1.0))
    decision = await route(jev, message, **kw)
    assert (decision.route, decision.reason) == ("agent", reason)
    assert jev.requests == []


async def test_it_never_raises_even_with_a_broken_client():
    decision = await route_request("make it blue", files=["index.html"], kind="html-app", base_url=BASE,
                                   api_key="k", http=object())
    assert (decision.route, decision.reason) == ("agent", "router_error")


# ── what the router is told ──────────────────────────────────────────


async def test_a_base_url_without_v1_still_reaches_the_decisions_route():
    jev = FakeJev(answer("edit", 0.9))
    await route(jev, base_url="https://hub.example/")
    assert str(jev.requests[0].url) == "https://hub.example/v1/decisions"


async def test_optional_context_and_only_the_users_recent_messages_are_sent():
    jev = FakeJev(answer("edit", 0.9))
    history = [{"role": "user", "text": f"message {i}"} for i in range(6)]
    history.insert(3, {"role": "assistant", "text": "Made the title blue."})
    await route(jev, "bigger", name="Sales", data='{"rows": [3 x {"region": "North"}]}', history=history,
                files=[f"f{i}.html" for i in range(50)])
    state = jev.body()["state"]
    assert state["earlier_messages"] == ["message 2", "message 3", "message 4", "message 5"]
    assert state["artifact"]["name"] == "Sales"
    assert state["artifact"]["data"].startswith('{"rows"')
    assert len(state["artifact"]["files"]) == 40 and state["artifact"]["more_files"] == 10


def test_describe_artifact_reads_the_name_and_the_data_shape():
    described = describe_artifact(FIXTURES / "regional-sales-dashboard-58f1c05b")
    assert described["name"] == "Regional Sales Dashboard"
    assert '"revenue_usd"' in described["data"] and len(described["data"]) <= 1200


def test_describe_artifact_of_an_empty_folder_is_empty(tmp_path):
    assert describe_artifact(tmp_path) == {"name": "", "description": "", "data": ""}


# ── is there a decisions model at all? ───────────────────────────────

_PROVIDER_ENV = ("ANTON_DECISIONS_MODEL", "ANTON_PLANNING_PROVIDER", "ANTON_CODING_PROVIDER", "ANTON_MINDS_URL",
                 "ANTON_MINDS_API_KEY", "ANTON_OPENAI_BASE_URL", "ANTON_OPENAI_API_KEY", "ANTON_ANTHROPIC_API_KEY")


@pytest.fixture
def settings_env(monkeypatch, tmp_path):
    """AntonSettings built from the environment alone (no ~/.anton/.env)."""
    from anton.config.settings import AntonSettings

    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)

    def build(**env):
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        return AntonSettings(_env_file=None)

    return build


def mindshub(**extra) -> dict:
    return {"ANTON_PLANNING_PROVIDER": "minds-cloud", "ANTON_CODING_PROVIDER": "minds-cloud",
            "ANTON_MINDS_URL": "https://api.mindshub.ai", "ANTON_MINDS_API_KEY": "k-hub", **extra}


def test_mindshub_gets_jev_by_default(settings_env):
    from anton.core.artifact_editor.router import decisions_available, decisions_config

    settings = settings_env(**mindshub())
    assert settings.decisions_model == "jev-1.13.0"
    assert decisions_available(settings)
    assert decisions_config(settings) == ("https://api.mindshub.ai/v1", "k-hub", "jev-1.13.0")


@pytest.mark.parametrize("env", [
    {"ANTON_PLANNING_PROVIDER": "anthropic", "ANTON_ANTHROPIC_API_KEY": "k"},
    {"ANTON_PLANNING_PROVIDER": "openai", "ANTON_CODING_PROVIDER": "openai", "ANTON_OPENAI_API_KEY": "k"},
    {"ANTON_PLANNING_PROVIDER": "openai-compatible", "ANTON_CODING_PROVIDER": "openai-compatible",
     "ANTON_OPENAI_BASE_URL": "https://llm.example.com/v1", "ANTON_OPENAI_API_KEY": "k"},
    # A look-alike host is not MindsHub.
    {"ANTON_PLANNING_PROVIDER": "openai-compatible", "ANTON_CODING_PROVIDER": "openai-compatible",
     "ANTON_OPENAI_BASE_URL": "https://api.mindshub.ai.example.com/v1", "ANTON_OPENAI_API_KEY": "k"},
    # The legacy mdb.ai host serves no decisions.
    {"ANTON_PLANNING_PROVIDER": "minds-cloud", "ANTON_CODING_PROVIDER": "minds-cloud",
     "ANTON_MINDS_URL": "https://mdb.ai", "ANTON_MINDS_API_KEY": "k"},
])
def test_without_mindshub_there_is_no_decisions_model(settings_env, env):
    from anton.core.artifact_editor.router import decisions_available, decisions_config

    settings = settings_env(**env)
    assert settings.decisions_model is None
    assert not decisions_available(settings) and decisions_config(settings) is None


@pytest.mark.parametrize("value", ["", "off", "none"])
def test_an_explicit_empty_value_turns_it_off_on_mindshub(settings_env, value):
    from anton.core.artifact_editor.router import decisions_available

    settings = settings_env(**mindshub(ANTON_DECISIONS_MODEL=value))
    assert not decisions_available(settings)


def test_an_explicit_model_is_used(settings_env):
    from anton.core.artifact_editor.router import decisions_config

    settings = settings_env(**mindshub(ANTON_DECISIONS_MODEL="jev-2.0.0"))
    assert settings.decisions_model == "jev-2.0.0"
    assert decisions_config(settings) == ("https://api.mindshub.ai/v1", "k-hub", "jev-2.0.0")


def test_an_explicit_model_with_a_mindshub_key_works_beside_another_llm(settings_env):
    from anton.core.artifact_editor.router import decisions_config

    settings = settings_env(ANTON_PLANNING_PROVIDER="anthropic", ANTON_ANTHROPIC_API_KEY="k",
                            ANTON_MINDS_URL="https://api.mindshub.ai", ANTON_MINDS_API_KEY="k-hub",
                            ANTON_DECISIONS_MODEL="jev-1.13.0")
    assert decisions_config(settings) == ("https://api.mindshub.ai/v1", "k-hub", "jev-1.13.0")


def test_an_explicit_model_without_mindshub_credentials_is_unavailable(settings_env):
    from anton.core.artifact_editor.router import decisions_available

    settings = settings_env(ANTON_PLANNING_PROVIDER="anthropic", ANTON_ANTHROPIC_API_KEY="k",
                            ANTON_DECISIONS_MODEL="jev-1.13.0")
    assert not decisions_available(settings)


def test_plain_core_settings_resolve_the_same_way():
    from types import SimpleNamespace

    from anton.core.artifact_editor.router import decisions_config

    hub = SimpleNamespace(decisions_model=None, planning_provider="openai-compatible", coding_provider="",
                          openai_base_url="https://api.mindshub.ai/v1", openai_api_key="k-hub",
                          minds_url=None, minds_api_key=None)
    assert decisions_config(hub) == ("https://api.mindshub.ai/v1", "k-hub", "jev-1.13.0")
    assert decisions_config(SimpleNamespace(decisions_model=None)) is None


async def test_no_decisions_model_means_no_http_call(settings_env):
    from anton.core.artifact_editor.router import decisions_config

    settings = settings_env(ANTON_PLANNING_PROVIDER="anthropic", ANTON_ANTHROPIC_API_KEY="k")
    assert decisions_config(settings) is None
    jev = FakeJev(answer("edit", 1.0))
    decision = await route(jev, model=getattr(settings, "decisions_model", None))
    assert (decision.route, decision.reason, decision.confidence) == ("agent", "no_decisions_model", 0.0)
    assert jev.requests == []
