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


async def test_optional_context_and_recent_turns_from_both_sides_are_sent():
    jev = FakeJev(answer("edit", 0.9))
    history = []
    for i in range(5):
        history += [{"role": "user", "text": f"ask {i}"}, {"role": "assistant", "text": f"did {i}"}]
    history.append({"role": "system", "text": "ignored"})
    await route(jev, "bigger", name="Sales", data='{"rows": [3 x {"region": "North"}]}', history=history,
                files=[f"f{i}.html" for i in range(50)])
    state = jev.body()["state"]
    assert state["earlier_messages"] == ["user: ask 2", "assistant: did 2", "user: ask 3", "assistant: did 3",
                                         "user: ask 4", "assistant: did 4"]
    assert state["artifact"]["name"] == "Sales"
    assert state["artifact"]["data"].startswith('{"rows"')
    assert len(state["artifact"]["files"]) == 40 and state["artifact"]["more_files"] == 10
    assert "outline" not in state["artifact"] and "recent_changes" not in state["artifact"] and "chat" not in state


async def test_outline_recent_changes_and_chat_reach_jev_clipped():
    jev = FakeJev(answer("edit", 0.95))
    changes = [f"Quick edit: change {i} " + "x" * 300 for i in range(9)]
    chat = [{"role": "user", "text": f"chat {i}"} for i in range(6)] + [{"role": "assistant", "text": "y" * 900}]
    decision = await route(jev, "make the unicorns poop rainbow", outline="o" * 3000, recent_changes=changes,
                           chat=chat)
    assert decision.route == "edit"
    state = jev.body()["state"]
    assert len(state["artifact"]["outline"]) <= 1500
    assert len(state["artifact"]["recent_changes"]) == 6
    assert state["artifact"]["recent_changes"][-1].startswith("Quick edit: change 8")
    assert all(len(c) <= 200 for c in state["artifact"]["recent_changes"])
    assert state["chat"][:3] == ["user: chat 3", "user: chat 4", "user: chat 5"]
    assert len(state["chat"]) == 4 and len(state["chat"][-1]) <= 300 + len("assistant: ")
    instructions = jev.body()["questions"]["route"]["instructions"]
    assert "state.artifact.outline" in instructions and "recent_changes" in instructions


def test_the_outline_shows_what_the_page_contains():
    from anton.core.artifact_editor.router import artifact_outline

    sky = artifact_outline(FIXTURES / "night-sky-unicorns-5b2e91c4")
    assert sky.startswith("files: index.html; title: Night Sky")
    for part in ("canvas", "animation", "moon-breathe", "drawUnicorn", "UNICORN_SPEED", "STAR_TWINKLE_SPEED"):
        assert part in sky
    timer = artifact_outline(FIXTURES / "focus-timer-9008a47d")
    assert "One thing at a time." in timer and "startButton" in timer and len(timer) <= 1500


def test_a_component_page_is_outlined_from_its_program_not_the_runtime():
    from anton.core.artifact_editor.router import artifact_outline

    sales = artifact_outline(FIXTURES / "regional-sales-dashboard-58f1c05b")
    assert sales.startswith("files: presentation-program.js")
    assert "Regional Sales Dashboard" in sales and "Monthly revenue" in sales
    assert "cowork-figure-checks" not in sales


def test_the_outline_leaves_out_embedded_data_and_housekeeping(tmp_path):
    from anton.core.artifact_editor.router import artifact_outline

    (tmp_path / "index.html").write_text(
        '<html><head><title>T</title></head><body><h1>Hello</h1>'
        '<script type="application/json" id="cowork-observed-json">{"secret_value": "ROW-DATA-XYZ"}</script>'
        '</body></html>')
    (tmp_path / "metadata.json").write_text('{"name": "n"}')
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "x.js").write_text("function sneaky() {}")
    outline = artifact_outline(tmp_path)
    assert "Hello" in outline and "ROW-DATA-XYZ" not in outline
    assert "metadata.json" not in outline and "sneaky" not in outline


def test_the_outline_of_an_empty_or_missing_folder_is_empty(tmp_path):
    from anton.core.artifact_editor.router import artifact_outline

    assert artifact_outline(tmp_path) == ""
    assert artifact_outline(tmp_path / "missing") == ""


def test_describe_artifact_reads_the_name_and_the_data_shape():
    described = describe_artifact(FIXTURES / "regional-sales-dashboard-58f1c05b")
    assert described["name"] == "Regional Sales Dashboard"
    assert '"revenue_usd"' in described["data"] and len(described["data"]) <= 1200


def test_describe_artifact_of_an_empty_folder_is_empty(tmp_path):
    assert describe_artifact(tmp_path) == {"name": "", "description": "", "data": "", "outline": ""}


def test_describe_artifact_includes_the_outline():
    described = describe_artifact(FIXTURES / "night-sky-unicorns-5b2e91c4")
    assert "drawUnicorn" in described["outline"] and described["name"] == "Night Sky"


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


def test_minds_cloud_is_available_with_jev_by_default(settings_env):
    from anton.core.artifact_editor.router import decisions_available, decisions_config

    settings = settings_env(**mindshub())
    assert settings.decisions_model == "jev-1.13.0"
    assert decisions_available(settings)
    assert decisions_config(settings) == ("https://api.mindshub.ai/v1", "k-hub", "jev-1.13.0")


def test_openai_compatible_on_api_mindshub_is_available(settings_env):
    from anton.core.artifact_editor.router import decisions_config

    settings = settings_env(ANTON_PLANNING_PROVIDER="openai-compatible", ANTON_CODING_PROVIDER="openai-compatible",
                            ANTON_OPENAI_BASE_URL="https://api.mindshub.ai/v1", ANTON_OPENAI_API_KEY="k-oc")
    assert settings.decisions_model == "jev-1.13.0"
    assert decisions_config(settings) == ("https://api.mindshub.ai/v1", "k-oc", "jev-1.13.0")


LEFTOVER_HUB = {"ANTON_MINDS_URL": "https://api.mindshub.ai", "ANTON_MINDS_API_KEY": "k-leftover"}


@pytest.mark.parametrize("env", [
    # Their own Anthropic or OpenAI key, with a MindsHub key and URL left over in the env file.
    {"ANTON_PLANNING_PROVIDER": "anthropic", "ANTON_CODING_PROVIDER": "anthropic", "ANTON_ANTHROPIC_API_KEY": "k",
     **LEFTOVER_HUB},
    {"ANTON_PLANNING_PROVIDER": "openai", "ANTON_CODING_PROVIDER": "openai", "ANTON_OPENAI_API_KEY": "k",
     **LEFTOVER_HUB},
    # A leftover key with the default minds_url.
    {"ANTON_PLANNING_PROVIDER": "anthropic", "ANTON_ANTHROPIC_API_KEY": "k", "ANTON_MINDS_API_KEY": "k-leftover"},
    {"ANTON_PLANNING_PROVIDER": "openai-compatible", "ANTON_CODING_PROVIDER": "openai-compatible",
     "ANTON_OPENAI_BASE_URL": "https://llm.example.com/v1", "ANTON_OPENAI_API_KEY": "k", **LEFTOVER_HUB},
    # Only the exact api.mindshub.ai host counts.
    {"ANTON_PLANNING_PROVIDER": "openai-compatible", "ANTON_CODING_PROVIDER": "openai-compatible",
     "ANTON_OPENAI_BASE_URL": "https://api.mindshub.ai.example.com/v1", "ANTON_OPENAI_API_KEY": "k"},
    {"ANTON_PLANNING_PROVIDER": "openai-compatible", "ANTON_CODING_PROVIDER": "openai-compatible",
     "ANTON_OPENAI_BASE_URL": "https://staging.mindshub.ai/v1", "ANTON_OPENAI_API_KEY": "k"},
    # minds-cloud on the legacy mdb.ai host serves no decisions.
    {"ANTON_PLANNING_PROVIDER": "minds-cloud", "ANTON_CODING_PROVIDER": "minds-cloud",
     "ANTON_MINDS_URL": "https://mdb.ai", "ANTON_MINDS_API_KEY": "k"},
])
def test_without_a_mindshub_provider_there_is_no_decisions_model(settings_env, env):
    from anton.core.artifact_editor.router import decisions_available, decisions_config

    settings = settings_env(**env)
    assert settings.decisions_model is None
    assert not decisions_available(settings) and decisions_config(settings) is None


def test_an_explicit_model_never_borrows_a_leftover_mindshub_key(settings_env):
    from anton.core.artifact_editor.router import decisions_available

    settings = settings_env(ANTON_PLANNING_PROVIDER="anthropic", ANTON_ANTHROPIC_API_KEY="k",
                            ANTON_DECISIONS_MODEL="jev-1.13.0", **LEFTOVER_HUB)
    assert not decisions_available(settings)


@pytest.mark.parametrize("value", ["", "off", "none"])
def test_an_explicit_empty_value_turns_it_off_on_mindshub(settings_env, value):
    from anton.core.artifact_editor.router import decisions_available

    settings = settings_env(**mindshub(ANTON_DECISIONS_MODEL=value))
    assert not decisions_available(settings)


def test_an_explicit_model_is_used_on_mindshub(settings_env):
    from anton.core.artifact_editor.router import decisions_config

    settings = settings_env(**mindshub(ANTON_DECISIONS_MODEL="jev-2.0.0"))
    assert settings.decisions_model == "jev-2.0.0"
    assert decisions_config(settings) == ("https://api.mindshub.ai/v1", "k-hub", "jev-2.0.0")


def test_plain_core_settings_resolve_the_same_way():
    from types import SimpleNamespace

    from anton.core.artifact_editor.router import decisions_config

    hub = SimpleNamespace(decisions_model=None, planning_provider="minds-cloud",
                          minds_url="https://api.mindshub.ai", minds_api_key="k-hub")
    assert decisions_config(hub) == ("https://api.mindshub.ai/v1", "k-hub", "jev-1.13.0")
    other = SimpleNamespace(decisions_model="jev-1.13.0", planning_provider="anthropic",
                            minds_url="https://api.mindshub.ai", minds_api_key="k-leftover",
                            openai_base_url="https://api.mindshub.ai/v1", openai_api_key="k-leftover")
    assert decisions_config(other) is None
    assert decisions_config(SimpleNamespace(decisions_model=None)) is None


async def test_no_decisions_model_means_no_http_call(settings_env):
    from anton.core.artifact_editor.router import decisions_config

    settings = settings_env(ANTON_PLANNING_PROVIDER="anthropic", ANTON_ANTHROPIC_API_KEY="k", **LEFTOVER_HUB)
    assert decisions_config(settings) is None
    jev = FakeJev(answer("edit", 1.0))
    decision = await route(jev, model=getattr(settings, "decisions_model", None))
    assert (decision.route, decision.reason, decision.confidence) == ("agent", "no_decisions_model", 0.0)
    assert jev.requests == []
