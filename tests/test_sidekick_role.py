"""The Chimera sidekick role on `LLMClient`, its settings, and its turn-cost slot.

The sidekick defaults to the coding role the way the router does, but a client
counts as having one (`has_sidekick`) only when a model was configured.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from anton.config.settings import AntonSettings
from anton.core.chimera.config import ChimeraConfig, resolve_main_cache_ttl
from anton.core.llm.client import LLMClient, ModelRoute, asked_model, model_route
from anton.core.llm.provider import (
    LLMProvider,
    LLMResponse,
    ProviderAuthError,
    StreamComplete,
    Usage,
)
from anton.core.llm.tracing import get_call_role
from anton.core.turn_cost import EVENT_ROLES, UNKNOWN_ROLE, TurnCost

_SIDEKICK_ENV = (
    "ANTON_SIDEKICK_PROVIDER", "ANTON_SIDEKICK_MODEL", "ANTON_SIDEKICK_REASONING_EFFORT",
    "ANTON_CHIMERA_ENABLED", "ANTON_MAIN_CACHE_TTL", "ANTON_CHIMERA_MAX_ROUNDS",
    "ANTON_CHIMERA_JEV_EVERY", "ANTON_CHIMERA_JEV_THRESHOLD",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _SIDEKICK_ENV:
        monkeypatch.delenv(name, raising=False)


async def _stream_of(response):
    yield StreamComplete(response=response)


def _provider(label: str) -> AsyncMock:
    provider = AsyncMock(spec=LLMProvider)
    provider.stream = Mock(side_effect=lambda **kw: _stream_of(
        LLMResponse(content=label, usage=Usage(input_tokens=10, output_tokens=2))
    ))
    return provider


class _RoleRecordingProvider(LLMProvider):
    def __init__(self) -> None:
        self.roles: list[str | None] = []
        self.models: list[str] = []

    async def complete(self, **kwargs) -> LLMResponse:
        self.roles.append(get_call_role())
        return LLMResponse(content="", usage=Usage())

    async def stream(self, *, model, **kwargs):
        self.roles.append(get_call_role())
        self.models.append(model)
        yield StreamComplete(response=LLMResponse(content="a", usage=Usage()))


# -- settings -------------------------------------------------------------------


class TestSettings:
    def test_defaults_leave_chimera_off(self):
        s = AntonSettings(_env_file=None)
        assert (s.sidekick_provider, s.sidekick_model, s.sidekick_reasoning_effort) == (None, None, None)
        assert s.chimera_enabled is False
        assert s.main_cache_ttl is None
        assert (s.chimera_max_rounds, s.chimera_jev_every, s.chimera_jev_threshold) == (20, 4, 0.8)

    def test_env_vars_are_read(self, monkeypatch):
        monkeypatch.setenv("ANTON_SIDEKICK_PROVIDER", "minds-cloud")
        monkeypatch.setenv("ANTON_SIDEKICK_MODEL", "gpt-sol")
        monkeypatch.setenv("ANTON_SIDEKICK_REASONING_EFFORT", "low")
        monkeypatch.setenv("ANTON_CHIMERA_ENABLED", "true")
        monkeypatch.setenv("ANTON_MAIN_CACHE_TTL", " 1H ")
        monkeypatch.setenv("ANTON_CHIMERA_MAX_ROUNDS", "12")
        monkeypatch.setenv("ANTON_CHIMERA_JEV_EVERY", "3")
        monkeypatch.setenv("ANTON_CHIMERA_JEV_THRESHOLD", "0.7")
        s = AntonSettings(_env_file=None)
        # The same provider normalisation as the other roles.
        assert s.sidekick_provider == "openai-compatible"
        assert (s.sidekick_model, s.sidekick_reasoning_effort) == ("gpt-sol", "low")
        assert s.chimera_enabled is True
        assert s.main_cache_ttl == "1h"
        assert (s.chimera_max_rounds, s.chimera_jev_every, s.chimera_jev_threshold) == (12, 3, 0.7)

    def test_a_blank_cache_ttl_is_unset(self, monkeypatch):
        monkeypatch.setenv("ANTON_MAIN_CACHE_TTL", "")
        assert AntonSettings(_env_file=None).main_cache_ttl is None

    def test_an_unknown_cache_ttl_is_refused(self, monkeypatch):
        monkeypatch.setenv("ANTON_MAIN_CACHE_TTL", "2h")
        with pytest.raises(ValueError):
            AntonSettings(_env_file=None)

    def test_chimera_config_reads_the_settings_and_caps_the_rounds(self):
        s = AntonSettings(chimera_enabled=True, chimera_max_rounds=100, chimera_jev_every=2,
                          chimera_jev_threshold=0.9, _env_file=None)
        config = ChimeraConfig.from_settings(s)
        assert config == ChimeraConfig(enabled=True, max_rounds=40, jev_every=2, jev_threshold=0.9)
        assert config.rounds_for(None) == 40
        assert config.rounds_for(5) == 5
        assert config.rounds_for(400) == 40
        assert config.rounds_for(0) == 1
        assert config.rounds_for(True) == 40

    def test_chimera_config_of_a_settings_object_without_the_fields_is_off(self):
        assert ChimeraConfig.from_settings(object()) == ChimeraConfig(enabled=False)

    @pytest.mark.parametrize(
        "fields,chimera,expected",
        [
            ({}, None, None),
            ({"main_cache_ttl": "5m"}, None, "5m"),
            ({"chimera_enabled": True}, None, "1h"),
            ({"chimera_enabled": True, "main_cache_ttl": "5m"}, None, "5m"),
            # The host's config wins over the settings' flag.
            ({"chimera_enabled": True}, ChimeraConfig(enabled=False), None),
            ({}, ChimeraConfig(enabled=True), "1h"),
        ],
    )
    def test_main_cache_ttl_defaults_to_1h_only_under_chimera(self, fields, chimera, expected):
        s = AntonSettings(_env_file=None, **fields)
        assert resolve_main_cache_ttl(s, chimera) == expected


# -- the client -----------------------------------------------------------------


class TestSidekickRole:
    def test_defaults_to_the_coding_role_and_counts_as_none(self):
        planning, coding = _provider("p"), _provider("c")
        client = LLMClient(planning_provider=planning, planning_model="m-plan",
                           coding_provider=coding, coding_model="m-code")
        assert client.sidekick_provider is coding
        assert client.sidekick_model == "m-code"
        assert client.has_sidekick is False

    def test_a_configured_model_is_a_sidekick(self):
        coding = _provider("c")
        client = LLMClient(planning_provider=_provider("p"), planning_model="m-plan",
                           coding_provider=coding, coding_model="m-code", sidekick_model="m-side")
        assert client.has_sidekick is True
        assert client.sidekick_provider is coding
        assert client.sidekick_model == "m-side"

    async def test_sidekick_stream_runs_on_the_sidekick_provider_and_model(self):
        planning, coding, sidekick = _provider("p"), _provider("c"), _provider("s")
        seen = []
        client = LLMClient(planning_provider=planning, planning_model="m-plan",
                           coding_provider=coding, coding_model="m-code",
                           sidekick_provider=sidekick, sidekick_model="m-side")
        client.usage_listener = lambda role, model, usage: seen.append((role, model, usage.input_tokens))

        events = [e async for e in client.sidekick_stream(system="s", messages=[])]

        assert events[-1].response.content == "s"
        assert sidekick.stream.call_args.kwargs["model"] == "m-side"
        coding.stream.assert_not_called()
        assert seen == [("sidekick", "m-side", 10)]

    async def test_sidekick_calls_name_the_sidekick_role(self):
        sidekick = _RoleRecordingProvider()
        client = LLMClient(planning_provider=_RoleRecordingProvider(), planning_model="m",
                           coding_provider=_RoleRecordingProvider(), coding_model="m",
                           sidekick_provider=sidekick, sidekick_model="m-side")
        async for _ in client.sidekick_stream(system="s", messages=[]):
            pass
        assert sidekick.roles == ["sidekick"]
        assert sidekick.trace_role == "sidekick"

    async def test_on_the_coding_fallback_it_is_still_the_sidekick_role(self):
        coding = _RoleRecordingProvider()
        client = LLMClient(planning_provider=_RoleRecordingProvider(), planning_model="m",
                           coding_provider=coding, coding_model="m", sidekick_model="m-side")
        async for _ in client.sidekick_stream(system="s", messages=[]):
            pass
        assert coding.roles == ["sidekick"]
        assert coding.models == ["m-side"]

    def test_a_client_without_a_sidekick_names_its_providers_as_before(self):
        planning, coding, router = _RoleRecordingProvider(), _RoleRecordingProvider(), _RoleRecordingProvider()
        LLMClient(planning_provider=planning, planning_model="m", coding_provider=coding,
                  coding_model="m", router_provider=router, router_model="m", sidekick_model="m-side")
        assert (planning.trace_role, coding.trace_role, router.trace_role) == ("planning", "coding", "router")

    async def test_a_second_auth_refusal_names_the_sidekick_role(self):
        sidekick = AsyncMock(spec=LLMProvider)

        def refuse(**kwargs):
            async def gen():
                raise ProviderAuthError("no")
                yield  # pragma: no cover
            return gen()

        sidekick.stream = Mock(side_effect=refuse)
        client = LLMClient(planning_provider=_provider("p"), planning_model="m",
                           coding_provider=_provider("c"), coding_model="m",
                           sidekick_provider=sidekick, sidekick_model="m-side")
        with pytest.raises(ProviderAuthError) as err:
            async for _ in client.sidekick_stream(system="s", messages=[]):
                pass
        assert sidekick.stream.call_count == 2
        assert err.value.role == "sidekick"

    async def test_model_route_applies_to_the_sidekick_role(self):
        sidekick = _RoleRecordingProvider()
        client = LLMClient(planning_provider=_RoleRecordingProvider(), planning_model="m",
                           coding_provider=_RoleRecordingProvider(), coding_model="m",
                           sidekick_provider=sidekick, sidekick_model="m-side")
        with model_route(ModelRoute(model="m-up")):
            assert asked_model(client, "sidekick") == "m-up"
            async for _ in client.sidekick_stream(system="s", messages=[]):
                pass
        assert sidekick.models == ["m-up"]
        assert asked_model(client, "sidekick") == "m-side"

    def test_asked_model_of_a_double_without_a_sidekick_is_none(self):
        assert asked_model(object(), "sidekick") is None

    async def test_aclose_closes_the_sidekick_provider_once(self):
        planning, coding, sidekick = _provider("p"), _provider("c"), _provider("s")
        client = LLMClient(planning_provider=planning, planning_model="m", coding_provider=coding,
                           coding_model="m", sidekick_provider=sidekick, sidekick_model="m-side")
        await client.aclose()
        sidekick.aclose.assert_awaited_once()


class TestFromSettings:
    def _client(self, **fields) -> LLMClient:
        with patch("anthropic.AsyncAnthropic"):
            settings = AntonSettings(planning_provider="anthropic", coding_provider="anthropic",
                                     anthropic_api_key="k", _env_file=None, **fields)
            return LLMClient.from_settings(settings)

    def test_no_sidekick_settings_build_no_sidekick(self):
        client = self._client()
        assert client.has_sidekick is False
        assert client.sidekick_provider is client.coding_provider
        assert client.main_cache_ttl is None

    def test_sidekick_settings_build_a_provider_with_its_own_effort(self):
        client = self._client(sidekick_provider="anthropic", sidekick_model="m-side",
                              sidekick_reasoning_effort="low")
        assert client.has_sidekick is True
        assert client.sidekick_model == "m-side"
        assert client.sidekick_provider is not client.coding_provider
        assert client.sidekick_provider.reasoning_effort == "low"

    def test_an_effort_without_a_provider_gets_a_provider_of_the_coding_kind(self):
        client = self._client(sidekick_model="m-side", sidekick_reasoning_effort="low")
        assert type(client.sidekick_provider) is type(client.coding_provider)
        assert client.sidekick_provider is not client.coding_provider
        assert client.sidekick_provider.reasoning_effort == "low"

    def test_a_model_alone_shares_the_coding_provider(self):
        client = self._client(sidekick_model="m-side")
        assert client.sidekick_provider is client.coding_provider

    def test_chimera_gives_the_planning_role_an_hour(self):
        assert self._client(chimera_enabled=True).main_cache_ttl == "1h"
        assert self._client(main_cache_ttl="5m").main_cache_ttl == "5m"

    def test_an_unknown_sidekick_provider_is_refused(self):
        with pytest.raises(ValueError, match="Unknown sidekick provider"):
            self._client(sidekick_provider="nope", sidekick_model="m-side")


# -- turn cost --------------------------------------------------------------------


def test_turn_cost_books_the_chimera_roles_under_their_own_names():
    assert {"sidekick", "decision"} <= set(EVENT_ROLES)
    tc = TurnCost()
    tc.add("sidekick", "gpt-sol", Usage(input_tokens=100, output_tokens=5))
    tc.add("decision", "jev-1.13.0", Usage(input_tokens=1700, output_tokens=0))
    assert tc.by_role["sidekick"].tokens == 105
    assert tc.by_role["decision"].model == "jev-1.13.0"
    assert UNKNOWN_ROLE not in tc.by_role
    assert sum(s.tokens for s in tc.by_role.values()) == tc.total_tokens
