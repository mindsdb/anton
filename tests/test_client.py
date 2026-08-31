from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anton.config.settings import AntonSettings
from anton.core.llm.client import LLMClient
from anton.core.llm.provider import (
    LLMProvider,
    LLMResponse,
    ProviderAuthError,
    StreamTextDelta,
    Usage,
)


@pytest.fixture()
def mock_providers():
    planning = AsyncMock(spec=LLMProvider)
    coding = AsyncMock(spec=LLMProvider)
    planning.complete = AsyncMock(
        return_value=LLMResponse(content="plan", usage=Usage())
    )
    coding.complete = AsyncMock(
        return_value=LLMResponse(content="code", usage=Usage())
    )
    return planning, coding


class TestLLMClient:
    async def test_plan_delegates_to_planning_provider(self, mock_providers):
        planning, coding = mock_providers
        client = LLMClient(
            planning_provider=planning,
            planning_model="model-a",
            coding_provider=coding,
            coding_model="model-b",
        )
        result = await client.plan(
            system="sys", messages=[{"role": "user", "content": "task"}]
        )
        planning.complete.assert_awaited_once()
        call_kwargs = planning.complete.call_args.kwargs
        assert call_kwargs["model"] == "model-a"
        assert result.content == "plan"

    async def test_code_delegates_to_coding_provider(self, mock_providers):
        planning, coding = mock_providers
        client = LLMClient(
            planning_provider=planning,
            planning_model="model-a",
            coding_provider=coding,
            coding_model="model-b",
        )
        result = await client.code(
            system="sys", messages=[{"role": "user", "content": "code this"}]
        )
        coding.complete.assert_awaited_once()
        call_kwargs = coding.complete.call_args.kwargs
        assert call_kwargs["model"] == "model-b"
        assert result.content == "code"

    async def test_plan_passes_tools(self, mock_providers):
        planning, coding = mock_providers
        client = LLMClient(
            planning_provider=planning,
            planning_model="m",
            coding_provider=coding,
            coding_model="m",
        )
        tools = [{"name": "test_tool"}]
        await client.plan(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
            tools=tools,
        )
        call_kwargs = planning.complete.call_args.kwargs
        assert call_kwargs["tools"] == tools

    async def test_plan_confirms_one_auth_refusal_then_returns_success(
        self, mock_providers
    ):
        planning, coding = mock_providers
        planning.complete = AsyncMock(
            side_effect=[
                ProviderAuthError("Invalid API key"),
                LLMResponse(content="recovered", usage=Usage()),
            ]
        )
        client = LLMClient(
            planning_provider=planning,
            planning_model="model-a",
            coding_provider=coding,
            coding_model="model-b",
        )

        response = await client.plan(system="sys", messages=[])

        assert response.content == "recovered"
        assert planning.complete.await_count == 2

    async def test_plan_propagates_second_auth_refusal(self, mock_providers):
        planning, coding = mock_providers
        planning.complete = AsyncMock(
            side_effect=ProviderAuthError("Invalid API key")
        )
        client = LLMClient(
            planning_provider=planning,
            planning_model="model-a",
            coding_provider=coding,
            coding_model="model-b",
        )

        with pytest.raises(ProviderAuthError) as err:
            await client.plan(system="sys", messages=[])

        assert planning.complete.await_count == 2
        assert err.value.role == "planning"

    async def test_plan_does_not_retry_unrelated_connection_error(
        self, mock_providers
    ):
        planning, coding = mock_providers
        planning.complete = AsyncMock(side_effect=ConnectionError("network down"))
        client = LLMClient(
            planning_provider=planning,
            planning_model="model-a",
            coding_provider=coding,
            coding_model="model-b",
        )

        with pytest.raises(ConnectionError, match="network down"):
            await client.plan(system="sys", messages=[])

        planning.complete.assert_awaited_once()

    async def test_code_marks_a_confirmed_auth_failure_with_its_role(
        self, mock_providers
    ):
        planning, coding = mock_providers
        coding.complete = AsyncMock(side_effect=ProviderAuthError("Invalid API key"))
        client = LLMClient(
            planning_provider=planning,
            planning_model="model-a",
            coding_provider=coding,
            coding_model="model-b",
        )

        with pytest.raises(ProviderAuthError) as err:
            await client.code(system="sys", messages=[])

        assert coding.complete.await_count == 2
        assert err.value.role == "coding"

    async def test_plan_stream_confirms_auth_before_first_event(
        self, mock_providers
    ):
        planning, coding = mock_providers
        calls = 0

        async def stream(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ProviderAuthError("Invalid API key")
            yield StreamTextDelta(text="recovered")

        planning.stream = MagicMock(side_effect=stream)
        client = LLMClient(
            planning_provider=planning,
            planning_model="model-a",
            coding_provider=coding,
            coding_model="model-b",
        )

        events = [
            event
            async for event in client.plan_stream(system="sys", messages=[])
        ]

        assert events == [StreamTextDelta(text="recovered")]
        assert calls == 2

    async def test_plan_stream_propagates_second_auth_refusal(
        self, mock_providers
    ):
        planning, coding = mock_providers
        calls = 0

        async def stream(**kwargs):
            nonlocal calls
            calls += 1
            raise ProviderAuthError("Invalid API key")
            yield  # pragma: no cover - preserve the async-iterator protocol

        planning.stream = MagicMock(side_effect=stream)
        client = LLMClient(
            planning_provider=planning,
            planning_model="model-a",
            coding_provider=coding,
            coding_model="model-b",
        )

        with pytest.raises(ProviderAuthError) as err:
            async for _ in client.plan_stream(system="sys", messages=[]):
                pass

        assert calls == 2
        assert err.value.role == "planning"

    async def test_plan_stream_never_replays_after_an_event_was_yielded(
        self, mock_providers
    ):
        planning, coding = mock_providers
        calls = 0

        async def stream(**kwargs):
            nonlocal calls
            calls += 1
            yield StreamTextDelta(text="partial")
            raise ProviderAuthError("Invalid API key")

        planning.stream = MagicMock(side_effect=stream)
        client = LLMClient(
            planning_provider=planning,
            planning_model="model-a",
            coding_provider=coding,
            coding_model="model-b",
        )
        events = []

        with pytest.raises(ProviderAuthError) as err:
            async for event in client.plan_stream(system="sys", messages=[]):
                events.append(event)

        assert events == [StreamTextDelta(text="partial")]
        assert calls == 1
        assert err.value.role == "planning"


class TestRouterRole:
    """Summarization runs on the router role, which defaults to the coding
    role when no distinct router model is configured (behavior-preserving)."""

    async def test_summarize_defaults_to_coding_role(self, mock_providers):
        planning, coding = mock_providers
        client = LLMClient(
            planning_provider=planning,
            planning_model="model-a",
            coding_provider=coding,
            coding_model="model-b",
        )
        result = await client.summarize(
            system="sys", messages=[{"role": "user", "content": "old turns"}]
        )
        coding.complete.assert_awaited_once()
        assert coding.complete.call_args.kwargs["model"] == "model-b"
        assert result.content == "code"

    async def test_summarize_fallback_attributes_auth_to_coding(
        self, mock_providers
    ):
        planning, coding = mock_providers
        coding.complete = AsyncMock(side_effect=ProviderAuthError("Invalid API key"))
        client = LLMClient(
            planning_provider=planning,
            planning_model="model-a",
            coding_provider=coding,
            coding_model="model-b",
        )

        with pytest.raises(ProviderAuthError) as err:
            await client.summarize(system="sys", messages=[])

        assert coding.complete.await_count == 2
        assert err.value.role == "coding"

    async def test_summarize_uses_distinct_router_model(self, mock_providers):
        planning, coding = mock_providers
        router = AsyncMock(spec=LLMProvider)
        router.complete = AsyncMock(
            return_value=LLMResponse(content="summary", usage=Usage())
        )
        client = LLMClient(
            planning_provider=planning,
            planning_model="model-a",
            coding_provider=coding,
            coding_model="model-b",
            router_provider=router,
            router_model="model-c",
        )
        result = await client.summarize(
            system="sys", messages=[{"role": "user", "content": "old turns"}]
        )
        router.complete.assert_awaited_once()
        assert router.complete.call_args.kwargs["model"] == "model-c"
        coding.complete.assert_not_awaited()
        assert result.content == "summary"
        assert client.router_provider is router
        assert client.router_model == "model-c"

    async def test_gate_confirms_one_router_auth_refusal(self, mock_providers):
        planning, coding = mock_providers
        router = AsyncMock(spec=LLMProvider)
        router.complete = AsyncMock(
            side_effect=[
                ProviderAuthError("Invalid API key"),
                LLMResponse(content="delegate", usage=Usage()),
            ]
        )
        client = LLMClient(
            planning_provider=planning,
            planning_model="model-a",
            coding_provider=coding,
            coding_model="model-b",
            router_provider=router,
            router_model="model-c",
        )

        response = await client.gate(system="sys", messages=[])

        assert response.content == "delegate"
        assert router.complete.await_count == 2

    async def test_gate_propagates_second_router_auth_refusal(self, mock_providers):
        planning, coding = mock_providers
        router = AsyncMock(spec=LLMProvider)
        router.complete = AsyncMock(
            side_effect=ProviderAuthError("Invalid API key")
        )
        client = LLMClient(
            planning_provider=planning,
            planning_model="model-a",
            coding_provider=coding,
            coding_model="model-b",
            router_provider=router,
            router_model="model-c",
        )

        with pytest.raises(ProviderAuthError) as err:
            await client.gate(system="sys", messages=[])

        assert router.complete.await_count == 2
        assert err.value.role == "router"

    def test_router_accessors_fall_back_to_coding(self, mock_providers):
        planning, coding = mock_providers
        client = LLMClient(
            planning_provider=planning,
            planning_model="model-a",
            coding_provider=coding,
            coding_model="model-b",
        )
        assert client.router_provider is coding
        assert client.router_model == "model-b"


class TestLLMClientFromSettings:
    def test_from_settings_creates_client(self):
        from anton.core.llm.anthropic import AnthropicProvider

        with patch("anthropic.AsyncAnthropic"):
            settings = AntonSettings(
                planning_provider="anthropic",
                coding_provider="anthropic",
                anthropic_api_key="test-key",
                _env_file=None,
            )
            client = LLMClient.from_settings(settings)
            assert isinstance(client, LLMClient)
            assert isinstance(client._planning_provider, AnthropicProvider)
            assert isinstance(client._coding_provider, AnthropicProvider)

    def _oc_vision_format(self, base_url: str) -> str:
        settings = AntonSettings(
            planning_provider="openai-compatible",
            coding_provider="openai-compatible",
            openai_api_key="test-key",
            openai_base_url=base_url,
            planning_model="m",
            coding_model="m",
            _env_file=None,
        )
        return LLMClient.from_settings(settings)._planning_provider._vision_format

    def test_openai_compatible_vision_format_is_always_openai(self):
        # ENG-1992: this used to guess "anthropic" for any mdb.ai/mindshub.ai
        # host, which assumed every model behind that gateway was Claude — it
        # broke non-Claude models sharing the same host (MindsHub Air routed
        # to an OpenAI Responses-backed model got Anthropic-shaped image
        # blocks and 400'd). The gateway itself already normalizes a standard
        # OpenAI image_url block into whatever the resolved backend needs, so
        # anton just sends OpenAI shape unconditionally now.
        assert self._oc_vision_format("https://api.mindshub.ai/v1") == "openai"
        assert self._oc_vision_format("https://mdb.ai/api/v1") == "openai"
        assert self._oc_vision_format(
            "https://generativelanguage.googleapis.com/v1beta/openai/"
        ) == "openai"
        assert self._oc_vision_format("https://my-proxy.example.com/v1") == "openai"

    def test_unknown_planning_provider_raises(self):
        settings = AntonSettings(
            planning_provider="unknown",
            anthropic_api_key="test",
            _env_file=None,
        )
        with pytest.raises(ValueError, match="Unknown planning provider"):
            LLMClient.from_settings(settings)

    def test_unknown_coding_provider_raises(self):
        settings = AntonSettings(
            coding_provider="unknown",
            anthropic_api_key="test",
            _env_file=None,
        )
        with pytest.raises(ValueError, match="Unknown coding provider"):
            LLMClient.from_settings(settings)

    def test_router_model_wired_to_router_role(self):
        with patch("anthropic.AsyncAnthropic"):
            settings = AntonSettings(
                planning_provider="anthropic",
                coding_provider="anthropic",
                coding_model="coding-m",
                router_provider="anthropic",
                router_model="router-m",
                anthropic_api_key="test-key",
                _env_file=None,
            )
            client = LLMClient.from_settings(settings)
            assert client.router_model == "router-m"

    def test_router_unset_falls_back_to_coding_role(self):
        with patch("anthropic.AsyncAnthropic"):
            settings = AntonSettings(
                planning_provider="anthropic",
                coding_provider="anthropic",
                coding_model="coding-m",
                anthropic_api_key="test-key",
                _env_file=None,
            )
            client = LLMClient.from_settings(settings)
            assert client.router_provider is client._coding_provider
            assert client.router_model == "coding-m"

    def test_unknown_router_provider_raises(self):
        settings = AntonSettings(
            router_provider="unknown",
            anthropic_api_key="test",
            _env_file=None,
        )
        with pytest.raises(ValueError, match="Unknown router provider"):
            LLMClient.from_settings(settings)
