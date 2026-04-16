from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from anton.config.settings import AntonSettings
from anton.core.llm.client import LLMClient
from anton.core.llm.provider import LLMProvider, LLMResponse, Usage


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
