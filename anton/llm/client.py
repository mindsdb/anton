from __future__ import annotations

from typing import TYPE_CHECKING

from anton.llm.provider import LLMProvider, LLMResponse

if TYPE_CHECKING:
    from anton.config.settings import AntonSettings


class LLMClient:
    def __init__(
        self,
        *,
        planning_provider: LLMProvider,
        planning_model: str,
        coding_provider: LLMProvider,
        coding_model: str,
    ) -> None:
        self._planning_provider = planning_provider
        self._planning_model = planning_model
        self._coding_provider = coding_provider
        self._coding_model = coding_model

    async def plan(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        return await self._planning_provider.complete(
            model=self._planning_model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
        )

    async def code(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        return await self._coding_provider.complete(
            model=self._coding_model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
        )

    @classmethod
    def from_settings(cls, settings: AntonSettings) -> LLMClient:
        from anton.llm.anthropic import AnthropicProvider

        providers = {
            "anthropic": lambda: AnthropicProvider(api_key=settings.anthropic_api_key),
        }

        planning_factory = providers.get(settings.planning_provider)
        coding_factory = providers.get(settings.coding_provider)

        if planning_factory is None:
            raise ValueError(f"Unknown planning provider: {settings.planning_provider}")
        if coding_factory is None:
            raise ValueError(f"Unknown coding provider: {settings.coding_provider}")

        return cls(
            planning_provider=planning_factory(),
            planning_model=settings.planning_model,
            coding_provider=coding_factory(),
            coding_model=settings.coding_model,
        )
