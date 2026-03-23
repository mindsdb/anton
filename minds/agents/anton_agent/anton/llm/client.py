from __future__ import annotations

from collections.abc import AsyncIterator

from .provider import LLMProvider, LLMResponse, StreamEvent


class LLMClient:
    def __init__(
        self,
        *,
        planning_provider: LLMProvider,
        planning_model: str,
        coding_provider: LLMProvider,
        coding_model: str,
        max_tokens: int = 8192,
    ) -> None:
        self._planning_provider = planning_provider
        self._planning_model = planning_model
        self._coding_provider = coding_provider
        self._coding_model = coding_model
        self._max_tokens = max_tokens

    async def plan(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return await self._planning_provider.complete(
            model=self._planning_model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens or self._max_tokens,
        )

    async def plan_stream(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        async for event in self._planning_provider.stream(
            model=self._planning_model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens or self._max_tokens,
        ):
            yield event

    @property
    def coding_provider(self) -> LLMProvider:
        """The LLM provider used for coding/skill execution."""
        return self._coding_provider

    @property
    def coding_model(self) -> str:
        """The model name used for coding/skill execution."""
        return self._coding_model

    async def code(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return await self._coding_provider.complete(
            model=self._coding_model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens or self._max_tokens,
        )
