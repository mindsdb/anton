from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from .provider import LLMProvider, LLMResponse, StreamEvent

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

    async def _generate_object_with(
        self,
        schema_class,
        *,
        provider: LLMProvider,
        model: str,
        system: str,
        messages: list[dict],
        max_tokens: int | None,
    ):
        """Internal: forced-tool-call structured output via any provider.

        Shared by `generate_object` (planning) and `generate_object_code`
        (coding). The schema-building/unwrapping logic is in
        `anton.core.llm.structured` so the scratchpad bridge can use the
        same primitives without depending on this class.
        """
        from anton.core.llm.structured import (
            build_structured_tool,
            unwrap_structured_response,
        )

        tool, validator_class, is_list = build_structured_tool(schema_class)

        response = await provider.complete(
            model=model,
            system=system,
            messages=messages,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            max_tokens=max_tokens or self._max_tokens,
        )

        if not response.tool_calls:
            raise ValueError(
                f"LLM did not return a tool call for forced schema {tool['name']}."
            )

        return unwrap_structured_response(
            response.tool_calls[0].input, validator_class, is_list
        )

    async def generate_object(
        self,
        schema_class,
        *,
        system: str,
        messages: list[dict],
        max_tokens: int | None = None,
    ):
        """Generate a structured object using the *planning* provider.

        Forces the planning LLM to call a synthetic tool whose
        input_schema is derived from the Pydantic model. The tool's
        input is then validated through `model_validate`, returning a
        typed instance (or a list of instances for `list[Model]`).

        This is the right primitive for any code that wants structured
        output from the LLM. It is more reliable than asking for JSON
        in the response text because:

          - The LLM is *forced* (via `tool_choice`) to call the tool
          - The tool's input is constrained by the JSON schema
          - Pydantic catches any structural drift via `model_validate`

        Use this method for any structured-output operation that
        currently uses `plan()`. For operations that should use the
        cheaper coding model (memory compaction, identity extraction,
        anything that ran via `code()` previously), use
        `generate_object_code()` instead.

        Args:
            schema_class: A Pydantic `BaseModel` subclass, or a
                `list[Model]` annotation for a homogeneous list.
            system: System prompt for the call.
            messages: Conversation messages.
            max_tokens: Token budget. Defaults to `self._max_tokens`.

        Returns:
            An instance of `schema_class`, or a `list[Model]` when the
            input was a list annotation.

        Raises:
            ValueError: If the LLM fails to produce a tool call (rare —
                forced tool_choice usually prevents this).
            pydantic.ValidationError: If the tool's input doesn't match
                the schema.

        The schema-building / unwrapping logic is shared with
        `_ScratchpadLLM.generate_object` (in `scratchpad_boot.py`) via
        `anton.core.llm.structured` — only the actual provider call
        differs between the two runtime contexts (async planning here,
        sync subprocess there).
        """
        return await self._generate_object_with(
            schema_class,
            provider=self._planning_provider,
            model=self._planning_model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
        )

    async def generate_object_code(
        self,
        schema_class,
        *,
        system: str,
        messages: list[dict],
        max_tokens: int | None = None,
    ):
        """Generate a structured object using the *coding* provider.

        Same forced-tool-call mechanism as `generate_object`, but routes
        through the coding provider/model. Use this when the operation
        is a fast, cheap structured task that previously called
        `code()` — e.g. memory compaction, identity extraction,
        scratchpad post-mortem analysis. The savings vs. the planning
        model add up across many small calls.
        """
        return await self._generate_object_with(
            schema_class,
            provider=self._coding_provider,
            model=self._coding_model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
        )

    @classmethod
    def from_settings(cls, settings: AntonSettings) -> LLMClient:
        from .anthropic import AnthropicProvider
        from .openai import OpenAIProvider

        providers = {
            "anthropic": lambda: AnthropicProvider(api_key=settings.anthropic_api_key),
            "openai": lambda: OpenAIProvider(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                ssl_verify=settings.minds_ssl_verify,
            ),
            "openai-compatible": lambda: OpenAIProvider(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                ssl_verify=settings.minds_ssl_verify,
            ),
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
            max_tokens=getattr(settings, "max_tokens", 8192),
        )
