from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from .provider import LLMProvider, LLMResponse, StreamEvent, StructuredOutputError

if TYPE_CHECKING:
    from anton.config.settings import AntonSettings


def _resolve_openai_compatible_flavor(settings: AntonSettings) -> str:
    """Distinguish mdb.ai passthrough from a generic openai-compatible endpoint.

    The "Minds-Enterprise-Cloud" setup path writes ``openai_base_url =
    f"{minds_url.rstrip('/')}/api/v1"`` and ``openai_api_key = minds_api_key``
    (see ``AntonSettings.model_post_init``). When that exact pairing matches
    the user's current settings, the OpenAI provider is talking to mdb.ai and
    can therefore use the chat.completions native web tool passthrough. Any
    other base URL is a generic third-party endpoint that needs the
    handler-dispatched fallback at the session layer.

    No new env var is introduced — we infer flavor purely from the existing
    config the setup flow already produces.
    """
    from .openai import OpenAIProvider

    base = (getattr(settings, "openai_base_url", None) or "").rstrip("/").lower()
    minds = (getattr(settings, "minds_url", None) or "").rstrip("/").lower()
    if minds and (base == minds or base == f"{minds}/api/v1"):
        return OpenAIProvider.FLAVOR_MINDS_PASSTHROUGH
    return OpenAIProvider.FLAVOR_OPENAI_COMPATIBLE_GENERIC


class LLMClient:
    def __init__(
        self,
        *,
        planning_provider: LLMProvider,
        planning_model: str,
        coding_provider: LLMProvider,
        coding_model: str,
        router_provider: LLMProvider | None = None,
        router_model: str | None = None,
        max_tokens: int = 8192,
    ) -> None:
        self._planning_provider = planning_provider
        self._planning_model = planning_model
        self._coding_provider = coding_provider
        self._coding_model = coding_model
        # Router role: the cheap model that owns history
        # summarization (and, later, per-turn respond-vs-delegate gating).
        # Defaults to the coding role so hosts that construct LLMClient
        # directly (cowork-server) get behavior-preserving summarization
        # with no changes.
        self._router_provider = router_provider or coding_provider
        self._router_model = router_model or coding_model
        self._max_tokens = max_tokens

    async def plan(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        native_web_tools: set[str] | None = None,
    ) -> LLMResponse:
        return await self._planning_provider.complete(
            model=self._planning_model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens or self._max_tokens,
            native_web_tools=native_web_tools,
        )

    async def plan_stream(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        native_web_tools: set[str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        async for event in self._planning_provider.stream(
            model=self._planning_model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens or self._max_tokens,
            native_web_tools=native_web_tools,
        ):
            yield event

    @property
    def planning_provider(self) -> LLMProvider:
        """The LLM provider used for planning / the user-facing turn loop."""
        return self._planning_provider

    @property
    def coding_provider(self) -> LLMProvider:
        """The LLM provider used for coding/skill execution."""
        return self._coding_provider

    @property
    def coding_model(self) -> str:
        """The model name used for coding/skill execution."""
        return self._coding_model

    @property
    def router_provider(self) -> LLMProvider:
        """The LLM provider used for the cheap router (summarization) role."""
        return self._router_provider

    @property
    def router_model(self) -> str:
        """The model name used for the cheap router (summarization) role."""
        return self._router_model

    async def code(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        native_web_tools: set[str] | None = None,
    ) -> LLMResponse:
        return await self._coding_provider.complete(
            model=self._coding_model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens or self._max_tokens,
            native_web_tools=native_web_tools,
        )

    async def summarize(
        self,
        *,
        system: str,
        messages: list[dict],
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """History-compaction call — runs on the cheap router role.

        Falls back to the coding role when no router model is configured
        (the router_* kwargs default to the coding role in __init__), so
        this is behavior-preserving unless a distinct model is selected.
        """
        return await self._router_provider.complete(
            model=self._router_model,
            system=system,
            messages=messages,
            max_tokens=max_tokens or self._max_tokens,
        )

    async def gate(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """One cheap gating call on the router role — see `anton.core.llm.thalamus`.

        No ``native_web_tools``: the thalamus must never do work itself,
        only answer from context or delegate.
        """
        return await self._router_provider.complete(
            model=self._router_model,
            system=system,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
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

        budget = max_tokens or self._max_tokens

        response = await provider.complete(
            model=model,
            system=system,
            messages=messages,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            max_tokens=budget,
        )

        if not response.tool_calls:
            # Report *why* there is no tool call, so callers can retry a
            # truncated call with more room instead of treating a budget
            # problem as a hard failure (ENG-1081). Token count is the
            # reliable signal here — the MindsHub gateway reports
            # `finish_reason: "stop"` at the cap for most aliases (ENG-1082),
            # so `stop_reason` alone would miss it.
            # Both provider dialects: OpenAI/gateway say "length", Anthropic says
            # "max_tokens" (passed through raw by AnthropicProvider).
            output_tokens = response.usage.output_tokens
            truncated = response.stop_reason in ("length", "max_tokens") or (
                budget > 0 and output_tokens >= budget
            )
            raise StructuredOutputError(
                f"LLM did not return a tool call for forced schema {tool['name']}"
                + (
                    f" (truncated: {output_tokens}/{budget} output tokens spent "
                    "on text before the call)."
                    if truncated
                    else "."
                ),
                truncated=truncated,
                output_tokens=output_tokens,
                max_tokens=budget,
                stop_reason=response.stop_reason,
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
            StructuredOutputError: If the LLM fails to produce a tool call.
                A ``ValueError`` subclass, so callers that catch
                ``ValueError`` are unaffected. Check ``.truncated`` to tell a
                blown ``max_tokens`` budget (retry with more room) from a
                genuine failure (ENG-1081).
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

        api_version = getattr(settings, "openai_api_version", None)
        compatible_flavor = _resolve_openai_compatible_flavor(settings)
        # Only Minds / MindsHub / MDB.AI proxy Anthropic over the OpenAI HTTP
        # envelope and need Anthropic-shaped image blocks. Gate on the base-URL
        # host, NOT on the "openai-compatible" provider name — other compatible
        # endpoints (Gemini at generativelanguage.googleapis.com, generic
        # proxies) expect standard OpenAI image_url. Mirrors the scratchpad_boot
        # gate; forcing anthropic format unconditionally mangled Gemini images.
        _oc_base_host = (settings.openai_base_url or "").lower()
        _oc_vision_format = (
            "anthropic"
            if any(h in _oc_base_host for h in ("mdb.ai", "mindshub.ai"))
            else "openai"
        )
        # Each factory takes the per-role effort so planning and coding stay
        # independent even when they resolve to the same provider type.
        providers = {
            "anthropic": lambda effort: AnthropicProvider(
                api_key=settings.anthropic_api_key,
                reasoning_effort=effort,
            ),
            "openai": lambda effort: OpenAIProvider(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                ssl_verify=settings.minds_ssl_verify,
                api_version=api_version,
                flavor=OpenAIProvider.FLAVOR_OPENAI,
                reasoning_effort=effort,
            ),
            "openai-compatible": lambda effort: OpenAIProvider(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                ssl_verify=settings.minds_ssl_verify,
                api_version=api_version,
                supports_vision=True,
                vision_format=_oc_vision_format,
                flavor=compatible_flavor,
                reasoning_effort=effort,
            ),
        }

        planning_factory = providers.get(settings.planning_provider)
        coding_factory = providers.get(settings.coding_provider)

        if planning_factory is None:
            raise ValueError(f"Unknown planning provider: {settings.planning_provider}")
        if coding_factory is None:
            raise ValueError(f"Unknown coding provider: {settings.coding_provider}")

        # Router role: the cheap model for summarization (and later gating).
        # Optional override; unset falls back to the coding provider/model
        # inside __init__. No reasoning effort: summarization is a single
        # cheap call.
        router_provider = None
        router_provider_name = getattr(settings, "router_provider", None)
        if router_provider_name:
            router_factory = providers.get(router_provider_name)
            if router_factory is None:
                raise ValueError(f"Unknown router provider: {router_provider_name}")
            router_provider = router_factory(None)

        return cls(
            planning_provider=planning_factory(
                getattr(settings, "planning_reasoning_effort", None)
            ),
            planning_model=settings.planning_model,
            coding_provider=coding_factory(
                getattr(settings, "coding_reasoning_effort", None)
            ),
            coding_model=settings.coding_model,
            router_provider=router_provider,
            router_model=getattr(settings, "router_model", None),
            max_tokens=getattr(settings, "max_tokens", 8192),
        )
