from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from .provider import LLMProvider, LLMResponse, StreamComplete, StreamEvent

if TYPE_CHECKING:
    from anton.config.settings import AntonSettings


def _resolve_openai_compatible_flavor(settings: AntonSettings) -> str:
    """Distinguish MindsHub/mdb.ai passthrough from a generic openai-compatible
    endpoint.

    ``AntonSettings.model_post_init`` derives ``openai_base_url`` from
    ``minds_url`` host-awarely: ``api.mindshub.ai`` serves the
    OpenAI-compatible API at ``/v1``, the legacy ``mdb.ai`` host at
    ``/api/v1`` (ENG-436). Both derivations, plus a ``minds_url`` that already
    carries its own suffix, mean the provider is talking to our gateway and
    can use the chat.completions native web-tool passthrough — it accepts
    ``{"type": "web_search"}`` / ``{"type": "fetch"}`` directly (matching the
    gateway's own ``GenericToolType``). Any other base URL is a generic
    third-party endpoint that needs the handler-dispatched fallback at the
    session layer.

    The ``/v1`` case was missing, so **every MindsHub install fell through to
    generic** and silently lost native web search — the session's fallback
    needs an Exa/Brave key that MindsHub users don't have (#317 review).
    Pinned by ``tests/test_openai_setup.py``.

    No new env var is introduced — we infer flavor purely from the existing
    config the setup flow already produces.
    """
    from .openai import OpenAIProvider

    base = (getattr(settings, "openai_base_url", None) or "").rstrip("/").lower()
    minds = (getattr(settings, "minds_url", None) or "").rstrip("/").lower()
    if minds and base in (minds, f"{minds}/v1", f"{minds}/api/v1"):
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
        # ENG-1638: the model the planning provider last reported SERVING (the
        # `model` on its response), not the id we asked for. The session reads
        # it when building the RUNTIME IDENTITY block so the agent's answer to
        # "which model are you?" is what actually answered, not the alias in
        # settings. None until the first planning response arrives.
        self.last_served_model: str | None = None
        # ENG-1288: optional per-call usage observer. Every LLM call this
        # client makes — plan/plan_stream (planning), code + structured
        # coding calls like the completion verifier (coding), summarize/gate
        # (router) — reports (role, model, usage) here. The session installs
        # its per-turn cost accumulator; None means nobody is counting.
        # Accounting must never break a call: notification is wrapped and
        # swallowed (see _notify_usage).
        self.usage_listener = None  # Callable[[str, str, Usage], None] | None

    def _notify_usage(self, role: str, model: str, usage, listener=None) -> None:
        """Report one call's usage to the turn-cost accumulator.

        ``listener`` must be the reference captured when the call was ISSUED,
        not read here at completion time. End-of-turn fire-and-forget work
        (cerebellum flush, identity update, scratchpad consolidation) shares
        this client, so on a long-lived-session host a fast follow-up message
        arms turn N+1's listener while turn N's flush is still in flight —
        resolving late booked that usage into the wrong turn (#309 review).
        Captured-at-issue means background calls started while disarmed book
        nowhere, which is what ``turn_cost``'s docstring already promises.
        """
        if listener is None:
            return
        try:
            listener(role, model, usage)
        except Exception:
            # A broken accumulator must never kill the turn it's counting.
            logging.getLogger(__name__).warning(
                "usage_listener raised — turn cost undercounted", exc_info=True
            )

    async def plan(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        native_web_tools: set[str] | None = None,
    ) -> LLMResponse:
        listener = self.usage_listener
        response = await self._planning_provider.complete(
            model=self._planning_model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens or self._max_tokens,
            native_web_tools=native_web_tools,
        )
        self._notify_usage("planning", self._planning_model, response.usage, listener)
        self._record_served(response)
        return response

    async def plan_stream(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        native_web_tools: set[str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        listener = self.usage_listener
        async for event in self._planning_provider.stream(
            model=self._planning_model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens or self._max_tokens,
            native_web_tools=native_web_tools,
        ):
            if isinstance(event, StreamComplete):
                self._notify_usage(
                    "planning", self._planning_model, event.response.usage, listener
                )
                self._record_served(event.response)
            yield event

    def _record_served(self, response) -> None:
        """Keep the last served model the planning provider reported.

        Only the planning role: it is the one talking to the user, so it is the
        one "which model are you?" is about. A response without the field
        leaves the previous value in place rather than erasing a known answer.
        """
        served = getattr(response, "model", None)
        if isinstance(served, str) and served.strip():
            self.last_served_model = served.strip()

    @property
    def planning_provider(self) -> LLMProvider:
        """The LLM provider used for planning / the user-facing turn loop."""
        return self._planning_provider

    @property
    def max_tokens(self) -> int:
        """Default output-token budget for calls that don't pass their own.

        Exposed so the session's truncation recovery can compare a
        response's ``output_tokens`` against the budget the call actually
        ran with (ENG-1042). This gate was added because the gateway once
        reported a normal stop at the cap (ENG-1082, fixed 2026-08-03); it is
        kept because a token count needs no dialect mapping and no provider
        can get it wrong.
        """
        return self._max_tokens

    @property
    def planning_model(self) -> str:
        """The model name used for planning / the user-facing turn loop."""
        return self._planning_model

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
        listener = self.usage_listener
        response = await self._coding_provider.complete(
            model=self._coding_model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens or self._max_tokens,
            native_web_tools=native_web_tools,
        )
        self._notify_usage("coding", self._coding_model, response.usage, listener)
        return response

    async def code_stream(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        native_web_tools: set[str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        listener = self.usage_listener
        async for event in self._coding_provider.stream(
            model=self._coding_model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens or self._max_tokens,
            native_web_tools=native_web_tools,
        ):
            if isinstance(event, StreamComplete):
                self._notify_usage(
                    "coding", self._coding_model, event.response.usage, listener
                )
            yield event

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
        listener = self.usage_listener
        response = await self._router_provider.complete(
            model=self._router_model,
            system=system,
            messages=messages,
            max_tokens=max_tokens or self._max_tokens,
        )
        self._notify_usage("router", self._router_model, response.usage, listener)
        return response

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
        listener = self.usage_listener
        response = await self._router_provider.complete(
            model=self._router_model,
            system=system,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens or self._max_tokens,
        )
        self._notify_usage("router", self._router_model, response.usage, listener)
        return response

    async def _generate_object_with(
        self,
        schema_class,
        *,
        provider: LLMProvider,
        model: str,
        role: str,
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
            looks_truncated,
            raise_unusable_tool_call,
            unwrap_structured_response,
        )

        tool, validator_class, is_list = build_structured_tool(schema_class)

        budget = max_tokens or self._max_tokens

        listener = self.usage_listener
        response = await provider.complete(
            model=model,
            system=system,
            messages=messages,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            max_tokens=budget,
        )
        # Count BEFORE the no-tool-call raise below: a structured call that
        # failed (and its bigger-budget retry) still spent real tokens
        # (ENG-1288). The role is PASSED, not inferred from model equality:
        # deployments exist where planning and coding resolve to the same
        # model (cowork-server's Gemini defaults are identical across all
        # three roles), and inference collapsed every generate_object_code
        # call — verifier verdicts, compaction, identity extraction — into
        # `planning`, defeating the split this exists to provide (#309 review).
        self._notify_usage(role, model, response.usage, listener)

        # No call at all, or one whose arguments were cut mid-value. `repaired`
        # cannot wait for the validation branch below: the dict parses, and it
        # validates whenever the missing field is optional or defaulted, so a
        # half-written object would be returned as the model's answer.
        #
        # `parse_error` deliberately stays out of this check — it reaches the
        # validation branch, which classifies it only when the budget ran out
        # and otherwise lets the schema error through as itself.
        if not response.tool_calls or any(tc.repaired for tc in response.tool_calls):
            # Shared with the scratchpad's sync twin so both paths classify the
            # failure identically — see `raise_unusable_tool_call` (ENG-1081).
            raise_unusable_tool_call(response, tool_name=tool["name"], budget=budget)

        try:
            return unwrap_structured_response(
                response.tool_calls[0].input, validator_class, is_list
            )
        except Exception:
            # The budget can also run out *inside* the tool call's JSON, so
            # validation fails here instead. Same cause, same retry (ENG-1081).
            if looks_truncated(response, budget):
                raise_unusable_tool_call(
                    response, tool_name=tool["name"], budget=budget
                )
            raise

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
            role="planning",
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
            role="coding",
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
                # vision_format defaults to "openai" — every gateway we route
                # openai-compatible requests through (MindsHub/mdb.ai included)
                # already normalizes a standard OpenAI image_url block into
                # whatever the resolved backend natively needs (Anthropic,
                # Responses API, ...). Guessing "anthropic" from the base-URL
                # host here used to assume every model behind that host was
                # Claude, which broke non-Claude models sharing the same
                # gateway (ENG-1992: MindsHub Air screenshots 400'd because
                # this sent Anthropic-shaped image blocks to an OpenAI
                # Responses-backed model).
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
