"""Fireworks provider adapter implementation.

Fireworks speaks the Anthropic Messages API shape but has no hosted search
index, so this adapter delegates to :mod:`minds.inference.providers.fireworks`
which drives a server-side external-search loop. The adapter is responsible for
building the SDK client and (when the request asks for web search) the
``SearchProvider`` it hands to the proxy.
"""

from __future__ import annotations

from typing import Any

from starlette.responses import JSONResponse, StreamingResponse

from minds.inference.adapter import ProviderAdapter
from minds.inference.providers import fireworks as fireworks_module
from minds.inference.providers.anthropic import _get_anthropic_client
from minds.inference.types import PassthroughModelConfig, UsageBox
from minds.schemas.chat import Message


class FireworksAdapter(ProviderAdapter):
    """Adapter for Fireworks-hosted models (Anthropic-shape + external search)."""

    def __init__(self) -> None:
        """Initialize the Fireworks adapter."""
        self._last_usage: tuple[int, int] | None = None
        self._last_output: dict[str, Any] | None = None
        self._last_artifacts: list[dict[str, Any]] = []
        self._usage_box: UsageBox | None = None

    async def complete(
        self,
        config: PassthroughModelConfig,
        messages: list[Message],
        stream: bool,
        request_id: str,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> StreamingResponse | JSONResponse:
        """Execute a Fireworks inference request."""
        # Create fresh usage box for this call
        usage_box = UsageBox()

        # Get client (Anthropic SDK pointed at the Fireworks base_url)
        client = _get_anthropic_client(config)

        # Build the search provider only when a web tool is requested; None
        # means the request degrades to a single-shot call with web tools
        # dropped (see build_search_provider_for_request).
        search_provider = fireworks_module.build_search_provider_for_request(config, tools)

        # Convert messages to dicts
        messages_dicts = [
            {
                "role": m.role.value,
                "content": m.content or "",
                **({"tool_calls": m.tool_calls} if m.tool_calls is not None else {}),
                **({"tool_call_id": m.tool_call_id} if m.tool_call_id is not None else {}),
                **({"name": m.name} if m.name is not None else {}),
            }
            for m in messages
        ]

        # Delegate to provider module
        response = await fireworks_module.proxy_fireworks(
            client=client,
            config=config,
            usage_box=usage_box,
            messages=messages_dicts,
            stream=stream,
            request_id=request_id,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            search_provider=search_provider,
        )

        # Capture state for later retrieval
        self._last_usage = usage_box.value
        self._last_output = usage_box.output_payload
        self._last_artifacts = list(usage_box.server_artifacts)
        self._usage_box = usage_box

        return response

    def get_last_usage(self) -> tuple[int, int] | None:
        """Return token usage from the most recent complete() call."""
        return self._last_usage

    def get_last_output(self) -> dict[str, Any] | None:
        """Return the OpenAI-format assistant message from the most recent call."""
        return self._last_output

    def get_last_artifacts(self) -> list[dict[str, Any]]:
        """Return server artifacts from the most recent call."""
        return list(self._last_artifacts)

    def get_usage_box(self) -> UsageBox | None:
        """Return the UsageBox from the most recent complete() call."""
        return self._usage_box
