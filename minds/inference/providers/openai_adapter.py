"""OpenAI provider adapter implementation."""

from __future__ import annotations

from typing import Any

from starlette.responses import JSONResponse, StreamingResponse

from minds.common.passthrough_config import PassthroughModelConfig
from minds.inference.adapter import ProviderAdapter
from minds.inference.providers import openai as openai_module
from minds.inference.types import UsageBox
from minds.schemas.chat import Message


class OpenAIAdapter(ProviderAdapter):
    """Adapter for OpenAI's Responses API."""

    def __init__(self) -> None:
        """Initialize the OpenAI adapter."""
        self._last_usage: tuple[int, int] | None = None
        self._last_output: dict[str, Any] | None = None
        self._last_artifacts: list[dict[str, Any]] = []

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
        """Execute an OpenAI inference request."""
        # Create fresh usage box for this call
        usage_box = UsageBox()

        # Get client
        client = openai_module._get_openai_client(config)

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
        response = await openai_module.proxy_openai(
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
        )

        # Capture state for later retrieval
        self._last_usage = usage_box.value
        self._last_output = usage_box.output_payload
        self._last_artifacts = list(usage_box.server_artifacts)

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
