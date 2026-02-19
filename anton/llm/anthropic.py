from __future__ import annotations

import json
from collections.abc import AsyncIterator

import anthropic

from anton.llm.provider import (
    LLMProvider,
    LLMResponse,
    StreamComplete,
    StreamEvent,
    StreamTextDelta,
    StreamToolUseDelta,
    StreamToolUseEnd,
    StreamToolUseStart,
    ToolCall,
    Usage,
)


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str | None = None) -> None:
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = anthropic.AsyncAnthropic(**kwargs)

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = await self._client.messages.create(**kwargs)

        content_text = ""
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, input=block.input)
                )

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            stop_reason=response.stop_reason,
        )

    async def stream(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamEvent]:
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        content_text = ""
        tool_calls: list[ToolCall] = []
        input_tokens = 0
        output_tokens = 0
        stop_reason: str | None = None

        # Track content blocks by index for tool correlation
        blocks: dict[int, dict] = {}

        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "message_start":
                    usage = event.message.usage
                    input_tokens = usage.input_tokens
                    output_tokens = getattr(usage, "output_tokens", 0)

                elif event.type == "content_block_start":
                    idx = event.index
                    block = event.content_block
                    if block.type == "tool_use":
                        blocks[idx] = {"type": "tool_use", "id": block.id, "name": block.name, "json_parts": []}
                        yield StreamToolUseStart(id=block.id, name=block.name)
                    else:
                        blocks[idx] = {"type": "text"}

                elif event.type == "content_block_delta":
                    idx = event.index
                    delta = event.delta
                    if delta.type == "text_delta":
                        content_text += delta.text
                        yield StreamTextDelta(text=delta.text)
                    elif delta.type == "input_json_delta":
                        info = blocks.get(idx, {})
                        if info.get("type") == "tool_use":
                            info["json_parts"].append(delta.partial_json)
                            yield StreamToolUseDelta(id=info["id"], json_delta=delta.partial_json)

                elif event.type == "content_block_stop":
                    idx = event.index
                    info = blocks.get(idx, {})
                    if info.get("type") == "tool_use":
                        raw_json = "".join(info["json_parts"])
                        parsed_input = json.loads(raw_json) if raw_json else {}
                        tool_calls.append(
                            ToolCall(id=info["id"], name=info["name"], input=parsed_input)
                        )
                        yield StreamToolUseEnd(id=info["id"])

                elif event.type == "message_delta":
                    stop_reason = event.delta.stop_reason
                    output_tokens = event.usage.output_tokens

        yield StreamComplete(
            response=LLMResponse(
                content=content_text,
                tool_calls=tool_calls,
                usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
                stop_reason=stop_reason,
            )
        )
