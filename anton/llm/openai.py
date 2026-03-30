from __future__ import annotations

import json
from collections.abc import AsyncIterator

import openai

from anton.llm.provider import (
    ContextOverflowError,
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
    compute_context_pressure,
)


def _translate_tools(tools: list[dict]) -> list[dict]:
    """Anthropic tool format -> OpenAI function-calling format."""
    result = []
    for tool in tools:
        result.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        })
    return result


def _translate_tool_choice(tool_choice: dict) -> dict | str:
    """Anthropic tool_choice -> OpenAI tool_choice."""
    tc_type = tool_choice.get("type")
    if tc_type == "tool":
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    if tc_type == "any":
        return "required"
    if tc_type == "auto":
        return "auto"
    return "auto"


def _translate_messages(system: str, messages: list[dict]) -> list[dict]:
    """Convert Anthropic-style messages to OpenAI chat format.

    Handles:
    - system prompt -> {"role": "system", ...}
    - plain text messages pass through
    - assistant messages with tool_use content blocks -> tool_calls array
    - user messages with tool_result content blocks -> role:tool messages
    """
    result: list[dict] = []
    if system:
        result.append({"role": "system", "content": system})

    for msg in messages:
        role = msg["role"]
        content = msg.get("content")

        # Plain string content — pass through
        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue

        # Content is a list of blocks (Anthropic format)
        if isinstance(content, list):
            if role == "assistant":
                result.extend(_translate_assistant_blocks(content))
            elif role == "user":
                result.extend(_translate_user_blocks(content))
            else:
                # Fallback: join text blocks
                text = " ".join(
                    b.get("text", "") for b in content if b.get("type") == "text"
                )
                result.append({"role": role, "content": text or ""})
            continue

        # Fallback
        result.append({"role": role, "content": str(content) if content else ""})

    return result


def _translate_assistant_blocks(blocks: list[dict]) -> list[dict]:
    """Convert assistant content blocks to OpenAI message(s)."""
    text_parts: list[str] = []
    tool_calls: list[dict] = []

    for block in blocks:
        if block.get("type") == "text":
            text_parts.append(block["text"])
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block["id"],
                "type": "function",
                "function": {
                    "name": block["name"],
                    "arguments": json.dumps(block.get("input", {})),
                },
            })

    msg: dict = {"role": "assistant"}
    content = "\n".join(text_parts) if text_parts else None
    msg["content"] = content
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return [msg]


def _translate_user_blocks(blocks: list[dict]) -> list[dict]:
    """Convert user content blocks (including tool_result and image) to OpenAI messages."""
    result: list[dict] = []
    content_parts: list[dict] = []  # Accumulates text + image_url blocks

    for block in blocks:
        if block.get("type") == "tool_result":
            # Flush any accumulated content parts first
            if content_parts:
                result.append({"role": "user", "content": content_parts})
                content_parts = []
            # tool_result -> role:tool message
            content = block.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    b.get("text", "") for b in content if b.get("type") == "text"
                )
            result.append({
                "role": "tool",
                "tool_call_id": block["tool_use_id"],
                "content": str(content),
            })
        elif block.get("type") == "text":
            content_parts.append({"type": "text", "text": block.get("text", "")})
        elif block.get("type") == "image":
            # Anthropic image block -> OpenAI image_url block
            source = block.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/png")
                data = source.get("data", "")
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{data}"},
                })

    if content_parts:
        # If only text parts, flatten to a simple string for compatibility
        if all(p.get("type") == "text" for p in content_parts):
            result.append({
                "role": "user",
                "content": "\n".join(p["text"] for p in content_parts),
            })
        else:
            result.append({"role": "user", "content": content_parts})

    return result


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        ssl_verify: bool = True,
    ) -> None:
        import httpx

        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        if not ssl_verify:
            kwargs["http_client"] = httpx.AsyncClient(verify=False)
        self._client = openai.AsyncOpenAI(**kwargs)

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        oai_messages = _translate_messages(system, messages)

        kwargs: dict = {
            "model": model,
            "messages": oai_messages,
            "max_completion_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = _translate_tools(tools)
        if tool_choice:
            kwargs["tool_choice"] = _translate_tool_choice(tool_choice)

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except openai.BadRequestError as exc:
            msg = str(exc).lower()
            if "context_length_exceeded" in msg or "maximum context length" in msg:
                raise ContextOverflowError(str(exc)) from exc
            raise
        except openai.APIStatusError as exc:
            if exc.status_code == 429 and isinstance(exc.body, dict) and exc.body.get("detail"):
                msg = f"Server returned 429 — {exc.body['detail']}"
                msg += " Visit https://mdb.ai to upgrade or to top up your tokens."
            else:
                msg = f"Server returned {exc.status_code} — the LLM endpoint may be temporarily unavailable. Try again in a moment."
            raise ConnectionError(msg) from exc
        except openai.APIConnectionError as exc:
            raise ConnectionError(
                "Could not reach the LLM server — check your connection or try again in a moment."
            ) from exc

        choice = response.choices[0]
        message = choice.message

        content_text = message.content or ""
        tool_calls: list[ToolCall] = []

        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        input=json.loads(tc.function.arguments) if tc.function.arguments else {},
                    )
                )

        usage_obj = response.usage
        input_tokens = usage_obj.prompt_tokens if usage_obj else 0
        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=usage_obj.completion_tokens if usage_obj else 0,
                context_pressure=compute_context_pressure(model, input_tokens),
            ),
            stop_reason=choice.finish_reason,
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
        oai_messages = _translate_messages(system, messages)

        kwargs: dict = {
            "model": model,
            "messages": oai_messages,
            "max_completion_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = _translate_tools(tools)

        content_text = ""
        tool_calls: list[ToolCall] = []
        input_tokens = 0
        output_tokens = 0
        stop_reason: str | None = None

        # Track tool call deltas by index
        tc_state: dict[int, dict] = {}

        try:
            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                finish = chunk.choices[0].finish_reason

                if finish:
                    stop_reason = finish

                # Text content
                if delta.content:
                    content_text += delta.content
                    yield StreamTextDelta(text=delta.content)

                # Tool call deltas
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tc_state:
                            # New tool call
                            tc_state[idx] = {
                                "id": tc_delta.id or "",
                                "name": tc_delta.function.name if tc_delta.function and tc_delta.function.name else "",
                                "args_parts": [],
                            }
                            if tc_state[idx]["id"] and tc_state[idx]["name"]:
                                yield StreamToolUseStart(
                                    id=tc_state[idx]["id"],
                                    name=tc_state[idx]["name"],
                                )
                        else:
                            # Update id/name if provided in later chunks
                            if tc_delta.id:
                                tc_state[idx]["id"] = tc_delta.id
                            if tc_delta.function and tc_delta.function.name:
                                tc_state[idx]["name"] = tc_delta.function.name

                        # Accumulate argument fragments
                        if tc_delta.function and tc_delta.function.arguments:
                            tc_state[idx]["args_parts"].append(tc_delta.function.arguments)
                            yield StreamToolUseDelta(
                                id=tc_state[idx]["id"],
                                json_delta=tc_delta.function.arguments,
                            )
        except openai.BadRequestError as exc:
            msg = str(exc).lower()
            if "context_length_exceeded" in msg or "maximum context length" in msg:
                raise ContextOverflowError(str(exc)) from exc
            raise
        except openai.APIStatusError as exc:
            if exc.status_code == 429 and isinstance(exc.body, dict) and exc.body.get("detail"):
                msg = f"Server returned 429 — {exc.body['detail']}"
                msg += " Visit https://mdb.ai to upgrade or top up your tokens."
            else:
                msg = f"Server returned {exc.status_code} — the LLM endpoint may be temporarily unavailable. Try again in a moment."
            raise ConnectionError(msg) from exc
        except openai.APIConnectionError as exc:
            raise ConnectionError(
                "Could not reach the LLM server — check your connection or try again in a moment."
            ) from exc

        # Finalize tool calls
        for idx in sorted(tc_state):
            info = tc_state[idx]
            raw_json = "".join(info["args_parts"])
            parsed = json.loads(raw_json) if raw_json else {}
            tool_calls.append(ToolCall(id=info["id"], name=info["name"], input=parsed))
            yield StreamToolUseEnd(id=info["id"])

        yield StreamComplete(
            response=LLMResponse(
                content=content_text,
                tool_calls=tool_calls,
                usage=Usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    context_pressure=compute_context_pressure(model, input_tokens),
                ),
                stop_reason=stop_reason,
            )
        )
