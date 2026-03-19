from __future__ import annotations

import json
from collections.abc import AsyncIterator

import openai

from anton.llm.openai_compat import (
    _translate_messages,
    _translate_tool_choice,
    _translate_tools,
)
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
        request_options: dict | None = None,
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
            raise ConnectionError(
                f"Server returned {exc.status_code} — the LLM endpoint may be temporarily unavailable. Try again in a moment."
            ) from exc
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
        request_options: dict | None = None,
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
            raise ConnectionError(
                f"Server returned {exc.status_code} — the LLM endpoint may be temporarily unavailable. Try again in a moment."
            ) from exc
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
