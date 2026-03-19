from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

import ollama

from anton.llm.openai_compat import translate_tools
from anton.llm.provider import (
    ContextOverflowError,
    LLMProvider,
    LLMResponse,
    StreamComplete,
    StreamEvent,
    StreamTaskProgress,
    StreamTextDelta,
    StreamToolUseDelta,
    StreamToolUseEnd,
    StreamToolUseStart,
    ToolCall,
    Usage,
    compute_context_pressure,
)

_DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


@dataclass(frozen=True)
class OllamaModelInfo:
    name: str
    size: str = ""
    quantization: str = ""
    parameter_size: str = ""

    @property
    def display_name(self) -> str:
        details = [detail for detail in (self.parameter_size, self.quantization) if detail]
        if details:
            return f"{self.name} ({', '.join(details)})"
        if self.size:
            return f"{self.name} ({self.size})"
        return self.name


def normalize_ollama_base_url(base_url: str | None) -> str:
    url = (base_url or _DEFAULT_OLLAMA_BASE_URL).strip()
    if not url:
        url = _DEFAULT_OLLAMA_BASE_URL
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    url = url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3].rstrip("/")
    return url


def list_ollama_models(base_url: str | None = None) -> list[OllamaModelInfo]:
    client = ollama.Client(host=normalize_ollama_base_url(base_url))
    response = client.list()
    models: list[OllamaModelInfo] = []
    for model in response.models:
        size = str(model.size) if model.size is not None else ""
        details = model.details
        models.append(
            OllamaModelInfo(
                name=model.model or "",
                size=size,
                quantization=details.quantization_level if details and details.quantization_level else "",
                parameter_size=details.parameter_size if details and details.parameter_size else "",
            )
        )
    return models


def translate_messages_to_ollama(system: str, messages: list[dict]) -> list[dict]:
    """Convert Anthropic-style messages to native Ollama chat format."""
    result: list[dict] = []
    tool_name_by_id: dict[str, str] = {}
    if system:
        result.append({"role": "system", "content": system})

    for message in messages:
        role = message["role"]
        content = message.get("content")

        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            result.append({"role": role, "content": str(content) if content else ""})
            continue

        if role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_name_by_id[block["id"]] = block["name"]
                    tool_calls.append({
                        "function": {
                            "name": block["name"],
                            "arguments": block.get("input", {}),
                        }
                    })
            msg: dict[str, object] = {"role": "assistant"}
            if text_parts:
                msg["content"] = "\n".join(text_parts)
            if tool_calls:
                msg["tool_calls"] = tool_calls
            result.append(msg)
            continue

        if role == "user":
            text_parts: list[str] = []
            images: list[str] = []
            for block in content:
                block_type = block.get("type")
                if block_type == "tool_result":
                    if text_parts or images:
                        result.append(_build_ollama_user_message(text_parts, images))
                        text_parts = []
                        images = []
                    tool_name = tool_name_by_id.get(block["tool_use_id"], block["tool_use_id"])
                    tool_content = block.get("content", "")
                    if isinstance(tool_content, list):
                        tool_content = "\n".join(
                            item.get("text", "")
                            for item in tool_content
                            if item.get("type") == "text"
                        )
                    result.append({
                        "role": "tool",
                        "tool_name": tool_name,
                        "content": str(tool_content),
                    })
                elif block_type == "text":
                    text_parts.append(block.get("text", ""))
                elif block_type == "image":
                    source = block.get("source", {})
                    if source.get("type") == "base64" and source.get("data"):
                        images.append(source["data"])
            if text_parts or images:
                result.append(_build_ollama_user_message(text_parts, images))
            continue

        text = " ".join(
            block.get("text", "")
            for block in content
            if block.get("type") == "text"
        )
        result.append({"role": role, "content": text or ""})

    return result


def _build_ollama_user_message(text_parts: list[str], images: list[str]) -> dict[str, object]:
    message: dict[str, object] = {"role": "user"}
    if text_parts:
        message["content"] = "\n".join(text_parts)
    if images:
        message["images"] = images
    return message


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = normalize_ollama_base_url(base_url)
        self._client = ollama.AsyncClient(host=self._base_url)

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
        kwargs = self._build_chat_kwargs(
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            request_options=request_options,
        )
        try:
            response = await self._client.chat(**kwargs)
        except ollama.ResponseError as exc:
            message = str(exc).lower()
            if "context" in message and "length" in message:
                raise ContextOverflowError(str(exc)) from exc
            raise ConnectionError(str(exc)) from exc
        except ConnectionError as exc:
            raise ConnectionError(str(exc)) from exc

        content_text = response.message.content or ""
        tool_calls = self._tool_calls_from_message(response.message.tool_calls or [])
        input_tokens = response.prompt_eval_count or 0
        output_tokens = response.eval_count or 0
        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                context_pressure=compute_context_pressure(model, input_tokens),
            ),
            stop_reason=response.done_reason,
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
        kwargs = self._build_chat_kwargs(
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            tool_choice=None,
            max_tokens=max_tokens,
            request_options=request_options,
        )
        kwargs["stream"] = True

        content_text = ""
        tool_calls: list[ToolCall] = []
        input_tokens = 0
        output_tokens = 0
        stop_reason: str | None = None
        showed_reasoning = False
        saw_content = False
        next_tool_index = 1

        try:
            stream = await self._client.chat(**kwargs)
            async for chunk in stream:
                message = chunk.message
                if not showed_reasoning and not saw_content and message.thinking:
                    showed_reasoning = True
                    yield StreamTaskProgress(phase="reasoning", message="Thinking...")

                if message.content:
                    saw_content = True
                    content_text += message.content
                    yield StreamTextDelta(text=message.content)

                if message.tool_calls:
                    for call in message.tool_calls:
                        tool_call = self._tool_call_from_ollama(call, next_tool_index)
                        next_tool_index += 1
                        tool_calls.append(tool_call)
                        yield StreamToolUseStart(id=tool_call.id, name=tool_call.name)
                        yield StreamToolUseDelta(
                            id=tool_call.id,
                            json_delta=json.dumps(tool_call.input),
                        )
                        yield StreamToolUseEnd(id=tool_call.id)

                if chunk.prompt_eval_count is not None:
                    input_tokens = chunk.prompt_eval_count
                if chunk.eval_count is not None:
                    output_tokens = chunk.eval_count
                if chunk.done_reason:
                    stop_reason = chunk.done_reason
        except ollama.ResponseError as exc:
            message = str(exc).lower()
            if "context" in message and "length" in message:
                raise ContextOverflowError(str(exc)) from exc
            raise ConnectionError(str(exc)) from exc
        except ConnectionError as exc:
            raise ConnectionError(str(exc)) from exc

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

    def _build_chat_kwargs(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict] | None,
        tool_choice: dict | None,
        max_tokens: int,
        request_options: dict | None,
    ) -> dict:
        kwargs: dict[str, object] = {
            "model": model,
            "messages": translate_messages_to_ollama(system, messages),
            "options": {"num_predict": max_tokens},
        }

        translated_tools = translate_tools(tools or []) if tools else None
        if translated_tools:
            kwargs["tools"] = self._apply_tool_choice(translated_tools, tool_choice)

        if request_options and "think" in request_options:
            kwargs["think"] = request_options["think"]

        return kwargs

    @staticmethod
    def _apply_tool_choice(tools: list[dict], tool_choice: dict | None) -> list[dict]:
        if not tool_choice or tool_choice.get("type") != "tool":
            return tools
        target_name = tool_choice.get("name")
        if not target_name:
            return tools
        filtered = [
            tool
            for tool in tools
            if tool.get("function", {}).get("name") == target_name
        ]
        return filtered or tools

    def _tool_calls_from_message(self, calls: list) -> list[ToolCall]:
        return [
            self._tool_call_from_ollama(call, index)
            for index, call in enumerate(calls, start=1)
        ]

    @staticmethod
    def _tool_call_from_ollama(call, index: int) -> ToolCall:
        function = call.function
        return ToolCall(
            id=f"ollama_tool_{index}",
            name=function.name,
            input=dict(function.arguments or {}),
        )
