from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator

import openai
from openai import AsyncAzureOpenAI

from .provider import safe_parse_tool_input
from .provider import (
    ContextOverflowError,
    LLMProvider,
    LLMResponse,
    ProviderConnectionInfo,
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
        result.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
        )
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


def _translate_messages(system: str, messages: list[dict], supports_vision: bool = True, vision_format: str = "openai") -> list[dict]:
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
                result.extend(_translate_user_blocks(content, supports_vision=supports_vision, vision_format=vision_format))
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
            tool_calls.append(
                {
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block.get("input", {})),
                    },
                }
            )

    msg: dict = {"role": "assistant"}
    content = "\n".join(text_parts) if text_parts else None
    msg["content"] = content
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return [msg]


def _translate_user_blocks(blocks: list[dict], supports_vision: bool = True, vision_format: str = "openai") -> list[dict]:
    """Convert user content blocks (including tool_result and image) to OpenAI messages.

    vision_format controls how image blocks are serialised:
    - "openai"     → {"type": "image_url", "image_url": {"url": "data:..."}}
    - "anthropic"  → kept as-is {"type": "image", "source": {...}}  (for
                     endpoints like MDB.AI that speak Anthropic content format
                     over an OpenAI-compatible HTTP envelope)
    """
    result: list[dict] = []
    content_parts: list[dict] = []  # Accumulates text + image blocks

    for block in blocks:
        if block.get("type") == "tool_result":
            # Flush any accumulated content parts first
            if content_parts:
                result.append({"role": "user", "content": content_parts})
                content_parts = []
            # tool_result -> role:tool message. The Chat Completions API only
            # accepts a string in `tool` messages, so when the tool returned
            # multimodal blocks (image + text), we split them: text → tool
            # message, image → a follow-up role:user message right after.
            raw = block.get("content", "")
            extra_images: list[dict] = []
            if isinstance(raw, list):
                text_parts_for_tool: list[str] = []
                for b in raw:
                    if b.get("type") == "text":
                        text_parts_for_tool.append(b.get("text", ""))
                    elif b.get("type") == "image" and supports_vision:
                        extra_images.append(b)
                if text_parts_for_tool:
                    tool_text = "\n".join(text_parts_for_tool)
                elif extra_images:
                    tool_text = "Image attached in next user message."
                else:
                    tool_text = ""
            else:
                tool_text = str(raw)

            result.append(
                {
                    "role": "tool",
                    "tool_call_id": block["tool_use_id"],
                    "content": tool_text,
                }
            )

            if extra_images:
                img_parts: list[dict] = [
                    {
                        "type": "text",
                        "text": "Image(s) returned by previous tool call:",
                    }
                ]
                for img in extra_images:
                    if vision_format == "anthropic":
                        img_parts.append(img)
                        continue
                    source = img.get("source", {})
                    if source.get("type") == "base64":
                        media_type = source.get("media_type", "image/png")
                        data = source.get("data", "")
                        img_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{data}"
                                },
                            }
                        )
                result.append({"role": "user", "content": img_parts})
        elif block.get("type") == "text":
            content_parts.append({"type": "text", "text": block.get("text", "")})
        elif block.get("type") == "image" and supports_vision:
            if vision_format == "anthropic":
                # Keep Anthropic-format image block as-is — endpoints like
                # MDB.AI pass content through to Claude and expect this format.
                content_parts.append(block)
            else:
                # Anthropic image block -> OpenAI image_url block
                source = block.get("source", {})
                if source.get("type") == "base64":
                    media_type = source.get("media_type", "image/png")
                    data = source.get("data", "")
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{data}"},
                        }
                    )
        elif block.get("type") == "image_url" and supports_vision:
            # Inbound OpenAI-format image (e.g. scratchpad code that built the
            # message in OpenAI shape). Translate to the configured outbound
            # format so it actually reaches the model.
            url = (block.get("image_url") or {}).get("url", "")
            if vision_format == "anthropic" and url.startswith("data:"):
                # data:<media_type>;base64,<data>  ->  Anthropic image block
                try:
                    header, data = url.split(",", 1)
                    media_type = header[len("data:") : header.index(";")]
                except (ValueError, IndexError):
                    media_type, data = "image/png", ""
                content_parts.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        },
                    }
                )
            else:
                # Already OpenAI format — pass through.
                content_parts.append(block)

    if content_parts:
        # If only text parts, flatten to a simple string for compatibility
        if all(p.get("type") == "text" for p in content_parts):
            result.append(
                {
                    "role": "user",
                    "content": "\n".join(p["text"] for p in content_parts),
                }
            )
        else:
            result.append({"role": "user", "content": content_parts})

    return result


def _is_azure_endpoint(url: str | None) -> bool:
    """Return True if the URL looks like an Azure OpenAI endpoint."""
    if not url:
        return False
    from urllib.parse import urlparse
    parsed = urlparse(url if "://" in url else f"https://{url}")
    host = (parsed.netloc or parsed.path).lower()
    return host.endswith(".openai.azure.com") or host.endswith(".cognitiveservices.azure.com")


def build_chat_completion_kwargs(
    *,
    model: str,
    messages: list[dict],
    max_tokens: int,
    stream: bool = False,
) -> dict:
    """Build chat.completions kwargs using modern OpenAI parameter names."""
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }
    if stream:
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
    return kwargs


class OpenAIProvider(LLMProvider):
    name: str = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        ssl_verify: bool = True,
        api_version: str | None = None,
        supports_vision: bool = True,
        vision_format: str = "openai",
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._ssl_verify = ssl_verify
        self._api_version = api_version
        self._supports_vision = supports_vision
        self._vision_format = vision_format
        # Whether to attach langfuse-style headers (Langfuse-Session-Id,
        # Langfuse-Tags, Langfuse-Metadata) to outbound requests. Default-on
        # only for the MindsHub-backed deployment, which is the curated
        # langfuse-aware router we ship against. For every other openai-
        # compatible endpoint (raw OpenAI, Azure, Gemini, self-hosted
        # vLLM/ollama/LM Studio) we skip by default so the cowork session
        # identity doesn't leak into third-party logs.
        #
        # Power-user opt-in: set `ANTON_LANGFUSE_HEADERS=1` to force-emit
        # the headers regardless of base URL — useful when the user has
        # pointed `base_url` at their own langfuse-instrumented proxy.
        self._emit_trace_headers = bool(base_url) and (
            "mindshub.ai" in base_url or "mdb.ai" in base_url
        )
        if os.environ.get("ANTON_LANGFUSE_HEADERS", "").strip().lower() in {
            "1", "true", "yes", "on",
        }:
            self._emit_trace_headers = True

        import httpx

        if api_version and _is_azure_endpoint(base_url):
            # Azure OpenAI: use the dedicated client which handles deployment
            # URL construction and api-version automatically.
            azure_kwargs: dict = {"api_version": api_version}
            if api_key:
                azure_kwargs["api_key"] = api_key
            if base_url:
                azure_kwargs["azure_endpoint"] = base_url
            if not ssl_verify:
                azure_kwargs["http_client"] = httpx.AsyncClient(verify=False)
            self._client = AsyncAzureOpenAI(**azure_kwargs)
        else:
            kwargs: dict = {}
            if api_key:
                kwargs["api_key"] = api_key
            if base_url:
                kwargs["base_url"] = base_url
            if not ssl_verify:
                kwargs["http_client"] = httpx.AsyncClient(verify=False)
            self._client = openai.AsyncOpenAI(**kwargs)

    def export_connection_info(self) -> ProviderConnectionInfo:
        return ProviderConnectionInfo(
            provider=self.name,
            api_key=self._api_key,
            base_url=self._base_url,
            ssl_verify=self._ssl_verify,
            api_version=self._api_version,
        )

    def _build_trace_headers(self) -> dict[str, str] | None:
        """Return langfuse-style headers for the active trace, or None.

        Returns None unless trace-header emission is enabled for this
        provider instance (default-on for MindsHub, opt-in for any other
        openai-compatible endpoint via `ANTON_LANGFUSE_HEADERS=1`) AND a
        `TraceContext` has been installed by `ChatSession.turn_stream`.
        """
        if not self._emit_trace_headers:
            return None
        from .tracing import get_trace_context

        ctx = get_trace_context()
        if ctx is None:
            return None
        headers: dict[str, str] = {}
        if ctx.session_id:
            headers["Langfuse-Session-Id"] = ctx.session_id
        if ctx.harness:
            headers["Langfuse-Tags"] = ctx.harness
        extra: dict[str, object] = {}
        if ctx.turn_id is not None:
            extra["turn_id"] = ctx.turn_id
        if ctx.harness:
            extra["harness"] = ctx.harness
        if extra:
            headers["Langfuse-Metadata"] = json.dumps(extra)
        return headers or None

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
        oai_messages = _translate_messages(system, messages, supports_vision=self._supports_vision, vision_format=self._vision_format)

        kwargs = build_chat_completion_kwargs(
            model=model,
            messages=oai_messages,
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = _translate_tools(tools)
        if tool_choice:
            kwargs["tool_choice"] = _translate_tool_choice(tool_choice)
        trace_headers = self._build_trace_headers()
        if trace_headers:
            kwargs["extra_headers"] = trace_headers

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except openai.BadRequestError as exc:
            msg = str(exc).lower()
            if "context_length_exceeded" in msg or "maximum context length" in msg:
                raise ContextOverflowError(str(exc)) from exc
            raise
        except openai.APIStatusError as exc:
            if (
                exc.status_code == 429
                and isinstance(exc.body, dict)
                and exc.body.get("detail")
            ):
                msg = f"Server returned 429 — {exc.body['detail']}"
                msg += " Visit https://mdb.ai to upgrade or to top up your tokens."
                from .provider import TokenLimitExceeded

                raise TokenLimitExceeded(msg) from exc
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
                # safe_parse_tool_input returns (parsed_dict,
                # parse_error). parse_error is forwarded to the
                # session dispatcher so the tool_use/tool_result
                # protocol can carry the recovery — see the streaming
                # path in this file for the same pattern.
                parsed_input, parse_error = safe_parse_tool_input(tc.function.arguments or "")
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        input=parsed_input,
                        parse_error=parse_error,
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
        oai_messages = _translate_messages(system, messages, supports_vision=self._supports_vision, vision_format=self._vision_format)

        kwargs = build_chat_completion_kwargs(
            model=model,
            messages=oai_messages,
            max_tokens=max_tokens,
            stream=True,
        )
        if tools:
            kwargs["tools"] = _translate_tools(tools)
        trace_headers = self._build_trace_headers()
        if trace_headers:
            kwargs["extra_headers"] = trace_headers

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
                                "name": tc_delta.function.name
                                if tc_delta.function and tc_delta.function.name
                                else "",
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
                            tc_state[idx]["args_parts"].append(
                                tc_delta.function.arguments
                            )
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
            if (
                exc.status_code == 429
                and isinstance(exc.body, dict)
                and exc.body.get("detail")
            ):
                msg = f"Server returned 429 — {exc.body['detail']}"
                msg += " Visit https://mdb.ai to upgrade or top up your tokens."
                from .provider import TokenLimitExceeded

                raise TokenLimitExceeded(msg) from exc
            else:
                msg = f"Server returned {exc.status_code} — the LLM endpoint may be temporarily unavailable. Try again in a moment."
            raise ConnectionError(msg) from exc
        except openai.APIConnectionError as exc:
            raise ConnectionError(
                "Could not reach the LLM server — check your connection or try again in a moment."
            ) from exc

        # Finalize tool calls. Same safe-parse protection as the
        # non-streaming path — a model cut off mid-JSON-arguments
        # would otherwise crash the whole turn here with an opaque
        # JSONDecodeError. parse_error rides along on the ToolCall so
        # the session dispatcher can short-circuit with a structured
        # recovery tool_result.
        for idx in sorted(tc_state):
            info = tc_state[idx]
            raw_json = "".join(info["args_parts"])
            parsed, parse_error = safe_parse_tool_input(raw_json)
            tool_calls.append(ToolCall(
                id=info["id"], name=info["name"], input=parsed,
                parse_error=parse_error,
            ))
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
