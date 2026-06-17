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


# ─────────────────────────────────────────────────────────────────────────────
# Responses API translation
#
# Used only for ``flavor="openai"`` (BYOK direct OpenAI). The Responses API is
# OpenAI's recommended transport going forward, supports native server-side
# web_search, and has a different request/response shape than chat.completions
# (flat function-tool params, ``input``/``instructions`` instead of
# ``messages``/``system``, ``output`` array instead of ``choices``).
# ─────────────────────────────────────────────────────────────────────────────


def _translate_tools_to_responses(tools: list[dict]) -> list[dict]:
    """Anthropic tool format -> OpenAI Responses API function-tool format.

    The Responses API uses a flat shape (``{"type": "function", "name": ...,
    "description": ..., "parameters": ...}``) rather than the chat.completions
    nested shape under a ``function`` key.
    """
    result: list[dict] = []
    for tool in tools:
        result.append(
            {
                "type": "function",
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            }
        )
    return result


def _translate_tool_choice_to_responses(tool_choice: dict) -> dict | str:
    """Anthropic tool_choice -> OpenAI Responses API tool_choice."""
    tc_type = tool_choice.get("type")
    if tc_type == "tool":
        return {"type": "function", "name": tool_choice["name"]}
    if tc_type == "any":
        return "required"
    if tc_type == "auto":
        return "auto"
    return "auto"


def _translate_messages_to_responses_input(
    messages: list[dict], supports_vision: bool = True
) -> list[dict]:
    """Convert Anthropic-style messages to Responses API ``input`` items.

    The Responses API accepts a list of items where:

    - User/assistant text messages → ``{"role": ..., "content": ..., "type": "message"}``
    - Assistant tool calls → ``{"type": "function_call", "call_id": ..., "name": ..., "arguments": ...}``
    - Tool results → ``{"type": "function_call_output", "call_id": ..., "output": ...}``

    The system prompt is passed via the top-level ``instructions`` parameter
    rather than as a message item, so it is *not* emitted here.
    """
    items: list[dict] = []

    for msg in messages:
        role = msg["role"]
        content = msg.get("content")

        # Plain string content — pass through as a simple message item.
        if isinstance(content, str):
            items.append({"role": role, "content": content, "type": "message"})
            continue

        if isinstance(content, list):
            if role == "assistant":
                items.extend(_translate_assistant_blocks_to_responses(content))
            elif role == "user":
                items.extend(
                    _translate_user_blocks_to_responses(
                        content, supports_vision=supports_vision
                    )
                )
            else:
                # Fallback: join text blocks
                text = " ".join(
                    b.get("text", "") for b in content if b.get("type") == "text"
                )
                items.append({"role": role, "content": text or "", "type": "message"})
            continue

        items.append(
            {"role": role, "content": str(content) if content else "", "type": "message"}
        )

    return items


def _translate_assistant_blocks_to_responses(blocks: list[dict]) -> list[dict]:
    """Convert assistant content blocks to Responses API input items.

    Tool-use blocks become ``function_call`` items; text blocks become a single
    assistant message item. The ordering matters less here than in
    chat.completions because each item is independent.
    """
    text_parts: list[str] = []
    tool_calls: list[dict] = []

    for block in blocks:
        if block.get("type") == "text":
            text_parts.append(block["text"])
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "type": "function_call",
                    "call_id": block["id"],
                    "name": block["name"],
                    "arguments": json.dumps(block.get("input", {})),
                }
            )

    items: list[dict] = []
    if text_parts:
        items.append(
            {"role": "assistant", "content": "\n".join(text_parts), "type": "message"}
        )
    items.extend(tool_calls)
    return items


def _translate_user_blocks_to_responses(
    blocks: list[dict], supports_vision: bool = True
) -> list[dict]:
    """Convert user content blocks (text, tool_result, image) to Responses API items."""
    result: list[dict] = []
    content_parts: list[dict] = []

    for block in blocks:
        if block.get("type") == "tool_result":
            # Flush any accumulated content parts first as a user message.
            if content_parts:
                result.append(_user_message_from_parts(content_parts))
                content_parts = []
            tool_content = block.get("content", "")
            if isinstance(tool_content, list):
                tool_content = "\n".join(
                    b.get("text", "") for b in tool_content if b.get("type") == "text"
                )
            result.append(
                {
                    "type": "function_call_output",
                    "call_id": block["tool_use_id"],
                    "output": str(tool_content),
                }
            )
        elif block.get("type") == "text":
            content_parts.append({"type": "input_text", "text": block.get("text", "")})
        elif block.get("type") == "image" and supports_vision:
            source = block.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/png")
                data = source.get("data", "")
                content_parts.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:{media_type};base64,{data}",
                    }
                )

    if content_parts:
        result.append(_user_message_from_parts(content_parts))

    return result


def _user_message_from_parts(parts: list[dict]) -> dict:
    """Build a Responses API user message from accumulated content parts.

    If the message is text-only, flatten to a plain string for compatibility;
    otherwise emit the structured content list (images + text).
    """
    if all(p.get("type") == "input_text" for p in parts):
        return {
            "role": "user",
            "content": "\n".join(p["text"] for p in parts),
            "type": "message",
        }
    return {"role": "user", "content": parts, "type": "message"}


def _native_web_entries_for_flavor(
    flavor: str, native_web_tools: set[str] | None
) -> list[dict]:
    """Build the list of native server-tool entries to append to the tools array.

    - ``flavor="openai"`` (Responses API): ``{"type": "web_search"}`` covers
      both search and fetch (per OpenAI docs, web_search handles fetch implicitly).
    - ``flavor="minds-passthrough"`` (chat.completions): mdb.ai accepts
      ``{"type": "web_search"}`` and ``{"type": "fetch"}`` directly in the
      OpenAI-shaped tools array.
    - ``flavor="openai-compatible-generic"``: never returns native entries —
      these endpoints get the handler-dispatched fallback at the session layer.
    """
    if not native_web_tools:
        return []
    if flavor == "openai":
        # Single Responses API tool covers search + fetch.
        if "web_search" in native_web_tools or "web_fetch" in native_web_tools:
            return [{"type": "web_search"}]
        return []
    if flavor == "minds-passthrough":
        entries: list[dict] = []
        if "web_search" in native_web_tools:
            entries.append({"type": "web_search"})
        if "web_fetch" in native_web_tools:
            entries.append({"type": "fetch"})
        return entries
    return []


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

    # Three flavors distinguish the transport + native-tool behavior. See
    # ``_native_web_entries_for_flavor`` for the per-flavor tool injection rules,
    # and the ``complete``/``stream`` methods for the per-flavor transport split.
    FLAVOR_OPENAI = "openai"  # Direct OpenAI BYOK — uses Responses API.
    FLAVOR_MINDS_PASSTHROUGH = "minds-passthrough"  # mdb.ai — chat.completions w/ native tools.
    FLAVOR_OPENAI_COMPATIBLE_GENERIC = "openai-compatible-generic"  # third-party.

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        ssl_verify: bool = True,
        api_version: str | None = None,
        supports_vision: bool = True,
        vision_format: str = "openai",
        flavor: str = FLAVOR_OPENAI_COMPATIBLE_GENERIC,
        reasoning_effort: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._ssl_verify = ssl_verify
        self._api_version = api_version
        self._supports_vision = supports_vision
        self._flavor = flavor
        self._vision_format = vision_format
        # Opaque effort level. Forwarded as top-level ``reasoning_effort`` on the
        # chat.completions path and as ``reasoning={"effort": ...}`` on the
        # Responses API path. None = the model's default.
        self._reasoning_effort = reasoning_effort
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

    def native_web_tools(self) -> set[str]:
        # BYOK OpenAI exposes web_search via Responses API (which covers fetch
        # implicitly). The mdb.ai passthrough accepts both web_search and fetch
        # directly in the chat.completions tools array. Generic OpenAI-compatible
        # endpoints have no native support — the session falls back to handler
        # ToolDefs (Exa/Brave for search, stdlib HTTP for fetch).
        if self._flavor in (self.FLAVOR_OPENAI, self.FLAVOR_MINDS_PASSTHROUGH):
            return {"web_search", "web_fetch"}
        return set()

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
        native_web_tools: set[str] | None = None,
    ) -> LLMResponse:
        if self._flavor == self.FLAVOR_OPENAI:
            return await self._complete_via_responses(
                model=model,
                system=system,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                native_web_tools=native_web_tools,
            )

        oai_messages = _translate_messages(system, messages, supports_vision=self._supports_vision, vision_format=self._vision_format)

        kwargs = build_chat_completion_kwargs(
            model=model,
            messages=oai_messages,
            max_tokens=max_tokens,
        )
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort
        merged_tools: list[dict] = []
        if tools:
            merged_tools.extend(_translate_tools(tools))
        # Native server-tool entries (mdb.ai passthrough) are appended *raw* so
        # they aren't routed through the function-shape translation.
        merged_tools.extend(_native_web_entries_for_flavor(self._flavor, native_web_tools))
        if merged_tools:
            kwargs["tools"] = merged_tools
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
            if exc.status_code == 401:
                msg = "Invalid API key — check your OpenAI API key configuration."
                raise ConnectionError(msg) from exc
            elif (
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
        native_web_tools: set[str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        if self._flavor == self.FLAVOR_OPENAI:
            async for event in self._stream_via_responses(
                model=model,
                system=system,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                native_web_tools=native_web_tools,
            ):
                yield event
            return

        oai_messages = _translate_messages(system, messages, supports_vision=self._supports_vision, vision_format=self._vision_format)

        kwargs = build_chat_completion_kwargs(
            model=model,
            messages=oai_messages,
            max_tokens=max_tokens,
            stream=True,
        )
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort
        merged_tools: list[dict] = []
        if tools:
            merged_tools.extend(_translate_tools(tools))
        merged_tools.extend(_native_web_entries_for_flavor(self._flavor, native_web_tools))
        if merged_tools:
            kwargs["tools"] = merged_tools
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
            if exc.status_code == 401:
                msg = "Invalid API key — check your OpenAI API key configuration."
                raise ConnectionError(msg) from exc
            elif (
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

    # ─────────────────────────────────────────────────────────────────────
    # Responses API path — used for ``flavor="openai"`` (BYOK direct OpenAI)
    # ─────────────────────────────────────────────────────────────────────

    def _build_responses_kwargs(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict] | None,
        tool_choice: dict | None,
        max_tokens: int,
        native_web_tools: set[str] | None,
    ) -> dict:
        """Common Responses API kwargs for both ``complete`` and ``stream``."""
        responses_input = _translate_messages_to_responses_input(
            messages, supports_vision=self._supports_vision
        )
        kwargs: dict = {
            "model": model,
            "input": responses_input,
            "max_output_tokens": max_tokens,
        }
        if system:
            kwargs["instructions"] = system
        if self._reasoning_effort:
            kwargs["reasoning"] = {"effort": self._reasoning_effort}

        merged_tools: list[dict] = []
        if tools:
            merged_tools.extend(_translate_tools_to_responses(tools))
        merged_tools.extend(_native_web_entries_for_flavor(self._flavor, native_web_tools))
        if merged_tools:
            kwargs["tools"] = merged_tools
        if tool_choice:
            kwargs["tool_choice"] = _translate_tool_choice_to_responses(tool_choice)
        return kwargs

    async def _complete_via_responses(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict] | None,
        tool_choice: dict | None,
        max_tokens: int,
        native_web_tools: set[str] | None,
    ) -> LLMResponse:
        kwargs = self._build_responses_kwargs(
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            native_web_tools=native_web_tools,
        )

        try:
            response = await self._client.responses.create(**kwargs)
        except openai.BadRequestError as exc:
            msg = str(exc).lower()
            if "context_length_exceeded" in msg or "maximum context length" in msg:
                raise ContextOverflowError(str(exc)) from exc
            raise
        except openai.APIStatusError as exc:
            if exc.status_code == 401:
                msg = "Invalid API key — check your OpenAI API key configuration."
                raise ConnectionError(msg) from exc
            elif (
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

        return _parse_response_object(response, model)

    async def _stream_via_responses(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict] | None,
        max_tokens: int,
        native_web_tools: set[str] | None,
    ) -> AsyncIterator[StreamEvent]:
        kwargs = self._build_responses_kwargs(
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            tool_choice=None,  # streaming path does not force tool_choice today
            max_tokens=max_tokens,
            native_web_tools=native_web_tools,
        )
        kwargs["stream"] = True

        content_text = ""
        tool_calls: list[ToolCall] = []
        input_tokens = 0
        output_tokens = 0
        stop_reason: str | None = None

        # Map output_index → in-flight function-call state. Responses API uses
        # a per-output_index stable handle for streaming arguments.
        fc_state: dict[int, dict] = {}

        try:
            stream = await self._client.responses.create(**kwargs)
            async for event in stream:
                etype = getattr(event, "type", "")

                # Text deltas
                if etype == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        content_text += delta
                        yield StreamTextDelta(text=delta)

                # New output item (could be a function_call, server-tool call,
                # or message). We only need to react when a function_call
                # appears so we can emit the StreamToolUseStart with id+name.
                elif etype == "response.output_item.added":
                    item = getattr(event, "item", None)
                    item_type = getattr(item, "type", None)
                    if item_type == "function_call":
                        idx = event.output_index
                        call_id = getattr(item, "call_id", "") or getattr(item, "id", "")
                        name = getattr(item, "name", "") or ""
                        fc_state[idx] = {"call_id": call_id, "name": name, "args_parts": []}
                        if call_id and name:
                            yield StreamToolUseStart(id=call_id, name=name)

                # Function-call argument deltas
                elif etype == "response.function_call_arguments.delta":
                    idx = event.output_index
                    delta = getattr(event, "delta", "")
                    info = fc_state.get(idx)
                    if info is None:
                        # output_item.added didn't surface this call yet — buffer
                        info = {"call_id": "", "name": "", "args_parts": []}
                        fc_state[idx] = info
                    info["args_parts"].append(delta)
                    if info["call_id"]:
                        yield StreamToolUseDelta(id=info["call_id"], json_delta=delta)

                # Function-call arguments complete — finalize this call.
                elif etype == "response.function_call_arguments.done":
                    idx = event.output_index
                    info = fc_state.get(idx)
                    if info is None:
                        continue
                    raw_json = "".join(info["args_parts"]) or getattr(
                        event, "arguments", ""
                    )
                    parsed = json.loads(raw_json) if raw_json else {}
                    tool_calls.append(
                        ToolCall(
                            id=info["call_id"], name=info["name"], input=parsed
                        )
                    )
                    if info["call_id"]:
                        yield StreamToolUseEnd(id=info["call_id"])

                # Final completion event carries the resolved Response object
                # with usage/stop_reason. We trust the structured parse here in
                # case the streamed deltas missed something (e.g. server-tool
                # calls produce text we already streamed but no function call).
                elif etype == "response.completed":
                    final_response = getattr(event, "response", None)
                    if final_response is not None:
                        usage = getattr(final_response, "usage", None)
                        if usage is not None:
                            input_tokens = getattr(usage, "input_tokens", 0) or 0
                            output_tokens = getattr(usage, "output_tokens", 0) or 0
                        stop_reason = getattr(final_response, "status", None)
        except openai.BadRequestError as exc:
            msg = str(exc).lower()
            if "context_length_exceeded" in msg or "maximum context length" in msg:
                raise ContextOverflowError(str(exc)) from exc
            raise
        except openai.APIStatusError as exc:
            if exc.status_code == 401:
                msg = "Invalid API key — check your OpenAI API key configuration."
                raise ConnectionError(msg) from exc
            elif (
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


def _parse_response_object(response, model: str) -> LLMResponse:
    """Convert a Responses API ``Response`` object into our unified ``LLMResponse``.

    The response contains an ``output`` array of items: messages (with
    ``output_text`` content blocks), function calls (with ``call_id``,
    ``name``, ``arguments``), and server-tool calls (web_search etc.) which we
    intentionally drop because their effects are already incorporated into the
    model's text content.
    """
    content_text = ""
    tool_calls: list[ToolCall] = []

    for item in response.output or []:
        item_type = getattr(item, "type", "")
        if item_type == "message":
            for content_block in getattr(item, "content", []) or []:
                if getattr(content_block, "type", "") == "output_text":
                    content_text += getattr(content_block, "text", "") or ""
        elif item_type == "function_call":
            call_id = getattr(item, "call_id", "") or getattr(item, "id", "")
            name = getattr(item, "name", "") or ""
            args_str = getattr(item, "arguments", "") or ""
            try:
                parsed = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                parsed = {}
            tool_calls.append(ToolCall(id=call_id, name=name, input=parsed))
        # Other item types (web_search_call, reasoning, etc.) are skipped —
        # the model's output_text already incorporates their effects.

    usage = getattr(response, "usage", None)
    # `or 0` guards an explicit None (the attr is present but null) — the
    # Responses API returns usage.input_tokens=None on web-search responses,
    # which a bare getattr default does NOT catch. Mirrors the streaming path.
    input_tokens = (getattr(usage, "input_tokens", 0) or 0) if usage else 0
    output_tokens = (getattr(usage, "output_tokens", 0) or 0) if usage else 0

    return LLMResponse(
        content=content_text,
        tool_calls=tool_calls,
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            context_pressure=compute_context_pressure(model, input_tokens),
        ),
        stop_reason=getattr(response, "status", None),
    )
