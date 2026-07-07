from __future__ import annotations

import json
from collections.abc import AsyncIterator

import anthropic

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
    SystemPrompt,
    ToolCall,
    Usage,
    compute_context_pressure,
)

# Native server-side web tool type strings exposed by the Anthropic Messages API.
# The model invokes these inside the provider — Anton's tool-dispatch loop never
# sees a tool_use for them; the model's final text content already incorporates
# the search/fetch results. Bump these constants when newer revisions ship.
ANTHROPIC_WEB_SEARCH_TOOL_TYPE = "web_search_20250305"
ANTHROPIC_WEB_FETCH_TOOL_TYPE = "web_fetch_20250910"
# web_fetch is gated behind a beta header; web_search is GA and needs no header.
ANTHROPIC_WEB_FETCH_BETA_HEADER = "web-fetch-2025-09-10"


def _build_native_web_tools(
    native_web_tools: set[str] | None,
) -> tuple[list[dict], list[str]]:
    """Translate the unified web-tool set into Anthropic server-tool entries.

    Returns ``(tool_entries, beta_headers)`` — entries to append to the
    Messages API ``tools`` array, and any ``anthropic-beta`` header values that
    must be set for the call.
    """
    if not native_web_tools:
        return [], []
    entries: list[dict] = []
    beta: list[str] = []
    if "web_search" in native_web_tools:
        entries.append({"type": ANTHROPIC_WEB_SEARCH_TOOL_TYPE, "name": "web_search"})
    if "web_fetch" in native_web_tools:
        entries.append({"type": ANTHROPIC_WEB_FETCH_TOOL_TYPE, "name": "web_fetch"})
        beta.append(ANTHROPIC_WEB_FETCH_BETA_HEADER)
    return entries, beta


_CACHE_MARKER = {"type": "ephemeral"}
# Block types cache_control may legally sit on in a message's content array.
_CACHEABLE_BLOCK_TYPES = {"text", "tool_result", "image"}


def _system_param(system: str | SystemPrompt) -> str | list[dict]:
    """SystemPrompt → content blocks with a cache marker on the stable prefix.

    The marker caches everything before it in the request (tools + the stable
    system block); the volatile tail rides after it, uncached, so the live
    clock and memory snapshot never invalidate the prefix. Plain strings pass
    through unchanged (uncached, pre-existing behavior).
    """
    if not isinstance(system, SystemPrompt):
        return system
    blocks = [{"type": "text", "text": system.stable, "cache_control": _CACHE_MARKER}]
    if system.volatile:
        blocks.append({"type": "text", "text": system.volatile})
    return blocks


def _mark_history_for_cache(messages: list[dict]) -> list[dict]:
    """Cache marker on the final content block of the last message.

    Each call marks its own last message; Anthropic still hits the entries
    created at earlier boundaries, so history caches incrementally — only the
    newest exchange is fresh each round. Copy-on-write: the caller's message
    list (the session's live history) is never mutated. Anything unexpected →
    return the input unchanged; caching is best-effort.
    """
    if not messages:
        return messages
    last = messages[-1]
    content = last.get("content") if isinstance(last, dict) else None
    marked_content = None
    if isinstance(content, str) and content:
        marked_content = [
            {"type": "text", "text": content, "cache_control": _CACHE_MARKER}
        ]
    elif isinstance(content, list) and content and isinstance(content[-1], dict):
        if content[-1].get("type") in _CACHEABLE_BLOCK_TYPES:
            marked_content = content[:-1] + [
                {**content[-1], "cache_control": _CACHE_MARKER}
            ]
    if marked_content is None:
        return messages
    return messages[:-1] + [{**last, "content": marked_content}]


def _usage_from(model: str, api_usage, input_tokens: int, output_tokens: int) -> Usage:
    """Build a Usage with cache stats; pressure uses the TOTAL context size.

    With caching active the API's ``input_tokens`` excludes cached tokens, so
    compaction decisions must add the cache read/write counts back in. Cache
    fields are coerced through ``isinstance(int)`` — gateways may omit them,
    return null, or (in tests) hand back mock objects.
    """
    def _int_or_zero(value) -> int:
        return value if isinstance(value, int) else 0

    cache_read = _int_or_zero(getattr(api_usage, "cache_read_input_tokens", 0))
    cache_creation = _int_or_zero(getattr(api_usage, "cache_creation_input_tokens", 0))
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        context_pressure=compute_context_pressure(
            model, (input_tokens or 0) + cache_read + cache_creation
        ),
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation,
    )


class AnthropicProvider(LLMProvider):
    name: str = "anthropic"

    def native_web_tools(self) -> set[str]:
        # Anthropic's Messages API ships both server-side web_search and
        # web_fetch tools; we route both through the provider when enabled.
        return {"web_search", "web_fetch"}

    def __init__(
        self, api_key: str | None = None, reasoning_effort: str | None = None
    ) -> None:
        self._api_key = api_key
        # Opaque effort level forwarded as ``output_config={"effort": ...}`` on
        # every call when set (None = the model's default). Sent via ``extra_body``
        # rather than a typed kwarg so it doesn't depend on the resolved
        # ``anthropic`` SDK version exposing ``output_config`` as a parameter
        # (the Stainless client rejects unknown kwargs, which would otherwise be a
        # hard TypeError on older SDKs the moment an effort is set).
        self._reasoning_effort = reasoning_effort
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = anthropic.AsyncAnthropic(**kwargs)

    def export_connection_info(self) -> ProviderConnectionInfo:
        return ProviderConnectionInfo(provider=self.name, api_key=self._api_key)

    async def complete(
        self,
        *,
        model: str,
        system: str | SystemPrompt,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        max_tokens: int = 4096,
        native_web_tools: set[str] | None = None,
    ) -> LLMResponse:
        web_entries, beta_headers = _build_native_web_tools(native_web_tools)
        merged_tools = list(tools or []) + web_entries

        use_cache = isinstance(system, SystemPrompt)
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "system": _system_param(system),
            "messages": _mark_history_for_cache(messages) if use_cache else messages,
        }
        if merged_tools:
            kwargs["tools"] = merged_tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        if self._reasoning_effort:
            kwargs["extra_body"] = {"output_config": {"effort": self._reasoning_effort}}
        if beta_headers:
            # Anthropic accepts a comma-separated list of beta features.
            kwargs["extra_headers"] = {"anthropic-beta": ",".join(beta_headers)}

        try:
            response = await self._client.messages.create(**kwargs)
        except anthropic.BadRequestError as exc:
            msg = str(exc).lower()
            if "prompt is too long" in msg or "context limit" in msg:
                raise ContextOverflowError(str(exc)) from exc
            raise
        except anthropic.APIStatusError as exc:
            if exc.status_code == 401:
                msg = "Invalid API key — check your ANTHROPIC_API_KEY environment variable."
                raise ConnectionError(msg) from exc
            elif (
                exc.status_code == 429
                and isinstance(exc.body, dict)
                and exc.body.get("detail")
            ):
                msg = f"Server returned 429 — {exc.body['detail']}"
                msg += " Visit https://console.mindshub.ai to upgrade or to top up your tokens."
                from .provider import TokenLimitExceeded

                raise TokenLimitExceeded(msg) from exc
            else:
                msg = f"Server returned {exc.status_code} — the LLM endpoint may be temporarily unavailable. Try again in a moment."
            raise ConnectionError(msg) from exc
        except anthropic.APIConnectionError as exc:
            raise ConnectionError(
                "Could not reach the LLM server — check your connection or try again in a moment."
            ) from exc

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
            usage=_usage_from(
                model,
                response.usage,
                response.usage.input_tokens,
                response.usage.output_tokens,
            ),
            stop_reason=response.stop_reason,
        )

    async def stream(
        self,
        *,
        model: str,
        system: str | SystemPrompt,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        native_web_tools: set[str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        web_entries, beta_headers = _build_native_web_tools(native_web_tools)
        merged_tools = list(tools or []) + web_entries

        use_cache = isinstance(system, SystemPrompt)
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "system": _system_param(system),
            "messages": _mark_history_for_cache(messages) if use_cache else messages,
        }
        if merged_tools:
            kwargs["tools"] = merged_tools
        if self._reasoning_effort:
            kwargs["extra_body"] = {"output_config": {"effort": self._reasoning_effort}}
        if beta_headers:
            kwargs["extra_headers"] = {"anthropic-beta": ",".join(beta_headers)}

        content_text = ""
        tool_calls: list[ToolCall] = []
        input_tokens = 0
        output_tokens = 0
        stop_reason: str | None = None
        start_usage = None

        # Track content blocks by index for tool correlation
        blocks: dict[int, dict] = {}

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if event.type == "message_start":
                        usage = event.message.usage
                        start_usage = usage
                        input_tokens = usage.input_tokens
                        output_tokens = getattr(usage, "output_tokens", 0)

                    elif event.type == "content_block_start":
                        idx = event.index
                        block = event.content_block
                        if block.type == "tool_use":
                            blocks[idx] = {
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "json_parts": [],
                            }
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
                                yield StreamToolUseDelta(
                                    id=info["id"], json_delta=delta.partial_json
                                )

                    elif event.type == "content_block_stop":
                        idx = event.index
                        info = blocks.get(idx, {})
                        if info.get("type") == "tool_use":
                            raw_json = "".join(info["json_parts"])
                            # safe_parse_tool_input never raises. It
                            # returns (parsed_dict, parse_error). When
                            # parse_error is set, the session
                            # dispatcher short-circuits with a tool
                            # result asking the LLM to re-emit a clean
                            # call — that recovery happens via the
                            # tool_use/tool_result protocol the LLM
                            # already understands, so it doesn't need
                            # to escalate to a session-level retry.
                            parsed_input, parse_error = safe_parse_tool_input(raw_json)
                            tool_calls.append(
                                ToolCall(
                                    id=info["id"], name=info["name"], input=parsed_input,
                                    parse_error=parse_error,
                                )
                            )
                            yield StreamToolUseEnd(id=info["id"])

                    elif event.type == "message_delta":
                        stop_reason = event.delta.stop_reason
                        output_tokens = event.usage.output_tokens
        except anthropic.BadRequestError as exc:
            msg = str(exc).lower()
            if "prompt is too long" in msg or "context limit" in msg:
                raise ContextOverflowError(str(exc)) from exc
            raise
        except anthropic.APIStatusError as exc:
            if exc.status_code == 401:
                msg = "Invalid API key — check your ANTHROPIC_API_KEY environment variable."
                raise ConnectionError(msg) from exc
            elif (
                exc.status_code == 429
                and isinstance(exc.body, dict)
                and exc.body.get("detail")
            ):
                msg = f"Server returned 429 — {exc.body['detail']}"
                msg += " Visit https://console.mindshub.ai to upgrade or to top up your tokens."
                from .provider import TokenLimitExceeded

                raise TokenLimitExceeded(msg) from exc
            else:
                msg = f"Server returned {exc.status_code} — the LLM endpoint may be temporarily unavailable. Try again in a moment."
            raise ConnectionError(msg) from exc
        except anthropic.APIConnectionError as exc:
            raise ConnectionError(
                "Could not reach the LLM server — check your connection or try again in a moment."
            ) from exc

        yield StreamComplete(
            response=LLMResponse(
                content=content_text,
                tool_calls=tool_calls,
                usage=_usage_from(model, start_usage, input_tokens, output_tokens),
                stop_reason=stop_reason,
            )
        )
