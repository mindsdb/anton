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


class AnthropicProvider(LLMProvider):
    name: str = "anthropic"

    def native_web_tools(self) -> set[str]:
        # Anthropic's Messages API ships both server-side web_search and
        # web_fetch tools; we route both through the provider when enabled.
        return {"web_search", "web_fetch"}

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
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
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        max_tokens: int = 4096,
        native_web_tools: set[str] | None = None,
    ) -> LLMResponse:
        web_entries, beta_headers = _build_native_web_tools(native_web_tools)
        merged_tools = list(tools or []) + web_entries

        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if merged_tools:
            kwargs["tools"] = merged_tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
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

        input_tokens = response.usage.input_tokens
        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=response.usage.output_tokens,
                context_pressure=compute_context_pressure(model, input_tokens),
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
        native_web_tools: set[str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        web_entries, beta_headers = _build_native_web_tools(native_web_tools)
        merged_tools = list(tools or []) + web_entries

        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if merged_tools:
            kwargs["tools"] = merged_tools
        if beta_headers:
            kwargs["extra_headers"] = {"anthropic-beta": ",".join(beta_headers)}

        content_text = ""
        tool_calls: list[ToolCall] = []
        input_tokens = 0
        output_tokens = 0
        stop_reason: str | None = None

        # Track content blocks by index for tool correlation
        blocks: dict[int, dict] = {}

        try:
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
        except anthropic.APIConnectionError as exc:
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
