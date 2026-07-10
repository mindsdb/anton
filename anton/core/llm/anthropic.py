from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import NoReturn

import anthropic

from anton.utils.datasources import scrub_credentials

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
    TokenLimitExceeded,
    ToolCall,
    TransientProviderError,
    Usage,
    classify_transient,
    compute_context_pressure,
)

logger = logging.getLogger(__name__)


def _raise_for_status_error(exc: anthropic.APIStatusError, *, provider: str = "Anthropic") -> NoReturn:
    """Map an Anthropic HTTP error onto anton's typed/curated exceptions.

    Shared by ``complete`` and ``stream`` so the mapping can't drift (the two
    were byte-identical copy-paste before ENG-673). Order matters — permanent
    failures are classified first; only what's left is offered to the transient
    classifier, and the generic "unavailable" copy is the last resort.

    - 401 → ConnectionError (invalid-key copy; cowork-server keys on this phrase).
    - 429 WITH a quota ``detail`` → TokenLimitExceeded (keeps its own card).
    - overloaded/api_error (incl. the mid-stream HTTP-200 case), 5xx, plain 429
      → TransientProviderError (retryable — see ENG-673).
    - anything else → the generic "temporarily unavailable" ConnectionError.
    """
    if exc.status_code == 401:
        raise ConnectionError(
            "Invalid API key — check your ANTHROPIC_API_KEY environment variable."
        ) from exc

    body = exc.body if isinstance(exc.body, dict) else {}
    if exc.status_code == 429 and body.get("detail"):
        msg = f"Server returned 429 — {body['detail']}"
        msg += " Visit https://console.mindshub.ai to upgrade or to top up your tokens."
        raise TokenLimitExceeded(msg) from exc

    transient = classify_transient(exc.status_code, body, provider=provider)
    if transient is not None:
        logger.warning(
            "transient provider error (%s): status=%s body=%s",
            transient.code, exc.status_code, scrub_credentials(str(exc.body))[:500],
        )
        raise transient from exc

    raise ConnectionError(
        f"Server returned {exc.status_code} — the LLM endpoint may be "
        "temporarily unavailable. Try again in a moment."
    ) from exc

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
            _raise_for_status_error(exc)
        except anthropic.APIConnectionError as exc:
            # Transient, but the SDK already retries connection errors at the
            # transport layer — so classify honestly and fail fast rather than
            # stacking the session budget on top (ENG-673).
            raise TransientProviderError(
                "Could not reach Anthropic — check your connection or try again in a moment.",
                provider="Anthropic", code="connection_error", session_backoff=False,
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
        if self._reasoning_effort:
            kwargs["extra_body"] = {"output_config": {"effort": self._reasoning_effort}}
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
            _raise_for_status_error(exc)
        except anthropic.APIConnectionError as exc:
            # Connection dropped mid-stream — transient, but the SDK retries
            # connection errors at the transport layer, so fail fast rather than
            # adding the session budget on top (ENG-673).
            raise TransientProviderError(
                "Lost the connection to Anthropic mid-response — try again in a moment.",
                provider="Anthropic", code="connection_error", session_backoff=False,
            ) from exc

        # A stream that ended without a stop_reason was cut off before the model
        # finished (a silent truncation — no error event, no message_stop). Raise
        # rather than yield a partial answer as if complete; classify transient so
        # the honest message surfaces and the turn retries quickly, but NOT with
        # the session budget — a persistently-malformed endpoint must fail fast,
        # not loop for 30s (ENG-673).
        if stop_reason is None:
            logger.warning("Anthropic stream ended with no stop_reason — treating as truncated")
            raise TransientProviderError(
                "Anthropic ended the response early — try again in a moment.",
                provider="Anthropic", code="truncated_stream", session_backoff=False,
            )

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
