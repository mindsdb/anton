"""Anthropic-side translation and proxy for the inference service.

Same module serves direct Anthropic and Fireworks-via-Anthropic-SDK (the
Fireworks Anthropic-compatible endpoint), dispatched by ``config.base_url``.

Module-name note: this file is intentionally named ``anthropic.py`` to mirror
its provider; Python 3 uses absolute imports by default, so ``from anthropic
import ...`` below resolves to the top-level Anthropic SDK package, not this
module. Callers reach this module via inference adapters in
``minds.inference.providers.anthropic_adapter``.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from anthropic import APIError as AnthropicAPIError
from anthropic import APIStatusError as AnthropicAPIStatusError
from anthropic import AsyncAnthropic
from starlette.responses import JSONResponse, StreamingResponse

from minds.common.logger import setup_logging
from minds.common.passthrough_config import PassthroughModelConfig, WebSearchMode
from minds.common.settings.app_settings import get_app_settings
from minds.inference.types import (
    AnthropicToolsTranslation,
    ChatCompletionsFunctionTool,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    GenericFetchTool,
    GenericWebSearchTool,
    UsageBox,
    _classify_tool,
    _emit_chunk,
    _only_web_tools,
)

__all__ = [
    "_anthropic_response_to_openai",
    "_collect_anthropic_server_artifacts",
    "_openai_messages_to_anthropic",
    "_openai_tool_choice_to_anthropic",
    "_translate_tools_for_anthropic",
    "_get_anthropic_client",
    "proxy_anthropic",
    "stream_anthropic_as_openai",
]

logger = setup_logging()
_settings = get_app_settings()


# ---------------------------------------------------------------------------
# Message + tool + tool_choice translators
# ---------------------------------------------------------------------------


def _openai_messages_to_anthropic(
    messages: list[dict],
) -> tuple[str | None, list[dict]]:
    """Convert OpenAI-format messages to Anthropic Messages API format.

    Returns ``(system_prompt, anthropic_messages)``.
    """
    system_prompt: str | None = None
    anthropic_msgs: list[dict] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            system_prompt = content
            continue

        if role == "tool":
            # Anthropic expects tool results as user messages with tool_result content blocks
            anthropic_msgs.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id", ""),
                            "content": content if isinstance(content, str) else json.dumps(content),
                        }
                    ],
                }
            )
            continue

        if role == "assistant" and msg.get("tool_calls"):
            # Convert OpenAI tool_calls to Anthropic tool_use content blocks
            content_blocks: list[dict] = []
            if content:
                content_blocks.append({"type": "text", "text": content})
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", str(uuid.uuid4())),
                        "name": fn.get("name", ""),
                        "input": json.loads(args) if isinstance(args, str) else args,
                    }
                )
            anthropic_msgs.append({"role": "assistant", "content": content_blocks})
            continue

        anthropic_msgs.append({"role": role, "content": content})

    return system_prompt, anthropic_msgs


def _openai_tool_choice_to_anthropic(
    tool_choice: str | dict | None,
) -> dict | None:
    """Convert OpenAI tool_choice to Anthropic tool_choice format."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        mapping = {
            "auto": {"type": "auto"},
            "none": None,
            "required": {"type": "any"},
        }
        return mapping.get(tool_choice)
    # {"type": "function", "function": {"name": "..."}}
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function", {})
        name = fn.get("name")
        if name:
            return {"type": "tool", "name": name}
    return {"type": "auto"}


def _anthropic_response_to_openai(response: Any, model_name: str) -> dict:
    """Convert Anthropic Messages response to OpenAI ChatCompletion dict."""
    content_text = ""
    tool_calls: list[dict] = []
    tc_index = 0

    for block in response.content:
        if block.type == "text":
            content_text += block.text
        elif block.type == "tool_use":
            tool_calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                }
            )
            tc_index += 1

    finish_reason = "stop"
    if response.stop_reason == "tool_use":
        finish_reason = "tool_calls"
    elif response.stop_reason == "end_turn":
        finish_reason = "stop"

    message: dict[str, Any] = {"role": "assistant", "content": content_text or None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage = response.usage
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": usage.input_tokens,
            "completion_tokens": usage.output_tokens,
            "total_tokens": usage.input_tokens + usage.output_tokens,
        },
    }


def _collect_anthropic_server_artifacts(content: Any) -> list[dict[str, Any]]:
    """Extract server-side tool blocks from an Anthropic Messages response.

    When the request opts into Anthropic's native web_search / web_fetch
    tools, the response carries ``server_tool_use`` and
    ``web_search_tool_result`` / ``web_fetch_tool_result`` content blocks
    alongside the regular text/tool_use blocks. The client-facing
    translator (``_anthropic_response_to_openai``) intentionally drops them,
    but they're the highest-signal artifacts for evals — capture them as
    plain dicts to attach as Langfuse metadata.
    """
    out: list[dict[str, Any]] = []
    for block in content or []:
        btype = getattr(block, "type", None)
        if btype == "server_tool_use":
            out.append(
                {
                    "type": btype,
                    "id": getattr(block, "id", None),
                    "name": getattr(block, "name", None),
                    "input": getattr(block, "input", None),
                }
            )
        elif btype in ("web_search_tool_result", "web_fetch_tool_result"):
            out.append(
                {
                    "type": btype,
                    "tool_use_id": getattr(block, "tool_use_id", None),
                    # ``content`` here is an SDK model; ``.model_dump()`` if
                    # present, else stringify so the trace stays JSON-safe.
                    "content": _safe_dump(getattr(block, "content", None)),
                }
            )
    return out


def _safe_dump(value: Any) -> Any:
    """Best-effort JSON-safe rendering for Anthropic SDK content blocks."""
    if value is None:
        return None
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:  # pragma: no cover - defensive
            pass
    if isinstance(value, list):
        return [_safe_dump(v) for v in value]
    if isinstance(value, dict | str | int | float | bool):
        return value
    return str(value)


def _translate_tools_for_anthropic(
    tools: list[dict] | None,
    web_search_mode: WebSearchMode = WebSearchMode.ANTHROPIC_NATIVE,
) -> AnthropicToolsTranslation:
    """Translate generic + function tools to Anthropic's native tool format.

    Returns an :class:`AnthropicToolsTranslation`. ``needs_web_fetch_beta``
    is True iff a generic ``fetch`` tool was translated, meaning the caller
    must add the ``anthropic-beta`` header (value from
    ``settings.anthropic.web_fetch_beta_header``) to the request.

    Generic web tools become Anthropic's versioned native types (from
    ``settings.anthropic.web_search_tool_type`` /
    ``settings.anthropic.web_fetch_tool_type``) when
    ``web_search_mode == "anthropic_native"``; with ``"drop"`` they are
    silently omitted (used for Fireworks routes, which share the Anthropic
    transport shape but have no hosted search index). Function tools are
    re-shaped via :func:`ChatCompletionsFunctionTool`'s typed fields;
    unrecognized types are skipped (matches today's behavior).
    """
    result = AnthropicToolsTranslation()
    if not tools:
        return result

    for tool in tools:
        parsed = _classify_tool(tool)
        if isinstance(parsed, GenericWebSearchTool):
            if web_search_mode == WebSearchMode.DROP:
                logger.debug("Dropping generic web_search (web_search_mode=DROP)")
                continue
            result.tools.append({"type": _settings.anthropic.web_search_tool_type, "name": "web_search"})
        elif isinstance(parsed, GenericFetchTool):
            if web_search_mode == WebSearchMode.DROP:
                logger.debug("Dropping generic fetch (web_search_mode=DROP)")
                continue
            result.tools.append({"type": _settings.anthropic.web_fetch_tool_type, "name": "web_fetch"})
            result.needs_web_fetch_beta = True
        elif isinstance(parsed, ChatCompletionsFunctionTool):
            result.tools.append(
                {
                    "name": parsed.function.name,
                    "description": parsed.function.description or "",
                    "input_schema": parsed.function.parameters,
                }
            )
        # else: unrecognized, already logged at debug by _classify_tool.

    logger.debug(
        "Translated tools for Anthropic",
        extra={
            "input_count": len(tools),
            "output_count": len(result.tools),
            "needs_web_fetch_beta": result.needs_web_fetch_beta,
            "web_search_mode": web_search_mode,
        },
    )
    return result


# ---------------------------------------------------------------------------
# Client + proxy + streaming
# ---------------------------------------------------------------------------


def _get_anthropic_client(config: PassthroughModelConfig) -> AsyncAnthropic:
    kwargs: dict[str, Any] = {"api_key": config.api_key}
    # base_url is set for Anthropic-compatible proxies (e.g., Fireworks);
    # leave unset to use the SDK default for direct Anthropic.
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return AsyncAnthropic(**kwargs)


async def proxy_anthropic(
    *,
    client: AsyncAnthropic,
    config: PassthroughModelConfig,
    usage_box: UsageBox,
    messages: list[dict],
    stream: bool,
    request_id: str,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> StreamingResponse | JSONResponse:
    """Proxy a chat-completions request to Anthropic's Messages API.

    Translates inbound OpenAI-shape messages/tools/tool_choice to Anthropic
    native format and translates the response back to a ChatCompletion dict
    on return so clients see no difference.

    The caller supplies the SDK ``client`` so tests can inject a fake without
    monkeypatching the SDK at the construction site.
    """
    system_prompt, anthropic_msgs = _openai_messages_to_anthropic(messages)

    kwargs: dict[str, Any] = {
        "model": config.model_name,
        "messages": anthropic_msgs,
        "max_tokens": max_tokens or 16384,
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    anthropic = _translate_tools_for_anthropic(tools, config.web_search_mode)
    if anthropic.tools:
        kwargs["tools"] = anthropic.tools
    if anthropic.needs_web_fetch_beta:
        # Anthropic's web_fetch tool is currently behind a beta header
        # (value configured via ANTHROPIC__WEB_FETCH_BETA_HEADER).
        kwargs["extra_headers"] = {"anthropic-beta": _settings.anthropic.web_fetch_beta_header}

    # tool_choice handling: pure pass-through, with one exception. Native
    # server-side tools (web_search_20250305 / web_fetch_20250910) cannot
    # be forced or pinned via tool_choice on Anthropic — they're invoked at
    # the model's discretion. If the request's only tools are web tools,
    # forwarding e.g. tool_choice="required" would be a no-op or upstream
    # error, so drop it. When function tools are also present, pass-through.
    effective_tool_choice = None if _only_web_tools(tools) else tool_choice
    if tool_choice is not None and effective_tool_choice is None:
        logger.debug(
            "Dropping tool_choice — only web tools present",
            extra={"request_id": request_id, "tool_choice": tool_choice},
        )
    anthropic_tc = _openai_tool_choice_to_anthropic(effective_tool_choice)
    if anthropic_tc is not None:
        kwargs["tool_choice"] = anthropic_tc
    if temperature is not None:
        kwargs["temperature"] = temperature
    if stream:
        kwargs["stream"] = True

    logger.debug(
        "Calling Anthropic messages",
        extra={
            "request_id": request_id,
            "model": config.model_name,
            "stream": stream,
            "has_tools": bool(anthropic.tools),
            "needs_web_fetch_beta": anthropic.needs_web_fetch_beta,
        },
    )
    try:
        response = await client.messages.create(**kwargs)
    except AnthropicAPIStatusError as exc:
        logger.error(
            "Anthropic API request failed",
            exc_info=True,
            extra={"model": config.model_name, "stream": stream, "request_id": request_id},
        )
        return JSONResponse(
            content={"error": {"message": exc.message, "type": "api_error", "code": exc.status_code}},
            status_code=exc.status_code,
        )
    except AnthropicAPIError as exc:
        logger.error(
            "Anthropic API request failed",
            exc_info=True,
            extra={"model": config.model_name, "stream": stream, "request_id": request_id},
        )
        return JSONResponse(
            content={"error": {"message": str(exc), "type": "api_error"}},
            status_code=502,
        )

    if stream:
        return StreamingResponse(
            stream_anthropic_as_openai(config, usage_box, response, request_id),
            media_type="text/event-stream",
        )
    usage_box.value = (response.usage.input_tokens, response.usage.output_tokens)
    completion = _anthropic_response_to_openai(response, config.model_name)
    usage_box.output_payload = completion["choices"][0]["message"]
    # Server-side ``web_search`` / ``web_fetch`` content blocks live on the
    # response itself for non-streaming requests; capture them as artifacts.
    usage_box.server_artifacts.extend(_collect_anthropic_server_artifacts(response.content))
    return JSONResponse(content=completion)


async def stream_anthropic_as_openai(
    config: PassthroughModelConfig,
    usage_box: UsageBox,
    stream,
    request_id: str,
):
    """Convert Anthropic streaming events to OpenAI SSE format on the fly.

    Note on native server-side tools: when generic ``web_search`` /
    ``fetch`` are translated to Anthropic's ``web_search_20250305`` /
    ``web_fetch_20250910``, the streaming response contains
    ``server_tool_use`` and ``web_search_tool_result`` /
    ``web_fetch_tool_result`` content blocks for the model's intermediate
    search/fetch artifacts. We intentionally do not surface these in the
    OpenAI-shaped output — clients see only the model's final ``text``
    deltas (with citations baked into the text). Surfacing structured
    citations / annotations is a possible follow-up.
    """
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())
    model = config.model_name
    emit_kwargs = {"completion_id": completion_id, "created": created, "model": model}

    input_tokens = 0
    output_tokens = 0

    # Track tool calls in progress
    current_tool_calls: dict[int, dict] = {}
    tool_call_index = 0
    # Mirror of the assistant message we'd emit non-streaming, plus server-side
    # tool blocks (web_search / web_fetch) we want on the Langfuse trace but
    # not in the client-facing stream.
    text_parts: list[str] = []
    tool_calls_accum: dict[int, dict] = {}
    server_artifacts_local: list[dict] = []
    # Track which content block index is the active server tool block so the
    # streamed input_json_delta on it lands on the right artifact entry.
    server_tool_block: dict[str, Any] | None = None

    logger.debug(
        "Anthropic stream begin",
        extra={"request_id": request_id, "model": model, "completion_id": completion_id},
    )

    async for event in stream:
        event_type = event.type

        if event_type == "message_start":
            if hasattr(event.message, "usage"):
                input_tokens = event.message.usage.input_tokens
            yield _emit_chunk(**emit_kwargs, role="assistant", content="")

        elif event_type == "content_block_start":
            block = event.content_block
            if block.type == "tool_use":
                current_tool_calls[tool_call_index] = {"id": block.id, "name": block.name}
                tool_calls_accum[tool_call_index] = {
                    "id": block.id,
                    "type": "function",
                    "function": {"name": block.name, "arguments": ""},
                }
                logger.debug(
                    "Opening Anthropic tool_use",
                    extra={"request_id": request_id, "tc_index": tool_call_index, "name": block.name},
                )
                yield _emit_chunk(
                    **emit_kwargs,
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=tool_call_index,
                            id=block.id,
                            type="function",
                            function=ChoiceDeltaToolCallFunction(name=block.name, arguments=""),
                        )
                    ],
                )
            elif block.type in ("server_tool_use", "web_search_tool_result", "web_fetch_tool_result"):
                server_tool_block = {
                    "type": block.type,
                    "id": getattr(block, "id", None),
                    "name": getattr(block, "name", None),
                    "input": "",
                }
                server_artifacts_local.append(server_tool_block)

        elif event_type == "content_block_delta":
            delta = event.delta
            if delta.type == "text_delta":
                text_parts.append(delta.text or "")
                yield _emit_chunk(**emit_kwargs, content=delta.text)
            elif delta.type == "input_json_delta":
                if server_tool_block is not None:
                    # Accumulate the server-side tool's input JSON for the trace.
                    server_tool_block["input"] += delta.partial_json or ""
                else:
                    accum = tool_calls_accum.get(tool_call_index)
                    if accum is not None:
                        accum["function"]["arguments"] += delta.partial_json or ""
                    yield _emit_chunk(
                        **emit_kwargs,
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=tool_call_index,
                                function=ChoiceDeltaToolCallFunction(arguments=delta.partial_json),
                            )
                        ],
                    )

        elif event_type == "content_block_stop":
            if server_tool_block is not None:
                server_tool_block = None
            elif tool_call_index in current_tool_calls:
                tool_call_index += 1

        elif event_type == "message_delta":
            if hasattr(event, "usage") and event.usage:
                output_tokens = event.usage.output_tokens
            finish_reason = "stop"
            if hasattr(event.delta, "stop_reason") and event.delta.stop_reason == "tool_use":
                finish_reason = "tool_calls"
            logger.debug(
                "Anthropic stream completed",
                extra={
                    "request_id": request_id,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "finish_reason": finish_reason,
                },
            )
            yield _emit_chunk(**emit_kwargs, finish_reason=finish_reason)

        elif event_type == "message_stop":
            pass

    usage_box.value = (input_tokens, output_tokens)
    final_text = "".join(text_parts)
    assistant_message: dict[str, Any] = {"role": "assistant", "content": final_text or None}
    final_tool_calls = [tool_calls_accum[i] for i in sorted(tool_calls_accum)] if tool_calls_accum else []
    if final_tool_calls:
        assistant_message["tool_calls"] = final_tool_calls
    usage_box.output_payload = assistant_message
    if server_artifacts_local:
        usage_box.server_artifacts.extend(server_artifacts_local)
    yield "data: [DONE]\n\n"
