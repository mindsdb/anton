"""OpenAI Responses-API translation and proxy for the passthrough agent.

Module-name note: this file is intentionally named ``openai.py`` to mirror its
provider. Python 3 uses absolute imports by default, so ``from openai import
...`` below resolves to the top-level OpenAI SDK package, not this module.
Callers reach this module via ``from minds.inference.providers import
openai as ...`` or relative imports.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from openai import APIError as OpenAIAPIError
from openai import APIStatusError as OpenAIAPIStatusError
from openai import AsyncOpenAI
from starlette.responses import JSONResponse, StreamingResponse

from minds.common.logger import setup_logging
from minds.inference.types import PassthroughModelConfig, WebSearchMode
from minds.inference.types import (
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
    "_chat_messages_to_responses_input",
    "_chat_tool_choice_to_responses",
    "_collect_responses_server_artifacts",
    "_get_openai_client",
    "_responses_response_to_chat_completion",
    "_translate_tools_for_openai",
    "proxy_openai",
    "stream_openai_responses_as_chat",
]

logger = setup_logging()


# ---------------------------------------------------------------------------
# Translators (messages, tools, tool_choice, response)
# ---------------------------------------------------------------------------


def _translate_tools_for_openai(
    tools: list[dict] | None,
    web_search_mode: WebSearchMode = WebSearchMode.OPENAI_NATIVE,
) -> list[dict]:
    """Translate generic + chat-completions function tools to OpenAI Responses API shape.

    The Responses API accepts ``{"type": "web_search"}`` directly as a
    ``tools[]`` entry, and uses a flatter function-tool shape than chat
    completions: ``{"type": "function", "name": ..., "parameters": ...}``
    (vs. chat completions' nested ``{"type": "function", "function": {...}}``).

    - Generic ``web_search`` / ``fetch`` → single ``{"type": "web_search"}``
      (deduped — OpenAI's web_search bundles URL fetching, no separate
      fetch tool exists). With ``web_search_mode == "drop"`` they are
      silently omitted instead.
    - Chat-completions function tool → flattened Responses shape, via
      :class:`ChatCompletionsFunctionTool` for typed field access.
    - Anything else passes through unchanged (covers tools the caller
      already shaped for the Responses API).
    """
    if not tools:
        return []

    out: list[dict] = []
    web_search_added = False
    for tool in tools:
        parsed = _classify_tool(tool)
        if isinstance(parsed, GenericWebSearchTool | GenericFetchTool):
            if web_search_mode == WebSearchMode.DROP:
                logger.debug(
                    "Dropping generic %s (web_search_mode=DROP)",
                    parsed.type,
                )
                continue
            if not web_search_added:
                out.append({"type": "web_search"})
                web_search_added = True
            continue
        if isinstance(parsed, ChatCompletionsFunctionTool):
            flat: dict[str, Any] = {
                "type": "function",
                "name": parsed.function.name,
                "parameters": parsed.function.parameters,
            }
            if parsed.function.description is not None:
                flat["description"] = parsed.function.description
            out.append(flat)
            continue
        # Pass through anything we didn't recognize (might be already-Responses-shaped).
        if isinstance(tool, dict):
            out.append(tool)

    logger.debug(
        "Translated tools for OpenAI (Responses)",
        extra={
            "input_count": len(tools),
            "output_count": len(out),
            "web_search_mode": web_search_mode,
        },
    )
    return out


def _chat_messages_to_responses_input(
    messages: list[dict],
) -> tuple[str | None, list[dict]]:
    """Convert chat-completions messages to OpenAI Responses API ``(instructions, input)``.

    System messages collapse into the top-level ``instructions`` string
    (multiple system messages are joined with blank lines). Other roles
    become Responses input items:

    - ``user`` / plain ``assistant`` text → ``{"role": ..., "content": ...}``
    - ``assistant`` with ``tool_calls`` → one ``{"role":"assistant"}`` item
      for any text content followed by a ``{"type":"function_call", ...}``
      item per tool call.
    - ``tool`` (a tool result) → ``{"type":"function_call_output",
      "call_id": ..., "output": ...}``.
    """
    instructions_parts: list[str] = []
    input_items: list[dict] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            if isinstance(content, str) and content:
                instructions_parts.append(content)
            continue

        if role == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": content if isinstance(content, str) else json.dumps(content),
                }
            )
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            text = content if isinstance(content, str) else ""
            if text:
                input_items.append({"role": "assistant", "content": text})
            for tc in tool_calls:
                fn = tc.get("function", {})
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", "{}"),
                    }
                )
            continue

        # user (or any other plain role)
        input_items.append({"role": role, "content": content})

    instructions = "\n\n".join(instructions_parts) if instructions_parts else None
    return instructions, input_items


def _chat_tool_choice_to_responses(tool_choice: str | dict | None) -> str | dict | None:
    """Convert chat-completions ``tool_choice`` to Responses API shape.

    String values (``auto``/``required``/``none``) pass through. Specific
    function selection changes shape:
    ``{"type": "function", "function": {"name": "X"}}`` →
    ``{"type": "function", "name": "X"}``.
    """
    if tool_choice is None or isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        fn = tool_choice.get("function", {})
        name = fn.get("name")
        if name:
            return {"type": "function", "name": name}
    return tool_choice


def _responses_response_to_chat_completion(response: Any, model_name: str) -> dict:
    """Convert an OpenAI Responses API response to a ChatCompletion dict.

    Walks ``response.output`` (list of typed items): ``message`` items'
    ``output_text`` parts become assistant text content; ``function_call``
    items become OpenAI ``tool_calls`` entries. ``web_search_call``,
    ``reasoning``, and other server-side intermediates are intentionally
    not surfaced — clients of ``/v1/chat/completions`` see only the final
    text plus any function calls the model wants the caller to execute.
    """
    text_parts: list[str] = []
    tool_calls: list[dict] = []

    for item in getattr(response, "output", []) or []:
        item_type = getattr(item, "type", None)
        if item_type == "message":
            for part in getattr(item, "content", []) or []:
                if getattr(part, "type", None) == "output_text":
                    text_parts.append(getattr(part, "text", ""))
        elif item_type == "function_call":
            tool_calls.append(
                {
                    "id": getattr(item, "call_id", "") or getattr(item, "id", ""),
                    "type": "function",
                    "function": {
                        "name": getattr(item, "name", ""),
                        "arguments": getattr(item, "arguments", "") or "",
                    },
                }
            )

    text = "".join(text_parts)
    finish_reason = "tool_calls" if tool_calls else "stop"

    message: dict[str, Any] = {"role": "assistant", "content": text or None}
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


# ---------------------------------------------------------------------------
# Client + proxy + streaming
# ---------------------------------------------------------------------------


def _get_openai_client(config: PassthroughModelConfig) -> AsyncOpenAI:
    kwargs: dict[str, Any] = {"api_key": config.api_key}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return AsyncOpenAI(**kwargs)


async def proxy_openai(
    *,
    client: AsyncOpenAI,
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
    """Proxy to OpenAI's Responses API (``/v1/responses``).

    Inbound messages/tools/tool_choice are in chat-completions wire
    format; we translate to the Responses API shape on the way out and
    translate the response back to a ChatCompletion dict on return so
    clients see no difference.

    The caller supplies the SDK ``client`` so tests can inject a fake
    without monkeypatching at construction time.
    """
    instructions, responses_input = _chat_messages_to_responses_input(messages)

    kwargs: dict[str, Any] = {
        "model": config.model_name,
        "input": responses_input,
        "stream": stream,
    }
    if instructions:
        kwargs["instructions"] = instructions

    openai_tools = _translate_tools_for_openai(tools, config.web_search_mode)
    if openai_tools:
        kwargs["tools"] = openai_tools
    # tool_choice handling: pure pass-through, with one exception. OpenAI's
    # native web_search hosted tool is invoked at the model's discretion;
    # it cannot be forced via tool_choice. If the request's only tools are
    # web tools, forwarding tool_choice would be a no-op or upstream error,
    # so drop it. When function tools are also present, pass-through (with
    # the chat-completions → Responses shape conversion).
    if tool_choice is not None and not _only_web_tools(tools):
        kwargs["tool_choice"] = _chat_tool_choice_to_responses(tool_choice)
    elif tool_choice is not None:
        logger.debug(
            "Dropping tool_choice — only web tools present",
            extra={"request_id": request_id, "tool_choice": tool_choice},
        )
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        # Chat completions ``max_tokens`` is ``max_output_tokens`` on Responses.
        kwargs["max_output_tokens"] = max_tokens
    if config.reasoning_effort:
        # Reasoning-capable models (``gpt-5.5`` etc.) accept a top-level
        # ``reasoning={"effort": ...}`` argument. Driven entirely by the
        # alias (``_gpt-5.5-low_`` etc.); not exposed to clients directly.
        kwargs["reasoning"] = {"effort": config.reasoning_effort}

    logger.debug(
        "Calling OpenAI Responses",
        extra={
            "request_id": request_id,
            "model": config.model_name,
            "stream": stream,
            "has_tools": bool(openai_tools),
            "has_instructions": bool(instructions),
            "reasoning_effort": config.reasoning_effort,
        },
    )
    try:
        response = await client.responses.create(**kwargs)
    except OpenAIAPIStatusError as exc:
        logger.error(
            "OpenAI API request failed",
            exc_info=True,
            extra={"model": config.model_name, "stream": stream, "request_id": request_id},
        )
        return JSONResponse(
            content={"error": {"message": exc.message, "type": "api_error", "code": exc.code}},
            status_code=exc.status_code,
        )
    except OpenAIAPIError as exc:
        logger.error(
            "OpenAI API request failed",
            exc_info=True,
            extra={"model": config.model_name, "stream": stream, "request_id": request_id},
        )
        return JSONResponse(
            content={"error": {"message": str(exc), "type": "api_error"}},
            status_code=502,
        )

    if stream:
        return StreamingResponse(
            stream_openai_responses_as_chat(config, usage_box, response, request_id),
            media_type="text/event-stream",
        )
    usage_box.value = (response.usage.input_tokens, response.usage.output_tokens)
    completion = _responses_response_to_chat_completion(response, config.model_name)
    # Capture the assistant message for Langfuse generation.output and any
    # server-side intermediates the translator skipped — both end up on the
    # parent generation as part of the eval-replay surface.
    usage_box.output_payload = completion["choices"][0]["message"]
    usage_box.server_artifacts.extend(_collect_responses_server_artifacts(response))
    return JSONResponse(content=completion)


def _collect_responses_server_artifacts(response: Any) -> list[dict[str, Any]]:
    """Pull server-side intermediates off a Responses API response/event.

    The Responses API returns ``web_search_call`` and ``reasoning`` items
    alongside the regular ``message`` / ``function_call`` items the
    translator surfaces to the client. They are intentionally not part of
    the ChatCompletion-shaped client response — but they carry the model's
    web-search queries and chain-of-thought summaries, which are exactly
    the artifacts we want available on Langfuse for evals and
    troubleshooting. Returns a list of plain dicts (one per intermediate)
    suitable for stuffing into ``metadata["server_artifacts"]``.
    """
    out: list[dict[str, Any]] = []
    for item in getattr(response, "output", []) or []:
        item_type = getattr(item, "type", None)
        if item_type == "web_search_call":
            out.append(
                {
                    "type": "web_search_call",
                    "id": getattr(item, "id", None),
                    "status": getattr(item, "status", None),
                    # Responses' web_search action carries the query under
                    # ``action.query`` on recent SDK versions; fall back to
                    # the top-level ``query`` field for older shapes.
                    "query": getattr(getattr(item, "action", None), "query", None) or getattr(item, "query", None),
                }
            )
        elif item_type == "reasoning":
            out.append(
                {
                    "type": "reasoning",
                    "id": getattr(item, "id", None),
                    # ``summary`` is a list of text parts; flatten to a
                    # single string for readability in the Langfuse UI.
                    "summary": _flatten_reasoning_summary(getattr(item, "summary", None)),
                }
            )
    return out


def _flatten_reasoning_summary(summary: Any) -> str | None:
    """Render the SDK's structured reasoning ``summary`` list as one string."""
    if not summary:
        return None
    parts: list[str] = []
    for entry in summary:
        text = getattr(entry, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts) if parts else None


async def stream_openai_responses_as_chat(
    config: PassthroughModelConfig,
    usage_box: UsageBox,
    stream,
    request_id: str,
):
    """Convert OpenAI Responses streaming events to ChatCompletion SSE chunks.

    Maps the relevant Responses events back to OpenAI ChatCompletion
    ``chat.completion.chunk`` deltas:

    - ``response.created`` / ``response.in_progress`` → initial role chunk
    - ``response.output_item.added`` (for ``function_call`` items) →
      opening ``tool_calls`` delta with ``id`` + ``name``
    - ``response.output_text.delta`` → text content delta
    - ``response.function_call_arguments.delta`` → tool_calls argument delta
    - ``response.completed`` → final usage + finish_reason chunk

    Server-side intermediates (``web_search_call.*``, ``reasoning``, etc.)
    are intentionally not surfaced — clients see only final text and any
    function_calls the caller is meant to execute.
    """
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())
    model = config.model_name
    emit_kwargs = {"completion_id": completion_id, "created": created, "model": model}

    # Map Responses output_index → chat-completion tool_call index/info.
    function_calls_by_output_index: dict[int, dict] = {}
    next_tool_call_index = 0
    sent_role_chunk = False
    finish_reason: str = "stop"
    input_tokens = 0
    output_tokens = 0
    # Mirror of the assistant message we'd emit non-streaming — built up as
    # text + arg deltas arrive so Langfuse can record the full output even
    # though the body runs after the @observe scope closes.
    text_parts: list[str] = []
    tool_calls_accum: dict[int, dict] = {}
    server_artifacts_local: list[dict] = []

    logger.debug(
        "OpenAI Responses stream begin",
        extra={"request_id": request_id, "model": model, "completion_id": completion_id},
    )

    async for event in stream:
        event_type = event.type

        if event_type in ("response.created", "response.in_progress"):
            if not sent_role_chunk:
                yield _emit_chunk(**emit_kwargs, role="assistant", content="")
                sent_role_chunk = True

        elif event_type == "response.output_item.added":
            item = event.item
            if getattr(item, "type", None) == "function_call":
                tc_index = next_tool_call_index
                next_tool_call_index += 1
                call_id = getattr(item, "call_id", "") or getattr(item, "id", "")
                name = getattr(item, "name", "")
                function_calls_by_output_index[event.output_index] = {
                    "tc_index": tc_index,
                    "id": call_id,
                    "name": name,
                }
                tool_calls_accum[tc_index] = {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": ""},
                }
                logger.debug(
                    "Opening tool_call delta",
                    extra={"request_id": request_id, "tc_index": tc_index, "name": name},
                )
                yield _emit_chunk(
                    **emit_kwargs,
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=tc_index,
                            id=call_id,
                            type="function",
                            function=ChoiceDeltaToolCallFunction(name=name, arguments=""),
                        )
                    ],
                )

        elif event_type == "response.output_text.delta":
            text_parts.append(event.delta or "")
            yield _emit_chunk(**emit_kwargs, content=event.delta)

        elif event_type == "response.function_call_arguments.delta":
            tc_info = function_calls_by_output_index.get(event.output_index)
            if tc_info is not None:
                accum = tool_calls_accum.get(tc_info["tc_index"])
                if accum is not None:
                    accum["function"]["arguments"] += event.delta or ""
                yield _emit_chunk(
                    **emit_kwargs,
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=tc_info["tc_index"],
                            function=ChoiceDeltaToolCallFunction(arguments=event.delta),
                        )
                    ],
                )

        elif event_type == "response.completed":
            full = event.response
            if getattr(full, "usage", None):
                input_tokens = full.usage.input_tokens
                output_tokens = full.usage.output_tokens
            # Determine finish reason from the final output items.
            for item in getattr(full, "output", []) or []:
                if getattr(item, "type", None) == "function_call":
                    finish_reason = "tool_calls"
                    break
            # Capture server-side intermediates for Langfuse metadata. The
            # client-facing stream stays clean (these events are not emitted
            # as ChatCompletion chunks), but they're the highest-signal
            # artifacts when grading whether the model grounded correctly.
            server_artifacts_local.extend(_collect_responses_server_artifacts(full))
            logger.debug(
                "OpenAI Responses stream completed",
                extra={
                    "request_id": request_id,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "finish_reason": finish_reason,
                },
            )
            yield _emit_chunk(**emit_kwargs, finish_reason=finish_reason)

        # All other event types (web_search_call.*, reasoning, content_part.*,
        # etc.) are intentionally not forwarded.

    usage_box.value = (input_tokens, output_tokens)
    # Reconstruct the OpenAI-shape assistant message so Langfuse records the
    # full output even though this generator runs after the @observe scope.
    final_text = "".join(text_parts)
    assistant_message: dict[str, Any] = {"role": "assistant", "content": final_text or None}
    final_tool_calls = [tool_calls_accum[i] for i in sorted(tool_calls_accum)] if tool_calls_accum else []
    if final_tool_calls:
        assistant_message["tool_calls"] = final_tool_calls
    usage_box.output_payload = assistant_message
    if server_artifacts_local:
        usage_box.server_artifacts.extend(server_artifacts_local)
    yield "data: [DONE]\n\n"
