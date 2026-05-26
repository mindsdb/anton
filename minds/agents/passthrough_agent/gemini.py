"""Google Gemini native-API translation and proxy for the passthrough agent.

Gemini's ``generateContent`` / ``generateContentStream`` uses a different
message + tool vocabulary than OpenAI/Anthropic. The translators below mirror
their OpenAI/Anthropic siblings structurally so the dispatcher stays the same
shape.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from starlette.responses import JSONResponse, StreamingResponse

from minds.agents.passthrough_agent.common import (
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    GenericToolType,
    UsageBox,
    _emit_chunk,
    _only_web_tools,
)
from minds.common.logger import setup_logging
from minds.common.passthrough_config import PassthroughModelConfig, WebSearchMode

__all__ = [
    "_chat_messages_to_gemini",
    "_chat_tool_choice_to_gemini",
    "_gemini_finish_reason_to_openai",
    "_gemini_first_candidate",
    "_gemini_parts_for",
    "_gemini_response_to_openai",
    "_get_gemini_client",
    "_translate_tools_for_gemini",
    "proxy_gemini",
    "stream_gemini_as_openai",
]

logger = setup_logging()


# ---------------------------------------------------------------------------
# Translators
# ---------------------------------------------------------------------------


def _chat_messages_to_gemini(messages: list[dict]) -> tuple[str | None, list]:
    """Convert OpenAI-format messages to ``(system_instruction, contents)`` for Gemini.

    Returns ``(system_instruction, gemini_contents)`` where ``gemini_contents``
    is a list of ``types.Content`` objects. System messages collapse into the
    top-level ``system_instruction`` string (multiple system messages joined
    with blank lines, matching the Responses-API translator).

    Mappings:
    - ``user`` text → ``Content(role="user", parts=[Part.from_text(...)])``
    - ``assistant`` text → ``Content(role="model", parts=[Part.from_text(...)])``
    - ``assistant`` with ``tool_calls`` → ``Content(role="model",
      parts=[Part.from_function_call(name, args)])`` (one part per call;
      preceding text content is preserved as a leading text part)
    - ``tool`` (a tool result) → ``Content(role="user",
      parts=[Part.from_function_response(name, response)])`` — Gemini puts
      tool results on the user role, not a dedicated tool role.
    """
    system_parts: list[str] = []
    gemini_contents: list = []

    # Map tool_call_id → function name so a later tool-result message can
    # name the function it's responding to (Gemini's function_response
    # requires the function name; OpenAI's tool message only carries the
    # call id).
    call_id_to_name: dict[str, str] = {}

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            if isinstance(content, str) and content:
                system_parts.append(content)
            continue

        if role == "tool":
            call_id = msg.get("tool_call_id", "")
            fn_name = call_id_to_name.get(call_id, msg.get("name") or "")
            response_payload = content
            if isinstance(response_payload, str):
                # Gemini expects function_response.response to be a dict.
                # Wrap raw text under a stable key so it's still readable.
                try:
                    parsed = json.loads(response_payload)
                    response_payload = parsed if isinstance(parsed, dict) else {"result": parsed}
                except (TypeError, ValueError):
                    response_payload = {"result": response_payload}
            gemini_contents.append(
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part.from_function_response(name=fn_name, response=response_payload)],
                )
            )
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            parts: list = []
            text = content if isinstance(content, str) else ""
            if text:
                parts.append(genai_types.Part.from_text(text=text))
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args else {}
                    except json.JSONDecodeError:
                        args = {"_raw_arguments": args}
                call_id_to_name[tc.get("id", "")] = name
                parts.append(genai_types.Part.from_function_call(name=name, args=args))
            if parts:
                gemini_contents.append(genai_types.Content(role="model", parts=parts))
            continue

        # user (or any other plain role; treat as user for Gemini).
        if isinstance(content, str):
            gemini_contents.append(genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=content)]))

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, gemini_contents


def _translate_tools_for_gemini(
    tools: list[dict] | None,
    web_search_mode: WebSearchMode = WebSearchMode.GEMINI_GOOGLE_SEARCH,
) -> list:
    """Translate generic + chat-completions function tools to Gemini ``Tool[]``.

    - ``{"type": "function", "function": {...}}`` → one
      ``Tool(function_declarations=[FunctionDeclaration(...)])`` per call.
    - Generic ``web_search`` → ``Tool(google_search=GoogleSearch())`` when
      ``web_search_mode == "gemini_google_search"``; dropped under
      ``"drop"``.
    - Generic ``fetch`` → dropped (Gemini's google_search tool grounds on
      retrieved URLs implicitly; no separate fetch primitive).
    - Unknown types → dropped.
    """
    if not tools:
        return []

    out: list = []
    google_search_added = False
    for tool in tools:
        ttype = tool.get("type") if isinstance(tool, dict) else None
        if ttype == GenericToolType.WEB_SEARCH:
            if web_search_mode != WebSearchMode.GEMINI_GOOGLE_SEARCH:
                continue
            if google_search_added:
                continue
            out.append(genai_types.Tool(google_search=genai_types.GoogleSearch()))
            google_search_added = True
            continue
        if ttype == GenericToolType.FETCH:
            # Gemini has no separate fetch primitive; google_search bundles
            # retrieval. Drop silently regardless of mode.
            continue
        if ttype == "function" and "function" in tool:
            fn = tool["function"]
            decl = genai_types.FunctionDeclaration(
                name=fn.get("name", ""),
                description=fn.get("description", ""),
                # The SDK accepts a JSON-schema dict here; using the dict
                # rather than typing it as genai_types.Schema avoids hand-wiring
                # Schema(type=Type.OBJECT, properties={...}) for every call.
                parameters=fn.get("parameters", {"type": "object", "properties": {}}),
            )
            out.append(genai_types.Tool(function_declarations=[decl]))

    logger.debug(
        "Translated tools for Gemini",
        extra={
            "input_count": len(tools),
            "output_count": len(out),
            "web_search_mode": web_search_mode,
        },
    )
    return out


def _chat_tool_choice_to_gemini(tool_choice: str | dict | None) -> genai_types.ToolConfig | None:
    """Convert OpenAI ``tool_choice`` to a Gemini ``ToolConfig`` (or ``None``).

    Mappings:
    - ``"auto"`` / ``None`` → ``None`` (default behavior; no config emitted)
    - ``"required"`` → mode=ANY (model must call a function)
    - ``"none"`` → mode=NONE (function calling disabled this turn)
    - ``{"type":"function","function":{"name":"X"}}`` → mode=ANY with
      ``allowed_function_names=["X"]``
    """
    if tool_choice is None or tool_choice == "auto":
        return None
    if tool_choice == "required":
        return genai_types.ToolConfig(function_calling_config=genai_types.FunctionCallingConfig(mode="ANY"))
    if tool_choice == "none":
        return genai_types.ToolConfig(function_calling_config=genai_types.FunctionCallingConfig(mode="NONE"))
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        fn = tool_choice.get("function", {})
        name = fn.get("name")
        if name:
            return genai_types.ToolConfig(
                function_calling_config=genai_types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=[name],
                )
            )
    return None


def _gemini_finish_reason_to_openai(
    finish_reason: genai_types.FinishReason | None,
    has_tool_calls: bool,
) -> str:
    """Map Gemini's ``FinishReason`` enum to an OpenAI ``finish_reason`` string.

    Tool-call finishes take precedence: if any function_call parts were
    emitted, surface ``"tool_calls"`` regardless of the upstream value
    (Gemini emits ``STOP`` even when the turn ends with a function call).
    """
    if has_tool_calls:
        return "tool_calls"
    if finish_reason == genai_types.FinishReason.MAX_TOKENS:
        return "length"
    return "stop"


def _gemini_first_candidate(response: genai_types.GenerateContentResponse) -> genai_types.Candidate | None:
    """Return the first candidate from a Gemini response, or None if empty.

    Gemini may return zero candidates (safety filters, etc.); calls into
    this helper consistently rather than indexing ``candidates[0]`` inline.
    """
    candidates = response.candidates or []
    return candidates[0] if candidates else None


def _gemini_parts_for(candidate: genai_types.Candidate | None) -> list[genai_types.Part]:
    """Return parts from a Gemini ``Candidate`` (or an empty list if missing)."""
    if candidate is None or candidate.content is None:
        return []
    return list(candidate.content.parts or [])


def _gemini_response_to_openai(response: genai_types.GenerateContentResponse, model_name: str) -> dict:
    """Convert a Gemini ``GenerateContentResponse`` to a ChatCompletion dict.

    Mirrors the Anthropic translator: walks the first candidate's parts,
    accumulating text and synthesizing OpenAI ``tool_calls`` for any
    ``function_call`` parts. Gemini doesn't emit tool_call ids — synthesize
    ``call_<8 hex>`` so callers can correlate tool result messages back.

    Field access is via the typed :class:`google.genai.types.GenerateContentResponse`
    pydantic model rather than ``getattr`` fallbacks — a missing field at this
    layer is a contract change with the SDK, not a runtime fallback path.
    """
    candidate = _gemini_first_candidate(response)
    text_parts: list[str] = []
    tool_calls: list[dict] = []

    for part in _gemini_parts_for(candidate):
        if part.text:
            text_parts.append(part.text)
        fn_call = part.function_call
        if fn_call is not None:
            args = fn_call.args or {}
            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": fn_call.name or "",
                        "arguments": json.dumps(args),
                    },
                }
            )

    text = "".join(text_parts)
    finish_reason = _gemini_finish_reason_to_openai(
        candidate.finish_reason if candidate else None,
        bool(tool_calls),
    )

    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage_metadata = response.usage_metadata
    prompt_tokens = (usage_metadata.prompt_token_count if usage_metadata is not None else 0) or 0
    completion_tokens = (usage_metadata.candidates_token_count if usage_metadata is not None else 0) or 0

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
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Client + proxy + streaming
# ---------------------------------------------------------------------------


def _get_gemini_client(config: PassthroughModelConfig) -> genai.Client:
    return genai.Client(api_key=config.api_key)


async def proxy_gemini(
    *,
    client: genai.Client,
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
    """Proxy to Gemini's native ``generateContent`` / ``generateContentStream``.

    Inbound messages/tools/tool_choice are in chat-completions wire
    format; we translate to Gemini's ``Content[]`` / ``Tool[]`` shape
    on the way out and translate the response back to a ChatCompletion
    dict on return so clients see no difference.

    The caller supplies the SDK ``client`` so tests can inject a fake.
    """
    system_instruction, contents = _chat_messages_to_gemini(messages)
    gemini_tools = _translate_tools_for_gemini(tools, config.web_search_mode)
    gemini_tool_config = None if _only_web_tools(tools) else _chat_tool_choice_to_gemini(tool_choice)

    config_kwargs: dict[str, Any] = {}
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    if gemini_tools:
        config_kwargs["tools"] = gemini_tools
    if gemini_tool_config is not None:
        config_kwargs["tool_config"] = gemini_tool_config
    if temperature is not None:
        config_kwargs["temperature"] = temperature
    if max_tokens is not None:
        config_kwargs["max_output_tokens"] = max_tokens

    gen_config = genai_types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    common_kwargs: dict[str, Any] = {
        "model": config.model_name,
        "contents": contents,
    }
    if gen_config is not None:
        common_kwargs["config"] = gen_config

    logger.debug(
        "Calling Gemini generate_content",
        extra={
            "request_id": request_id,
            "model": config.model_name,
            "stream": stream,
            "has_tools": bool(gemini_tools),
            "has_system_instruction": bool(system_instruction),
            "tool_config_set": gemini_tool_config is not None,
        },
    )
    try:
        if stream:
            return StreamingResponse(
                stream_gemini_as_openai(config, usage_box, client, common_kwargs, request_id),
                media_type="text/event-stream",
            )
        response = await client.aio.models.generate_content(**common_kwargs)
    except genai_errors.APIError as exc:
        logger.error(
            "Gemini API request failed",
            exc_info=True,
            extra={"model": config.model_name, "stream": stream, "request_id": request_id},
        )
        # ``APIError.code`` is the upstream HTTP status (int) when set.
        status_code = exc.code if isinstance(exc.code, int) else 502
        return JSONResponse(
            content={"error": {"message": str(exc), "type": "api_error"}},
            status_code=status_code,
        )

    usage_metadata = response.usage_metadata
    usage_box.value = (
        (usage_metadata.prompt_token_count if usage_metadata is not None else 0) or 0,
        (usage_metadata.candidates_token_count if usage_metadata is not None else 0) or 0,
    )
    completion = _gemini_response_to_openai(response, config.model_name)
    usage_box.output_payload = completion["choices"][0]["message"]
    return JSONResponse(content=completion)


async def stream_gemini_as_openai(
    config: PassthroughModelConfig,
    usage_box: UsageBox,
    client,
    common_kwargs: dict,
    request_id: str,
):
    """Convert Gemini streaming chunks to OpenAI ChatCompletion SSE format.

    Gemini emits each ``function_call`` part atomically (full args at
    once) rather than as incremental argument deltas, so each
    function_call part is forwarded as a single ``delta.tool_calls``
    chunk carrying the entire ``arguments`` JSON. Text parts forward
    as individual ``delta.content`` chunks. Per-chunk
    ``usage_metadata`` is captured from the final chunk.
    """
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())
    model = config.model_name
    emit_kwargs = {"completion_id": completion_id, "created": created, "model": model}

    sent_role_chunk = False
    tool_call_index = 0
    tool_calls_emitted = False
    prompt_tokens = 0
    completion_tokens = 0
    last_finish_reason: genai_types.FinishReason | None = None
    text_parts: list[str] = []
    tool_calls_accum: list[dict] = []

    def _role_chunk_sse() -> str | None:
        nonlocal sent_role_chunk
        if sent_role_chunk:
            return None
        sent_role_chunk = True
        return _emit_chunk(**emit_kwargs, role="assistant", content="")

    logger.debug(
        "Gemini stream begin",
        extra={"request_id": request_id, "model": model, "completion_id": completion_id},
    )

    try:
        stream = await client.aio.models.generate_content_stream(**common_kwargs)
        async for chunk in stream:
            role_chunk = _role_chunk_sse()
            if role_chunk is not None:
                yield role_chunk

            candidate = _gemini_first_candidate(chunk)
            if candidate is None:
                continue
            if candidate.finish_reason is not None:
                last_finish_reason = candidate.finish_reason

            for part in _gemini_parts_for(candidate):
                if part.text:
                    text_parts.append(part.text)
                    yield _emit_chunk(**emit_kwargs, content=part.text)

                fn_call = part.function_call
                if fn_call is not None:
                    tool_calls_emitted = True
                    args = fn_call.args or {}
                    call_id = f"call_{uuid.uuid4().hex[:8]}"
                    args_json = json.dumps(args)
                    tool_calls_accum.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": fn_call.name or "", "arguments": args_json},
                        }
                    )
                    # Whole tool_call atomically — Gemini doesn't stream
                    # incremental argument deltas.
                    yield _emit_chunk(
                        **emit_kwargs,
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=tool_call_index,
                                id=call_id,
                                type="function",
                                function=ChoiceDeltaToolCallFunction(
                                    name=fn_call.name or "",
                                    arguments=args_json,
                                ),
                            )
                        ],
                    )
                    tool_call_index += 1

            # Gemini's streaming usage is cumulative; latest chunk wins.
            usage_metadata = chunk.usage_metadata
            if usage_metadata is not None:
                if usage_metadata.prompt_token_count is not None:
                    prompt_tokens = usage_metadata.prompt_token_count
                if usage_metadata.candidates_token_count is not None:
                    completion_tokens = usage_metadata.candidates_token_count
    except Exception:
        logger.error(
            "Gemini streaming request failed",
            exc_info=True,
            extra={"model": config.model_name, "request_id": request_id},
        )
        raise

    finish_reason = _gemini_finish_reason_to_openai(last_finish_reason, tool_calls_emitted)
    # If nothing came through at all, still emit a role chunk so the
    # client sees a valid (empty) completion rather than a half-stream.
    if not sent_role_chunk:
        empty_chunk = _role_chunk_sse()
        if empty_chunk is not None:
            yield empty_chunk
    logger.debug(
        "Gemini stream completed",
        extra={
            "request_id": request_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "finish_reason": finish_reason,
        },
    )
    yield _emit_chunk(**emit_kwargs, finish_reason=finish_reason)

    usage_box.value = (prompt_tokens, completion_tokens)
    final_text = "".join(text_parts)
    assistant_message: dict[str, Any] = {"role": "assistant", "content": final_text or None}
    if tool_calls_accum:
        assistant_message["tool_calls"] = tool_calls_accum
    usage_box.output_payload = assistant_message
    yield "data: [DONE]\n\n"
