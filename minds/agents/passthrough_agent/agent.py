"""Passthrough LLM agent — proxies requests to upstream OpenAI / Anthropic / Gemini.

Routing is driven by ``self.config.api_kind``:

- ``openai_responses`` — OpenAI Responses API (also covers any vendor exposing
  a Responses-compatible surface).
- ``anthropic_messages`` — Anthropic Messages API. Same path also serves
  Fireworks.ai when its ``base_url`` is set on the config (Fireworks exposes
  an Anthropic-compatible endpoint for hosted open-source models).
- ``gemini_native`` — Google Gemini ``generateContent`` /
  ``generateContentStream`` via ``google-genai``.

Tool translation per destination is gated by ``self.config.web_search_mode``;
provider-specific native search hooks in for OpenAI/Anthropic/Gemini, and
``"drop"`` silently strips generic ``web_search``/``fetch`` entries (used for
Fireworks today, which has no hosted search index).
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from starlette.responses import JSONResponse, StreamingResponse

from minds.common.logger import setup_logging
from minds.common.passthrough_config import PassthroughModelConfig
from minds.schemas.chat import Message, Role

logger = setup_logging()


# ---------------------------------------------------------------------------
# Format converters
# ---------------------------------------------------------------------------


def _messages_to_dicts(messages: list[Message]) -> list[dict]:
    """Convert our Message objects to plain dicts for the OpenAI client."""
    out: list[dict] = []
    for m in messages:
        d: dict[str, Any] = {"role": m.role.value, "content": m.content or ""}
        if m.tool_calls is not None:
            d["tool_calls"] = m.tool_calls
        if m.tool_call_id is not None:
            d["tool_call_id"] = m.tool_call_id
        if m.name is not None:
            d["name"] = m.name
        out.append(d)
    return out


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


def _openai_tools_to_anthropic(tools: list[dict]) -> list[dict]:
    """Convert OpenAI function-calling tools to Anthropic tools format."""
    anthropic_tools: list[dict] = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        fn = tool["function"]
        anthropic_tools.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return anthropic_tools


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


# ---------------------------------------------------------------------------
# Native web-tool support
# ---------------------------------------------------------------------------
#
# Clients may opt into provider-native server-side web tools via a uniform
# generic shape in the request's ``tools`` array, regardless of which
# provider the passthrough config resolves to:
#
#     {"type": "web_search"}   # → Anthropic: tools=[{"type":"web_search_20250305", ...}]
#                              # → OpenAI: tools=[{"type":"web_search"}] on Responses API
#     {"type": "fetch"}        # → Anthropic: tools=[{"type":"web_fetch_20250910", ...}]
#                              #   (requires anthropic-beta: web-fetch-2025-09-10 header)
#                              # → OpenAI: bundled into web_search (no separate fetch
#                              #   tool exists), so requesting fetch alone still emits
#                              #   a single web_search.
#
# Note on the OpenAI request shape: this proxy talks to OpenAI's **Responses
# API** (``/v1/responses``) for the OpenAI provider, not chat-completions.
# Chat-completions only supports native web search on the deprecated
# ``gpt-4o-search-preview`` / ``gpt-4o-mini-search-preview`` models via a
# top-level ``web_search_options`` parameter; the broader GPT-5+ lineup
# requires the Responses API. We translate the inbound chat-completions wire
# format (messages, tool_calls, tool_choice) → Responses input on the way
# out, and translate Responses output (output items, output_text, function
# calls) → ChatCompletion shape on the way back, so clients of
# ``/v1/chat/completions`` see no difference.
#
# These coexist with the existing OpenAI function-calling shape
# (``{"type": "function", "function": {...}}``) and any number of each may
# be mixed.

# Generic wire-format type names that clients send to opt into native web tools.
WEB_SEARCH_TYPE = "web_search"
FETCH_TYPE = "fetch"
_GENERIC_WEB_TOOL_TYPES = {WEB_SEARCH_TYPE, FETCH_TYPE}

# Provider-native equivalents.
_ANTHROPIC_WEB_SEARCH_TYPE = "web_search_20250305"
_ANTHROPIC_WEB_FETCH_TYPE = "web_fetch_20250910"
_ANTHROPIC_WEB_FETCH_BETA_HEADER = "web-fetch-2025-09-10"


def _is_generic_web_tool(tool: dict) -> bool:
    """True iff ``tool`` uses our generic web-tool shape (``web_search`` / ``fetch``)."""
    return isinstance(tool, dict) and tool.get("type") in _GENERIC_WEB_TOOL_TYPES


def _only_web_tools(tools: list[dict] | None) -> bool:
    """True iff ``tools`` is non-empty and every entry is a generic web tool.

    Drives whether ``tool_choice`` is dropped before forwarding upstream — see
    the ``_proxy_*`` methods for the rationale.
    """
    if not tools:
        return False
    return all(_is_generic_web_tool(t) for t in tools)


def _translate_tools_for_anthropic(
    tools: list[dict] | None,
    web_search_mode: str = "anthropic_native",
) -> tuple[list[dict], bool]:
    """Translate generic + function tools to Anthropic's native tool format.

    Returns ``(anthropic_tools, needs_web_fetch_beta)``. The boolean is True
    iff a generic ``fetch`` tool was translated, meaning the caller must add
    the ``anthropic-beta: web-fetch-2025-09-10`` header to the request (the
    Anthropic web_fetch tool is currently in beta).

    Generic web tools become Anthropic's versioned native types when
    ``web_search_mode == "anthropic_native"``; with ``"drop"`` they are
    silently omitted (used for Fireworks routes, which share the Anthropic
    transport shape but have no hosted search index). Function tools fall
    through to the existing :func:`_openai_tools_to_anthropic` shape;
    unrecognized types are skipped (matches today's behavior).
    """
    if not tools:
        return [], False

    out: list[dict] = []
    needs_web_fetch_beta = False
    for tool in tools:
        ttype = tool.get("type") if isinstance(tool, dict) else None
        if ttype == WEB_SEARCH_TYPE:
            if web_search_mode == "drop":
                continue
            out.append({"type": _ANTHROPIC_WEB_SEARCH_TYPE, "name": "web_search"})
        elif ttype == FETCH_TYPE:
            if web_search_mode == "drop":
                continue
            out.append({"type": _ANTHROPIC_WEB_FETCH_TYPE, "name": "web_fetch"})
            needs_web_fetch_beta = True
        elif ttype == "function":
            # Reuse existing function-tool conversion for one-tool input.
            out.extend(_openai_tools_to_anthropic([tool]))
        # else: silently skip unknown tool types (preserves prior behavior).

    return out, needs_web_fetch_beta


def _translate_tools_for_openai(
    tools: list[dict] | None,
    web_search_mode: str = "openai_native",
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
    - Chat-completions function tool → flattened Responses shape.
    - Anything else passes through unchanged (covers tools the caller
      already shaped for the Responses API).
    """
    if not tools:
        return []

    out: list[dict] = []
    web_search_added = False
    for tool in tools:
        ttype = tool.get("type") if isinstance(tool, dict) else None
        if ttype in _GENERIC_WEB_TOOL_TYPES:
            if web_search_mode == "drop":
                continue
            if not web_search_added:
                out.append({"type": "web_search"})
                web_search_added = True
            continue
        if ttype == "function" and "function" in tool:
            fn = tool["function"]
            flat: dict[str, Any] = {
                "type": "function",
                "name": fn.get("name", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            }
            if "description" in fn:
                flat["description"] = fn["description"]
            out.append(flat)
            continue
        out.append(tool)

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
# Gemini translation
# ---------------------------------------------------------------------------
#
# Gemini's native ``generateContent`` API uses a different message+tool
# vocabulary than OpenAI/Anthropic. The translators below mirror the
# OpenAI/Anthropic ones structurally so the dispatch in ``_proxy_gemini``
# stays the same shape. ``google.genai.types`` is imported lazily inside
# functions that need it so the import only fires for callers that hit a
# Gemini route.


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
    from google.genai import types

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
                types.Content(
                    role="user",
                    parts=[types.Part.from_function_response(name=fn_name, response=response_payload)],
                )
            )
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            parts: list = []
            text = content if isinstance(content, str) else ""
            if text:
                parts.append(types.Part.from_text(text=text))
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
                parts.append(types.Part.from_function_call(name=name, args=args))
            if parts:
                gemini_contents.append(types.Content(role="model", parts=parts))
            continue

        # user (or any other plain role; treat as user for Gemini).
        if isinstance(content, str):
            gemini_contents.append(types.Content(role="user", parts=[types.Part.from_text(text=content)]))

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, gemini_contents


def _translate_tools_for_gemini(
    tools: list[dict] | None,
    web_search_mode: str = "gemini_google_search",
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
    from google.genai import types

    if not tools:
        return []

    out: list = []
    google_search_added = False
    for tool in tools:
        ttype = tool.get("type") if isinstance(tool, dict) else None
        if ttype == WEB_SEARCH_TYPE:
            if web_search_mode != "gemini_google_search":
                continue
            if google_search_added:
                continue
            out.append(types.Tool(google_search=types.GoogleSearch()))
            google_search_added = True
            continue
        if ttype == FETCH_TYPE:
            # Gemini has no separate fetch primitive; google_search bundles
            # retrieval. Drop silently regardless of mode.
            continue
        if ttype == "function" and "function" in tool:
            fn = tool["function"]
            decl = types.FunctionDeclaration(
                name=fn.get("name", ""),
                description=fn.get("description", ""),
                # The SDK accepts a JSON-schema dict here; using the dict
                # rather than typing it as types.Schema avoids hand-wiring
                # Schema(type=Type.OBJECT, properties={...}) for every call.
                parameters=fn.get("parameters", {"type": "object", "properties": {}}),
            )
            out.append(types.Tool(function_declarations=[decl]))

    return out


def _chat_tool_choice_to_gemini(tool_choice: str | dict | None):
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

    from google.genai import types

    if tool_choice == "required":
        return types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="ANY"))
    if tool_choice == "none":
        return types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="NONE"))
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        fn = tool_choice.get("function", {})
        name = fn.get("name")
        if name:
            return types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=[name],
                )
            )
    return None


def _gemini_finish_reason_to_openai(finish_reason: Any, has_tool_calls: bool) -> str:
    """Map Gemini's ``finish_reason`` enum to an OpenAI ``finish_reason`` string.

    Tool-call finishes take precedence: if any function_call parts were
    emitted, surface ``"tool_calls"`` regardless of the upstream value
    (Gemini emits ``STOP`` even when the turn ends with a function call).
    """
    if has_tool_calls:
        return "tool_calls"
    # finish_reason is a typed enum; ``.name`` gives the canonical token,
    # ``str(...)`` includes the enum name. Normalize defensively.
    name = getattr(finish_reason, "name", None) or str(finish_reason or "")
    name = name.upper()
    if "MAX_TOKENS" in name:
        return "length"
    return "stop"


def _gemini_response_to_openai(response: Any, model_name: str) -> dict:
    """Convert a Gemini ``GenerateContentResponse`` to a ChatCompletion dict.

    Mirrors :func:`_anthropic_response_to_openai`: walks the first candidate's
    parts, accumulating text and synthesizing OpenAI ``tool_calls`` for any
    ``function_call`` parts. Gemini doesn't emit tool_call ids — synthesize
    ``call_<8 hex>`` so callers can correlate tool result messages back.
    """
    candidate = (getattr(response, "candidates", None) or [None])[0]
    text_parts: list[str] = []
    tool_calls: list[dict] = []

    if candidate is not None:
        gemini_content = getattr(candidate, "content", None)
        for part in getattr(gemini_content, "parts", None) or []:
            text = getattr(part, "text", None)
            if text:
                text_parts.append(text)
            fn_call = getattr(part, "function_call", None)
            if fn_call is not None:
                args = getattr(fn_call, "args", {}) or {}
                tool_calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": getattr(fn_call, "name", "") or "",
                            "arguments": json.dumps(args),
                        },
                    }
                )

    text = "".join(text_parts)
    finish_reason = _gemini_finish_reason_to_openai(
        getattr(candidate, "finish_reason", None) if candidate else None,
        bool(tool_calls),
    )

    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage_metadata = getattr(response, "usage_metadata", None)
    prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
    completion_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0

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
# PassthroughAgent
# ---------------------------------------------------------------------------


class PassthroughAgent:
    """Fully OpenAI-compatible proxy to the configured LLM provider.

    Forwards the complete request — including tools, tool_choice, temperature,
    max_tokens — to the upstream provider and returns the raw response.
    Supports both streaming and non-streaming for OpenAI and Anthropic.
    """

    def __init__(
        self,
        config: PassthroughModelConfig,
        instrument: bool = True,
    ):
        self.config = config
        self._usage: tuple[int, int] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def proxy(
        self,
        messages: list[Message],
        stream: bool,
        request_id: str,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> StreamingResponse | JSONResponse:
        logger.debug(
            "proxy called",
            extra={
                "api_kind": self.config.api_kind,
                "provider_label": self.config.label,
                "model": self.config.model_name,
                "message_count": len(messages),
                "stream": stream,
                "request_id": request_id,
            },
        )
        msg_dicts = _messages_to_dicts(messages)

        kwargs: dict[str, Any] = {
            "messages": msg_dicts,
            "stream": stream,
            "request_id": request_id,
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if self.config.api_kind == "anthropic_messages":
            return await self._proxy_anthropic(**kwargs)
        if self.config.api_kind == "gemini_native":
            return await self._proxy_gemini(**kwargs)
        # openai_responses (default)
        return await self._proxy_openai(**kwargs)

    async def run(
        self,
        messages: list[Message],
        streamer: Any,
        stream: bool,
        run_context: Any = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ):
        """Legacy run method for backward compat with MessageStreamer pipeline.

        Note: this code path is currently unreachable for passthrough models
        (``OpenAIRequestHandler.create`` short-circuits to ``proxy()``); the
        ``tools`` / ``tool_choice`` parameters exist for feature parity with
        ``proxy()`` so future callers can use generic ``web_search`` / ``fetch``
        here too. Only the model's final-text content is pushed to ``streamer``;
        any ``server_tool_use`` / ``web_search_tool_result`` /
        ``web_fetch_tool_result`` blocks Anthropic emits for native tools are
        intentionally not surfaced (they're internal to the server-side tool).
        """
        logger.debug(
            "run called",
            extra={
                "api_kind": self.config.api_kind,
                "provider_label": self.config.label,
                "model": self.config.model_name,
                "message_count": len(messages),
                "stream": stream,
                "request_id": None,
            },
        )
        msg_dicts = _messages_to_dicts(messages)

        # See _proxy_anthropic / _proxy_openai for the tool_choice rationale:
        # native web tools cannot be forced via tool_choice, so drop it when
        # the request's only tools are web tools.
        effective_tool_choice = None if _only_web_tools(tools) else tool_choice

        if self.config.api_kind == "gemini_native":
            # The legacy run() path is currently unreachable for passthrough
            # models (handler short-circuits to proxy()). Keep the dispatch
            # symmetric so future callers don't silently fall through to the
            # OpenAI branch with a Gemini config; surface explicitly instead.
            raise NotImplementedError(
                "PassthroughAgent.run() does not yet support api_kind='gemini_native'. "
                "Use proxy() (the default request path) for Gemini routes."
            )

        if self.config.api_kind == "anthropic_messages":
            client = self._get_anthropic_client()
            system_prompt, anthropic_msgs = _openai_messages_to_anthropic(msg_dicts)

            kwargs: dict[str, Any] = {
                "model": self.config.model_name,
                "messages": anthropic_msgs,
                "max_tokens": 16384,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            anthropic_tools, needs_web_fetch_beta = _translate_tools_for_anthropic(tools, self.config.web_search_mode)
            if anthropic_tools:
                kwargs["tools"] = anthropic_tools
            if needs_web_fetch_beta:
                kwargs["extra_headers"] = {"anthropic-beta": _ANTHROPIC_WEB_FETCH_BETA_HEADER}
            anthropic_tc = _openai_tool_choice_to_anthropic(effective_tool_choice)
            if anthropic_tc is not None:
                kwargs["tool_choice"] = anthropic_tc

            try:
                response = await client.messages.create(**kwargs)
            except Exception:
                logger.error(
                    "Anthropic API request failed in run",
                    exc_info=True,
                    extra={"model": self.config.model_name},
                )
                raise
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text
            self._usage = (response.usage.input_tokens, response.usage.output_tokens)
            await streamer.push(role=Role.assistant, content=text)
        else:
            client = self._get_openai_client()
            instructions, responses_input = _chat_messages_to_responses_input(msg_dicts)

            kwargs: dict[str, Any] = {
                "model": self.config.model_name,
                "input": responses_input,
            }
            if instructions:
                kwargs["instructions"] = instructions

            openai_tools = _translate_tools_for_openai(tools, self.config.web_search_mode)
            if openai_tools:
                kwargs["tools"] = openai_tools
            if effective_tool_choice is not None:
                kwargs["tool_choice"] = _chat_tool_choice_to_responses(effective_tool_choice)

            try:
                response = await client.responses.create(**kwargs)
            except Exception:
                logger.error(
                    "OpenAI API request failed in run",
                    exc_info=True,
                    extra={"model": self.config.model_name},
                )
                raise
            # Extract assistant text from output items; ignore function_call,
            # web_search_call, reasoning, etc. (consistent with the Anthropic
            # branch — the streamer only sees final text).
            text_parts: list[str] = []
            for item in getattr(response, "output", []) or []:
                if getattr(item, "type", None) == "message":
                    for part in getattr(item, "content", []) or []:
                        if getattr(part, "type", None) == "output_text":
                            text_parts.append(getattr(part, "text", ""))
            self._usage = (response.usage.input_tokens, response.usage.output_tokens)
            await streamer.push(role=Role.assistant, content="".join(text_parts))

    async def get_last_run_usage(self) -> tuple[int, int] | None:
        return self._usage

    # ------------------------------------------------------------------
    # Clients
    # ------------------------------------------------------------------

    def _get_openai_client(self):
        from openai import AsyncOpenAI

        kwargs: dict[str, Any] = {"api_key": self.config.api_key}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        return AsyncOpenAI(**kwargs)

    def _get_anthropic_client(self):
        from anthropic import AsyncAnthropic

        kwargs: dict[str, Any] = {"api_key": self.config.api_key}
        # base_url is set for Anthropic-compatible proxies (e.g., Fireworks);
        # leave unset to use the SDK default for direct Anthropic.
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        return AsyncAnthropic(**kwargs)

    def _get_gemini_client(self):
        from google import genai

        return genai.Client(api_key=self.config.api_key)

    # ------------------------------------------------------------------
    # OpenAI proxy
    # ------------------------------------------------------------------

    async def _proxy_openai(
        self,
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
        """
        from openai import APIError, APIStatusError

        client = self._get_openai_client()

        instructions, responses_input = _chat_messages_to_responses_input(messages)

        kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "input": responses_input,
            "stream": stream,
        }
        if instructions:
            kwargs["instructions"] = instructions

        openai_tools = _translate_tools_for_openai(tools, self.config.web_search_mode)
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
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            # Chat completions ``max_tokens`` is ``max_output_tokens`` on Responses.
            kwargs["max_output_tokens"] = max_tokens

        try:
            response = await client.responses.create(**kwargs)
        except APIStatusError as exc:
            logger.error(
                "OpenAI API request failed",
                exc_info=True,
                extra={"model": self.config.model_name, "stream": stream, "request_id": request_id},
            )
            return JSONResponse(
                content={"error": {"message": exc.message, "type": "api_error", "code": exc.code}},
                status_code=exc.status_code,
            )
        except APIError as exc:
            logger.error(
                "OpenAI API request failed",
                exc_info=True,
                extra={"model": self.config.model_name, "stream": stream, "request_id": request_id},
            )
            return JSONResponse(
                content={"error": {"message": str(exc), "type": "api_error"}},
                status_code=502,
            )

        if stream:
            return StreamingResponse(
                self._stream_openai_responses_as_chat(response, request_id),
                media_type="text/event-stream",
            )
        else:
            self._usage = (response.usage.input_tokens, response.usage.output_tokens)
            return JSONResponse(content=_responses_response_to_chat_completion(response, self.config.model_name))

    async def _stream_openai_responses_as_chat(self, stream, request_id: str):
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
        model = self.config.model_name

        # Map Responses output_index → chat-completion tool_call index/info.
        function_calls_by_output_index: dict[int, dict] = {}
        next_tool_call_index = 0
        sent_role_chunk = False
        finish_reason: str = "stop"
        input_tokens = 0
        output_tokens = 0

        async for event in stream:
            event_type = event.type

            if event_type in ("response.created", "response.in_progress"):
                if not sent_role_chunk:
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": ""},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
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
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": tc_index,
                                            "id": call_id,
                                            "type": "function",
                                            "function": {"name": name, "arguments": ""},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

            elif event_type == "response.output_text.delta":
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": event.delta},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            elif event_type == "response.function_call_arguments.delta":
                tc_info = function_calls_by_output_index.get(event.output_index)
                if tc_info is not None:
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": tc_info["tc_index"],
                                            "function": {"arguments": event.delta},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

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
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            # All other event types (web_search_call.*, reasoning, content_part.*,
            # etc.) are intentionally not forwarded.

        self._usage = (input_tokens, output_tokens)
        yield "data: [DONE]\n\n"

    # ------------------------------------------------------------------
    # Anthropic proxy
    # ------------------------------------------------------------------

    async def _proxy_anthropic(
        self,
        messages: list[dict],
        stream: bool,
        request_id: str,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> StreamingResponse | JSONResponse:
        from anthropic import APIError, APIStatusError

        client = self._get_anthropic_client()
        system_prompt, anthropic_msgs = _openai_messages_to_anthropic(messages)

        kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": anthropic_msgs,
            "max_tokens": max_tokens or 16384,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        anthropic_tools, needs_web_fetch_beta = _translate_tools_for_anthropic(tools, self.config.web_search_mode)
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        if needs_web_fetch_beta:
            # Anthropic's web_fetch tool is currently behind a beta header.
            kwargs["extra_headers"] = {"anthropic-beta": _ANTHROPIC_WEB_FETCH_BETA_HEADER}

        # tool_choice handling: pure pass-through, with one exception. Native
        # server-side tools (web_search_20250305 / web_fetch_20250910) cannot
        # be forced or pinned via tool_choice on Anthropic — they're invoked at
        # the model's discretion. If the request's only tools are web tools,
        # forwarding e.g. tool_choice="required" would be a no-op or upstream
        # error, so drop it. When function tools are also present, pass-through.
        effective_tool_choice = None if _only_web_tools(tools) else tool_choice
        anthropic_tc = _openai_tool_choice_to_anthropic(effective_tool_choice)
        if anthropic_tc is not None:
            kwargs["tool_choice"] = anthropic_tc
        if temperature is not None:
            kwargs["temperature"] = temperature
        if stream:
            kwargs["stream"] = True

        try:
            response = await client.messages.create(**kwargs)
        except APIStatusError as exc:
            logger.error(
                "Anthropic API request failed",
                exc_info=True,
                extra={"model": self.config.model_name, "stream": stream, "request_id": request_id},
            )
            return JSONResponse(
                content={"error": {"message": exc.message, "type": "api_error", "code": exc.status_code}},
                status_code=exc.status_code,
            )
        except APIError as exc:
            logger.error(
                "Anthropic API request failed",
                exc_info=True,
                extra={"model": self.config.model_name, "stream": stream, "request_id": request_id},
            )
            return JSONResponse(
                content={"error": {"message": str(exc), "type": "api_error"}},
                status_code=502,
            )

        if stream:
            return StreamingResponse(
                self._stream_anthropic_as_openai(response, request_id),
                media_type="text/event-stream",
            )
        else:
            self._usage = (response.usage.input_tokens, response.usage.output_tokens)
            openai_response = _anthropic_response_to_openai(response, self.config.model_name)
            return JSONResponse(content=openai_response)

    async def _stream_anthropic_as_openai(self, stream, request_id: str):
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
        model = self.config.model_name

        input_tokens = 0
        output_tokens = 0

        # Track tool calls in progress
        current_tool_calls: dict[int, dict] = {}
        tool_call_index = 0

        async for event in stream:
            event_type = event.type

            if event_type == "message_start":
                if hasattr(event.message, "usage"):
                    input_tokens = event.message.usage.input_tokens
                # Send initial role chunk
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            elif event_type == "content_block_start":
                block = event.content_block
                if block.type == "tool_use":
                    tc = {
                        "id": block.id,
                        "type": "function",
                        "function": {"name": block.name, "arguments": ""},
                    }
                    current_tool_calls[tool_call_index] = tc
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"tool_calls": [{"index": tool_call_index, **tc}]},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

            elif event_type == "content_block_delta":
                delta = event.delta
                if delta.type == "text_delta":
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": delta.text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                elif delta.type == "input_json_delta":
                    if tool_call_index in current_tool_calls:
                        current_tool_calls[tool_call_index]["function"]["arguments"] += delta.partial_json
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": tool_call_index,
                                            "function": {"arguments": delta.partial_json},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

            elif event_type == "content_block_stop":
                if tool_call_index in current_tool_calls:
                    tool_call_index += 1

            elif event_type == "message_delta":
                if hasattr(event, "usage") and event.usage:
                    output_tokens = event.usage.output_tokens
                finish_reason = "stop"
                if hasattr(event.delta, "stop_reason") and event.delta.stop_reason == "tool_use":
                    finish_reason = "tool_calls"
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            elif event_type == "message_stop":
                pass

        self._usage = (input_tokens, output_tokens)
        yield "data: [DONE]\n\n"

    # ------------------------------------------------------------------
    # Gemini proxy
    # ------------------------------------------------------------------

    async def _proxy_gemini(
        self,
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
        """
        from google.genai import errors as genai_errors
        from google.genai import types

        client = self._get_gemini_client()

        system_instruction, contents = _chat_messages_to_gemini(messages)
        gemini_tools = _translate_tools_for_gemini(tools, self.config.web_search_mode)
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

        gen_config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        common_kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "contents": contents,
        }
        if gen_config is not None:
            common_kwargs["config"] = gen_config

        try:
            if stream:
                return StreamingResponse(
                    self._stream_gemini_as_openai(client, common_kwargs, request_id),
                    media_type="text/event-stream",
                )
            response = await client.aio.models.generate_content(**common_kwargs)
        except genai_errors.APIError as exc:
            logger.error(
                "Gemini API request failed",
                exc_info=True,
                extra={"model": self.config.model_name, "stream": stream, "request_id": request_id},
            )
            status_code = getattr(exc, "code", None) or 502
            return JSONResponse(
                content={"error": {"message": str(exc), "type": "api_error"}},
                status_code=status_code if isinstance(status_code, int) else 502,
            )

        usage_metadata = getattr(response, "usage_metadata", None)
        self._usage = (
            getattr(usage_metadata, "prompt_token_count", 0) or 0,
            getattr(usage_metadata, "candidates_token_count", 0) or 0,
        )
        return JSONResponse(content=_gemini_response_to_openai(response, self.config.model_name))

    async def _stream_gemini_as_openai(self, client, common_kwargs: dict, request_id: str):
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
        model = self.config.model_name

        sent_role_chunk = False
        tool_call_index = 0
        text_emitted = False
        tool_calls_emitted = False
        prompt_tokens = 0
        completion_tokens = 0
        last_finish_reason: Any = None

        async def _emit_role_chunk():
            nonlocal sent_role_chunk
            if sent_role_chunk:
                return None
            sent_role_chunk = True
            return {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            }

        try:
            stream = await client.aio.models.generate_content_stream(**common_kwargs)
            async for chunk in stream:
                role_chunk = await _emit_role_chunk()
                if role_chunk is not None:
                    yield f"data: {json.dumps(role_chunk)}\n\n"

                candidate = (getattr(chunk, "candidates", None) or [None])[0]
                if candidate is None:
                    continue
                if getattr(candidate, "finish_reason", None):
                    last_finish_reason = candidate.finish_reason

                gemini_content = getattr(candidate, "content", None)
                for part in getattr(gemini_content, "parts", None) or []:
                    text = getattr(part, "text", None)
                    if text:
                        text_emitted = True
                        delta_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(delta_chunk)}\n\n"

                    fn_call = getattr(part, "function_call", None)
                    if fn_call is not None:
                        tool_calls_emitted = True
                        args = getattr(fn_call, "args", {}) or {}
                        # Whole tool_call atomically — Gemini doesn't stream
                        # incremental argument deltas.
                        tc_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": tool_call_index,
                                                "id": f"call_{uuid.uuid4().hex[:8]}",
                                                "type": "function",
                                                "function": {
                                                    "name": getattr(fn_call, "name", "") or "",
                                                    "arguments": json.dumps(args),
                                                },
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(tc_chunk)}\n\n"
                        tool_call_index += 1

                usage_metadata = getattr(chunk, "usage_metadata", None)
                if usage_metadata is not None:
                    # Gemini's streaming usage is cumulative; latest chunk wins.
                    prompt_tokens = getattr(usage_metadata, "prompt_token_count", prompt_tokens) or prompt_tokens
                    completion_tokens = (
                        getattr(usage_metadata, "candidates_token_count", completion_tokens) or completion_tokens
                    )
        except Exception:
            logger.error(
                "Gemini streaming request failed",
                exc_info=True,
                extra={"model": self.config.model_name, "request_id": request_id},
            )
            raise

        finish_reason = _gemini_finish_reason_to_openai(last_finish_reason, tool_calls_emitted)
        # If nothing came through at all, still emit a role chunk so the
        # client sees a valid (empty) completion rather than a half-stream.
        if not sent_role_chunk:
            empty_chunk = await _emit_role_chunk()
            if empty_chunk is not None:
                yield f"data: {json.dumps(empty_chunk)}\n\n"
        _ = text_emitted  # currently unused but retained for clarity
        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"

        self._usage = (prompt_tokens, completion_tokens)
        yield "data: [DONE]\n\n"
