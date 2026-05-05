"""Passthrough LLM agent — proxies requests to upstream OpenAI / Anthropic."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from starlette.responses import JSONResponse, StreamingResponse

from minds.common.logger import setup_logging
from minds.common.passthrough_config import PassthroughModelConfig
from minds.common.settings.app_settings import get_app_settings
from minds.schemas.chat import Message, Role

logger = setup_logging()
settings = get_app_settings()


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
#                              # → OpenAI: web_search_options={} (search-preview models)
#     {"type": "fetch"}        # → Anthropic: tools=[{"type":"web_fetch_20250910", ...}]
#                              #   (requires anthropic-beta: web-fetch-2025-09-10 header)
#                              # → OpenAI: bundled into web_search_options (no separate
#                              #   fetch tool exists), so requesting fetch alone still
#                              #   sets web_search_options.
#
# Note on the OpenAI request shape: chat-completions does NOT accept
# ``{"type": "web_search"}`` as a ``tools[]`` entry — its tools array only
# accepts ``function`` and ``custom`` types. Native web search on chat
# completions is enabled via the top-level ``web_search_options`` parameter
# (on search-preview models such as ``gpt-4o-search-preview`` /
# ``gpt-4o-mini-search-preview``). Per the user's "let upstream error" rule,
# we don't gate on model name; non-search-preview models will return an
# upstream error which we surface.
#
# These coexist with the existing OpenAI function-calling shape
# (``{"type": "function", ...}``) and any number of each may be mixed.

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
) -> tuple[list[dict], bool]:
    """Translate generic + function tools to Anthropic's native tool format.

    Returns ``(anthropic_tools, needs_web_fetch_beta)``. The boolean is True
    iff a generic ``fetch`` tool was translated, meaning the caller must add
    the ``anthropic-beta: web-fetch-2025-09-10`` header to the request (the
    Anthropic web_fetch tool is currently in beta).

    Generic web tools become Anthropic's versioned native types; function
    tools fall through to the existing :func:`_openai_tools_to_anthropic`
    shape; unrecognized types are skipped (matches today's behavior).
    """
    if not tools:
        return [], False

    out: list[dict] = []
    needs_web_fetch_beta = False
    for tool in tools:
        ttype = tool.get("type") if isinstance(tool, dict) else None
        if ttype == WEB_SEARCH_TYPE:
            out.append({"type": _ANTHROPIC_WEB_SEARCH_TYPE, "name": "web_search"})
        elif ttype == FETCH_TYPE:
            out.append({"type": _ANTHROPIC_WEB_FETCH_TYPE, "name": "web_fetch"})
            needs_web_fetch_beta = True
        elif ttype == "function":
            # Reuse existing function-tool conversion for one-tool input.
            out.extend(_openai_tools_to_anthropic([tool]))
        # else: silently skip unknown tool types (preserves prior behavior).

    return out, needs_web_fetch_beta


def _translate_tools_for_openai(
    tools: list[dict] | None,
) -> tuple[list[dict], bool]:
    """Filter generic web tools out of the OpenAI ``tools`` array.

    Returns ``(chat_tools, wants_web_search)``. The chat-completions API does
    not accept ``{"type": "web_search"}`` as a ``tools[]`` entry — its tools
    array only accepts ``function`` and ``custom`` types. Native web search
    is enabled instead via the top-level ``web_search_options`` parameter on
    search-preview models, so we strip generic ``web_search`` / ``fetch``
    here and signal the caller to set ``web_search_options={}`` on the
    request.

    OpenAI's web_search bundles URL fetching internally, so both generic
    ``web_search`` and ``fetch`` collapse to the same signal. Function tools
    and any other unrecognized entries pass through unchanged.
    """
    if not tools:
        return [], False

    out: list[dict] = []
    wants_web_search = False
    for tool in tools:
        ttype = tool.get("type") if isinstance(tool, dict) else None
        if ttype in _GENERIC_WEB_TOOL_TYPES:
            wants_web_search = True
            continue
        out.append(tool)

    return out, wants_web_search


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
                "provider": self.config.provider,
                "model": self.config.model_name,
                "message_count": len(messages),
                "stream": stream,
                "request_id": request_id,
            },
        )
        msg_dicts = _messages_to_dicts(messages)

        if self.config.provider == "anthropic":
            return await self._proxy_anthropic(
                messages=msg_dicts,
                stream=stream,
                request_id=request_id,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            return await self._proxy_openai(
                messages=msg_dicts,
                stream=stream,
                request_id=request_id,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens,
            )

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
                "provider": self.config.provider,
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

        if self.config.provider == "anthropic":
            client = self._get_anthropic_client()
            system_prompt, anthropic_msgs = _openai_messages_to_anthropic(msg_dicts)

            kwargs: dict[str, Any] = {
                "model": self.config.model_name,
                "messages": anthropic_msgs,
                "max_tokens": 16384,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            anthropic_tools, needs_web_fetch_beta = _translate_tools_for_anthropic(tools)
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
            kwargs = {
                "model": self.config.model_name,
                "messages": msg_dicts,
            }
            openai_tools, wants_web_search = _translate_tools_for_openai(tools)
            if openai_tools:
                kwargs["tools"] = openai_tools
            if wants_web_search:
                # See _proxy_openai: native web search is enabled via the
                # top-level web_search_options param, not the tools array.
                kwargs["web_search_options"] = {}
            if effective_tool_choice is not None:
                kwargs["tool_choice"] = effective_tool_choice

            try:
                response = await client.chat.completions.create(**kwargs)
            except Exception:
                logger.error(
                    "OpenAI API request failed in run",
                    exc_info=True,
                    extra={"model": self.config.model_name},
                )
                raise
            choice = response.choices[0]
            self._usage = (
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )
            await streamer.push(role=Role.assistant, content=choice.message.content or "")

    async def get_last_run_usage(self) -> tuple[int, int] | None:
        return self._usage

    # ------------------------------------------------------------------
    # Clients
    # ------------------------------------------------------------------

    def _get_openai_client(self):
        from openai import AsyncOpenAI

        return AsyncOpenAI(
            api_key=settings.openai.api_key,
            base_url=settings.openai.api_url,
        )

    def _get_anthropic_client(self):
        from anthropic import AsyncAnthropic

        return AsyncAnthropic(api_key=settings.anthropic.api_key)

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
        from openai import APIError, APIStatusError

        client = self._get_openai_client()

        kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": stream,
        }
        openai_tools, wants_web_search = _translate_tools_for_openai(tools)
        if openai_tools:
            kwargs["tools"] = openai_tools
        if wants_web_search:
            # Native web search on chat-completions is enabled via the
            # top-level ``web_search_options`` parameter (on search-preview
            # models), not as a tools[] entry. Empty dict = default config.
            kwargs["web_search_options"] = {}
        # tool_choice handling: pure pass-through, with one exception. OpenAI's
        # native web_search hosted tool is invoked at the model's discretion;
        # it cannot be forced via tool_choice. If the request's only tools are
        # web tools, forwarding tool_choice would be a no-op or upstream error,
        # so drop it. When function tools are also present, pass-through.
        if tool_choice is not None and not _only_web_tools(tools):
            kwargs["tool_choice"] = tool_choice
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        if stream:
            kwargs["stream_options"] = {"include_usage": True}

        try:
            response = await client.chat.completions.create(**kwargs)
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
                self._stream_openai_chunks(response, request_id),
                media_type="text/event-stream",
            )
        else:
            self._usage = (
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )
            return JSONResponse(content=response.model_dump())

    async def _stream_openai_chunks(self, response, request_id: str):
        """Forward OpenAI SSE chunks as-is."""
        async for chunk in response:
            if chunk.usage:
                self._usage = (chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
            data = chunk.model_dump_json()
            yield f"data: {data}\n\n"
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

        anthropic_tools, needs_web_fetch_beta = _translate_tools_for_anthropic(tools)
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
