"""Fireworks proxy with a server-side external-search loop.

Fireworks-hosted models (kimi/deepseek/qwen) speak the Anthropic Messages API
shape — so this module reuses the Anthropic transport + translation helpers
from :mod:`minds.inference.providers.anthropic` — but they have **no hosted
search index**. Where direct Anthropic uses native ``web_search_20250305`` /
``web_fetch_20250910`` server tools, Fireworks gets web search by *us* running
the tool loop server-side:

    model emits tool_use(web_search) → we run the configured SearchProvider
    → feed the results back as a tool_result → re-call until the model stops
    requesting tools (or we hit the iteration cap, then force a final answer).

The loop is always non-streaming (we need the complete tool_use blocks before
we can run a tool); for streaming clients the final completion is replayed as
synthetic OpenAI SSE chunks (:func:`_replay_completion_as_sse`). When no search
provider is available (no web tool requested, or search disabled/misconfigured)
the request falls back to a single-shot call with web tools dropped — i.e. the
old ``WebSearchMode.DROP`` behavior.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from anthropic import APIError as AnthropicAPIError
from anthropic import APIStatusError as AnthropicAPIStatusError
from anthropic import AsyncAnthropic
from starlette.responses import JSONResponse, StreamingResponse

from minds.common.logger import setup_logging
from minds.common.search.base import FetchedContent, SearchProvider, SearchResult
from minds.common.settings.app_settings import get_app_settings
from minds.inference.providers.anthropic import (
    _anthropic_response_to_openai,
    _openai_messages_to_anthropic,
    _openai_tool_choice_to_anthropic,
    stream_anthropic_as_openai,
)
from minds.inference.types import (
    ChatCompletionsFunctionTool,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    GenericFetchTool,
    GenericWebSearchTool,
    PassthroughModelConfig,
    UsageBox,
    _classify_tool,
    _emit_chunk,
    _is_generic_web_tool,
    _only_web_tools,
)

__all__ = [
    "EXTERNAL_FETCH_TOOL_NAME",
    "EXTERNAL_SEARCH_TOOL_NAME",
    "build_search_provider_for_request",
    "proxy_fireworks",
]

logger = setup_logging()
_settings = get_app_settings()


# ---------------------------------------------------------------------------
# External-search tool definitions
# ---------------------------------------------------------------------------
#
# The generic ``web_search`` / ``fetch`` tools are exposed to the Fireworks
# model as *ordinary* Anthropic function tools it calls explicitly; we execute
# them server-side and feed results back as tool_result blocks. The names are
# referenced by the loop's dispatch, so they live here as constants rather than
# string literals scattered across the module.

EXTERNAL_SEARCH_TOOL_NAME = "web_search"
EXTERNAL_FETCH_TOOL_NAME = "fetch_url"

_EXTERNAL_SEARCH_TOOL = {
    "name": EXTERNAL_SEARCH_TOOL_NAME,
    "description": "Search the web for current information. Returns a list of results with title, url, and snippet.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "num_results": {
                "type": "integer",
                "description": "Max number of results to return.",
            },
        },
        "required": ["query"],
    },
}

_EXTERNAL_FETCH_TOOL = {
    "name": EXTERNAL_FETCH_TOOL_NAME,
    "description": "Fetch and read the full text contents of a web page by URL.",
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to fetch."},
        },
        "required": ["url"],
    },
}


# ---------------------------------------------------------------------------
# Tool translation + provider selection
# ---------------------------------------------------------------------------


def _translate_tools_for_fireworks(tools: list[dict] | None, *, external_search: bool) -> list[dict]:
    """Translate generic + function tools to Fireworks (Anthropic-shape) tools.

    Generic ``web_search`` / ``fetch`` become the external function tools the
    loop executes (when ``external_search`` is True) or are dropped (when no
    search provider is available). Function tools pass through reshaped to
    Anthropic's ``{"name", "description", "input_schema"}`` form; unrecognized
    types are skipped (matches the Anthropic translator's behavior).
    """
    out: list[dict] = []
    if not tools:
        return out
    for tool in tools:
        parsed = _classify_tool(tool)
        if isinstance(parsed, GenericWebSearchTool):
            if external_search:
                out.append(dict(_EXTERNAL_SEARCH_TOOL))
            else:
                logger.debug("Dropping generic web_search (no search provider available)")
        elif isinstance(parsed, GenericFetchTool):
            if external_search:
                out.append(dict(_EXTERNAL_FETCH_TOOL))
            else:
                logger.debug("Dropping generic fetch (no search provider available)")
        elif isinstance(parsed, ChatCompletionsFunctionTool):
            out.append(
                {
                    "name": parsed.function.name,
                    "description": parsed.function.description or "",
                    "input_schema": parsed.function.parameters,
                }
            )
        # else: unrecognized, already logged at debug by _classify_tool.
    return out


def build_search_provider_for_request(
    config: PassthroughModelConfig, tools: list[dict] | None
) -> SearchProvider | None:
    """Build the search provider for this request, or ``None`` to skip search.

    Returns ``None`` (so the request degrades to a single-shot call with web
    tools dropped) when the per-user search kill switch is off, when no generic
    web tool was requested, or when the provider can't be constructed (e.g. the
    exa key isn't set). A construction failure is logged as a warning rather
    than raised, so a search misconfig never 5xxs an otherwise-valid Fireworks
    request. ``config.search_provider_name`` (a per-user Statsig override) wins
    over the env-configured ``SEARCH__PROVIDER`` when set.
    """
    if not config.search_enabled:
        logger.debug("External search disabled for this user; dropping web tools")
        return None
    if not any(_is_generic_web_tool(t) for t in (tools or [])):
        return None
    from minds.common.search import get_search_provider

    try:
        return get_search_provider(get_app_settings(), provider_name=config.search_provider_name)
    except Exception as exc:  # noqa: BLE001 - degrade to no-search rather than failing the request
        logger.warning("Could not build search provider; dropping web tools", extra={"error": str(exc)})
        return None


# ---------------------------------------------------------------------------
# External-search tool-execution loop
# ---------------------------------------------------------------------------


def _format_search_results(results: list[SearchResult]) -> str:
    if not results:
        return "No results found."
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        date = f" ({r.published_date})" if r.published_date else ""
        lines.append(f"[{i}] {r.title}{date}\n{r.url}\n{r.snippet}")
    return "\n\n".join(lines)


def _format_fetched_content(content: FetchedContent) -> str:
    header = content.title or content.url
    suffix = "\n\n[content truncated]" if content.truncated else ""
    return f"{header}\n{content.url}\n\n{content.text}{suffix}"


async def _exec_search_tool(
    provider: SearchProvider,
    block: Any,
    settings: Any,
    usage_box: UsageBox,
) -> dict[str, Any]:
    """Execute one tool_use block and return its tool_result block.

    Provider/transport errors are caught and returned as a recoverable
    ``tool_result`` string (rather than aborting the request), so the model
    can react to a failed search and still answer. Each call is recorded as a
    server artifact for Langfuse.
    """
    tool_use_id = getattr(block, "id", "")
    name = getattr(block, "name", "")
    args = getattr(block, "input", {}) or {}
    # ``type="external_search"`` marks these for the handler, which renders them
    # as nested Langfuse tool spans (not token-bearing generations) so the
    # search provider's cost never mixes with the upstream model's token cost.
    artifact: dict[str, Any] = {
        "type": "external_search",
        "tool": name,
        "provider": settings.search.provider,
        "input": args,
    }
    try:
        if name == EXTERNAL_SEARCH_TOOL_NAME:
            query = str(args.get("query", ""))
            num_results = int(args.get("num_results") or settings.search.max_results)
            results = await provider.search(query, num_results=num_results)
            text = _format_search_results(results)
            artifact["results"] = [{"title": r.title, "url": r.url} for r in results]
        elif name == EXTERNAL_FETCH_TOOL_NAME:
            url = str(args.get("url", ""))
            content = await provider.fetch(url, char_limit=settings.search.fetch_char_limit)
            text = _format_fetched_content(content)
            artifact["results"] = [{"url": content.url, "truncated": content.truncated}]
        else:
            text = f"Unknown tool: {name}"
            artifact["error"] = "unknown_tool"
    except Exception as exc:  # noqa: BLE001 - any provider error becomes a recoverable tool result
        logger.warning("External search tool failed", extra={"tool": name, "error": str(exc)})
        text = f"Search failed: {exc}"
        artifact["error"] = str(exc)
    usage_box.server_artifacts.append(artifact)
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": text}


async def _run_external_search_loop(
    *,
    client: AsyncAnthropic,
    base_kwargs: dict[str, Any],
    provider: SearchProvider,
    settings: Any,
    usage_box: UsageBox,
    request_id: str,
) -> tuple[Any, tuple[int, int]]:
    """Drive the model<->search loop; return ``(final_response, (in, out))``.

    ``base_kwargs`` is the fully-built ``messages.create`` kwargs WITHOUT
    ``stream`` (the loop is always non-streaming). Token usage is summed
    across every turn so the caller records the true total, not just the
    final turn's.
    """
    convo: list[dict[str, Any]] = list(base_kwargs["messages"])
    per_turn = {**base_kwargs, "max_tokens": settings.search.per_turn_max_tokens}
    total_in = total_out = 0

    for iteration in range(settings.search.max_iterations):
        resp = await client.messages.create(**{**per_turn, "messages": convo})
        total_in += resp.usage.input_tokens
        total_out += resp.usage.output_tokens

        tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
        if resp.stop_reason != "tool_use" or not tool_uses:
            return resp, (total_in, total_out)

        logger.debug(
            "External search loop: executing tools",
            extra={"request_id": request_id, "iteration": iteration, "tool_count": len(tool_uses)},
        )
        # Echo the assistant turn (its tool_use blocks) back verbatim — the
        # Anthropic SDK re-serializes its own response content blocks, and the
        # follow-up tool_result blocks attach to them by id.
        convo.append({"role": "assistant", "content": resp.content})
        tool_results = await asyncio.gather(*[_exec_search_tool(provider, b, settings, usage_box) for b in tool_uses])
        convo.append({"role": "user", "content": list(tool_results)})

    # Iteration cap reached: force a final answer with no tools so the model
    # stops searching and commits to a response.
    logger.debug("External search loop hit iteration cap", extra={"request_id": request_id})
    final_kwargs = {k: v for k, v in per_turn.items() if k not in ("tools", "tool_choice")}
    resp = await client.messages.create(**{**final_kwargs, "messages": convo})
    total_in += resp.usage.input_tokens
    total_out += resp.usage.output_tokens
    return resp, (total_in, total_out)


async def _replay_completion_as_sse(config: PassthroughModelConfig, completion: dict[str, Any]):
    """Replay an already-computed ChatCompletion dict as OpenAI SSE chunks.

    Used for streaming clients on the external-search path: the loop ran
    non-streaming, so there's no upstream token stream to forward — we emit
    the final assistant message as role/content/tool_call/finish chunks. The
    caller has already populated the usage_box; this just terminates the stream.
    """
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())
    emit_kwargs = {"completion_id": completion_id, "created": created, "model": config.model_name}

    message = completion["choices"][0]["message"]
    tool_calls = message.get("tool_calls") or []

    yield _emit_chunk(**emit_kwargs, role="assistant", content="")
    if message.get("content"):
        yield _emit_chunk(**emit_kwargs, content=message["content"])
    for idx, tc in enumerate(tool_calls):
        fn = tc.get("function", {})
        yield _emit_chunk(
            **emit_kwargs,
            tool_calls=[
                ChoiceDeltaToolCall(
                    index=idx,
                    id=tc.get("id"),
                    type="function",
                    function=ChoiceDeltaToolCallFunction(name=fn.get("name"), arguments=fn.get("arguments", "")),
                )
            ],
        )
    finish_reason = "tool_calls" if tool_calls else "stop"
    yield _emit_chunk(**emit_kwargs, finish_reason=finish_reason)
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Proxy
# ---------------------------------------------------------------------------


async def proxy_fireworks(
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
    search_provider: SearchProvider | None = None,
) -> StreamingResponse | JSONResponse:
    """Proxy a chat-completions request to a Fireworks (Anthropic-shape) model.

    When ``search_provider`` is supplied, generic web tools are exposed as
    function tools and executed server-side via the search loop; otherwise the
    request is a single-shot call with web tools dropped. The caller supplies
    the SDK ``client`` so tests can inject a fake without monkeypatching.
    """
    system_prompt, anthropic_msgs = _openai_messages_to_anthropic(messages)

    kwargs: dict[str, Any] = {
        "model": config.model_name,
        "messages": anthropic_msgs,
        "max_tokens": max_tokens or 16384,
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    use_loop = search_provider is not None
    translated_tools = _translate_tools_for_fireworks(tools, external_search=use_loop)
    if translated_tools:
        kwargs["tools"] = translated_tools

    # Native server-side tools can't be forced via tool_choice; if the only
    # tools are web tools, drop tool_choice (matches the Anthropic path).
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
    if config.reasoning_effort:
        # Fireworks' Anthropic-compatible endpoint takes the OpenAI-style
        # ``reasoning_effort`` as an extension field (mutually exclusive with
        # ``thinking``), so it rides in the request body via ``extra_body``.
        # The level string was validated by the resolver and forwarded
        # verbatim; ``base_kwargs`` spreads into every search-loop turn, so
        # the loop path inherits it too.
        kwargs["extra_body"] = {"reasoning_effort": config.reasoning_effort}
    if stream and not use_loop:
        kwargs["stream"] = True

    logger.debug(
        "Calling Fireworks (Anthropic-shape) messages",
        extra={
            "request_id": request_id,
            "model": config.model_name,
            "stream": stream,
            "use_search_loop": use_loop,
            "has_tools": bool(translated_tools),
            "reasoning_effort": config.reasoning_effort,
        },
    )

    loop_usage: tuple[int, int] | None = None
    try:
        if use_loop:
            # The loop is always non-streaming — it needs complete tool_use
            # blocks before executing a tool — regardless of the client's
            # stream flag. Replay to streaming clients happens below.
            response, loop_usage = await _run_external_search_loop(
                client=client,
                base_kwargs=kwargs,
                provider=search_provider,
                settings=_settings,
                usage_box=usage_box,
                request_id=request_id,
            )
        else:
            response = await client.messages.create(**kwargs)
    except AnthropicAPIStatusError as exc:
        logger.error(
            "Fireworks API request failed",
            exc_info=True,
            extra={"model": config.model_name, "stream": stream, "request_id": request_id},
        )
        return JSONResponse(
            content={"error": {"message": exc.message, "type": "api_error", "code": exc.status_code}},
            status_code=exc.status_code,
        )
    except AnthropicAPIError as exc:
        logger.error(
            "Fireworks API request failed",
            exc_info=True,
            extra={"model": config.model_name, "stream": stream, "request_id": request_id},
        )
        return JSONResponse(
            content={"error": {"message": str(exc), "type": "api_error"}},
            status_code=502,
        )

    # Streaming + external-search: the loop already produced the final
    # (non-streaming) response; replay it as synthetic SSE chunks. The
    # client-facing shape is identical to the live-stream path.
    if stream and use_loop:
        usage_box.value = loop_usage
        completion = _anthropic_response_to_openai(response, config.model_name)
        usage_box.output_payload = completion["choices"][0]["message"]
        return StreamingResponse(
            _replay_completion_as_sse(config, completion),
            media_type="text/event-stream",
        )
    if stream:
        return StreamingResponse(
            stream_anthropic_as_openai(config, usage_box, response, request_id),
            media_type="text/event-stream",
        )
    usage_box.value = loop_usage or (response.usage.input_tokens, response.usage.output_tokens)
    completion = _anthropic_response_to_openai(response, config.model_name)
    usage_box.output_payload = completion["choices"][0]["message"]
    return JSONResponse(content=completion)
