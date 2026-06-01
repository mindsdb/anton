"""Shared types and helpers for the inference layer.

Pure data + utilities; no provider SDK imports live here so the per-provider
modules (``anthropic``, ``openai``, ``gemini``) can each depend on this one
without dragging the other providers' SDKs into their import graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice as ChunkChoice,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from pydantic import BaseModel, Field, ValidationError

from minds.common.logger import setup_logging
from minds.schemas.chat import Message

__all__ = [
    "AnthropicToolsTranslation",
    "ChatCompletionsFunctionDef",
    "ChatCompletionsFunctionTool",
    "ChoiceDeltaToolCall",
    "ChoiceDeltaToolCallFunction",
    "GenericFetchTool",
    "GenericToolType",
    "GenericWebSearchTool",
    "ParsedTool",
    "UsageBox",
    "_classify_tool",
    "_emit_chunk",
    "_is_generic_web_tool",
    "_messages_to_dicts",
    "_only_web_tools",
    "_GENERIC_WEB_TOOL_TYPES",
]

logger = setup_logging()

@dataclass
class UsageBox:
    """Mutable container for request-scoped state tracking.

    The per-provider proxies write to this as the request runs.

    Streaming bodies run on the ASGI server *after* the handler returns, so
    the agent can't read a method-local return value to capture token counts
    or the assistant response. A shared box lets the proxy assign on
    completion (whether streaming or not) and the adapter's accessors read it
    later.

    Fields:
        value: ``(input_tokens, output_tokens)`` once known; ``None`` until then.
        output_payload: The OpenAI-shaped assistant message dict
            (``{"role": "assistant", "content": ..., "tool_calls": [...]}``).
            Populated by the non-streaming response translator and by the
            streaming converter at end-of-stream so Langfuse can record the
            full response as the generation's ``output``.
        server_artifacts: Provider-side intermediates that are intentionally
            not surfaced to the client (Anthropic ``server_tool_use`` /
            ``web_search_tool_result`` / ``web_fetch_tool_result``, OpenAI
            ``web_search_call`` / ``reasoning``). Captured here so we can
            attach them as Langfuse metadata for evals / troubleshooting
            without changing the client-facing response shape.
    """

    value: tuple[int, int] | None = None
    output_payload: dict[str, Any] | None = None
    server_artifacts: list[dict[str, Any]] = field(default_factory=list)


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


class GenericToolType(StrEnum):
    """Wire-format ``type`` values for native web tools.

    Defined as a ``StrEnum`` so clients can reference enum members
    rather than passing bare string literals.
    """

    WEB_SEARCH = "web_search"
    FETCH = "fetch"


_GENERIC_WEB_TOOL_TYPES = {GenericToolType.WEB_SEARCH, GenericToolType.FETCH}


class GenericWebSearchTool(BaseModel):
    """Wire-format opt-in for native web search (provider-agnostic)."""

    type: Literal[GenericToolType.WEB_SEARCH]


class GenericFetchTool(BaseModel):
    """Wire-format opt-in for native URL fetching (provider-agnostic)."""

    type: Literal[GenericToolType.FETCH]


class ChatCompletionsFunctionDef(BaseModel):
    """The ``function`` payload nested inside a chat-completions function tool."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
    )


class ChatCompletionsFunctionTool(BaseModel):
    """A chat-completions function tool: ``{"type":"function","function":{...}}``."""

    type: Literal["function"]
    function: ChatCompletionsFunctionDef


ParsedTool = GenericWebSearchTool | GenericFetchTool | ChatCompletionsFunctionTool


class AnthropicToolsTranslation(BaseModel):
    """Result of translating a client tool list to Anthropic-native format."""

    tools: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Tools shaped for Anthropic's ``messages.create(tools=...)``.",
    )
    needs_web_fetch_beta: bool = Field(
        default=False,
        description="True iff a generic ``fetch`` tool requires the anthropic-beta header.",
    )


def _classify_tool(tool: Any) -> ParsedTool | None:
    """Return a typed wrapper for a known tool shape, or ``None`` if unrecognized."""
    if not isinstance(tool, dict):
        logger.debug("Skipping non-dict tool entry", extra={"tool_type": type(tool).__name__})
        return None
    ttype = tool.get("type")
    if ttype == GenericToolType.WEB_SEARCH:
        try:
            return GenericWebSearchTool.model_validate(tool)
        except ValidationError as exc:
            logger.warning("Malformed generic web_search tool, skipping", extra={"errors": exc.errors()})
            return None
    if ttype == GenericToolType.FETCH:
        try:
            return GenericFetchTool.model_validate(tool)
        except ValidationError as exc:
            logger.warning("Malformed generic fetch tool, skipping", extra={"errors": exc.errors()})
            return None
    if ttype == "function":
        try:
            return ChatCompletionsFunctionTool.model_validate(tool)
        except ValidationError as exc:
            logger.warning(
                "Malformed function tool, skipping",
                extra={"errors": exc.errors(), "tool_name": (tool.get("function") or {}).get("name")},
            )
            return None
    logger.debug("Unrecognized tool type, will pass through to upstream", extra={"type": ttype})
    return None


def _is_generic_web_tool(tool: Any) -> bool:
    """True iff ``tool`` is a recognized generic web tool (``web_search`` / ``fetch``)."""
    parsed = _classify_tool(tool)
    return isinstance(parsed, GenericWebSearchTool | GenericFetchTool)


def _only_web_tools(tools: list[dict] | None) -> bool:
    """True iff ``tools`` is non-empty and every entry is a generic web tool."""
    if not tools:
        return False
    return all(_is_generic_web_tool(t) for t in tools)


def _emit_chunk(
    *,
    completion_id: str,
    created: int,
    model: str,
    role: str | None = None,
    content: str | None = None,
    tool_calls: list[ChoiceDeltaToolCall] | None = None,
    finish_reason: str | None = None,
) -> str:
    """Build a serialized ``chat.completion.chunk`` SSE line via OpenAI SDK types.

    Constructing chunks through ``ChatCompletionChunk`` / ``ChoiceDelta`` /
    ``ChoiceDeltaToolCall`` keeps the wire shape correct by construction.
    """
    delta = ChoiceDelta(role=role, content=content, tool_calls=tool_calls)
    chunk = ChatCompletionChunk(
        id=completion_id,
        object="chat.completion.chunk",
        created=created,
        model=model,
        choices=[ChunkChoice(index=0, delta=delta, finish_reason=finish_reason)],
    )
    return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
