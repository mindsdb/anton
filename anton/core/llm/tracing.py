"""Per-turn trace identity for outbound LLM telemetry.

`ChatSession.turn_stream` sets the active `TraceContext` for the
duration of a turn. The OpenAI provider reads it when talking to
MindsHub and attaches langfuse-style headers so every LLM call (and
any nested tool/scratchpad LLM call made within the same asyncio
task) is attributed to the same session + turn server-side.

A `ContextVar` is used so that nested calls — `_stream_and_handle_tools`,
`generate_object` (structured output), the cerebellum's diff call,
and the scratchpad's `coding_provider` calls — all inherit the same
trace automatically without threading kwargs through every layer.

Scope: only consumed by the OpenAI provider when its base URL points
at MindsHub. Other providers (direct Anthropic, raw OpenAI, Azure,
Gemini) ignore the context entirely.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass


@dataclass(frozen=True)
class TraceContext:
    """Identifiers attached to outbound LLM calls during a turn."""

    session_id: str | None = None
    turn_id: int | None = None
    harness: str | None = None


_trace_ctx: ContextVar[TraceContext | None] = ContextVar(
    "anton_trace_ctx", default=None
)


def get_trace_context() -> TraceContext | None:
    """Return the active trace context, or None if no turn is in flight."""
    return _trace_ctx.get()


def set_trace_context(ctx: TraceContext | None) -> Token:
    """Install a trace context for the current task; pair with `reset_trace_context`."""
    return _trace_ctx.set(ctx)


def reset_trace_context(token: Token) -> None:
    """Restore the previous trace context. Pass the token returned by `set_trace_context`."""
    _trace_ctx.reset(token)
