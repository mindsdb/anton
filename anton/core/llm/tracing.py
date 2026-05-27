"""Per-turn trace identity for outbound LLM telemetry.

`ChatSession.turn_stream` sets the active `TraceContext` for the
duration of a turn. The OpenAI provider reads it when talking to
MindsHub and attaches langfuse-style headers so every LLM call (and
any nested tool/scratchpad LLM call made within the same asyncio
task) is attributed to the same session + turn server-side.

The per-turn `trace_id` (a 32-char hex id) is sent as `Langfuse-Trace-Id`
on every passthrough call so the gateway *groups and nests* a whole turn's
calls under one trace, and `trace_input` (the user's request) is sent as
`Langfuse-Trace-Input` so the trace has clean input → output for evals. The
gateway takes a trace's output from the last call to finish, so post-answer
bookkeeping calls (completion verification, memory consolidation) must run
inside `detached_trace()` — which drops the trace id so they neither group
under nor clobber the turn trace.

A `ContextVar` is used so that nested calls — `_stream_and_handle_tools`,
`generate_object` (structured output), the cerebellum's diff call,
and the scratchpad's `coding_provider` calls — all inherit the same
trace automatically without threading kwargs through every layer.

Scope: only consumed by the OpenAI provider when its base URL points
at MindsHub. Other providers (direct Anthropic, raw OpenAI, Azure,
Gemini) ignore the context entirely.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class TraceContext:
    """Identifiers attached to outbound LLM calls during a turn."""

    session_id: str | None = None
    turn_id: int | None = None
    harness: str | None = None
    # Per-turn trace id (32-char hex). Groups every passthrough call in a turn
    # under one server-side trace. None = ungrouped (each call its own trace).
    trace_id: str | None = None
    # The user's request for this turn; surfaced as the trace's input for evals.
    trace_input: str | None = None


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


@contextmanager
def detached_trace():
    """Run a block off the current turn's eval trace.

    Drops `trace_id`/`trace_input` for the duration so bookkeeping or
    post-answer calls (completion verification, memory consolidation) don't
    group under — or clobber the output of — the turn trace, while keeping
    session/turn/harness attribution intact. No-op when no context is active.

    Because `asyncio.create_task` snapshots the current context at creation,
    wrapping a `create_task(...)` call in this manager detaches the spawned
    task too.
    """
    ctx = get_trace_context()
    if ctx is None:
        yield
        return
    token = set_trace_context(replace(ctx, trace_id=None, trace_input=None))
    try:
        yield
    finally:
        reset_trace_context(token)
