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

import contextlib
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace


# Where the user was when the turn happened (ENG-1459). This is the CANONICAL
# definition; cowork-server imports these rather than repeating the strings, so
# the field cannot pick up a second vocabulary the way ``harness`` did.
#
# `surface` answers WHERE, `harness` answers WHICH AGENT. They are orthogonal:
# a `web` turn and a `desktop` turn can both run the anton agent, and after
# ENG-1694 `harness` will carry only the agent identity (`anton` / `hermes`).
#
#   SURFACE_DESKTOP  the Electron app
#   SURFACE_WEB      the SaaS build (cowork-server derives it from org tenancy)
#   SURFACE_CLI      anton's own interactive chat
#
# Deliberately NOT a value here: the cloud one-turn-per-pod path. That is an
# *execution mode*, not a place a user sits — cowork is its caller, so one web
# turn could legitimately be both — and folding it in would recreate exactly
# the two-vocabularies-in-one-field problem this pair of fields is undoing.
SURFACE_DESKTOP = "desktop"
SURFACE_WEB = "web"
SURFACE_CLI = "cli"
VALID_SURFACES = frozenset({SURFACE_DESKTOP, SURFACE_WEB, SURFACE_CLI})

# Tags are the only Langfuse dimension a filter can reach cheaply, but they are
# a flat namespace shared with caller-supplied tags — so the value is prefixed.
# Two reasons it is not emitted bare: an unprefixed `cli` would be
# indistinguishable from the `harness` tag of the same name while ENG-1694 is
# still in flight, and `origin:` (ENG-1289) already set this convention one
# change earlier.
#
# NOTE for anyone querying it: tags are FILTERABLE, not groupable.
# ``dimensions: [{field: "tags"}]`` keys on the whole tag array as a tuple and
# returns roughly one row per trace. Use
# ``filters: [{column: "tags", operator: "any of", value: ["surface:web"], type: "arrayOptions"}]``
# and count, one query per surface.
SURFACE_TAG_PREFIX = "surface:"


def surface_tag(surface: str) -> str:
    """Render a surface as its Langfuse tag (``web`` -> ``surface:web``)."""
    return f"{SURFACE_TAG_PREFIX}{surface}"


@dataclass(frozen=True)
class TraceContext:
    """Identifiers attached to outbound LLM calls during a turn."""

    session_id: str | None = None
    turn_id: int | None = None
    harness: str | None = None
    # Where the user was: one of VALID_SURFACES, or None when the host did not
    # say. None is a real answer ("unknown host"), not a default to be guessed
    # at — the same reservation ``harness`` makes, and the reason ENG-1495 had
    # to stop the CLI reporting an empty string.
    surface: str | None = None
    # Optional, caller-supplied trace annotations forwarded verbatim to the
    # langfuse-style headers (see ``OpenAIProvider._build_trace_headers``).
    # `tags` are appended to ``Langfuse-Tags``; `metadata` is merged into
    # ``Langfuse-Metadata`` (built-in keys win on collision). Kept generic so
    # hosts can attach arbitrary correlation data — e.g. an eval harness adding
    # an eval-run id — without changing this structure.
    tags: tuple[str, ...] = ()
    metadata: dict[str, str] | None = None


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


@contextlib.contextmanager
def tagged_trace(*tags: str):
    """Append `tags` to the active trace for the duration of the block (ENG-1390).

    For isolating ONE call site among the many that share an `LLMClient` entry
    point. Tags ride the ``Langfuse-Tags`` header, and tags are one of the few
    dimensions the Langfuse metrics API can group by — so a tagged call stays
    countable without reading trace payloads, which matters while ENG-1392 is
    redacting them.

    Deliberately a no-op rather than an error when there is nothing to annotate:
    no turn in flight (the CLI `turn()` path never installs a context), or a
    provider that ignores trace context entirely (anything but the
    MindsHub-routed OpenAI provider — a BYOK user on direct Anthropic has no
    gateway trace to tag in the first place). Callers can therefore annotate
    unconditionally without branching on either condition.
    """
    ctx = _trace_ctx.get()
    if ctx is None or not tags:
        yield
        return
    token = _trace_ctx.set(replace(ctx, tags=ctx.tags + tuple(tags)))
    try:
        yield
    finally:
        try:
            _trace_ctx.reset(token)
        except ValueError:
            # Cross-context teardown: when an abandoned async generator is
            # finalized, the cleanup can run in a COPIED context, where resetting
            # a token created elsewhere raises. `ChatSession.turn_stream` already
            # carries this exact guard (#309 review), where an unguarded reset
            # aborted a `finally` and dropped the whole turn's books. The copy
            # dies with the context, so there is nothing to restore.
            pass
