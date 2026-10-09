"""Streamed model calls for the artifact pipeline: draining and drop retry."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from anton.core.llm.provider import is_stream_drop

if TYPE_CHECKING:
    from anton.core.llm.provider import LLMResponse

logger = logging.getLogger(__name__)


async def _drain_stream(events, on_text=None) -> "LLMResponse":
    """Consume a `plan_stream()`/`code_stream()` iterator and return the final
    assembled response; hand each text delta to ``on_text`` when given.

    Used in place of the one-shot `plan()`/`code()` calls. This pipeline runs
    headless — its own progress surface is the step-level `ToolProgress`
    protocol — so nothing needs the intermediate `StreamToolUse*` events,
    only the terminal `StreamComplete`. The text deltas have one taker:
    `GenState.peek_for` feeds them to the live tail in the spinner footer
    (`progress.LivePeek`), so a minute-long write is not a frozen screen.
    The reason to stream at all is transport, not UX: a
    non-streaming call sends no bytes over the wire until the whole response
    is ready, and `api.mindshub.ai` sits behind Cloudflare, which kills a
    connection that has been silent for ~100s with a 524 — a real failure on
    long spec/code generations.

    For TEXT that works: bytes flow continuously and the proxy never observes
    silence. For a large TOOL-CALL argument it does NOT, and the original
    version of this docstring was wrong to claim otherwise. Measured
    2026-08-28: generating a 59 000-character `write_file` argument produced
    its first stream event at ~2s, then nothing for 112 seconds, then every
    remaining event in a single burst. The same profile appears when talking
    straight to `api.anthropic.com`, so it is not the gateway's doing and
    cannot be fixed on our side — the argument simply is not streamed
    incrementally.

    Consequence: a long tool-call generation IS a silent connection, and
    whether it survives is a race against the proxy's idle timeout. Hence
    `_call_with_stream_retry` below.
    """
    from anton.core.llm.provider import StreamComplete, StreamTextDelta

    result = None
    async for event in events:
        if isinstance(event, StreamComplete):
            result = event.response
        elif on_text is not None and isinstance(event, StreamTextDelta) and event.text:
            on_text(event.text)
    if result is None:
        raise RuntimeError("LLM stream ended without a StreamComplete event")
    return result


# Floor for the halved retry budget — below this a chunk is too small to make
# progress and the round is wasted either way.
_RETRY_BUDGET_FLOOR = 2048


async def _call_with_stream_retry(
    llm_call,
    *,
    system: str,
    messages: list[dict],
    tools: list[dict] | None,
    max_tokens: int | None,
    default_cap: int | None,
    on_text=None,
) -> tuple["LLMResponse", int | None]:
    """One LLM round, retried once if the connection dies mid-stream.

    Returns the response and the budget it actually ran on — the caller needs
    the latter to judge truncation, and the retry deliberately does not run on
    the same budget as the first try.

    Retrying with IDENTICAL parameters would mostly reproduce the failure: the
    drop is a race between how long the generation stays silent (see
    `_drain_stream`) and the proxy's idle timeout, and neither changes on a
    re-run. So the retry halves the budget, which halves the silence. If the
    shorter budget truncates instead, that is a strictly better outcome — the
    loop already recovers from truncation by asking for a smaller chunk, while
    a dropped connection would otherwise end the whole generation.

    Nothing has been executed when a drop happens: tool calls run only after
    the stream is fully drained, so a retry cannot double-apply a write.
    """
    budget = max_tokens
    try:
        return await _drain_stream(
            llm_call(system=system, messages=messages, tools=tools, max_tokens=budget),
            on_text,
        ), budget
    except Exception as exc:
        if not is_stream_drop(exc):
            raise
        effective = budget or default_cap
        retry_budget = max(_RETRY_BUDGET_FLOOR, effective // 2) if effective else None
        logger.warning(
            "generate_artifact: stream dropped mid-generation; retrying once "
            "with a halved output budget (%s -> %s)", effective, retry_budget,
        )
    return await _drain_stream(
        llm_call(system=system, messages=messages, tools=tools, max_tokens=retry_budget),
        on_text,
    ), retry_budget
