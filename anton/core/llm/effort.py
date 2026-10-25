"""The output budgets that depend on the reasoning effort."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

# Hidden reasoning spends from the same max_tokens as the visible answer.
# Measured on real agent steps: high and xhigh stay under 2.5k output tokens,
# max spends 8k-23k. Only streamed calls get these: a non-streamed call this
# large is refused by the Anthropic SDK (expected duration over ten minutes)
# and outlives the gateway's non-streaming time budget.
STREAM_BUDGET_BY_EFFORT: dict[str, int] = {"high": 16384, "xhigh": 32768, "max": 65536}
MAX_STREAM_BUDGET = 65536

# A file-writing round at max thinks through the whole page before it writes a
# line: measured 2026-10-24 through MindsHub, `sonnet-5-5` wrote the page in
# 103553 tokens and `opus-5-5` in 67465, and neither finished at 65536. So a
# write round at max gets 128000, the output ceiling of every model MindsHub
# offers max on; MindsHub clamps a larger request to the model's own ceiling.
WRITE_BUDGET_BY_EFFORT: dict[str, int] = {**STREAM_BUDGET_BY_EFFORT, "max": 128000}

# Hidden thinking sends nothing on MindsHub's chat completions, so a call can be
# silent for most of its budget: before writing, `sonnet-5-5` was silent for
# 655 s and `opus-5-5` for 536 s (2026-10-24, about 150 tokens a second). A call
# with a budget above SILENCE_COVERED_BUDGET may stay silent one second per
# SILENT_TOKENS_PER_SECOND tokens of it; the default 600 s deadline already
# covers a budget up to that one.
SILENT_TOKENS_PER_SECOND = 100
SILENCE_COVERED_BUDGET = 600 * SILENT_TOKENS_PER_SECOND
# The longest silence any call may get, which the SDK clients' read timeout must
# not cut short.
LONGEST_SILENCE_S = max(WRITE_BUDGET_BY_EFFORT.values()) / SILENT_TOKENS_PER_SECOND

TRUNCATION_RETRY_NOTE = "The answer was cut off — trying again"


def stream_budget_for(
    effort: object, default: int, by_effort: Mapping[str, int] = STREAM_BUDGET_BY_EFFORT
) -> int:
    """Output budget for a streamed call at ``effort``; never below ``default``."""
    table = by_effort.get(effort, 0) if isinstance(effort, str) else 0
    return max(default, table)


def retry_budget(budget: int) -> int:
    """Budget for the one retry after a truncation; never below ``budget``."""
    return max(budget, min(budget * 2, MAX_STREAM_BUDGET))


def client_stream_budget(
    llm: Any, role: str, *, default: int = 8192, by_effort: Mapping[str, int] | None = None
) -> int:
    """The budget ``llm`` gives a streamed call of ``role``, from ``by_effort``
    when given, else from the client's own table.

    Host clients and test doubles do not all have ``stream_budget``; an async
    mock has one that returns a coroutine. Those fall back to ``max_tokens``.
    """
    method = getattr(llm, "stream_budget", None)
    if callable(method) and not inspect.iscoroutinefunction(method):
        budget = method(role) if by_effort is None else method(role, by_effort=by_effort)
        if isinstance(budget, int) and budget > 0:
            return budget
    budget = getattr(llm, "max_tokens", None)
    return budget if isinstance(budget, int) and budget > 0 else default
