"""The output budgets that depend on the reasoning effort."""

from __future__ import annotations

import inspect
from typing import Any

# Hidden reasoning spends from the same max_tokens as the visible answer.
# Measured on real agent steps: high and xhigh stay under 2.5k output tokens,
# max spends 8k-23k. Only streamed calls get these: a non-streamed call this
# large is refused by the Anthropic SDK (expected duration over ten minutes)
# and outlives the gateway's non-streaming time budget.
STREAM_BUDGET_BY_EFFORT: dict[str, int] = {"high": 16384, "xhigh": 32768, "max": 65536}
MAX_STREAM_BUDGET = 65536

TRUNCATION_RETRY_NOTE = "The answer was cut off — trying again"


def stream_budget_for(effort: object, default: int) -> int:
    """Output budget for a streamed call at ``effort``; never below ``default``."""
    table = STREAM_BUDGET_BY_EFFORT.get(effort, 0) if isinstance(effort, str) else 0
    return max(default, table)


def retry_budget(budget: int) -> int:
    """Budget for the one retry after a truncation; never below ``budget``."""
    return max(budget, min(budget * 2, MAX_STREAM_BUDGET))


def client_stream_budget(llm: Any, role: str, *, default: int = 8192) -> int:
    """The budget ``llm`` gives a streamed call of ``role``.

    Host clients and test doubles do not all have ``stream_budget``; an async
    mock has one that returns a coroutine. Those fall back to ``max_tokens``.
    """
    method = getattr(llm, "stream_budget", None)
    if callable(method) and not inspect.iscoroutinefunction(method):
        budget = method(role)
        if isinstance(budget, int) and budget > 0:
            return budget
    budget = getattr(llm, "max_tokens", None)
    return budget if isinstance(budget, int) and budget > 0 else default
