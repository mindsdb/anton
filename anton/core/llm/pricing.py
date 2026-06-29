"""Per-model USD price table + cost computation for LLM token usage.

This module is *purely additive* telemetry: it turns the token counts Anton
already records on every LLM call (see ``Usage`` in ``provider.py``) into a
dollar figure a host can surface as "$ this turn / $ this task". It does not
gate, cap, or alter any request — there is no budget object and no enforcement
here by design (those are later slices).

Rates are USD per **one million** tokens, matching how providers publish them.
The table is matched by model-ID prefix in declaration order — exact/most
specific IDs first, family fallbacks last — mirroring the ``_CONTEXT_WINDOWS``
table in ``provider.py`` so the two stay stylistically aligned. Models not in
the table price at ``0.0`` rather than raising: an unpriced model must never
break a turn, and a zero cost is an honest "we don't have a rate for this".

Cache rates follow Anthropic's published multipliers relative to the base
input rate: a 5-minute cache *write* costs ~1.25x input, a cache *read* ~0.1x.
Anton does not enable prompt caching today (no ``cache_control`` is sent), so
``cache_write_tokens`` / ``cache_read_tokens`` are 0 in practice and the cache
terms contribute nothing — but the table and ``compute_cost`` carry them so the
meter stays correct the moment caching is turned on upstream.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    """USD rates for one model, per **one million** tokens.

    ``cache_write`` / ``cache_read`` default to the Anthropic-published
    multiples of ``input`` (1.25x write, 0.1x read) so a table entry only has
    to specify the two base rates unless a provider prices caching differently.
    """

    input: float
    output: float
    cache_write: float | None = None
    cache_read: float | None = None

    def cache_write_rate(self) -> float:
        return self.cache_write if self.cache_write is not None else self.input * 1.25

    def cache_read_rate(self) -> float:
        return self.cache_read if self.cache_read is not None else self.input * 0.1


# Matched by prefix in order — exact model IDs first, family fallbacks last.
# Rates are USD per 1M tokens. Keep this list maintained as the set of models
# Anton actually runs changes; an absent model prices at 0.0 (see compute_cost).
_MODEL_PRICES: list[tuple[str, ModelPrice]] = [
    # Anton defaults (exact model IDs first — see anton/config/settings.py)
    ("claude-sonnet-4-6", ModelPrice(input=3.00, output=15.00)),
    ("claude-haiku-4-5-20251001", ModelPrice(input=1.00, output=5.00)),
    # Claude families (most specific prefix first)
    ("claude-opus-4", ModelPrice(input=5.00, output=25.00)),
    ("claude-sonnet-4", ModelPrice(input=3.00, output=15.00)),
    ("claude-haiku-4", ModelPrice(input=1.00, output=5.00)),
    # OpenAI families (rates per 1M tokens)
    ("gpt-5", ModelPrice(input=1.25, output=10.00)),
    ("gpt-4.1", ModelPrice(input=2.00, output=8.00)),
    ("gpt-4o", ModelPrice(input=2.50, output=10.00)),
    ("o3", ModelPrice(input=2.00, output=8.00)),
    ("o1", ModelPrice(input=15.00, output=60.00)),
]

_PER_MILLION = 1_000_000.0


def get_model_price(model: str) -> ModelPrice | None:
    """Return the price entry whose prefix matches ``model``, or None.

    Matching is longest-declared-first by prefix, the same scheme
    ``compute_context_pressure`` uses for context windows.
    """
    if not model:
        return None
    for prefix, price in _MODEL_PRICES:
        if model.startswith(prefix):
            return price
    return None


def compute_cost(
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    cache_write_tokens: int | None = 0,
    cache_read_tokens: int | None = 0,
) -> float:
    """Return the USD cost of one LLM call's token usage.

    Any token count may be ``None`` — some providers omit a usage field (e.g.
    the MindsHub passthrough returns ``input_tokens=None`` on web-search
    responses), so a missing count is treated as 0 rather than crashing, exactly
    as ``compute_context_pressure`` does. An unknown/unpriced model returns
    ``0.0``. The result is always a non-negative float in dollars.
    """
    price = get_model_price(model)
    if price is None:
        return 0.0
    cost = (
        (input_tokens or 0) * price.input
        + (output_tokens or 0) * price.output
        + (cache_write_tokens or 0) * price.cache_write_rate()
        + (cache_read_tokens or 0) * price.cache_read_rate()
    )
    return cost / _PER_MILLION
