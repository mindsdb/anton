from __future__ import annotations

from anton.core.llm.pricing import ModelPrice, compute_cost, get_model_price
from anton.core.llm.provider import Usage


class TestGetModelPrice:
    def test_exact_default_models_match(self):
        # Anton's two default models (anton/config/settings.py) must be priced.
        assert get_model_price("claude-sonnet-4-6") is not None
        assert get_model_price("claude-haiku-4-5-20251001") is not None

    def test_dated_haiku_id_matches_exact_entry_before_family(self):
        # The coding default is the dated id; it must resolve to the dated
        # entry, not fall through to a looser family prefix.
        price = get_model_price("claude-haiku-4-5-20251001")
        assert price is not None and price.input == 1.00 and price.output == 5.00

    def test_family_prefix_fallback(self):
        # An opus model with a date suffix still matches the "claude-opus-4"
        # family prefix.
        price = get_model_price("claude-opus-4-8")
        assert price is not None and price.input == 5.00 and price.output == 25.00

    def test_unknown_model_returns_none(self):
        assert get_model_price("totally-made-up-model") is None

    def test_empty_model_returns_none(self):
        assert get_model_price("") is None


class TestModelPriceCacheRates:
    def test_cache_rates_default_to_input_multiples(self):
        # 1.25x write, 0.1x read, relative to the base input rate.
        p = ModelPrice(input=4.00, output=20.00)
        assert p.cache_write_rate() == 5.00
        assert p.cache_read_rate() == 0.40

    def test_explicit_cache_rates_win(self):
        p = ModelPrice(input=4.00, output=20.00, cache_write=9.99, cache_read=0.01)
        assert p.cache_write_rate() == 9.99
        assert p.cache_read_rate() == 0.01


class TestComputeCost:
    def test_input_and_output_priced_per_million(self):
        # Sonnet: $3/1M in, $15/1M out → 1M each = $18.00 exactly.
        assert compute_cost("claude-sonnet-4-6", 1_000_000, 1_000_000) == 18.00

    def test_small_token_counts(self):
        # Haiku: 1000 in @ $1/1M = $0.001, 2000 out @ $5/1M = $0.010 → $0.011.
        assert compute_cost("claude-haiku-4-5-20251001", 1000, 2000) == 0.011

    def test_opus_family(self):
        # Opus: 100 in @ $5/1M + 50 out @ $25/1M = 0.0005 + 0.00125 = 0.00175.
        assert compute_cost("claude-opus-4-8", 100, 50) == 0.00175

    def test_cache_tokens_priced_additively(self):
        # Anthropic reports cache tokens separately from input, so compute_cost
        # adds them: sonnet cache_read 1M = 0.1 * $3 = $0.30.
        cost = compute_cost("claude-sonnet-4-6", 0, 0, 0, 1_000_000)
        assert round(cost, 6) == 0.30

    def test_cache_write_more_expensive_than_read(self):
        write_only = compute_cost("claude-sonnet-4-6", 0, 0, 1_000_000, 0)
        read_only = compute_cost("claude-sonnet-4-6", 0, 0, 0, 1_000_000)
        # 1.25x input vs 0.1x input.
        assert round(write_only, 6) == 3.75
        assert write_only > read_only

    def test_full_breakdown_sums_all_four_components(self):
        # input + output + cache_write + cache_read, all per-million.
        cost = compute_cost("claude-sonnet-4-6", 1_000_000, 1_000_000, 1_000_000, 1_000_000)
        # 3 + 15 + 3.75 + 0.30 = 22.05
        assert round(cost, 6) == 22.05

    def test_zero_tokens_is_zero(self):
        assert compute_cost("claude-sonnet-4-6", 0, 0) == 0.0

    def test_none_tokens_treated_as_zero_not_crash(self):
        # Mirrors compute_context_pressure: the MindsHub passthrough can return
        # usage.input_tokens=None; cost must not raise on None * float.
        assert compute_cost("claude-sonnet-4-6", None, None) == 0.0
        assert compute_cost("claude-sonnet-4-6", None, 1_000_000) == 15.00

    def test_unknown_model_costs_zero_not_crash(self):
        # An unpriced model must never break a turn — price it at 0.0.
        assert compute_cost("some-unlisted-model", 1_000_000, 1_000_000) == 0.0

    def test_empty_model_costs_zero(self):
        assert compute_cost("", 1000, 1000) == 0.0


class TestUsageCarriesCost:
    def test_usage_has_additive_cost_fields_with_safe_defaults(self):
        # The cost meter rides on the existing Usage dataclass; defaults keep
        # every prior construction site valid (additive-only change).
        u = Usage(input_tokens=10, output_tokens=20)
        assert u.cost_usd == 0.0
        assert u.cache_write_tokens == 0
        assert u.cache_read_tokens == 0

    def test_usage_accepts_populated_cost(self):
        u = Usage(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cost_usd=compute_cost("claude-sonnet-4-6", 1_000_000, 1_000_000),
        )
        assert u.cost_usd == 18.00
