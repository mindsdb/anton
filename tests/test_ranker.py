"""Tests for the BM25 ranker (Layer 3 — Phase B).

The ranker decides which `## When` rules land in the system prompt
when the rule corpus exceeds the token budget. These tests pin the
core BM25 contract (relevance signals lift the right rules), the
edge cases (empty inputs), and the budget arithmetic (floor + cap).
"""

from __future__ import annotations

from anton.core.memory.ranker import Ranker, RankedRule, tokenize


class TestTokenize:
    def test_lowercases_and_drops_punct(self):
        assert tokenize("Use ONE scratchpad name per task!") == [
            "use", "one", "scratchpad", "name", "per", "task"
        ]

    def test_drops_stopwords(self):
        # 'the' / 'and' / 'of' / 'to' are stopwords.
        assert "the" not in tokenize("the scratchpad")
        assert "and" not in tokenize("save and exit")
        assert "of" not in tokenize("a lot of csv")
        assert "to" not in tokenize("write to disk")

    def test_keeps_numbers(self):
        # "5 KB" should keep "5" because rules like "cells over 5 KB"
        # leverage numeric matching against user messages.
        toks = tokenize("Keep cells under 5 KB")
        assert "5" in toks
        assert "kb" in toks

    def test_empty_input_returns_empty(self):
        assert tokenize("") == []
        assert tokenize(None) == []  # type: ignore[arg-type]


class TestRankRelevance:
    """The headline behaviour: relevant rules outrank irrelevant ones."""

    def test_pandas_query_lifts_pandas_rule(self):
        rules = [
            "Use httpx instead of requests for HTTP calls",
            "For CSV files with mixed types, pass low_memory=False to pd.read_csv",
            "Render HTML reports with explicit charset utf-8",
        ]
        ranked = Ranker().rank(rules, "process the sales.csv with pandas")
        # The pandas/csv rule should be at the top.
        assert "pd.read_csv" in ranked[0].text

    def test_html_query_lifts_html_rule(self):
        rules = [
            "Use httpx instead of requests for HTTP calls",
            "For CSV files with mixed types, pass low_memory=False to pd.read_csv",
            "Render HTML reports with explicit charset utf-8",
        ]
        ranked = Ranker().rank(rules, "build a dashboard HTML report")
        assert "HTML" in ranked[0].text

    def test_completely_irrelevant_query_still_returns_all_rules(self):
        rules = [
            "Use httpx instead of requests",
            "For CSV files use low_memory=False",
        ]
        ranked = Ranker().rank(rules, "blueberries are tasty")
        # No matches → all scores zero, but we still return every rule
        # so the budget step has something to pick from.
        assert len(ranked) == 2
        assert all(r.score == 0.0 for r in ranked)


class TestRankEdgeCases:
    def test_empty_corpus(self):
        assert Ranker().rank([], "anything") == []

    def test_empty_query_returns_input_order(self):
        rules = ["alpha rule", "beta rule", "gamma rule"]
        ranked = Ranker().rank(rules, "")
        assert [r.text for r in ranked] == rules
        assert all(r.score == 0.0 for r in ranked)

    def test_query_with_only_stopwords_returns_input_order(self):
        # "the and of to" tokenizes to nothing → cold-start fallback.
        rules = ["alpha rule", "beta rule"]
        ranked = Ranker().rank(rules, "the and of to")
        assert [r.text for r in ranked] == rules

    def test_token_estimate_is_positive(self):
        ranked = Ranker().rank(["short rule"], "rule")
        assert ranked[0].token_estimate >= 1

    def test_stable_tie_preserves_input_order(self):
        # Identical-on-the-query rules should keep input order.
        rules = ["pandas a", "pandas b", "pandas c"]
        ranked = Ranker().rank(rules, "pandas")
        # All three contain only "pandas" → identical scores → input order.
        assert [r.text for r in ranked] == rules


class TestSelectWithinBudget:
    def _make(self, n: int, tokens_each: int = 10) -> list[RankedRule]:
        # Descending scores so the order in the input is the rank order.
        return [
            RankedRule(text=f"rule-{i}", score=float(n - i), token_estimate=tokens_each)
            for i in range(n)
        ]

    def test_budget_caps_selection(self):
        ranked = self._make(10, tokens_each=20)
        out = Ranker().select_within_budget(ranked, budget_tokens=85)
        # Budget = 85 tokens, each rule 20 → 4 rules fit (80 used; 5th
        # would push to 100). With floor_k=3 the first 3 are auto-loaded,
        # then the 4th fits within budget.
        assert len(out) == 4

    def test_floor_k_loaded_even_when_budget_too_small(self):
        ranked = self._make(10, tokens_each=50)
        out = Ranker().select_within_budget(ranked, budget_tokens=10, floor_k=3)
        # Budget too small for even one rule, but floor_k=3 forces top-3.
        assert len(out) == 3

    def test_cap_k_limits_even_with_huge_budget(self):
        ranked = self._make(50, tokens_each=1)
        out = Ranker().select_within_budget(ranked, budget_tokens=10_000, cap_k=20)
        # 50 rules × 1 token each easily fits, but cap_k caps at 20.
        assert len(out) == 20

    def test_floor_k_capped_by_input_size(self):
        ranked = self._make(2, tokens_each=10)
        out = Ranker().select_within_budget(ranked, budget_tokens=5, floor_k=5)
        # Only 2 rules to choose from → floor_k can't add more.
        assert len(out) == 2

    def test_empty_ranked_returns_empty(self):
        assert Ranker().select_within_budget([], budget_tokens=100) == []
