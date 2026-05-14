"""BM25-style lexical ranker over rule text — Layer 3 of the ACC path.

Scores each rule against the current turn's query (the user message)
and returns them sorted by relevance. The result drives which rules
land in the system prompt under the ``## When`` section when the
total rule corpus exceeds the token budget.

Why BM25 and not embeddings (v1):

  - No new dependencies. anton has no embeddings client today, and
    adding one means another API surface (or a local model + tokenizer)
    that has to live alongside the existing planning/coding split.
  - No LLM call on the prompt-assembly hot path. Build-memory-context
    runs at the start of every turn; a synchronous, deterministic
    ranker keeps the cold-start latency low.
  - For 1–3 sentence rules + short user messages, lexical matching
    works because rules contain domain nouns ('pandas', 'CSV',
    'publish', 'scratchpad', 'datavault') and user messages mention
    those same nouns. The cases where pure lexical matching fails
    (paraphrase, abstract relevance) also happen to be the cases
    where the corpus is small enough to fit in the budget without
    ranking — so the failure mode is graceful.
  - Microsecond-cheap: O(rules × query_terms). For ~50 rules × ~20
    query terms the entire rank() call is sub-millisecond.

Designed to be **drop-in replaceable** by an embedding-based ranker
in v2. Keep this interface narrow (``rank`` + ``select_within_budget``)
so swapping implementations doesn't touch the cortex call site.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


# Minimal stopword list. Stemming deliberately omitted — adds dep + bugs
# for marginal lift on these short documents. The list is intentionally
# small; over-stopping (e.g. removing "when", "if") would hurt rule
# texts that pivot on those words.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at",
    "to", "from", "for", "with", "by", "is", "are", "was", "were",
    "be", "been", "being", "do", "does", "did", "doing", "have",
    "has", "had", "having", "will", "would", "could", "should",
    "may", "might", "can", "this", "that", "these", "those",
    "you", "your", "yours", "it", "its", "as", "so", "not",
})


_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase → strip punctuation → drop stopwords. Preserves
    digits so rule text like 'over 5 KB' keeps '5' as a queryable
    term. Public for testing and for the Phase C outcome bridge,
    which compares rule text against detector lesson text."""
    if not text:
        return []
    return [t for t in _WORD_RE.findall(text.lower()) if t not in _STOPWORDS]


@dataclass(frozen=True)
class RankedRule:
    """One rule + its BM25 score for the current query.

    ``token_estimate`` is a rough word count used for budget
    arithmetic in ``select_within_budget``. We deliberately don't
    import a real tokenizer here — that would couple the ranker to
    the LLM provider. The 1.3× word-count multiplier matches typical
    English BPE expansion well enough for budget enforcement that
    only needs to be approximately right.
    """

    text: str
    score: float
    token_estimate: int


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


class Ranker:
    """BM25 ranker with budget-aware selection.

    Tunable parameters use the standard Robertson defaults
    (``k1=1.5``, ``b=0.75``). We haven't tuned them against real rule
    corpora — the defaults are fine for v1 because the corpus is
    tiny (typically <50 rules). When the corpus grows and tuning
    matters, switch to embeddings rather than tuning BM25 — the
    upgrade path is more productive than the tuning one.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self._k1 = k1
        self._b = b

    def rank(self, rule_texts: list[str], query: str) -> list[RankedRule]:
        """Rank rules by BM25 score against ``query``.

        Empty query → return rules in their input order with
        ``score=0.0``. This is the cold-start case (no user message
        yet, or the message contains no scorable terms after stopword
        removal). Falling back to insertion order rather than dropping
        all rules keeps the system prompt intact for turn 1.

        Stable for ties: rules with equal scores keep their input
        order (Python's sort is stable). When the consolidator
        eventually wants "most recently added rule wins ties", it can
        pre-sort the input list to match.
        """
        if not rule_texts:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return [
                RankedRule(text=t, score=0.0, token_estimate=_estimate_tokens(t))
                for t in rule_texts
            ]

        docs = [tokenize(t) for t in rule_texts]
        doc_lens = [len(d) for d in docs]
        avgdl = (sum(doc_lens) / len(doc_lens)) if doc_lens else 1.0
        avgdl = max(avgdl, 1.0)

        # Document frequency per term across the rule corpus.
        df: dict[str, int] = {}
        for d in docs:
            for term in set(d):
                df[term] = df.get(term, 0) + 1

        N = len(docs)

        def idf(term: str) -> float:
            n = df.get(term, 0)
            # BM25's "plus-one" IDF — strictly non-negative, well-defined
            # when a term is in every doc (n == N) or none (n == 0).
            return math.log((N - n + 0.5) / (n + 0.5) + 1.0)

        scored: list[RankedRule] = []
        for i, doc in enumerate(docs):
            tfs = Counter(doc)
            doc_len = doc_lens[i] or 1
            score = 0.0
            for term in query_terms:
                tf = tfs.get(term, 0)
                if tf == 0:
                    continue
                numer = tf * (self._k1 + 1)
                denom = tf + self._k1 * (1 - self._b + self._b * doc_len / avgdl)
                score += idf(term) * (numer / denom)
            scored.append(RankedRule(
                text=rule_texts[i],
                score=score,
                token_estimate=_estimate_tokens(rule_texts[i]),
            ))

        # Stable sort: ties preserve input order. reverse=True → desc.
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored

    def select_within_budget(
        self,
        ranked: list[RankedRule],
        budget_tokens: int,
        *,
        floor_k: int = 3,
        cap_k: int = 20,
    ) -> list[RankedRule]:
        """Take rules in ranked order until we hit the budget.

        - ``floor_k``: always load the top N rules even if they
          exceed the budget. The whole point of the ranker is that
          the most-relevant rules are valuable; under a tiny budget,
          a hard cutoff would defeat the purpose. Skip floor
          enforcement implicitly by passing fewer than ``floor_k``
          rules.
        - ``cap_k``: never load more than this many even if the
          budget allows. A handful of well-chosen rules is more
          useful to the LLM than a wall of them, and BM25 scores
          drop off sharply past the top results.
        """
        out: list[RankedRule] = []
        used = 0
        for i, r in enumerate(ranked):
            if i >= cap_k:
                break
            if i < floor_k:
                out.append(r)
                used += r.token_estimate
                continue
            if used + r.token_estimate > budget_tokens:
                break
            out.append(r)
            used += r.token_estimate
        return out
