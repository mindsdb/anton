"""MemoryService — scoring, budget accounting, and prompt formatting."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from minds.model.memory_rule import MemoryRule, RuleType
from minds.model.memory_topic import MemoryTopic

from .repository import MemoryRepository


def _normalize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters."""
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def _token_count(text: str) -> int:
    """Approximate token count (1 token ≈ 4 characters)."""
    return len(text) // 4


@dataclass
class MemoryBlock:
    rules: list[MemoryRule] = field(default_factory=list)
    topics: list[MemoryTopic] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.rules and not self.topics


class MemoryService:
    """
    Loads and scores mind memory for a ChatSession.

    Call load_for_session() once per session — results are deterministic and
    should be cached by the caller for the session lifetime.
    """

    def __init__(
        self,
        repo: MemoryRepository,
        token_budget: int = 3000,
        max_topics: int = 5,
    ) -> None:
        self.repo = repo
        self.token_budget = token_budget
        self.max_topics = max_topics

    def load_for_session(self, query: str) -> MemoryBlock:
        """
        Load and select memory relevant to query.

        Rules: all active rules are always included. Their token cost is
        deducted before topic selection.

        Topics: scored by keyword match against title/tags/description,
        capped at max_topics and the remaining token budget.
        """
        rules = self.repo.get_active_rules()
        topics = self.repo.get_active_topics()

        remaining_budget = self.token_budget - sum(_token_count(r.content) for r in rules)

        scored_topics = self._score_topics(topics, query)
        selected_topics = self._apply_caps(scored_topics, remaining_budget)

        return MemoryBlock(rules=rules, topics=selected_topics)

    def _score_topics(self, topics: list[MemoryTopic], query: str) -> list[MemoryTopic]:
        """
        Score topics by keyword overlap with query.

        Weights: title ×3, each matching tag ×2, description ×1.
        Returns topics sorted by score descending (zero-score topics included last).
        """
        query_tokens = set(_normalize(query))
        if not query_tokens:
            return list(topics)

        def score(topic: MemoryTopic) -> float:
            s = len(query_tokens & set(_normalize(topic.title))) * 3
            if topic.tags:
                for tag in topic.tags:
                    if query_tokens & set(_normalize(tag)):
                        s += 2
            if topic.description:
                s += len(query_tokens & set(_normalize(topic.description))) * 1
            return s

        return sorted(topics, key=score, reverse=True)

    def _apply_caps(self, topics: list[MemoryTopic], budget: int) -> list[MemoryTopic]:
        """Apply count cap (max_topics) then token budget cap."""
        selected: list[MemoryTopic] = []
        remaining = budget
        for topic in topics[: self.max_topics]:
            cost = _token_count(topic.body)
            if remaining - cost < 0:
                continue
            selected.append(topic)
            remaining -= cost
        return selected

    @staticmethod
    def format_block(block: MemoryBlock) -> str:
        """
        Render memory as a markdown string for system prompt injection.
        Returns an empty string if the block is empty.
        """
        if block.is_empty:
            return ""

        parts: list[str] = []

        if block.rules:
            sections: list[str] = []
            for rule_type in (RuleType.always, RuleType.never, RuleType.when):
                matching = [r.content for r in block.rules if r.rule_type == rule_type]
                if matching:
                    header = f"# {rule_type.value.capitalize()}"
                    sections.append(header + "\n" + "\n".join(f"- {c}" for c in matching))
            parts.append("## Memory Rules\n\n" + "\n\n".join(sections))

        if block.topics:
            topic_sections = [f"### {t.title}\n{t.body}" for t in block.topics]
            parts.append("## Memory Topics\n" + "\n\n".join(topic_sections))

        return "\n\n".join(parts)
