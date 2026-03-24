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

    def load_raw(self) -> tuple[list[MemoryRule], list[MemoryTopic]]:
        """
        Fetch all active rules and topics from the DB.

        Callers should cache this result and pass it to select_for_query()
        on each turn to avoid repeated DB round-trips.
        """
        return self.repo.get_active_rules(), self.repo.get_active_topics()

    def select_for_query(
        self,
        rules: list[MemoryRule],
        topics: list[MemoryTopic],
        query: str,
    ) -> MemoryBlock:
        """
        Score pre-loaded topics against a query and apply budget/count caps.

        No DB call — pass the output of load_raw() here.
        """
        remaining_budget = self.token_budget - sum(_token_count(r.content) for r in rules)
        scored_topics = self._score_topics(topics, query)
        selected_topics = self._apply_caps(scored_topics, remaining_budget)
        return MemoryBlock(rules=rules, topics=selected_topics)

    def load_for_session(self, query: str) -> MemoryBlock:
        """
        Load and select memory relevant to query.

        Convenience wrapper around load_raw() + select_for_query() for
        callers that only need a single-shot load. For per-turn re-scoring
        without repeated DB calls, use load_raw() once and call
        select_for_query() on each turn.
        """
        rules, topics = self.load_raw()
        return self.select_for_query(rules, topics, query)

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
    def _sanitize_rule_content(content: str) -> str:
        """Collapse rule content into a single line safe for a markdown bullet."""
        lines = [line.strip() for line in content.splitlines()]
        return " ".join(line for line in lines if line)

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
                items = [
                    MemoryService._sanitize_rule_content(r.content) for r in block.rules if r.rule_type == rule_type
                ]
                items = [i for i in items if i]
                if items:
                    header = f"### {rule_type.value.capitalize()}"
                    sections.append(header + "\n" + "\n".join(f"- {c}" for c in items))
            if sections:
                parts.append(
                    "## MANDATORY RULES\nApply every rule below carefully where relevant.\n\n" + "\n\n".join(sections)
                )

        if block.topics:
            topic_sections = [f"### {t.title}\n{t.body}" for t in block.topics]
            parts.append("## Memory Topics\n" + "\n\n".join(topic_sections))

        return "\n\n".join(parts)

    @staticmethod
    def format_rules_reminder(rules: list[MemoryRule]) -> str:
        """Short end-of-prompt reminder referencing rules without repeating them."""
        if not rules:
            return ""
        count = sum(1 for r in rules if MemoryService._sanitize_rule_content(r.content))
        if count == 0:
            return ""
        return f"REMINDER: You have {count} mandatory rule(s) defined above. Apply each one carefully where relevant."
