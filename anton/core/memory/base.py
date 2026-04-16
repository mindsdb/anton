"""HippocampusProtocol — structural interface for memory backend swappability.

The Protocol defines the public contract of a Hippocampus instance so that
Enterprise adapters can provide alternate backends (e.g. database-backed,
cloud-synced) without inheriting from the file-based implementation.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable


@runtime_checkable
class HippocampusProtocol(Protocol):
    """Structural protocol for a single-scope memory store.

    Implementors handle read/write at one scope (global or project).
    The concrete ``Hippocampus`` class in ``core/memory/hippocampus.py``
    satisfies this protocol automatically via structural sub-typing.
    """

    def recall_identity(self) -> str:
        """Return the identity snapshot (profile.md equivalent)."""
        ...

    def recall_rules(self) -> str:
        """Return behavioral gates (rules.md equivalent)."""
        ...

    def recall_lessons(self, token_budget: int = 1000) -> str:
        """Return semantic facts within the given token budget."""
        ...

    def recall_topic(self, slug: str) -> str:
        """Return deep domain expertise for a topic slug."""
        ...

    def recall_scratchpad_wisdom(self) -> str:
        """Return procedural knowledge relevant to scratchpad execution."""
        ...

    def encode_rule(
        self,
        text: str,
        kind: Literal["always", "never", "when"],
        confidence: str = "medium",
        source: str = "llm",
    ) -> None:
        """Write a behavioral gate to storage."""
        ...

    def encode_lesson(
        self,
        text: str,
        topic: str = "",
        source: str = "llm",
    ) -> None:
        """Write a semantic fact to storage."""
        ...

    def rewrite_identity(self, entries: list[str]) -> None:
        """Replace the identity snapshot in full."""
        ...

    def entry_count(self) -> int:
        """Count total entries across rules and lessons stores."""
        ...
