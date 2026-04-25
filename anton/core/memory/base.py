"""HippocampusProtocol — structural interface for memory backend swappability.

The Protocol defines the public contract of a Hippocampus instance so that
Enterprise adapters can provide alternate backends (e.g. database-backed,
cloud-synced) without inheriting from the file-based implementation.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable


@dataclass
class Engram:
    """A single memory trace — the fundamental unit of memory.

    Named for Karl Lashley's 'engram' — the physical substrate of a memory.
    Each engram carries its content plus metadata about confidence, origin,
    and topic for later retrieval and consolidation.
    """

    text: str
    kind: Literal["always", "never", "when", "lesson", "profile"] | None = None
    scope: Literal["global", "project"] | None = None
    confidence: Literal["high", "medium", "low"] | None = None
    topic: str = None
    source: Literal["user", "consolidation", "llm"] | None = None
    updated_at: dt.datetime = None

    def __post_init__(self):
        self.id = hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@runtime_checkable
class HippocampusProtocol(Protocol):
    """Structural protocol for a single-scope memory store.

    Implementors handle read/write at one scope (global or project).
    The concrete ``Hippocampus`` class in ``core/memory/hippocampus.py``
    satisfies this protocol automatically via structural sub-typing.
    """

    # --- read (string, for context assembly) ---

    def recall_identities(self) -> str:
        """Return the identity snapshot (profile.md equivalent)."""
        ...

    def recall_rules(self) -> str:
        """Return behavioral gates (rules.md equivalent)."""
        ...

    def recall_lessons(self, token_budget: int = 1000) -> str:
        """Return semantic facts within the given token budget."""
        ...

    def recall_topic(self, slug: str) -> str:
        """Return lessons tagged with the given topic slug."""
        ...

    def recall_scratchpad_wisdom(self) -> str:
        """Return procedural knowledge relevant to scratchpad execution."""
        ...

    # --- read (Engrams, for inspection / CRUD UI) ---

    def get_identities(self) -> list[Engram]:
        """Return identity entries as Engrams."""
        ...

    def get_rules(self) -> list[Engram]:
        """Return behavioral rules as Engrams."""
        ...

    def get_lessons(self, token_budget: int = None) -> list[Engram]:
        """Return semantic facts as Engrams, optionally budget-limited."""
        ...

    # --- write ---

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
        """Append a semantic fact to storage."""
        ...

    def rewrite_identity(self, entries: list[str]) -> None:
        """Merge new entries into the identity snapshot and rewrite."""
        ...

    def clear_identity(self) -> None:
        """Delete the identity snapshot file if present."""
        ...

    # --- update / delete ---

    def del_rule(self, id: str) -> None:
        """Delete a rule by its Engram id."""
        ...

    def update_rule(self, id: str, text: str) -> None:
        """Update a rule's text by its Engram id."""
        ...

    def del_lesson(self, id: str) -> None:
        """Delete a lesson by its Engram id."""
        ...

    def update_lesson(self, id: str, text: str) -> None:
        """Update a lesson's text by its Engram id."""
        ...

    def del_identity(self, id: str) -> None:
        """Delete an identity entry by its Engram id."""
        ...

    def update_identity(self, id: str, text: str) -> None:
        """Update an identity entry's text by its Engram id."""
        ...

    # --- maintenance ---

    def clear(self) -> None:
        """Wipe all memory files in this scope."""
        ...

    def entry_count(self) -> int:
        """Count total entries across rules and lessons stores."""
        ...
