from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anton.memory.learnings import LearningStore
    from anton.memory.store import SessionStore


class MemoryContext:
    """Builds the memory context string injected into the planner's system prompt."""

    def __init__(self, session_store: SessionStore, learning_store: LearningStore) -> None:
        self._sessions = session_store
        self._learnings = learning_store

    def build(self, task: str) -> str:
        sections: list[str] = []

        # Recent session summaries
        summaries = self._sessions.get_recent_summaries(limit=3)
        if summaries:
            lines = []
            for i, s in enumerate(summaries, 1):
                # Truncate long summaries
                preview = s[:300] + "..." if len(s) > 300 else s
                lines.append(f"{i}. {preview}")
            sections.append("## Recent Activity\n" + "\n".join(lines))

        # Relevant learnings
        learnings = self._learnings.find_relevant(task, limit=3)
        if learnings:
            lines = []
            for item in learnings:
                lines.append(f"### {item['topic']}\n{item['content']}")
            sections.append("## Relevant Learnings\n" + "\n".join(lines))

        return "\n\n".join(sections)
