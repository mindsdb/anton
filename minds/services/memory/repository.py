"""MemoryRepository — thin data access layer for mind memory."""

from __future__ import annotations

from uuid import UUID

from sqlmodel import Session, and_, select

from minds.model.memory_rule import MemoryRule
from minds.model.memory_topic import MemoryTopic


class MemoryRepository:
    def __init__(self, session: Session, mind_id: str | UUID) -> None:
        self.session = session
        self.mind_id = mind_id

    def get_active_rules(self) -> list[MemoryRule]:
        """Return all non-deleted rules for the mind."""
        statement = select(MemoryRule).where(
            and_(
                MemoryRule.mind_id == self.mind_id,
                MemoryRule.deleted_at.is_(None),  # type: ignore[union-attr]
            )
        )
        return list(self.session.exec(statement).all())

    def get_active_topics(self) -> list[MemoryTopic]:
        """Return all non-deleted topics for the mind, including body."""
        statement = select(MemoryTopic).where(
            and_(
                MemoryTopic.mind_id == self.mind_id,
                MemoryTopic.deleted_at.is_(None),  # type: ignore[union-attr]
            )
        )
        return list(self.session.exec(statement).all())
