from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, and_, select

from minds.model.memory_rule import MemoryRule, RuleType
from minds.model.memory_topic import MemoryTopic


class MemoryNotFoundError(Exception):
    pass


class MemoryConflictError(Exception):
    pass


class MemoryAdminService:
    """CRUD operations for admin-managed memory rules and topics."""

    def __init__(self, session: Session, user_id: UUID, organization_id: UUID) -> None:
        self.session = session
        self.user_id = user_id
        self.organization_id = organization_id

    def list_rules(self, mind_id: UUID) -> list[MemoryRule]:
        statement = select(MemoryRule).where(
            and_(MemoryRule.mind_id == mind_id, MemoryRule.deleted_at.is_(None))  # type: ignore[union-attr]
        )
        return list(self.session.exec(statement).all())

    def create_rule(self, mind_id: UUID, rule_type: RuleType, content: str) -> MemoryRule:
        rule = MemoryRule(
            mind_id=mind_id,
            organization_id=self.organization_id,
            user_id=self.user_id,
            rule_type=rule_type,
            content=content,
        )
        self.session.add(rule)
        self.session.commit()
        self.session.refresh(rule)
        return rule

    def update_rule(
        self,
        mind_id: UUID,
        rule_id: UUID,
        rule_type: RuleType | None,
        content: str | None,
    ) -> MemoryRule:
        rule = self._get_active_rule(mind_id, rule_id)
        if rule_type is not None:
            rule.rule_type = rule_type
        if content is not None:
            rule.content = content
        self.session.add(rule)
        self.session.commit()
        self.session.refresh(rule)
        return rule

    def delete_rule(self, mind_id: UUID, rule_id: UUID) -> None:
        rule = self._get_active_rule(mind_id, rule_id)
        rule.deleted_at = datetime.now(timezone.utc)
        self.session.add(rule)
        self.session.commit()

    def _get_active_rule(self, mind_id: UUID, rule_id: UUID) -> MemoryRule:
        statement = select(MemoryRule).where(
            and_(
                MemoryRule.id == rule_id,
                MemoryRule.mind_id == mind_id,
                MemoryRule.deleted_at.is_(None),  # type: ignore[union-attr]
            )
        )
        rule = self.session.exec(statement).first()
        if rule is None:
            raise MemoryNotFoundError(f"Rule {rule_id} not found for mind {mind_id}")
        return rule

    def list_topics(self, mind_id: UUID) -> list[MemoryTopic]:
        statement = select(MemoryTopic).where(
            and_(MemoryTopic.mind_id == mind_id, MemoryTopic.deleted_at.is_(None))  # type: ignore[union-attr]
        )
        return list(self.session.exec(statement).all())

    def create_topic(
        self,
        mind_id: UUID,
        title: str,
        body: str,
        tags: list[str] | None,
        description: str | None,
    ) -> MemoryTopic:
        topic = MemoryTopic(
            mind_id=mind_id,
            organization_id=self.organization_id,
            user_id=self.user_id,
            title=title,
            body=body,
            tags=tags,
            description=description,
        )
        self.session.add(topic)
        try:
            self.session.commit()
        except IntegrityError:
            self.session.rollback()
            raise MemoryConflictError(f"A topic with title '{title}' already exists for this mind") from None
        self.session.refresh(topic)
        return topic

    def update_topic(
        self,
        mind_id: UUID,
        topic_id: UUID,
        title: str | None,
        body: str | None,
        tags: list[str] | None,
        description: str | None,
    ) -> MemoryTopic:
        topic = self._get_active_topic(mind_id, topic_id)
        if title is not None:
            topic.title = title
        if body is not None:
            topic.body = body
        if tags is not None:
            topic.tags = tags
        if description is not None:
            topic.description = description
        self.session.add(topic)
        try:
            self.session.commit()
        except IntegrityError:
            self.session.rollback()
            raise MemoryConflictError(f"A topic with title '{title}' already exists for this mind") from None
        self.session.refresh(topic)
        return topic

    def delete_topic(self, mind_id: UUID, topic_id: UUID) -> None:
        topic = self._get_active_topic(mind_id, topic_id)
        topic.deleted_at = datetime.now(timezone.utc)
        self.session.add(topic)
        self.session.commit()

    def _get_active_topic(self, mind_id: UUID, topic_id: UUID) -> MemoryTopic:
        statement = select(MemoryTopic).where(
            and_(
                MemoryTopic.id == topic_id,
                MemoryTopic.mind_id == mind_id,
                MemoryTopic.deleted_at.is_(None),  # type: ignore[union-attr]
            )
        )
        topic = self.session.exec(statement).first()
        if topic is None:
            raise MemoryNotFoundError(f"Topic {topic_id} not found for mind {mind_id}")
        return topic
