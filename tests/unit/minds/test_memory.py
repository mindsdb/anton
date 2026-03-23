"""Unit tests for MemoryRule, MemoryTopic models and MemoryRepository."""

from datetime import datetime, timezone
from unittest.mock import Mock
from uuid import uuid4

import pytest
from sqlmodel import Session

from minds.model.memory_rule import MemoryRule, RuleType
from minds.model.memory_topic import MemoryTopic
from minds.services.memory import MemoryRepository

MIND_ID = uuid4()
ORG_ID = uuid4()
USER_ID = uuid4()


class TestRuleType:
    def test_values(self):
        assert RuleType.always == "always"
        assert RuleType.never == "never"
        assert RuleType.when == "when"

    def test_is_str(self):
        assert isinstance(RuleType.always, str)


class TestMemoryRule:
    def test_instantiation(self):
        rule = MemoryRule(
            mind_id=MIND_ID,
            organization_id=ORG_ID,
            user_id=USER_ID,
            rule_type=RuleType.always,
            content="Always respond in the user's language.",
        )
        assert rule.rule_type == RuleType.always
        assert rule.content == "Always respond in the user's language."
        assert rule.deleted_at is None

    def test_soft_delete_field(self):
        rule = MemoryRule(
            mind_id=MIND_ID,
            organization_id=ORG_ID,
            user_id=USER_ID,
            rule_type=RuleType.never,
            content="Never share internal cost data.",
        )
        assert rule.deleted_at is None
        now = datetime.now(timezone.utc)
        rule.deleted_at = now
        assert rule.deleted_at == now

    def test_all_rule_types(self):
        for rule_type in RuleType:
            rule = MemoryRule(
                mind_id=MIND_ID,
                organization_id=ORG_ID,
                user_id=USER_ID,
                rule_type=rule_type,
                content="Some content.",
            )
            assert rule.rule_type == rule_type


class TestMemoryTopic:
    def test_instantiation(self):
        topic = MemoryTopic(
            mind_id=MIND_ID,
            organization_id=ORG_ID,
            user_id=USER_ID,
            title="SQL Query Optimization",
            body="# SQL Query Optimization\n\nUse indexes...",
        )
        assert topic.title == "SQL Query Optimization"
        assert topic.body == "# SQL Query Optimization\n\nUse indexes..."
        assert topic.tags is None
        assert topic.description is None
        assert topic.deleted_at is None

    def test_optional_fields(self):
        topic = MemoryTopic(
            mind_id=MIND_ID,
            organization_id=ORG_ID,
            user_id=USER_ID,
            title="Customer Data Policy",
            tags=["privacy", "pii", "customer"],
            description="How to handle customer PII data.",
            body="# Customer Data Policy\n\nNever log PII...",
        )
        assert topic.tags == ["privacy", "pii", "customer"]
        assert topic.description == "How to handle customer PII data."

    def test_soft_delete_field(self):
        topic = MemoryTopic(
            mind_id=MIND_ID,
            organization_id=ORG_ID,
            user_id=USER_ID,
            title="Some Topic",
            body="Content.",
        )
        assert topic.deleted_at is None
        now = datetime.now(timezone.utc)
        topic.deleted_at = now
        assert topic.deleted_at == now


class TestMemoryRepository:
    @pytest.fixture
    def mock_session(self):
        session = Mock(spec=Session)
        session.exec = Mock()
        return session

    @pytest.fixture
    def repo(self, mock_session):
        return MemoryRepository(session=mock_session, mind_id=MIND_ID)

    def _make_rule(self, rule_type=RuleType.always, content="Do this."):
        return MemoryRule(
            mind_id=MIND_ID,
            organization_id=ORG_ID,
            user_id=USER_ID,
            rule_type=rule_type,
            content=content,
        )

    def _make_topic(self, title="Some Topic", tags=None):
        return MemoryTopic(
            mind_id=MIND_ID,
            organization_id=ORG_ID,
            user_id=USER_ID,
            title=title,
            tags=tags,
            body="Body content.",
        )

    def test_get_active_rules_returns_list(self, repo, mock_session):
        rule = self._make_rule()
        mock_session.exec.return_value.all.return_value = [rule]

        result = repo.get_active_rules()

        assert result == [rule]
        mock_session.exec.assert_called_once()

    def test_get_active_rules_empty(self, repo, mock_session):
        mock_session.exec.return_value.all.return_value = []

        result = repo.get_active_rules()

        assert result == []

    def test_get_active_rules_multiple(self, repo, mock_session):
        rules = [
            self._make_rule(RuleType.always, "Always do X."),
            self._make_rule(RuleType.never, "Never do Y."),
            self._make_rule(RuleType.when, "When Z, do W."),
        ]
        mock_session.exec.return_value.all.return_value = rules

        result = repo.get_active_rules()

        assert len(result) == 3
        assert result[0].rule_type == RuleType.always
        assert result[1].rule_type == RuleType.never
        assert result[2].rule_type == RuleType.when

    def test_get_active_topics_returns_list(self, repo, mock_session):
        topic = self._make_topic()
        mock_session.exec.return_value.all.return_value = [topic]

        result = repo.get_active_topics()

        assert result == [topic]
        mock_session.exec.assert_called_once()

    def test_get_active_topics_empty(self, repo, mock_session):
        mock_session.exec.return_value.all.return_value = []

        result = repo.get_active_topics()

        assert result == []

    def test_get_active_topics_includes_body(self, repo, mock_session):
        topic = self._make_topic(title="Pricing")
        topic.body = "# Pricing\n\nTier 1: $10/month"
        mock_session.exec.return_value.all.return_value = [topic]

        result = repo.get_active_topics()

        assert result[0].body == "# Pricing\n\nTier 1: $10/month"

    def test_get_active_topics_with_tags(self, repo, mock_session):
        topic = self._make_topic(title="Security Policy", tags=["security", "compliance"])
        mock_session.exec.return_value.all.return_value = [topic]

        result = repo.get_active_topics()

        assert result[0].tags == ["security", "compliance"]

    def test_repo_stores_mind_id(self, mock_session):
        mind_id = uuid4()
        repo = MemoryRepository(session=mock_session, mind_id=mind_id)
        assert repo.mind_id == mind_id

    def test_repo_accepts_string_mind_id(self, mock_session):
        mind_id = str(uuid4())
        repo = MemoryRepository(session=mock_session, mind_id=mind_id)
        assert repo.mind_id == mind_id
