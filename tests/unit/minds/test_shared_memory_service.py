from datetime import datetime
from unittest.mock import Mock
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

from minds.model.memory_rule import MemoryRule, RuleType
from minds.model.memory_topic import MemoryTopic
from minds.services.memory import (
    MemoryAdminService,
    MemoryBlock,
    MemoryConflictError,
    MemoryNotFoundError,
    MemoryRepository,
    MemoryService,
    _normalize,
    _token_count,
)

MIND_ID = uuid4()
ORG_ID = uuid4()
USER_ID = uuid4()
RULE_ID = uuid4()
TOPIC_ID = uuid4()


def make_rule(rule_type=RuleType.always, content="Rule content.", rule_id=None):
    rule = MemoryRule(
        mind_id=MIND_ID,
        organization_id=ORG_ID,
        user_id=USER_ID,
        rule_type=rule_type,
        content=content,
    )
    if rule_id is not None:
        rule.id = rule_id
    return rule


def make_topic(title="Generic Topic", tags=None, description=None, body=None, topic_id=None):
    topic = MemoryTopic(
        mind_id=MIND_ID,
        organization_id=ORG_ID,
        user_id=USER_ID,
        title=title,
        tags=tags,
        description=description,
        body=body or f"Body of {title}.",
    )
    if topic_id is not None:
        topic.id = topic_id
    return topic


def make_service(rules=None, topics=None, token_budget=3000, max_topics=5):
    repo = Mock(spec=MemoryRepository)
    repo.get_active_rules.return_value = rules or []
    repo.get_active_topics.return_value = topics or []
    return MemoryService(repo=repo, token_budget=token_budget, max_topics=max_topics)


@pytest.fixture
def admin_session():
    s = Mock(spec=Session)
    s.exec = Mock()
    return s


@pytest.fixture
def admin_service(admin_session):
    return MemoryAdminService(session=admin_session, user_id=USER_ID, organization_id=ORG_ID)


class TestNormalize:
    def test_lowercase(self):
        assert _normalize("SQL") == ["sql"]

    def test_splits_on_punctuation(self):
        assert _normalize("sql-query optimization") == ["sql", "query", "optimization"]

    def test_empty_string(self):
        assert _normalize("") == []

    def test_strips_empty_tokens(self):
        assert _normalize("  hello  world  ") == ["hello", "world"]


class TestTokenCount:
    def test_approximation(self):
        assert _token_count("abcd") == 1
        assert _token_count("a" * 400) == 100

    def test_empty(self):
        assert _token_count("") == 0


class TestMemoryBlock:
    def test_is_empty_with_no_content(self):
        block = MemoryBlock()
        assert block.is_empty

    def test_is_empty_false_with_rules(self):
        block = MemoryBlock(rules=[make_rule()])
        assert not block.is_empty

    def test_is_empty_false_with_topics(self):
        block = MemoryBlock(topics=[make_topic()])
        assert not block.is_empty


class TestScoreTopics:
    def setup_method(self):
        self.service = make_service()

    def test_title_match_scores_highest(self):
        sql_topic = make_topic(title="SQL Query Optimization")
        pricing_topic = make_topic(title="Pricing Tiers")
        result = self.service._score_topics([pricing_topic, sql_topic], "how to write fast sql queries")
        assert result[0].title == "SQL Query Optimization"

    def test_tag_match_contributes(self):
        tagged = make_topic(title="Generic", tags=["sql", "database"])
        untagged = make_topic(title="Other Topic")
        result = self.service._score_topics([untagged, tagged], "sql")
        assert result[0].title == "Generic"

    def test_description_match_contributes(self):
        described = make_topic(title="Policy", description="This covers SQL database rules.")
        plain = make_topic(title="Other")
        result = self.service._score_topics([plain, described], "sql")
        assert result[0].title == "Policy"

    def test_empty_query_returns_original_order(self):
        topics = [make_topic("A"), make_topic("B"), make_topic("C")]
        result = self.service._score_topics(topics, "")
        assert [t.title for t in result] == ["A", "B", "C"]

    def test_no_match_topics_still_returned(self):
        topics = [make_topic("Pricing"), make_topic("Onboarding")]
        result = self.service._score_topics(topics, "unrelated query xyz")
        assert len(result) == 2


class TestApplyCaps:
    def setup_method(self):
        self.service = make_service(max_topics=3, token_budget=1000)

    def test_count_cap_respected(self):
        topics = [make_topic(f"Topic {i}", body="x" * 40) for i in range(10)]
        result = self.service._apply_caps(topics, budget=9999)
        assert len(result) <= 3

    def test_budget_cap_respected(self):
        # Each topic body = 400 chars = 100 tokens. Budget = 150 → only 1 fits.
        topics = [make_topic(f"Topic {i}", body="x" * 400) for i in range(5)]
        result = self.service._apply_caps(topics, budget=150)
        assert len(result) == 1

    def test_topic_exceeding_budget_skipped(self):
        # First topic is huge, second is tiny.
        huge = make_topic("Huge", body="x" * 4000)  # 1000 tokens
        tiny = make_topic("Tiny", body="x" * 40)  # 10 tokens
        result = self.service._apply_caps([huge, tiny], budget=50)
        assert len(result) == 1
        assert result[0].title == "Tiny"

    def test_empty_topics(self):
        assert self.service._apply_caps([], budget=9999) == []


class TestLoadForSession:
    def test_all_rules_always_included(self):
        service = make_service(
            rules=[
                make_rule(RuleType.always, "Use the production schema."),
                make_rule(RuleType.never, "Never query events without a date filter."),
                make_rule(RuleType.when, "When calculating revenue, subtract refunds."),
            ]
        )
        block = service.load_for_session("any query")
        assert len(block.rules) == 3

    def test_rules_included_regardless_of_query(self):
        service = make_service(rules=[make_rule(RuleType.when, "Conditional rule.")])
        block = service.load_for_session("completely unrelated query xyz")
        assert len(block.rules) == 1

    def test_topics_selected_by_relevance(self):
        service = make_service(
            topics=[
                make_topic("SQL Optimization", body="x" * 40),
                make_topic("Pricing Policy", body="x" * 40),
            ]
        )
        block = service.load_for_session("sql performance")
        assert block.topics[0].title == "SQL Optimization"

    def test_returns_empty_block_when_no_data(self):
        service = make_service()
        block = service.load_for_session("anything")
        assert block.is_empty

    def test_rules_budget_reduces_topic_budget(self):
        # Rule consumes 200 tokens (800 chars), budget=250 → 50 left for topics.
        # Topic body = 400 chars = 100 tokens → too large, skipped.
        service = make_service(
            rules=[make_rule(content="x" * 800)],
            topics=[make_topic(body="x" * 400)],
            token_budget=250,
        )
        block = service.load_for_session("query")
        assert block.topics == []


class TestFormatBlock:
    def test_empty_block_returns_empty_string(self):
        assert MemoryService.format_block(MemoryBlock()) == ""

    def test_rules_section_present(self):
        block = MemoryBlock(rules=[make_rule(RuleType.always, "Always be helpful.")])
        output = MemoryService.format_block(block)
        assert "## MANDATORY RULES" in output
        assert "### Always" in output
        assert "- Always be helpful." in output

    def test_topics_section_present(self):
        block = MemoryBlock(topics=[make_topic("SQL Guide", body="Use indexes.")])
        output = MemoryService.format_block(block)
        assert "## Memory Topics" in output
        assert "### SQL Guide" in output
        assert "Use indexes." in output

    def test_rules_before_topics(self):
        block = MemoryBlock(
            rules=[make_rule(RuleType.never, "Never share PII.")],
            topics=[make_topic("Privacy Policy", body="Body.")],
        )
        output = MemoryService.format_block(block)
        assert output.index("## MANDATORY RULES") < output.index("## Memory Topics")

    def test_never_rule_label(self):
        block = MemoryBlock(rules=[make_rule(RuleType.never, "Never do X.")])
        output = MemoryService.format_block(block)
        assert "### Never" in output
        assert "- Never do X." in output

    def test_when_rule_label(self):
        block = MemoryBlock(rules=[make_rule(RuleType.when, "When Y, do Z.")])
        output = MemoryService.format_block(block)
        assert "### When" in output
        assert "- When Y, do Z." in output

    def test_rules_grouped_by_type(self):
        block = MemoryBlock(
            rules=[
                make_rule(RuleType.always, "Use production schema."),
                make_rule(RuleType.never, "Never skip date filter."),
                make_rule(RuleType.when, "When revenue, subtract refunds."),
            ]
        )
        output = MemoryService.format_block(block)
        assert output.index("### Always") < output.index("### Never") < output.index("### When")

    def test_empty_type_section_omitted(self):
        block = MemoryBlock(rules=[make_rule(RuleType.always, "Do this.")])
        output = MemoryService.format_block(block)
        assert "### Never" not in output
        assert "### When" not in output

    def test_whitespace_only_rules_skipped(self):
        block = MemoryBlock(rules=[make_rule(RuleType.always, "  \n  \n  ")])
        output = MemoryService.format_block(block)
        assert output == ""

    def test_multiline_content_collapsed_to_single_bullet(self):
        block = MemoryBlock(rules=[make_rule(RuleType.always, "Line one.\nLine two.\n\nLine three.")])
        output = MemoryService.format_block(block)
        assert "- Line one. Line two. Line three." in output
        assert output.count("\n- ") == 1

    def test_multiple_topics_all_present(self):
        block = MemoryBlock(
            topics=[
                make_topic("Topic A", body="Body A."),
                make_topic("Topic B", body="Body B."),
            ]
        )
        output = MemoryService.format_block(block)
        assert "### Topic A" in output
        assert "### Topic B" in output


class TestAdminListRules:
    def test_returns_rules(self, admin_service, admin_session):
        rule = make_rule(rule_id=RULE_ID)
        admin_session.exec.return_value.all.return_value = [rule]
        assert admin_service.list_rules(MIND_ID) == [rule]

    def test_returns_empty_list(self, admin_service, admin_session):
        admin_session.exec.return_value.all.return_value = []
        assert admin_service.list_rules(MIND_ID) == []


class TestAdminCreateRule:
    def test_creates_and_returns_rule(self, admin_service, admin_session):
        admin_session.refresh = Mock()
        result = admin_service.create_rule(MIND_ID, RuleType.always, "Always be helpful.")
        admin_session.add.assert_called_once()
        admin_session.commit.assert_called_once()
        assert result.content == "Always be helpful."
        assert result.rule_type == RuleType.always
        assert result.mind_id == MIND_ID
        assert result.organization_id == ORG_ID


class TestAdminUpdateRule:
    def test_updates_content(self, admin_service, admin_session):
        rule = make_rule(content="Old.", rule_id=RULE_ID)
        admin_session.exec.return_value.first.return_value = rule
        admin_session.refresh = Mock()
        result = admin_service.update_rule(MIND_ID, RULE_ID, rule_type=None, content="New.")
        assert result.content == "New."

    def test_updates_rule_type(self, admin_service, admin_session):
        rule = make_rule(rule_type=RuleType.always, rule_id=RULE_ID)
        admin_session.exec.return_value.first.return_value = rule
        admin_session.refresh = Mock()
        result = admin_service.update_rule(MIND_ID, RULE_ID, rule_type=RuleType.never, content=None)
        assert result.rule_type == RuleType.never

    def test_none_values_not_applied(self, admin_service, admin_session):
        rule = make_rule(content="Keep this.", rule_id=RULE_ID)
        admin_session.exec.return_value.first.return_value = rule
        admin_session.refresh = Mock()
        admin_service.update_rule(MIND_ID, RULE_ID, rule_type=None, content=None)
        assert rule.content == "Keep this."

    def test_raises_not_found(self, admin_service, admin_session):
        admin_session.exec.return_value.first.return_value = None
        with pytest.raises(MemoryNotFoundError):
            admin_service.update_rule(MIND_ID, RULE_ID, rule_type=None, content="x")


class TestAdminDeleteRule:
    def test_soft_deletes_rule(self, admin_service, admin_session):
        rule = make_rule(rule_id=RULE_ID)
        admin_session.exec.return_value.first.return_value = rule
        admin_service.delete_rule(MIND_ID, RULE_ID)
        assert isinstance(rule.deleted_at, datetime)
        admin_session.commit.assert_called_once()

    def test_raises_not_found(self, admin_service, admin_session):
        admin_session.exec.return_value.first.return_value = None
        with pytest.raises(MemoryNotFoundError):
            admin_service.delete_rule(MIND_ID, RULE_ID)


class TestAdminListTopics:
    def test_returns_topics(self, admin_service, admin_session):
        topic = make_topic(topic_id=TOPIC_ID)
        admin_session.exec.return_value.all.return_value = [topic]
        assert admin_service.list_topics(MIND_ID) == [topic]

    def test_returns_empty_list(self, admin_service, admin_session):
        admin_session.exec.return_value.all.return_value = []
        assert admin_service.list_topics(MIND_ID) == []


class TestAdminCreateTopic:
    def test_creates_topic(self, admin_service, admin_session):
        admin_session.refresh = Mock()
        result = admin_service.create_topic(
            MIND_ID, "SQL Guide", "Use indexes.", tags=["sql"], description="About SQL."
        )
        admin_session.add.assert_called_once()
        admin_session.commit.assert_called_once()
        assert result.title == "SQL Guide"
        assert result.tags == ["sql"]
        assert result.description == "About SQL."

    def test_creates_topic_without_optional_fields(self, admin_service, admin_session):
        admin_session.refresh = Mock()
        result = admin_service.create_topic(MIND_ID, "Minimal", "Body.", tags=None, description=None)
        assert result.tags is None
        assert result.description is None

    def test_raises_conflict_on_duplicate_title(self, admin_service, admin_session):
        admin_session.commit.side_effect = IntegrityError("duplicate", {}, Exception())
        with pytest.raises(MemoryConflictError, match="SQL Guide"):
            admin_service.create_topic(MIND_ID, "SQL Guide", "Body.", tags=None, description=None)
        admin_session.rollback.assert_called_once()


class TestAdminUpdateTopic:
    def test_updates_title(self, admin_service, admin_session):
        topic = make_topic(title="Old Title", topic_id=TOPIC_ID)
        admin_session.exec.return_value.first.return_value = topic
        admin_session.refresh = Mock()
        result = admin_service.update_topic(
            MIND_ID, TOPIC_ID, title="New Title", body=None, tags=None, description=None
        )
        assert result.title == "New Title"

    def test_updates_body(self, admin_service, admin_session):
        topic = make_topic(body="Old body.", topic_id=TOPIC_ID)
        admin_session.exec.return_value.first.return_value = topic
        admin_session.refresh = Mock()
        result = admin_service.update_topic(
            MIND_ID, TOPIC_ID, title=None, body="New body.", tags=None, description=None
        )
        assert result.body == "New body."

    def test_updates_tags(self, admin_service, admin_session):
        topic = make_topic(topic_id=TOPIC_ID)
        admin_session.exec.return_value.first.return_value = topic
        admin_session.refresh = Mock()
        result = admin_service.update_topic(
            MIND_ID, TOPIC_ID, title=None, body=None, tags=["new", "tags"], description=None
        )
        assert result.tags == ["new", "tags"]

    def test_updates_description(self, admin_service, admin_session):
        topic = make_topic(topic_id=TOPIC_ID)
        admin_session.exec.return_value.first.return_value = topic
        admin_session.refresh = Mock()
        result = admin_service.update_topic(
            MIND_ID, TOPIC_ID, title=None, body=None, tags=None, description="New desc."
        )
        assert result.description == "New desc."

    def test_none_values_not_applied(self, admin_service, admin_session):
        topic = make_topic(title="Keep This", body="Keep body.", topic_id=TOPIC_ID)
        admin_session.exec.return_value.first.return_value = topic
        admin_session.refresh = Mock()
        admin_service.update_topic(MIND_ID, TOPIC_ID, title=None, body=None, tags=None, description=None)
        assert topic.title == "Keep This"
        assert topic.body == "Keep body."

    def test_raises_not_found(self, admin_service, admin_session):
        admin_session.exec.return_value.first.return_value = None
        with pytest.raises(MemoryNotFoundError):
            admin_service.update_topic(MIND_ID, TOPIC_ID, title="x", body=None, tags=None, description=None)

    def test_raises_conflict_on_duplicate_title(self, admin_service, admin_session):
        topic = make_topic(topic_id=TOPIC_ID)
        admin_session.exec.return_value.first.return_value = topic
        admin_session.commit.side_effect = IntegrityError("duplicate", {}, Exception())
        with pytest.raises(MemoryConflictError, match="Duplicate"):
            admin_service.update_topic(MIND_ID, TOPIC_ID, title="Duplicate", body=None, tags=None, description=None)
        admin_session.rollback.assert_called_once()


class TestAdminDeleteTopic:
    def test_soft_deletes_topic(self, admin_service, admin_session):
        topic = make_topic(topic_id=TOPIC_ID)
        admin_session.exec.return_value.first.return_value = topic
        admin_service.delete_topic(MIND_ID, TOPIC_ID)
        assert isinstance(topic.deleted_at, datetime)
        admin_session.commit.assert_called_once()

    def test_raises_not_found(self, admin_service, admin_session):
        admin_session.exec.return_value.first.return_value = None
        with pytest.raises(MemoryNotFoundError):
            admin_service.delete_topic(MIND_ID, TOPIC_ID)
