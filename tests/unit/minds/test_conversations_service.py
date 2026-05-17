"""
Unit tests for ConversationsService.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID, uuid4

import pandas as pd
import pytest
from mindsdb_sdk.server import Server
from sqlmodel import Session

from minds.model.conversation import Conversation
from minds.model.message import Message
from minds.model.mind import Mind
from minds.schemas.charts import (
    AxisSpec,
    ChartImageResponse,
    ChartMeta,
    ChartResponse,
    RenderChartType,
    RenderPlan,
    SeriesSpec,
    XYIntent,
)
from minds.schemas.chat import Role
from minds.schemas.conversations import ConversationCreateRequest, ConversationMetadata, ConversationResponse
from minds.schemas.messages import MessageContentType, MessageResponse, MessageResultResponse
from minds.services.conversations import (
    AgentNotAntonError,
    ConversationNotFoundError,
    ConversationsService,
    ConversationsServiceError,
    MessageNoSQLQueryError,
    MessageNotAssistantError,
)
from minds.services.minds import MindsService


class TestConversationsService:
    """Test cases for ConversationsService."""

    @pytest.fixture
    def mock_session(self):
        """Mock database session."""
        session = Mock(spec=Session)
        session.add = Mock()
        session.add_all = Mock()
        session.commit = Mock()
        session.rollback = Mock()
        session.flush = Mock()
        session.merge = Mock()
        return session

    @pytest.fixture
    def mock_mindsdb_client(self):
        """Mock MindsDB client."""
        client = Mock(spec=Server)
        client.query = Mock()
        return client

    @pytest.fixture
    def user_id(self):
        """Test user ID."""
        return str(uuid4())

    @pytest.fixture
    def organization_id(self):
        """Test organization ID."""
        return str(uuid4())

    @pytest.fixture
    def service(self, mock_session, mock_mindsdb_client, user_id, organization_id):
        """Create ConversationsService instance."""
        return ConversationsService(
            session=mock_session,
            mindsdb_client=mock_mindsdb_client,
            user_id=user_id,
            organization_id=organization_id,
        )

    @pytest.fixture
    def sample_conversation(self, user_id, organization_id):
        """Sample conversation for testing."""
        mind_id = uuid4()
        mind = Mind(
            id=mind_id,
            name="test-model",
            provider="openai",
            model_name="gpt-4o",
            user_id=UUID(user_id),
        )
        conversation = Conversation(
            id=uuid4(),
            topic="Test Conversation",
            user_id=UUID(user_id),
            organization_id=UUID(organization_id),
            mind_id=mind_id,
            created_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
        )
        # Set the mind relationship
        conversation.mind = mind
        return conversation

    @pytest.fixture
    def sample_message(self, sample_conversation, organization_id):
        """Sample message for testing."""
        return Message(
            id=uuid4(),
            conversation_id=sample_conversation.id,
            organization_id=UUID(organization_id),
            role=Role.user,
            content="Hello, world!",
            request_id="test-request-id",
            created_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
        )

    @pytest.fixture
    def sample_create_request(self):
        """Sample create request."""
        return ConversationCreateRequest(
            metadata=ConversationMetadata(topic="New Conversation", model_name="test-model"),
        )

    @pytest.fixture
    def mock_mind_service(self):
        """Mock MindsService instance."""
        service = Mock(spec=MindsService)
        return service

    @pytest.fixture
    def mock_mind(self, user_id):
        """Mock Mind object."""
        mind = Mind(
            id=uuid4(),
            name="test-model",
            provider="openai",
            model_name="gpt-4o",
            user_id=UUID(user_id),
        )
        return mind

    def test_service_initialization(self, mock_session, mock_mindsdb_client, user_id, organization_id):
        """Test service initialization."""
        service = ConversationsService(
            session=mock_session,
            mindsdb_client=mock_mindsdb_client,
            user_id=user_id,
            organization_id=organization_id,
        )

        assert service.session == mock_session
        assert service.user_id == user_id
        assert service.organization_id == organization_id
        assert service.mindsdb_client == mock_mindsdb_client

    @pytest.mark.asyncio
    async def test_list_conversations_empty(self, service, mock_session):
        """Test listing conversations when none exist."""
        mock_result = Mock()
        mock_result.all.return_value = []
        mock_session.exec.return_value = mock_result

        result = await service.list_conversations()

        assert result == []
        mock_session.exec.assert_called()

    @pytest.mark.asyncio
    async def test_list_conversations_success(self, service, mock_session, sample_conversation):
        """Test successful conversations listing."""
        mock_result = Mock()
        mock_result.all.return_value = [sample_conversation]
        mock_session.exec.return_value = mock_result

        result = await service.list_conversations()

        assert len(result) == 1
        assert isinstance(result[0], ConversationResponse)
        assert result[0].metadata.topic == "Test Conversation"

    @pytest.mark.asyncio
    async def test_list_conversations_with_filters(self, service, mock_session, sample_conversation):
        """Test listing conversations with filters."""
        mock_result = Mock()
        mock_result.all.return_value = [sample_conversation]
        mock_count_result = Mock()
        mock_count_result.one.return_value = 1
        mock_session.exec.side_effect = [mock_count_result, mock_result]

        result = await service.list_conversations(topic="Test", limit=10, offset=5, include_total=True)

        assert len(result) == 2  # Returns tuple (conversations, total)
        conversations, total = result
        assert len(conversations) == 1
        assert total == 1

    @pytest.mark.asyncio
    async def test_list_conversations_with_sorting(self, service, mock_session, sample_conversation):
        """Test listing conversations with sorting."""
        mock_result = Mock()
        mock_result.all.return_value = [sample_conversation]
        mock_session.exec.return_value = mock_result

        result = await service.list_conversations(sort_by="created_at", sort_order="asc")

        assert len(result) == 1
        mock_session.exec.assert_called()

    @pytest.mark.asyncio
    async def test_list_conversations_include_deleted(self, service, mock_session, sample_conversation):
        """Test listing conversations including deleted ones."""
        mock_result = Mock()
        mock_result.all.return_value = [sample_conversation]
        mock_session.exec.return_value = mock_result

        result = await service.list_conversations(include_deleted=True)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_list_conversations_database_error(self, service, mock_session):
        """Test list conversations with database error."""
        mock_session.exec.side_effect = Exception("Database error")

        with pytest.raises(ConversationsServiceError, match="Failed to list conversations"):
            await service.list_conversations()

    @pytest.mark.asyncio
    async def test_get_conversation_success(self, service, mock_session, sample_conversation):
        """Test successful conversation retrieval."""
        mock_result = Mock()
        mock_result.first.return_value = sample_conversation
        mock_session.exec.return_value = mock_result

        result = await service.get_conversation(sample_conversation.id)

        assert isinstance(result, ConversationResponse)
        assert result.id == sample_conversation.id
        assert result.metadata.topic == "Test Conversation"

    @pytest.mark.asyncio
    async def test_get_conversation_not_found(self, service, mock_session):
        """Test get conversation when not found."""
        mock_result = Mock()
        mock_result.first.return_value = None
        mock_session.exec.return_value = mock_result

        conversation_id = uuid4()
        with pytest.raises(ConversationNotFoundError, match=f"Conversation with ID '{conversation_id}' not found"):
            await service.get_conversation(conversation_id)

    @pytest.mark.asyncio
    async def test_get_conversation_database_error(self, service, mock_session):
        """Test get conversation with database error."""
        mock_session.exec.side_effect = Exception("Database error")

        conversation_id = uuid4()
        with pytest.raises(ConversationsServiceError, match="Failed to get conversation"):
            await service.get_conversation(conversation_id)

    @pytest.mark.asyncio
    async def test_create_conversation_mind_not_found_rolls_back_and_reraises(
        self,
        service,
        mock_session,
        sample_create_request,
        mock_mind_service,
    ):
        """MindNotFoundError should rollback and bubble up unchanged."""
        from minds.services.minds import MindNotFoundError

        mock_mind_service.get_mind_model = AsyncMock(side_effect=MindNotFoundError("missing"))

        with pytest.raises(MindNotFoundError, match="missing"):
            await service.create_conversation(sample_create_request, mock_mind_service)

        mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_conversation_message_unexpected_error_wrapped(self, service, mock_session):
        """create_conversation_message should wrap unexpected exceptions."""
        conversation_id = uuid4()
        service._get_conversation = AsyncMock(return_value=SimpleNamespace(id=conversation_id))  # type: ignore[method-assign]
        mock_session.commit.side_effect = Exception("commit failed")

        with pytest.raises(ConversationsServiceError, match="Failed to create message: commit failed"):
            await service.create_conversation_message(conversation_id, role=Role.user, content="hi")

    @pytest.mark.asyncio
    async def test_create_conversation_message_placeholder_conversation_missing_raises_not_found(self, service):
        """create_conversation_message_placeholder should raise if conversation lookup returns falsy."""
        conversation_id = uuid4()
        service._get_conversation = AsyncMock(return_value=None)  # type: ignore[method-assign]
        with pytest.raises(ConversationNotFoundError, match=f"Conversation with ID '{conversation_id}' not found"):
            await service.create_conversation_message_placeholder(conversation_id, role=Role.user)

    @pytest.mark.asyncio
    async def test_create_conversation_message_placeholder_unexpected_error_wrapped(self, service, mock_session):
        """create_conversation_message_placeholder should wrap unexpected exceptions."""
        conversation_id = uuid4()
        service._get_conversation = AsyncMock(return_value=SimpleNamespace(id=conversation_id))  # type: ignore[method-assign]
        mock_session.flush.side_effect = Exception("flush failed")

        with pytest.raises(ConversationsServiceError, match="Failed to create message placeholder: flush failed"):
            await service.create_conversation_message_placeholder(conversation_id, role=Role.user)

    @pytest.mark.asyncio
    async def test_create_conversation_message_event_unexpected_error_wrapped(self, service, mock_session):
        """create_conversation_message_event should wrap unexpected exceptions."""
        mock_session.flush.side_effect = Exception("flush failed")
        with pytest.raises(ConversationsServiceError, match="Failed to create message event: flush failed"):
            await service.create_conversation_message_event(
                message_id=uuid4(),
                sequence_number=1,
                event_data={"x": 1},
            )

    @pytest.mark.asyncio
    async def test_update_conversation_message_content_pending_rollback_recovers_by_creating_message(
        self, service, mock_session
    ):
        """PendingRollbackError path should rollback, create message, commit, and return response."""
        from sqlalchemy.exc import PendingRollbackError

        conversation_id = uuid4()
        message_id = uuid4()
        msg = SimpleNamespace(
            id=message_id,
            conversation_id=conversation_id,
            role=Role.assistant,
            content="old",
            sql_query=None,
            model_name=None,
            request_id=None,
            langfuse_trace_id=None,
            input_tokens=0,
            output_tokens=0,
        )

        mock_session.merge.return_value = msg
        mock_session.commit.side_effect = [PendingRollbackError("rolled back"), None]

        service._get_message = AsyncMock(return_value=None)  # type: ignore[method-assign]
        expected = SimpleNamespace(id=message_id, ok=True)
        service._message_to_response = AsyncMock(return_value=expected)  # type: ignore[method-assign]

        out = await service.update_conversation_message_content(
            msg,
            content="new",
            sql_query="SELECT 1",
            model_name="m",
            request_id="r",
            langfuse_trace_id="t",
            input_tokens=1,
            output_tokens=2,
        )

        assert out is expected
        mock_session.rollback.assert_called_once()
        assert mock_session.commit.call_count == 2
        mock_session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_conversation_message_content_unexpected_error_wrapped(self, service, mock_session):
        """update_conversation_message_content should rollback and wrap unexpected exceptions."""
        msg = SimpleNamespace(id=uuid4(), conversation_id=uuid4(), role=Role.user)
        mock_session.merge.side_effect = Exception("merge failed")

        with pytest.raises(ConversationsServiceError, match="Failed to update message content: merge failed"):
            await service.update_conversation_message_content(msg, content="x")

        mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_conversation_message_result_unexpected_error_wrapped(self, service, monkeypatch):
        """Unexpected exceptions should be wrapped as ConversationsServiceError."""
        service._get_conversation = AsyncMock(return_value=SimpleNamespace(id=uuid4()))  # type: ignore[method-assign]
        service._get_message = AsyncMock(  # type: ignore[method-assign]
            return_value=SimpleNamespace(role=Role.assistant, sql_query="SELECT 1")
        )
        service._validate_and_parse_sql_query = Mock(return_value=SimpleNamespace(order_by=[]))  # type: ignore[method-assign]
        service.mindsdb_client.query = Mock(side_effect=RuntimeError("boom"))

        with pytest.raises(ConversationsServiceError, match="Failed to get message result: boom"):
            await service.get_conversation_message_result(uuid4(), uuid4())

    @pytest.mark.asyncio
    async def test_export_conversation_message_result_unexpected_error_wrapped(self, service, mock_mindsdb_client):
        """export_conversation_message_result should wrap unexpected exceptions."""
        service._get_conversation = AsyncMock(return_value=SimpleNamespace(id=uuid4()))  # type: ignore[method-assign]
        service._get_message = AsyncMock(  # type: ignore[method-assign]
            return_value=SimpleNamespace(role=Role.assistant, sql_query="SELECT 1")
        )
        service._validate_and_parse_sql_query = Mock(return_value=SimpleNamespace())  # type: ignore[method-assign]
        mock_mindsdb_client.query.side_effect = Exception("query failed")

        with pytest.raises(ConversationsServiceError, match="Failed to export message result: query failed"):
            await service.export_conversation_message_result(uuid4(), uuid4())

    @pytest.mark.asyncio
    async def test_get_conversation_message_chart_mssql_uses_top_and_returns_chart_response(
        self, service, mock_mindsdb_client, monkeypatch
    ):
        """MSSQL branch should use TOP in query and return ChartResponse."""
        import minds.services.chart_compiler as chart_compiler_mod
        import minds.services.conversations as conversations_mod

        service._get_conversation = AsyncMock(return_value=SimpleNamespace(id=uuid4()))  # type: ignore[method-assign]
        service._get_message = AsyncMock(  # type: ignore[method-assign]
            return_value=SimpleNamespace(role=Role.assistant, sql_query="SELECT * FROM t")
        )
        service._validate_and_parse_sql_query = Mock(return_value=SimpleNamespace(order_by=[]))  # type: ignore[method-assign]

        monkeypatch.setattr(conversations_mod, "extract_database_engines_from_select", lambda *_a, **_k: {"mssql"})

        df = pd.DataFrame({"a": [1]})
        mock_mindsdb_client.query.return_value.fetch.return_value = df

        from minds.schemas.charts import XYIntent

        called = {}

        def _fake_compile_chart(result, intent):
            called["rows"] = len(result)
            called["intent"] = intent
            return (
                RenderPlan(
                    chart_type=RenderChartType.BAR,
                    title=None,
                    show_legend=False,
                    labels=["1"],
                    series=[SeriesSpec(label="a", values=[1])],
                    x_axis=AxisSpec(title="a", scale_type="category"),
                    y_axis=AxisSpec(title="a", scale_type="linear"),
                ),
                [],
                ChartMeta(
                    row_count=len(result),
                    used_rows=len(result),
                    points=len(result),
                    series=1,
                    fields=None,
                ),
            )

        monkeypatch.setattr(chart_compiler_mod, "compile_chart", _fake_compile_chart)
        monkeypatch.setattr(chart_compiler_mod, "MAX_ROWS_TO_PROCESS", 10)

        out = await service.get_conversation_message_chart(uuid4(), uuid4(), XYIntent(type="bar", x="a", y="a"))
        assert out.config["type"] == "bar"
        q = mock_mindsdb_client.query.call_args.args[0]
        assert "SELECT TOP 10" in q
        assert called["rows"] == 1

    @pytest.mark.asyncio
    async def test_get_conversation_message_chart_non_mssql_uses_limit(self, service, mock_mindsdb_client, monkeypatch):
        """Non-MSSQL branch should use LIMIT in query."""
        import minds.services.chart_compiler as chart_compiler_mod
        import minds.services.conversations as conversations_mod

        service._get_conversation = AsyncMock(return_value=SimpleNamespace(id=uuid4()))  # type: ignore[method-assign]
        service._get_message = AsyncMock(  # type: ignore[method-assign]
            return_value=SimpleNamespace(role=Role.assistant, sql_query="SELECT * FROM t")
        )
        service._validate_and_parse_sql_query = Mock(return_value=SimpleNamespace(order_by=[]))  # type: ignore[method-assign]

        monkeypatch.setattr(conversations_mod, "extract_database_engines_from_select", lambda *_a, **_k: set())
        monkeypatch.setattr(
            chart_compiler_mod,
            "compile_chartjs",
            lambda result, _intent: (
                {"type": "line"},
                [],
                {"row_count": len(result), "used_rows": len(result), "points": len(result), "series": 1},
            ),
        )
        monkeypatch.setattr(chart_compiler_mod, "MAX_ROWS_TO_PROCESS", 7)

        df = pd.DataFrame({"a": [1]})
        mock_mindsdb_client.query.return_value.fetch.return_value = df

        from minds.schemas.charts import XYIntent

        out = await service.get_conversation_message_chart(uuid4(), uuid4(), XYIntent(type="line", x="a", y="a"))
        assert out.config["type"] == "line"
        q = mock_mindsdb_client.query.call_args.args[0]
        assert "LIMIT 7" in q

    @pytest.mark.asyncio
    async def test_get_conversation_message_chart_unexpected_error_wrapped(
        self, service, mock_mindsdb_client, monkeypatch
    ):
        """Unexpected exceptions in chart generation should be wrapped."""
        import minds.services.conversations as conversations_mod

        service._get_conversation = AsyncMock(return_value=SimpleNamespace(id=uuid4()))  # type: ignore[method-assign]
        service._get_message = AsyncMock(  # type: ignore[method-assign]
            return_value=SimpleNamespace(role=Role.assistant, sql_query="SELECT * FROM t")
        )
        service._validate_and_parse_sql_query = Mock(return_value=SimpleNamespace(order_by=[]))  # type: ignore[method-assign]
        monkeypatch.setattr(conversations_mod, "extract_database_engines_from_select", lambda *_a, **_k: set())
        mock_mindsdb_client.query.side_effect = Exception("boom")

        from minds.schemas.charts import XYIntent

        with pytest.raises(ConversationsServiceError, match="Failed to generate chart: boom"):
            await service.get_conversation_message_chart(uuid4(), uuid4(), XYIntent(type="bar", x="a", y="a"))

    def test_validate_and_parse_sql_query_empty_raises(self, service):
        from minds.services.conversations import InvalidSQLQueryError

        with pytest.raises(InvalidSQLQueryError, match="Empty input"):
            service._validate_and_parse_sql_query("")

    def test_validate_and_parse_sql_query_parse_error_raises(self, service, monkeypatch):
        from mindsdb_sql_parser.exceptions import ParsingException

        import minds.services.conversations as conversations_mod
        from minds.services.conversations import InvalidSQLQueryError

        monkeypatch.setattr(conversations_mod, "parse_sql", lambda _s: (_ for _ in ()).throw(ParsingException("bad")))
        with pytest.raises(InvalidSQLQueryError, match="Invalid SQL query:"):
            service._validate_and_parse_sql_query("SELECT")

    def test_validate_and_parse_sql_query_non_select_raises(self, service, monkeypatch):
        import minds.services.conversations as conversations_mod
        from minds.services.conversations import InvalidSQLQueryError

        class _FakeSelect:  # local stand-in for isinstance check
            pass

        monkeypatch.setattr(conversations_mod, "Select", _FakeSelect)
        monkeypatch.setattr(conversations_mod, "parse_sql", lambda _s: object())

        with pytest.raises(InvalidSQLQueryError, match="not a SELECT query"):
            service._validate_and_parse_sql_query("DELETE FROM t")

    @pytest.mark.asyncio
    async def test_get_message_not_found_raises(self, service, mock_session):
        """_get_message should raise MessageNotFoundError when not found."""
        from minds.services.conversations import MessageNotFoundError

        mock_result = Mock()
        mock_result.first.return_value = None
        mock_session.exec.return_value = mock_result

        with pytest.raises(MessageNotFoundError):
            await service._get_message(uuid4(), uuid4())

    @pytest.mark.asyncio
    async def test_get_conversation_messages_with_events_includes_events_and_adds_loader_options(
        self,
        service,
        mock_session,
        sample_conversation,
        sample_message,
    ):
        """with_events=True should attach loader options and include event_data in responses."""
        from minds.model.message_event import MessageEvent

        # Avoid re-testing _get_conversation() here; focus on message query + event shaping.
        service._get_conversation = AsyncMock(return_value=sample_conversation)  # type: ignore[method-assign]

        # Attach events on the message as the ORM would when selectinload() is used.
        sample_message.user_id = UUID(service.user_id)
        sample_message.sql_query = "SELECT 1"
        sample_message.message_events = [
            MessageEvent(
                message_id=sample_message.id,
                organization_id=sample_message.organization_id,
                user_id=sample_message.user_id,
                sequence_number=1,
                event_data={"type": "tool", "value": 1},
            ),
            MessageEvent(
                message_id=sample_message.id,
                organization_id=sample_message.organization_id,
                user_id=sample_message.user_id,
                sequence_number=2,
                event_data={"type": "tool", "value": 2},
            ),
        ]

        mock_result = Mock()
        mock_result.all.return_value = [sample_message]
        mock_session.exec.return_value = mock_result

        out = await service.get_conversation_messages(
            sample_conversation.id,
            with_sql_query=True,
            with_events=True,
        )

        assert len(out) == 1
        msg = out[0]
        assert isinstance(msg, MessageResponse)
        assert msg.sql_query == "SELECT 1"
        assert msg.events == [{"type": "tool", "value": 1}, {"type": "tool", "value": 2}]
        assert msg.content.type == MessageContentType.input_text

        # Ensure we exercised the "statement.options(...)" branch.
        stmt = mock_session.exec.call_args.args[0]
        with_opts = getattr(stmt, "_with_options", ())
        assert len(with_opts) > 0

    @pytest.mark.asyncio
    async def test_get_conversation_messages_without_events_does_not_include_events_or_loader_options(
        self,
        service,
        mock_session,
        sample_conversation,
        sample_message,
    ):
        """with_events=False should not add loader options and should leave events unset."""
        from minds.model.message_event import MessageEvent

        service._get_conversation = AsyncMock(return_value=sample_conversation)  # type: ignore[method-assign]

        sample_message.user_id = UUID(service.user_id)
        sample_message.sql_query = "SELECT 1"
        sample_message.message_events = [
            MessageEvent(
                message_id=sample_message.id,
                organization_id=sample_message.organization_id,
                user_id=sample_message.user_id,
                sequence_number=1,
                event_data={"x": 1},
            )
        ]

        mock_result = Mock()
        mock_result.all.return_value = [sample_message]
        mock_session.exec.return_value = mock_result

        out = await service.get_conversation_messages(
            sample_conversation.id,
            with_sql_query=True,
            with_events=False,
        )

        assert len(out) == 1
        msg = out[0]
        assert msg.sql_query == "SELECT 1"
        assert msg.events is None

        stmt = mock_session.exec.call_args.args[0]
        with_opts = getattr(stmt, "_with_options", ())
        assert len(with_opts) == 0

    @pytest.mark.asyncio
    async def test_get_conversation_messages_conversation_not_found_propagates(self, service):
        """ConversationNotFoundError should bubble up unchanged."""
        service._get_conversation = AsyncMock(side_effect=ConversationNotFoundError("nope"))  # type: ignore[method-assign]
        with pytest.raises(ConversationNotFoundError, match="nope"):
            await service.get_conversation_messages(uuid4(), with_events=True)

    @pytest.mark.asyncio
    async def test_get_conversation_messages_unexpected_error_wrapped(self, service, mock_session, sample_conversation):
        """Unexpected errors should be wrapped as ConversationsServiceError."""
        service._get_conversation = AsyncMock(return_value=sample_conversation)  # type: ignore[method-assign]
        mock_session.exec.side_effect = Exception("db down")

        with pytest.raises(ConversationsServiceError, match="Failed to get conversation with messages"):
            await service.get_conversation_messages(sample_conversation.id, with_events=True)

    @pytest.mark.asyncio
    async def test_create_conversation_message_event_commit_flag_controls_commit(self, service, mock_session):
        """create_conversation_message_event should flush always and commit only when commit=True."""
        message_id = uuid4()

        await service.create_conversation_message_event(
            message_id=message_id,
            sequence_number=1,
            event_data={"k": "v"},
            commit=False,
        )
        mock_session.add.assert_called()
        mock_session.flush.assert_called()
        mock_session.commit.assert_not_called()

        mock_session.add.reset_mock()
        mock_session.flush.reset_mock()
        mock_session.commit.reset_mock()

        await service.create_conversation_message_event(
            message_id=message_id,
            sequence_number=2,
            event_data={"k2": "v2"},
            commit=True,
        )
        mock_session.add.assert_called()
        mock_session.flush.assert_called()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_conversation_message_report_exists_success(self, service, monkeypatch):
        """check_conversation_message_report_exists returns None when report exists."""
        import minds.services.conversations as conversations_mod

        conversation_id = uuid4()
        message_id = uuid4()

        service._get_conversation = AsyncMock(  # type: ignore[method-assign]
            return_value=SimpleNamespace(
                mind=SimpleNamespace(agent="anton_agent", name="anton", parameters={"agent_name": "anton_agent"})
            )
        )
        service._get_message = AsyncMock(return_value=SimpleNamespace(role=Role.assistant))  # type: ignore[method-assign]

        factory = SimpleNamespace(report_exists=AsyncMock(return_value=True))
        monkeypatch.setattr(conversations_mod, "ScratchpadRuntimeFactory", lambda: factory)
        monkeypatch.setattr(conversations_mod, "AntonAgentSettings", lambda: SimpleNamespace(backend="docker"))

        result = await service.check_conversation_message_report_exists(conversation_id, message_id)
        assert result is None
        factory.report_exists.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_check_conversation_message_report_exists_false_raises_file_not_found(self, service, monkeypatch):
        """check_conversation_message_report_exists raises FileNotFoundError when report does not exist."""
        import minds.services.conversations as conversations_mod

        conversation_id = uuid4()
        message_id = uuid4()

        service._get_conversation = AsyncMock(  # type: ignore[method-assign]
            return_value=SimpleNamespace(
                mind=SimpleNamespace(agent="anton_agent", name="anton", parameters={"agent_name": "anton_agent"})
            )
        )
        service._get_message = AsyncMock(return_value=SimpleNamespace(role=Role.assistant))  # type: ignore[method-assign]

        factory = SimpleNamespace(report_exists=AsyncMock(return_value=False))
        monkeypatch.setattr(conversations_mod, "ScratchpadRuntimeFactory", lambda: factory)
        monkeypatch.setattr(conversations_mod, "AntonAgentSettings", lambda: SimpleNamespace(backend="docker"))

        with pytest.raises(FileNotFoundError, match="report is not available"):
            await service.check_conversation_message_report_exists(conversation_id, message_id)

    @pytest.mark.asyncio
    async def test_check_conversation_message_report_exists_non_anton_conversation_raises(self, service, monkeypatch):
        """If default agent and mind agent aren't Anton, raise AgentNotAntonError."""
        import minds.services.conversations as conversations_mod

        service._get_conversation = AsyncMock(  # type: ignore[method-assign]
            return_value=SimpleNamespace(
                mind=SimpleNamespace(agent="other", name="other", parameters={"agent_name": "candidate_sql_agent"})
            )
        )
        service._get_message = AsyncMock(return_value=SimpleNamespace(role=Role.assistant))  # type: ignore[method-assign]

        monkeypatch.setattr(
            conversations_mod, "app_settings", SimpleNamespace(agents=SimpleNamespace(default_agent="not_anton"))
        )

        with pytest.raises(AgentNotAntonError, match="not using the Anton agent"):
            await service.check_conversation_message_report_exists(uuid4(), uuid4())

    @pytest.mark.asyncio
    async def test_get_conversation_message_report_success(self, service, monkeypatch):
        """get_conversation_message_report returns HTML string when available."""
        import minds.services.conversations as conversations_mod

        conversation_id = uuid4()
        message_id = uuid4()

        service._get_conversation = AsyncMock(  # type: ignore[method-assign]
            return_value=SimpleNamespace(
                mind=SimpleNamespace(agent="anton_agent", name="anton", parameters={"agent_name": "anton_agent"})
            )
        )
        service._get_message = AsyncMock(return_value=SimpleNamespace(role=Role.assistant))  # type: ignore[method-assign]

        factory = SimpleNamespace(get_report=AsyncMock(return_value="<html>ok</html>"))
        monkeypatch.setattr(conversations_mod, "ScratchpadRuntimeFactory", lambda: factory)
        monkeypatch.setattr(conversations_mod, "AntonAgentSettings", lambda: SimpleNamespace(backend="docker"))

        report = await service.get_conversation_message_report(conversation_id, message_id)
        assert report == "<html>ok</html>"
        factory.get_report.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_conversation_message_report_message_not_assistant_raises(self, service, monkeypatch):
        """get_conversation_message_report raises MessageNotAssistantError before calling backend."""
        import minds.services.conversations as conversations_mod

        service._get_conversation = AsyncMock(return_value=SimpleNamespace(mind=SimpleNamespace(agent="anton_agent")))  # type: ignore[method-assign]
        service._get_message = AsyncMock(return_value=SimpleNamespace(role=Role.user))  # type: ignore[method-assign]

        monkeypatch.setattr(
            conversations_mod,
            "get_app_settings",
            lambda: SimpleNamespace(agents=SimpleNamespace(default_agent="anton_agent")),
        )

        with pytest.raises(MessageNotAssistantError):
            await service.get_conversation_message_report(uuid4(), uuid4())

    @pytest.mark.asyncio
    async def test_create_conversation_success(
        self, service, mock_session, sample_create_request, mock_mind_service, mock_mind
    ):
        """Test successful conversation creation."""
        # Mock the check for existing conversation
        mock_result = Mock()
        mock_result.first.return_value = None
        mock_session.exec.return_value = mock_result

        # Mock mind_service.get_mind_model
        mock_mind_service.get_mind_model = AsyncMock(return_value=mock_mind)

        # Mock flush to set ID
        def mock_flush():
            if mock_session.add.call_args:
                conversation = mock_session.add.call_args[0][0]
                conversation.id = uuid4()
                conversation.created_at = datetime.now(timezone.utc)
                conversation.modified_at = datetime.now(timezone.utc)
                # Set mind relationship
                conversation.mind = mock_mind

        mock_session.flush.side_effect = mock_flush

        result = await service.create_conversation(sample_create_request, mock_mind_service)

        assert isinstance(result, ConversationResponse)
        assert result.metadata.topic == "New Conversation"
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_conversation_with_items(
        self, service, mock_session, sample_create_request, mock_mind_service, mock_mind
    ):
        """Test creating conversation with initial items."""
        from minds.schemas.chat import Message as ChatMessage

        # Add items to the request
        sample_create_request.items = [
            ChatMessage(role=Role.user, content="Hello"),
            ChatMessage(role=Role.assistant, content="Hi there!"),
        ]

        # Mock the check for existing conversation
        mock_result = Mock()
        mock_result.first.return_value = None
        mock_session.exec.return_value = mock_result

        # Mock mind_service.get_mind_model
        mock_mind_service.get_mind_model = AsyncMock(return_value=mock_mind)

        # Mock flush to set ID
        def mock_flush():
            if mock_session.add.call_args:
                conversation = mock_session.add.call_args[0][0]
                conversation.id = uuid4()
                conversation.created_at = datetime.now(timezone.utc)
                conversation.modified_at = datetime.now(timezone.utc)
                # Set mind relationship
                conversation.mind = mock_mind

        mock_session.flush.side_effect = mock_flush

        result = await service.create_conversation(sample_create_request, mock_mind_service)

        assert isinstance(result, ConversationResponse)
        mock_session.add.assert_called_once()  # For conversation
        mock_session.add_all.assert_called_once()  # For messages
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_conversation_database_error(
        self, service, mock_session, sample_create_request, mock_mind_service, mock_mind
    ):
        """Test create conversation with database error."""
        mock_result = Mock()
        mock_result.first.return_value = None
        mock_session.exec.return_value = mock_result
        mock_session.add.side_effect = Exception("Database error")

        # Mock mind_service.get_mind_model
        mock_mind_service.get_mind_model = AsyncMock(return_value=mock_mind)

        with pytest.raises(ConversationsServiceError, match="Failed to create conversation"):
            await service.create_conversation(sample_create_request, mock_mind_service)

    @pytest.mark.asyncio
    async def test_delete_conversation_success(self, service, mock_session, sample_conversation):
        """Test successful conversation deletion."""
        mock_result = Mock()
        mock_result.first.return_value = sample_conversation
        mock_session.exec.return_value = mock_result

        await service.delete_conversation(sample_conversation.id)

        assert sample_conversation.deleted_at is not None
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_conversation_not_found(self, service, mock_session):
        """Test delete conversation when not found."""
        mock_result = Mock()
        mock_result.first.return_value = None
        mock_session.exec.return_value = mock_result

        conversation_id = uuid4()
        with pytest.raises(ConversationNotFoundError, match=f"Conversation with ID '{conversation_id}' not found"):
            await service.delete_conversation(conversation_id)

    @pytest.mark.asyncio
    async def test_delete_conversation_database_error(self, service, mock_session, sample_conversation):
        """Test delete conversation with database error."""
        mock_result = Mock()
        mock_result.first.return_value = sample_conversation
        mock_session.exec.return_value = mock_result
        mock_session.commit.side_effect = Exception("Database error")

        conversation_id = uuid4()
        with pytest.raises(ConversationsServiceError, match="Failed to delete conversation"):
            await service.delete_conversation(conversation_id)

    @pytest.mark.asyncio
    async def test_get_conversation_messages_success(self, service, mock_session, sample_conversation, sample_message):
        """Test successful message retrieval."""
        # Mock conversation lookup
        mock_conv_result = Mock()
        mock_conv_result.first.return_value = sample_conversation
        # Mock messages lookup
        mock_msg_result = Mock()
        mock_msg_result.all.return_value = [sample_message]
        mock_session.exec.side_effect = [mock_conv_result, mock_msg_result]

        result = await service.get_conversation_messages(sample_conversation.id)

        assert len(result) == 1
        assert isinstance(result[0], MessageResponse)
        assert result[0].role == Role.user

    @pytest.mark.asyncio
    async def test_get_conversation_messages_not_found(self, service, mock_session):
        """Test get messages when conversation not found."""
        mock_result = Mock()
        mock_result.first.return_value = None
        mock_session.exec.return_value = mock_result

        conversation_id = uuid4()
        with pytest.raises(ConversationNotFoundError, match=f"Conversation with ID '{conversation_id}' not found"):
            await service.get_conversation_messages(conversation_id)

    @pytest.mark.asyncio
    async def test_create_conversation_message_success(self, service, mock_session, sample_conversation):
        """Test successful message creation."""
        mock_result = Mock()
        mock_result.first.return_value = sample_conversation
        mock_session.exec.return_value = mock_result

        # Create a message object that will be returned by the service
        created_message = Message(
            id=uuid4(),
            conversation_id=sample_conversation.id,
            organization_id=UUID(service.organization_id),
            role=Role.user,
            content="Test message",
            request_id="test-request-id",
            created_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
        )

        # Mock add to capture the message and set attributes
        def mock_add(message):
            message.id = created_message.id
            message.created_at = created_message.created_at
            message.modified_at = created_message.modified_at

        mock_session.add.side_effect = mock_add

        result = await service.create_conversation_message(sample_conversation.id, Role.user, "Test message")

        assert isinstance(result, MessageResponse)
        assert result.role == Role.user
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_conversation_message_placeholder_success(self, service, mock_session, sample_conversation):
        """Test successful message placeholder creation."""
        mock_result = Mock()
        mock_result.first.return_value = sample_conversation
        mock_session.exec.return_value = mock_result

        # Mock flush to set ID
        def mock_flush():
            if mock_session.add.call_args:
                message = mock_session.add.call_args[0][0]
                message.id = uuid4()
                message.created_at = datetime.now(timezone.utc)
                message.modified_at = datetime.now(timezone.utc)

        mock_session.flush.side_effect = mock_flush

        result = await service.create_conversation_message_placeholder(sample_conversation.id, Role.assistant)

        assert isinstance(result, Message)
        assert result.role == Role.assistant
        assert result.content == ""
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_conversation_message_content_success(self, service, mock_session, sample_message):
        """Test successful message content update."""
        mock_session.merge.return_value = sample_message

        result = await service.update_conversation_message_content(
            sample_message,
            "Updated content",
            sql_query="SELECT * FROM test",
            request_id="test-request-id",
        )

        assert isinstance(result, MessageResponse)
        assert sample_message.content == "Updated content"
        assert sample_message.sql_query == "SELECT * FROM test"
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_conversation_message_content_with_rollback(self, service, mock_session, sample_message):
        """Test message content update with rollback recovery."""
        from sqlalchemy.exc import PendingRollbackError

        # First call raises PendingRollbackError
        mock_session.merge.side_effect = [PendingRollbackError("Rollback needed"), sample_message]
        mock_session.exec.return_value.first.return_value = sample_message

        result = await service.update_conversation_message_content(
            sample_message,
            "Updated content",
            request_id="test-request-id",
        )

        assert isinstance(result, MessageResponse)
        mock_session.rollback.assert_called()
        mock_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_get_conversation_message_result_success(
        self, service, mock_session, mock_mindsdb_client, sample_conversation
    ):
        """Test successful message result retrieval."""
        # Create assistant message with SQL query
        assistant_message = Message(
            id=uuid4(),
            conversation_id=sample_conversation.id,
            organization_id=UUID(service.organization_id),
            role=Role.assistant,
            content="Query result",
            sql_query="SELECT * FROM test_table ORDER BY col1",
            request_id="test-request-id",
            created_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
        )

        # Mock conversation and message lookups
        mock_conv_result = Mock()
        mock_conv_result.first.return_value = sample_conversation
        mock_msg_result = Mock()
        mock_msg_result.first.return_value = assistant_message
        mock_session.exec.side_effect = [mock_conv_result, mock_msg_result]

        # Mock MindsDB query results
        # The service calls query twice: count query and paginated query
        # COUNT(*) returns a DataFrame with values array
        count_df = pd.DataFrame([[10]], columns=["count"])
        data_df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})

        # Create separate mock queries for each call
        mock_count_query = Mock()
        mock_count_query.fetch.return_value = count_df

        mock_data_query = Mock()
        mock_data_query.fetch.return_value = data_df

        # The service calls query twice: count and paginated
        mock_mindsdb_client.query.side_effect = [mock_count_query, mock_data_query]

        # Mock databases.get() to return a mock database with engine attribute
        # The service accesses: self.mindsdb_client.databases.get(parsed_sql_query.from_table.parts[0]).engine
        # For SQL "SELECT * FROM test_table ORDER BY col1", from_table.parts[0] is "test_table"
        mock_database = Mock()
        mock_database.engine = "postgresql"  # Not "mssql", so it uses LIMIT/OFFSET path
        mock_mindsdb_client.databases = Mock()
        mock_mindsdb_client.databases.get.return_value = mock_database

        result = await service.get_conversation_message_result(sample_conversation.id, assistant_message.id)

        assert isinstance(result, tuple)
        message_result, total_rows, is_pagination_consistent = result
        assert isinstance(message_result, MessageResultResponse)
        assert total_rows == 10
        assert isinstance(is_pagination_consistent, bool)

    @pytest.mark.asyncio
    async def test_get_conversation_message_result_not_assistant(
        self, service, mock_session, sample_conversation, sample_message
    ):
        """Test get message result when message is not assistant."""
        mock_conv_result = Mock()
        mock_conv_result.first.return_value = sample_conversation
        mock_msg_result = Mock()
        mock_msg_result.first.return_value = sample_message  # User message, not assistant
        mock_session.exec.side_effect = [mock_conv_result, mock_msg_result]

        with pytest.raises(MessageNotAssistantError):
            await service.get_conversation_message_result(sample_conversation.id, sample_message.id)

    @pytest.mark.asyncio
    async def test_get_conversation_message_result_no_sql_query(self, service, mock_session, sample_conversation):
        """Test get message result when message has no SQL query."""
        assistant_message = Message(
            id=uuid4(),
            conversation_id=sample_conversation.id,
            organization_id=UUID(service.organization_id),
            role=Role.assistant,
            content="No SQL",
            sql_query=None,
            request_id="test-request-id",
            created_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
        )

        mock_conv_result = Mock()
        mock_conv_result.first.return_value = sample_conversation
        mock_msg_result = Mock()
        mock_msg_result.first.return_value = assistant_message
        mock_session.exec.side_effect = [mock_conv_result, mock_msg_result]

        with pytest.raises(MessageNoSQLQueryError):
            await service.get_conversation_message_result(sample_conversation.id, assistant_message.id)

    @pytest.mark.asyncio
    async def test_export_conversation_message_result_success(
        self, service, mock_session, mock_mindsdb_client, sample_conversation
    ):
        """Test successful message result export."""
        assistant_message = Message(
            id=uuid4(),
            conversation_id=sample_conversation.id,
            organization_id=UUID(service.organization_id),
            role=Role.assistant,
            content="Query result",
            sql_query="SELECT * FROM test_table",
            request_id="test-request-id",
            created_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
        )

        mock_conv_result = Mock()
        mock_conv_result.first.return_value = sample_conversation
        mock_msg_result = Mock()
        mock_msg_result.first.return_value = assistant_message
        mock_session.exec.side_effect = [mock_conv_result, mock_msg_result]

        # Mock MindsDB query result
        data_df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        mock_query = Mock()
        mock_query.fetch.return_value = data_df
        mock_mindsdb_client.query.return_value = mock_query

        result = await service.export_conversation_message_result(sample_conversation.id, assistant_message.id)

        assert isinstance(result, bytes)
        assert b"col1" in result
        assert b"col2" in result

    @pytest.mark.asyncio
    async def test_export_conversation_message_result_not_assistant(
        self, service, mock_session, sample_conversation, sample_message
    ):
        """Test export message result when message is not assistant."""
        mock_conv_result = Mock()
        mock_conv_result.first.return_value = sample_conversation
        mock_msg_result = Mock()
        mock_msg_result.first.return_value = sample_message
        mock_session.exec.side_effect = [mock_conv_result, mock_msg_result]

        with pytest.raises(MessageNotAssistantError):
            await service.export_conversation_message_result(sample_conversation.id, sample_message.id)

    @pytest.mark.asyncio
    async def test_get_conversation_message_chart_chartjs_success(
        self, service, mock_session, mock_mindsdb_client, sample_conversation
    ):
        """Test chart generation in chartjs mode."""
        assistant_message = Message(
            id=uuid4(),
            conversation_id=sample_conversation.id,
            organization_id=UUID(service.organization_id),
            role=Role.assistant,
            content="Chart result",
            sql_query="SELECT * FROM sales",
            request_id="test-request-id",
            created_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
        )

        mock_conv_result = Mock()
        mock_conv_result.first.return_value = sample_conversation
        mock_msg_result = Mock()
        mock_msg_result.first.return_value = assistant_message
        mock_session.exec.side_effect = [mock_conv_result, mock_msg_result]

        data_df = pd.DataFrame({"month": ["jan", "feb"], "revenue": [100, 150]})
        mock_query = Mock()
        mock_query.fetch.return_value = data_df
        mock_mindsdb_client.query.return_value = mock_query
        mock_database = Mock()
        mock_database.engine = "postgresql"
        mock_mindsdb_client.databases = Mock()
        mock_mindsdb_client.databases.get.return_value = mock_database

        intent = XYIntent(type="bar", x="month", y="revenue")

        with patch("minds.services.chart_compiler.compile_chart") as mock_compile:
            mock_compile.return_value = (
                RenderPlan(
                    chart_type=RenderChartType.BAR,
                    title=None,
                    show_legend=False,
                    labels=["jan"],
                    series=[SeriesSpec(label="revenue", values=[100])],
                    x_axis=AxisSpec(title="month", scale_type="category"),
                    y_axis=AxisSpec(title="revenue", scale_type="linear"),
                ),
                [],
                {"row_count": 2, "used_rows": 2, "points": 1, "series": 1, "fields": None},
            )

            result = await service.get_conversation_message_chart(
                sample_conversation.id,
                assistant_message.id,
                intent,
            )

        assert isinstance(result, ChartResponse)
        assert result.config is not None
        assert result.meta.row_count == 2

    @pytest.mark.asyncio
    async def test_render_conversation_message_chart_png_success(
        self, service, mock_session, mock_mindsdb_client, sample_conversation
    ):
        """Test direct chart PNG rendering."""
        assistant_message = Message(
            id=uuid4(),
            conversation_id=sample_conversation.id,
            organization_id=UUID(service.organization_id),
            role=Role.assistant,
            content="Chart result",
            sql_query="SELECT * FROM sales",
            request_id="test-request-id",
            created_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
        )

        mock_conv_result = Mock()
        mock_conv_result.first.return_value = sample_conversation
        mock_msg_result = Mock()
        mock_msg_result.first.return_value = assistant_message
        mock_session.exec.side_effect = [mock_conv_result, mock_msg_result]

        data_df = pd.DataFrame({"month": ["jan", "feb"], "revenue": [100, 150]})
        mock_query = Mock()
        mock_query.fetch.return_value = data_df
        mock_mindsdb_client.query.return_value = mock_query
        mock_database = Mock()
        mock_database.engine = "postgresql"
        mock_mindsdb_client.databases = Mock()
        mock_mindsdb_client.databases.get.return_value = mock_database

        intent = XYIntent(type="bar", x="month", y="revenue")

        with (
            patch("minds.services.chart_compiler.compile_chart") as mock_compile,
            patch("minds.services.chart_renderer.render_chart_image", return_value=b"png-bytes") as mock_render,
        ):
            mock_compile.return_value = (
                RenderPlan(
                    chart_type=RenderChartType.BAR,
                    title=None,
                    show_legend=False,
                    labels=["jan"],
                    series=[SeriesSpec(label="revenue", values=[100])],
                    x_axis=AxisSpec(title="month", scale_type="category"),
                    y_axis=AxisSpec(title="revenue", scale_type="linear"),
                ),
                [],
                {"row_count": 2, "used_rows": 2, "points": 1, "series": 1, "fields": None},
            )

            result = await service.render_conversation_message_chart_png(
                sample_conversation.id,
                assistant_message.id,
                intent,
            )

        assert result == b"png-bytes"
        mock_render.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_conversation_message_chart_image_success(
        self, service, mock_session, mock_mindsdb_client, sample_conversation
    ):
        """Test image_url generation returns an opaque tokenized URL."""
        assistant_message = Message(
            id=uuid4(),
            conversation_id=sample_conversation.id,
            organization_id=UUID(service.organization_id),
            role=Role.assistant,
            content="Chart result",
            sql_query="SELECT * FROM sales",
            request_id="test-request-id",
            created_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
        )

        mock_conv_result = Mock()
        mock_conv_result.first.return_value = sample_conversation
        mock_msg_result = Mock()
        mock_msg_result.first.return_value = assistant_message
        mock_session.exec.side_effect = [mock_conv_result, mock_msg_result]

        data_df = pd.DataFrame({"month": ["jan", "feb"], "revenue": [100, 150]})
        mock_query = Mock()
        mock_query.fetch.return_value = data_df
        mock_mindsdb_client.query.return_value = mock_query
        mock_database = Mock()
        mock_database.engine = "postgresql"
        mock_mindsdb_client.databases = Mock()
        mock_mindsdb_client.databases.get.return_value = mock_database

        intent = XYIntent(type="bar", x="month", y="revenue")

        with patch("minds.services.chart_compiler.compile_chart") as mock_compile:
            mock_compile.return_value = (
                RenderPlan(
                    chart_type=RenderChartType.BAR,
                    title=None,
                    show_legend=False,
                    labels=["jan"],
                    series=[SeriesSpec(label="revenue", values=[100])],
                    x_axis=AxisSpec(title="month", scale_type="category"),
                    y_axis=AxisSpec(title="revenue", scale_type="linear"),
                ),
                [],
                ChartMeta(row_count=2, used_rows=2, points=1, series=1, fields=None),
            )

            result = await service.get_conversation_message_chart_image(
                sample_conversation.id,
                assistant_message.id,
                intent,
            )

        token = result.image_url.split("token=")[1]
        assert isinstance(result, ChartImageResponse)
        assert result.image_url.startswith(
            f"/v1/conversations/{sample_conversation.id}/items/{assistant_message.id}/chart?token="
        )
        assert service._parse_chart_image_token(token) == intent

    @pytest.mark.asyncio
    async def test_render_conversation_message_chart_by_token_success(self, service, mock_session, sample_conversation):
        """Test token-backed image GET renders from the encoded intent."""
        assistant_message = Message(
            id=uuid4(),
            conversation_id=sample_conversation.id,
            organization_id=UUID(service.organization_id),
            role=Role.assistant,
            content="Chart result",
            sql_query="SELECT * FROM sales",
            request_id="test-request-id",
            created_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
        )

        mock_conv_result = Mock()
        mock_conv_result.first.return_value = sample_conversation
        mock_msg_result = Mock()
        mock_msg_result.first.return_value = assistant_message
        mock_session.exec.side_effect = [mock_conv_result, mock_msg_result]

        token = service._build_chart_image_token(XYIntent(type="bar", x="month", y="revenue"))

        with patch.object(
            service,
            "render_conversation_message_chart_png",
            AsyncMock(return_value=b"png-bytes"),
        ) as mock_render:
            result = await service.render_conversation_message_chart_by_token(
                sample_conversation.id,
                assistant_message.id,
                token,
            )

        assert result == b"png-bytes"
        mock_render.assert_awaited_once_with(
            conversation_id=sample_conversation.id,
            message_id=assistant_message.id,
            intent=XYIntent(type="bar", x="month", y="revenue"),
        )

    def test_build_chart_image_token_changes_with_intent(self, service):
        """Test token changes when the chart intent changes."""
        first_intent = XYIntent(type="bar", x="month", y="revenue")
        second_intent = XYIntent(type="line", x="month", y="revenue")

        first_key = service._build_chart_image_token(first_intent)
        second_key = service._build_chart_image_token(second_intent)

        assert first_key != second_key

    def test_build_chart_image_token_round_trips_same_intent(self, service):
        """Test token round-trips the same intent."""
        intent = XYIntent(type="bar", x="month", y="revenue")

        token = service._build_chart_image_token(intent)

        assert service._parse_chart_image_token(token) == intent

    def test_parse_chart_image_token_rejects_invalid_payload(self, service):
        """Test invalid tokens are rejected."""
        with pytest.raises(ValueError, match="Invalid chart image token"):
            service._parse_chart_image_token("not-a-valid-token")

    @pytest.mark.asyncio
    async def test_conversation_to_response(self, service, sample_conversation):
        """Test conversation_to_response conversion."""
        result = await service.conversation_to_response(sample_conversation)

        assert isinstance(result, ConversationResponse)
        assert result.id == sample_conversation.id
        assert result.metadata.topic == "Test Conversation"
        assert result.metadata.model_name == "test-model"

    @pytest.mark.asyncio
    async def test_message_to_response(self, service, sample_message):
        """Test _message_to_response conversion."""
        result = await service._message_to_response(sample_message)

        assert isinstance(result, MessageResponse)
        assert result.id == sample_message.id
        assert result.role == Role.user
        assert result.content.type == MessageContentType.input_text

    @pytest.mark.asyncio
    async def test_message_to_response_assistant(self, service, sample_conversation):
        """Test _message_to_response conversion for assistant message."""
        assistant_message = Message(
            id=uuid4(),
            conversation_id=sample_conversation.id,
            organization_id=UUID(service.organization_id),
            role=Role.assistant,
            content="Assistant response",
            request_id="test-request-id",
            created_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
        )

        result = await service._message_to_response(assistant_message)

        assert isinstance(result, MessageResponse)
        assert result.role == Role.assistant
        assert result.content.type == MessageContentType.output_text
