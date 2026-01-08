"""
Unit tests for ConversationsService.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

import pandas as pd
import pytest
from mindsdb_sdk.server import Server
from sqlmodel import Session

from minds.model.conversation import Conversation
from minds.model.message import Message
from minds.model.mind import Mind
from minds.schemas.chat import Role
from minds.schemas.conversations import ConversationCreateRequest, ConversationMetadata, ConversationResponse
from minds.schemas.messages import MessageContentType, MessageResponse, MessageResultResponse
from minds.services.conversations import (
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
    def tenant_id(self):
        """Test tenant ID."""
        return str(uuid4())

    @pytest.fixture
    def service(self, mock_session, mock_mindsdb_client, user_id, tenant_id):
        """Create ConversationsService instance."""
        return ConversationsService(
            session=mock_session,
            mindsdb_client=mock_mindsdb_client,
            user_id=user_id,
            tenant_id=tenant_id,
        )

    @pytest.fixture
    def sample_conversation(self, user_id, tenant_id):
        """Sample conversation for testing."""
        mind_id = uuid4()
        mind = Mind(
            id=mind_id,
            name="test-model",
            provider="openai",
            model_name="gpt-4",
            user_id=UUID(user_id),
        )
        conversation = Conversation(
            id=uuid4(),
            topic="Test Conversation",
            user_id=UUID(user_id),
            tenant_id=UUID(tenant_id),
            mind_id=mind_id,
            created_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
        )
        # Set the mind relationship
        conversation.mind = mind
        return conversation

    @pytest.fixture
    def sample_message(self, sample_conversation, tenant_id):
        """Sample message for testing."""
        return Message(
            id=uuid4(),
            conversation_id=sample_conversation.id,
            tenant_id=UUID(tenant_id),
            role=Role.user,
            content="Hello, world!",
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
            model_name="gpt-4",
            user_id=UUID(user_id),
        )
        return mind

    def test_service_initialization(self, mock_session, mock_mindsdb_client, user_id, tenant_id):
        """Test service initialization."""
        service = ConversationsService(
            session=mock_session,
            mindsdb_client=mock_mindsdb_client,
            user_id=user_id,
            tenant_id=tenant_id,
        )

        assert service.session == mock_session
        assert service.user_id == user_id
        assert service.tenant_id == tenant_id
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
            tenant_id=UUID(service.tenant_id),
            role=Role.user,
            content="Test message",
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
            sample_message, "Updated content", "SELECT * FROM test"
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

        result = await service.update_conversation_message_content(sample_message, "Updated content")

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
            tenant_id=UUID(service.tenant_id),
            role=Role.assistant,
            content="Query result",
            sql_query="SELECT * FROM test_table ORDER BY col1",
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
            tenant_id=UUID(service.tenant_id),
            role=Role.assistant,
            content="No SQL",
            sql_query=None,
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
            tenant_id=UUID(service.tenant_id),
            role=Role.assistant,
            content="Query result",
            sql_query="SELECT * FROM test_table",
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
            tenant_id=UUID(service.tenant_id),
            role=Role.assistant,
            content="Assistant response",
            created_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
        )

        result = await service._message_to_response(assistant_message)

        assert isinstance(result, MessageResponse)
        assert result.role == Role.assistant
        assert result.content.type == MessageContentType.output_text
