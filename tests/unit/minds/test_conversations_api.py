"""
Unit tests for conversations API endpoints.
"""

from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

from minds.api.v1.endpoints.conversations import (
    create_conversation,
    delete_conversation,
    export_conversation_message_result,
    get_conversation,
    get_conversation_message_result,
    get_conversation_messages,
    get_conversations_service,
    list_conversations,
)
from minds.schemas.conversations import ConversationCreateRequest, ConversationMetadata, ConversationResponse
from minds.schemas.messages import MessageContent, MessageContentType, MessageResponse, MessageResultResponse
from minds.services.conversations import (
    ConversationAlreadyExistsError,
    ConversationNotFoundError,
    ConversationsService,
    ConversationsServiceError,
    MessageNoSQLQueryError,
    MessageNotAssistantError,
    MessageNotFoundError,
)
from minds.services.minds import MindsService


class TestConversationsAPI:
    """Test conversations API endpoints."""

    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI request object."""
        request = Mock()
        request.headers = {"x-user-id": "test-user-123"}
        return request

    @pytest.fixture
    def mock_session(self):
        """Mock database session."""
        session = Mock()
        session.commit = Mock()
        session.rollback = Mock()
        session.close = Mock()
        return session

    @pytest.fixture
    def mock_mindsdb_client(self):
        """Mock MindsDB client."""
        return Mock()

    @pytest.fixture
    def mock_conversations_service(self):
        """Mock ConversationsService instance."""
        service = Mock(spec=ConversationsService)
        service.user_id = "test-user-123"
        service.tenant_id = "test-tenant-456"
        return service

    @pytest.fixture
    def mock_mind_service(self):
        """Mock MindsService instance."""
        return Mock(spec=MindsService)

    @pytest.fixture
    def test_uuid(self):
        """Test UUID for responses."""
        return uuid4()

    @pytest.fixture
    def sample_conversation_response(self, test_uuid):
        """Sample ConversationResponse for testing."""
        return ConversationResponse(
            id=test_uuid,
            metadata=ConversationMetadata(topic="Test Conversation", model_name="test-model"),
            created_at="2023-01-01T12:00:00",
            modified_at="2023-01-01T12:00:00",
        )

    @pytest.fixture
    def sample_message_response(self, test_uuid):
        """Sample MessageResponse for testing."""
        return MessageResponse(
            id=test_uuid,
            role="user",
            content=MessageContent(type=MessageContentType.input_text, text="Hello"),
            created_at="2023-01-01T12:00:00",
            modified_at="2023-01-01T12:00:00",
        )

    def test_get_conversations_service_dependency(self, mock_request, mock_session, mock_mindsdb_client):
        """Test the get_conversations_service dependency function."""
        with (
            patch("minds.api.v1.endpoints.conversations.extract_context_from_request") as mock_extract,
            patch("minds.api.v1.endpoints.conversations.create_mindsdb_client_from_request") as mock_create_client,
        ):
            mock_context = Mock()
            mock_context.user_id = "test-user-123"
            mock_context.tenant_id = "test-tenant-456"
            mock_extract.return_value = mock_context
            mock_create_client.return_value = mock_mindsdb_client

            service = get_conversations_service(mock_request, mock_session)

            assert isinstance(service, ConversationsService)
            assert service.session == mock_session
            assert service.user_id == "test-user-123"
            assert service.tenant_id == "test-tenant-456"

    @pytest.mark.asyncio
    async def test_list_conversations_success(self, mock_conversations_service, sample_conversation_response):
        """Test successful conversations listing."""
        mock_conversations_service.list_conversations = AsyncMock(return_value=[sample_conversation_response])

        result = await list_conversations(
            conversations_service=mock_conversations_service,
            topic=None,
            limit=50,
            offset=0,
            include_total=False,
            sort_by=None,
            sort_order="desc",
        )

        assert len(result) == 1
        assert result[0].metadata.topic == "Test Conversation"
        mock_conversations_service.list_conversations.assert_called_once_with(
            topic=None,
            limit=50,
            offset=0,
            include_total=False,
            sort_by=None,
            sort_order="desc",
        )

    @pytest.mark.asyncio
    async def test_list_conversations_with_total(self, mock_conversations_service, sample_conversation_response):
        """Test listing conversations with total count."""
        mock_conversations_service.list_conversations = AsyncMock(return_value=([sample_conversation_response], 1))

        result = await list_conversations(
            conversations_service=mock_conversations_service,
            topic=None,
            limit=50,
            offset=0,
            include_total=True,
            sort_by=None,
            sort_order="desc",
        )

        assert isinstance(result, dict)
        assert "conversations" in result
        assert "total" in result
        assert len(result["conversations"]) == 1
        assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_list_conversations_empty(self, mock_conversations_service):
        """Test listing empty conversations."""
        mock_conversations_service.list_conversations = AsyncMock(return_value=[])

        result = await list_conversations(
            conversations_service=mock_conversations_service,
            include_total=False,
            sort_by=None,
            sort_order="desc",
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_list_conversations_with_filters(self, mock_conversations_service, sample_conversation_response):
        """Test listing conversations with filters."""
        mock_conversations_service.list_conversations = AsyncMock(return_value=[sample_conversation_response])

        result = await list_conversations(
            conversations_service=mock_conversations_service,
            topic="Test",
            limit=10,
            offset=5,
            include_total=False,
            sort_by="created_at",
            sort_order="asc",
        )

        assert len(result) == 1
        mock_conversations_service.list_conversations.assert_called_once_with(
            topic="Test",
            limit=10,
            offset=5,
            include_total=False,
            sort_by="created_at",
            sort_order="asc",
        )

    @pytest.mark.asyncio
    async def test_list_conversations_service_error(self, mock_conversations_service):
        """Test list conversations with service error."""
        mock_conversations_service.list_conversations = AsyncMock(
            side_effect=ConversationsServiceError("Service error")
        )

        with pytest.raises(HTTPException) as exc_info:
            await list_conversations(
                conversations_service=mock_conversations_service,
                include_total=False,
                sort_by=None,
                sort_order="desc",
            )

        assert exc_info.value.status_code == 500
        assert "Service error" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_list_conversations_unexpected_error(self, mock_conversations_service):
        """Test list conversations with unexpected error."""
        mock_conversations_service.list_conversations = AsyncMock(side_effect=Exception("Unexpected error"))

        with pytest.raises(HTTPException) as exc_info:
            await list_conversations(
                conversations_service=mock_conversations_service,
                include_total=False,
                sort_by=None,
                sort_order="desc",
            )

        assert exc_info.value.status_code == 500
        assert "Internal server error" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_conversation_success(self, mock_conversations_service, sample_conversation_response):
        """Test successful conversation retrieval."""
        mock_conversations_service.get_conversation = AsyncMock(return_value=sample_conversation_response)

        conversation_id = uuid4()
        result = await get_conversation(
            conversation_id=conversation_id,
            conversations_service=mock_conversations_service,
        )

        assert result.metadata.topic == "Test Conversation"
        mock_conversations_service.get_conversation.assert_called_once_with(conversation_id)

    @pytest.mark.asyncio
    async def test_get_conversation_not_found(self, mock_conversations_service):
        """Test get conversation when not found."""
        mock_conversations_service.get_conversation = AsyncMock(
            side_effect=ConversationNotFoundError("Conversation not found")
        )

        conversation_id = uuid4()
        with pytest.raises(HTTPException) as exc_info:
            await get_conversation(
                conversation_id=conversation_id,
                conversations_service=mock_conversations_service,
            )

        assert exc_info.value.status_code == 404
        assert "Conversation not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_conversation_service_error(self, mock_conversations_service):
        """Test get conversation with service error."""
        mock_conversations_service.get_conversation = AsyncMock(side_effect=ConversationsServiceError("Service error"))

        conversation_id = uuid4()
        with pytest.raises(HTTPException) as exc_info:
            await get_conversation(
                conversation_id=conversation_id,
                conversations_service=mock_conversations_service,
            )

        assert exc_info.value.status_code == 500
        assert "Service error" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_conversation_unexpected_error(self, mock_conversations_service):
        """Test get conversation with unexpected error."""
        mock_conversations_service.get_conversation = AsyncMock(side_effect=Exception("Unexpected error"))

        conversation_id = uuid4()
        with pytest.raises(HTTPException) as exc_info:
            await get_conversation(
                conversation_id=conversation_id,
                conversations_service=mock_conversations_service,
            )

        assert exc_info.value.status_code == 500
        assert "Internal server error" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_conversation_messages_success(self, mock_conversations_service, sample_message_response):
        """Test successful conversation messages retrieval."""
        mock_conversations_service.get_conversation_messages = AsyncMock(return_value=[sample_message_response])

        conversation_id = uuid4()
        result = await get_conversation_messages(
            conversation_id=conversation_id,
            conversations_service=mock_conversations_service,
        )

        assert isinstance(result, dict)
        assert "object" in result
        assert "data" in result
        assert result["object"] == "list"
        assert len(result["data"]) == 1
        mock_conversations_service.get_conversation_messages.assert_called_once_with(conversation_id)

    @pytest.mark.asyncio
    async def test_get_conversation_messages_service_error(self, mock_conversations_service):
        """Test get conversation messages with service error."""
        mock_conversations_service.get_conversation_messages = AsyncMock(
            side_effect=ConversationsServiceError("Service error")
        )

        conversation_id = uuid4()
        with pytest.raises(HTTPException) as exc_info:
            await get_conversation_messages(
                conversation_id=conversation_id,
                conversations_service=mock_conversations_service,
            )

        assert exc_info.value.status_code == 500
        assert "Service error" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_conversation_messages_unexpected_error(self, mock_conversations_service):
        """Test get conversation messages with unexpected error."""
        mock_conversations_service.get_conversation_messages = AsyncMock(side_effect=Exception("Unexpected error"))

        conversation_id = uuid4()
        with pytest.raises(HTTPException) as exc_info:
            await get_conversation_messages(
                conversation_id=conversation_id,
                conversations_service=mock_conversations_service,
            )

        assert exc_info.value.status_code == 500
        assert "Internal server error" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_create_conversation_success(
        self, mock_conversations_service, mock_mind_service, sample_conversation_response
    ):
        """Test successful conversation creation."""
        mock_conversations_service.create_conversation = AsyncMock(return_value=sample_conversation_response)

        create_request = ConversationCreateRequest(
            metadata=ConversationMetadata(topic="New Conversation", model_name="test-model"),
        )

        result = await create_conversation(
            conversation_data=create_request,
            conversations_service=mock_conversations_service,
            mind_service=mock_mind_service,
        )

        assert result.metadata.topic == "Test Conversation"
        mock_conversations_service.create_conversation.assert_called_once_with(create_request, mock_mind_service)

    @pytest.mark.asyncio
    async def test_create_conversation_already_exists(self, mock_conversations_service):
        """Test create conversation when it already exists."""
        mock_conversations_service.create_conversation = AsyncMock(
            side_effect=ConversationAlreadyExistsError("Already exists")
        )

        create_request = ConversationCreateRequest(
            metadata=ConversationMetadata(topic="Existing Conversation", model_name="test-model"),
        )

        with pytest.raises(HTTPException) as exc_info:
            await create_conversation(
                conversation_data=create_request,
                conversations_service=mock_conversations_service,
            )

        assert exc_info.value.status_code == 409
        assert "Already exists" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_create_conversation_service_error(self, mock_conversations_service):
        """Test create conversation with service error."""
        mock_conversations_service.create_conversation = AsyncMock(
            side_effect=ConversationsServiceError("Service error")
        )

        create_request = ConversationCreateRequest(
            metadata=ConversationMetadata(topic="New Conversation", model_name="test-model"),
        )

        with pytest.raises(HTTPException) as exc_info:
            await create_conversation(
                conversation_data=create_request,
                conversations_service=mock_conversations_service,
            )

        assert exc_info.value.status_code == 500
        assert "Service error" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_create_conversation_unexpected_error(self, mock_conversations_service):
        """Test create conversation with unexpected error."""
        mock_conversations_service.create_conversation = AsyncMock(side_effect=Exception("Unexpected error"))

        create_request = ConversationCreateRequest(
            metadata=ConversationMetadata(topic="New Conversation", model_name="test-model"),
        )

        with pytest.raises(HTTPException) as exc_info:
            await create_conversation(
                conversation_data=create_request,
                conversations_service=mock_conversations_service,
            )

        assert exc_info.value.status_code == 500
        assert "Internal server error" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_delete_conversation_success(self, mock_conversations_service):
        """Test successful conversation deletion."""
        mock_conversations_service.delete_conversation = AsyncMock()

        conversation_id = uuid4()
        result = await delete_conversation(
            conversation_id=conversation_id,
            conversations_service=mock_conversations_service,
        )

        assert result is None
        mock_conversations_service.delete_conversation.assert_called_once_with(conversation_id)

    @pytest.mark.asyncio
    async def test_delete_conversation_not_found(self, mock_conversations_service):
        """Test delete conversation when not found."""
        mock_conversations_service.delete_conversation = AsyncMock(
            side_effect=ConversationNotFoundError("Conversation not found")
        )

        conversation_id = uuid4()
        with pytest.raises(HTTPException) as exc_info:
            await delete_conversation(
                conversation_id=conversation_id,
                conversations_service=mock_conversations_service,
            )

        assert exc_info.value.status_code == 404
        assert "Conversation not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_delete_conversation_service_error(self, mock_conversations_service):
        """Test delete conversation with service error."""
        mock_conversations_service.delete_conversation = AsyncMock(
            side_effect=ConversationsServiceError("Service error")
        )

        conversation_id = uuid4()
        with pytest.raises(HTTPException) as exc_info:
            await delete_conversation(
                conversation_id=conversation_id,
                conversations_service=mock_conversations_service,
            )

        assert exc_info.value.status_code == 500
        assert "Service error" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_delete_conversation_unexpected_error(self, mock_conversations_service):
        """Test delete conversation with unexpected error."""
        mock_conversations_service.delete_conversation = AsyncMock(side_effect=Exception("Unexpected error"))

        conversation_id = uuid4()
        with pytest.raises(HTTPException) as exc_info:
            await delete_conversation(
                conversation_id=conversation_id,
                conversations_service=mock_conversations_service,
            )

        assert exc_info.value.status_code == 500
        assert "Internal server error" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_conversation_message_result_success(self, mock_conversations_service):
        """Test successful message result retrieval."""
        message_result = MessageResultResponse(
            data=[[1, "a"], [2, "b"]],
            column_names=["col1", "col2"],
        )
        mock_conversations_service.get_conversation_message_result = AsyncMock(return_value=(message_result, 2, True))

        conversation_id = uuid4()
        message_id = uuid4()
        result = await get_conversation_message_result(
            conversation_id=conversation_id,
            message_id=message_id,
            limit=100,
            offset=0,
            conversations_service=mock_conversations_service,
        )

        assert isinstance(result, dict)
        assert "result" in result
        assert "total" in result
        assert "is_pagination_consistent" in result
        assert result["total"] == 2
        assert result["is_pagination_consistent"] is True
        mock_conversations_service.get_conversation_message_result.assert_called_once_with(
            conversation_id, message_id, limit=100, offset=0
        )

    @pytest.mark.asyncio
    async def test_get_conversation_message_result_conversation_not_found(self, mock_conversations_service):
        """Test get message result when conversation not found."""
        mock_conversations_service.get_conversation_message_result = AsyncMock(
            side_effect=ConversationNotFoundError("Conversation not found")
        )

        conversation_id = uuid4()
        message_id = uuid4()
        with pytest.raises(HTTPException) as exc_info:
            await get_conversation_message_result(
                conversation_id=conversation_id,
                message_id=message_id,
                conversations_service=mock_conversations_service,
            )

        assert exc_info.value.status_code == 404
        assert "Conversation not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_conversation_message_result_message_not_found(self, mock_conversations_service):
        """Test get message result when message not found."""
        mock_conversations_service.get_conversation_message_result = AsyncMock(
            side_effect=MessageNotFoundError("Message not found")
        )

        conversation_id = uuid4()
        message_id = uuid4()
        with pytest.raises(HTTPException) as exc_info:
            await get_conversation_message_result(
                conversation_id=conversation_id,
                message_id=message_id,
                conversations_service=mock_conversations_service,
            )

        assert exc_info.value.status_code == 404
        assert "Message not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_conversation_message_result_not_assistant(self, mock_conversations_service):
        """Test get message result when message is not assistant."""
        mock_conversations_service.get_conversation_message_result = AsyncMock(
            side_effect=MessageNotAssistantError("Message is not an assistant message")
        )

        conversation_id = uuid4()
        message_id = uuid4()
        with pytest.raises(HTTPException) as exc_info:
            await get_conversation_message_result(
                conversation_id=conversation_id,
                message_id=message_id,
                conversations_service=mock_conversations_service,
            )

        assert exc_info.value.status_code == 400
        assert "Message is not an assistant message" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_conversation_message_result_no_sql_query(self, mock_conversations_service):
        """Test get message result when message has no SQL query."""
        mock_conversations_service.get_conversation_message_result = AsyncMock(
            side_effect=MessageNoSQLQueryError("Message does not have a SQL query")
        )

        conversation_id = uuid4()
        message_id = uuid4()
        with pytest.raises(HTTPException) as exc_info:
            await get_conversation_message_result(
                conversation_id=conversation_id,
                message_id=message_id,
                conversations_service=mock_conversations_service,
            )

        assert exc_info.value.status_code == 400
        assert "Message does not have a SQL query" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_conversation_message_result_service_error(self, mock_conversations_service):
        """Test get message result with service error."""
        mock_conversations_service.get_conversation_message_result = AsyncMock(
            side_effect=ConversationsServiceError("Service error")
        )

        conversation_id = uuid4()
        message_id = uuid4()
        with pytest.raises(HTTPException) as exc_info:
            await get_conversation_message_result(
                conversation_id=conversation_id,
                message_id=message_id,
                conversations_service=mock_conversations_service,
            )

        assert exc_info.value.status_code == 500
        assert "Service error" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_conversation_message_result_unexpected_error(self, mock_conversations_service):
        """Test get message result with unexpected error."""
        mock_conversations_service.get_conversation_message_result = AsyncMock(
            side_effect=Exception("Unexpected error")
        )

        conversation_id = uuid4()
        message_id = uuid4()
        with pytest.raises(HTTPException) as exc_info:
            await get_conversation_message_result(
                conversation_id=conversation_id,
                message_id=message_id,
                conversations_service=mock_conversations_service,
            )

        assert exc_info.value.status_code == 500
        assert "Internal server error" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_export_conversation_message_result_success(self, mock_conversations_service):
        """Test successful message result export."""
        csv_data = b"col1,col2\n1,a\n2,b\n"
        mock_conversations_service.export_conversation_message_result = AsyncMock(return_value=csv_data)

        conversation_id = uuid4()
        message_id = uuid4()
        result = await export_conversation_message_result(
            conversation_id=conversation_id,
            message_id=message_id,
            conversations_service=mock_conversations_service,
        )

        # FastAPI Response object uses body attribute, not content
        assert result.body == csv_data
        assert result.media_type == "text/csv"
        mock_conversations_service.export_conversation_message_result.assert_called_once_with(
            conversation_id, message_id
        )

    @pytest.mark.asyncio
    async def test_export_conversation_message_result_conversation_not_found(self, mock_conversations_service):
        """Test export message result when conversation not found."""
        mock_conversations_service.export_conversation_message_result = AsyncMock(
            side_effect=ConversationNotFoundError("Conversation not found")
        )

        conversation_id = uuid4()
        message_id = uuid4()
        with pytest.raises(HTTPException) as exc_info:
            await export_conversation_message_result(
                conversation_id=conversation_id,
                message_id=message_id,
                conversations_service=mock_conversations_service,
            )

        assert exc_info.value.status_code == 404
        assert "Conversation not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_export_conversation_message_result_message_not_found(self, mock_conversations_service):
        """Test export message result when message not found."""
        mock_conversations_service.export_conversation_message_result = AsyncMock(
            side_effect=MessageNotFoundError("Message not found")
        )

        conversation_id = uuid4()
        message_id = uuid4()
        with pytest.raises(HTTPException) as exc_info:
            await export_conversation_message_result(
                conversation_id=conversation_id,
                message_id=message_id,
                conversations_service=mock_conversations_service,
            )

        assert exc_info.value.status_code == 404
        assert "Message not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_export_conversation_message_result_not_assistant(self, mock_conversations_service):
        """Test export message result when message is not assistant."""
        mock_conversations_service.export_conversation_message_result = AsyncMock(
            side_effect=MessageNotAssistantError("Message is not an assistant message")
        )

        conversation_id = uuid4()
        message_id = uuid4()
        with pytest.raises(HTTPException) as exc_info:
            await export_conversation_message_result(
                conversation_id=conversation_id,
                message_id=message_id,
                conversations_service=mock_conversations_service,
            )

        assert exc_info.value.status_code == 400
        assert "Message is not an assistant message" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_export_conversation_message_result_service_error(self, mock_conversations_service):
        """Test export message result with service error."""
        mock_conversations_service.export_conversation_message_result = AsyncMock(
            side_effect=ConversationsServiceError("Service error")
        )

        conversation_id = uuid4()
        message_id = uuid4()
        with pytest.raises(HTTPException) as exc_info:
            await export_conversation_message_result(
                conversation_id=conversation_id,
                message_id=message_id,
                conversations_service=mock_conversations_service,
            )

        assert exc_info.value.status_code == 500
        assert "Service error" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_export_conversation_message_result_unexpected_error(self, mock_conversations_service):
        """Test export message result with unexpected error."""
        mock_conversations_service.export_conversation_message_result = AsyncMock(
            side_effect=Exception("Unexpected error")
        )

        conversation_id = uuid4()
        message_id = uuid4()
        with pytest.raises(HTTPException) as exc_info:
            await export_conversation_message_result(
                conversation_id=conversation_id,
                message_id=message_id,
                conversations_service=mock_conversations_service,
            )

        assert exc_info.value.status_code == 500
        assert "Internal server error" in str(exc_info.value.detail)
