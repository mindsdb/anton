"""
Unit tests for MindsService.

Tests the business logic layer for minds management including:
- CRUD operations
- Error handling
- Validation
- Transaction management
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from sqlmodel import Session

from minds.model.mind import Mind
from minds.schemas.minds import DatasourceConfig, MindCreateRequest, MindUpdateRequest
from minds.services.conversations import ConversationsService
from minds.services.minds import (
    MindAlreadyExistsError,
    MindNotFoundError,
    MindsService,
    MindsServiceError,
)


class TestMindsService:
    """Test suite for MindsService."""

    @pytest.fixture
    def mock_session(self):
        """Mock database session."""
        session = Mock(spec=Session)
        session.exec = Mock()
        session.add = Mock()
        session.commit = Mock()
        session.rollback = Mock()
        session.refresh = Mock()
        return session

    @pytest.fixture
    def mock_mindsdb_client(self):
        """Mock MindsDB client."""
        return Mock()

    @pytest.fixture
    def mock_conversations_service(self):
        """Mock ConversationsService instance."""
        return Mock(spec=ConversationsService)

    @pytest.fixture
    def minds_service(self, mock_session, mock_mindsdb_client, user_id=uuid4(), organization_id=uuid4()):
        """Create MindsService instance with mocked session."""
        # Mock the datasource validation by patching the validation method
        service = MindsService(
            session=mock_session,
            mindsdb_client=mock_mindsdb_client,
            user_id=user_id,
            organization_id=organization_id,
        )
        service._validate_datasources = AsyncMock()
        service._add_datasources_to_mind = AsyncMock()
        return service

    @pytest.fixture
    def sample_mind(self, user_id=uuid4(), organization_id=uuid4()):
        """Sample Mind instance for testing."""
        mind = Mind(
            name="test-mind",
            provider="openai",
            model_name="gpt-4o",
            user_id=user_id,
            organization_id=organization_id,
            parameters={"temperature": 0.7},
            deleted_at=None,
            created_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
        )
        # Mock the mind_datasources relationship as an empty list
        mind.mind_datasources = []
        return mind

    @pytest.fixture
    def create_request(self):
        """Sample mind creation request."""
        return MindCreateRequest(
            name="new-mind",
            provider="openai",
            model_name="gpt-4o",
            parameters={"temperature": 0.8},
            datasources=[
                DatasourceConfig(name="datasource1", tables=["table1"]),
                DatasourceConfig(name="datasource2", tables=None),
            ],
        )

    @pytest.fixture
    def update_request(self):
        """Sample mind update request."""
        return MindUpdateRequest(name="updated-mind", parameters={"temperature": 0.9})

    @pytest.fixture
    def mock_data_catalog_loader(self):
        """Mock DataCatalogLoader instance."""
        loader = Mock()
        loader.load = AsyncMock()
        return loader

    def test_service_initialization(self, mock_session, mock_mindsdb_client):
        """Test MindsService initialization."""
        service = MindsService(
            session=mock_session,
            mindsdb_client=mock_mindsdb_client,
            user_id="user-123",
            organization_id="organization-123",
        )

        assert service.session == mock_session
        assert service.mindsdb_client == mock_mindsdb_client
        assert service.user_id == "user-123"

    @pytest.mark.asyncio
    async def test_list_minds_success(self, minds_service, mock_session, sample_mind, mock_conversations_service):
        """Test successful minds listing."""
        mock_session.exec.return_value.all.return_value = [sample_mind]

        result = await minds_service.list_minds(mock_conversations_service, limit=10, offset=0)

        assert len(result) == 1
        assert result[0].name == "test-mind"
        assert result[0].provider == "openai"
        mock_session.exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_minds_with_filters(self, minds_service, mock_session, sample_mind, mock_conversations_service):
        """Test minds listing with filters."""
        mock_session.exec.return_value.all.return_value = [sample_mind]

        result = await minds_service.list_minds(
            mock_conversations_service,
            provider="openai",
            include_deleted=False,
            limit=5,
            offset=10,
            with_detailed_data=False,
        )

        assert len(result) == 1
        mock_session.exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_minds_empty_result(self, minds_service, mock_session, mock_conversations_service):
        """Test minds listing with no results."""
        mock_session.exec.return_value.all.return_value = []

        result = await minds_service.list_minds(mock_conversations_service)

        assert result == []
        mock_session.exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_minds_database_error(self, minds_service, mock_session, mock_conversations_service):
        """Test minds listing with database error."""
        mock_session.exec.side_effect = Exception("Database error")

        with pytest.raises(MindsServiceError) as exc_info:
            await minds_service.list_minds(mock_conversations_service)

        assert "Failed to list minds" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_mind_success(self, minds_service, mock_session, sample_mind, mock_conversations_service):
        """Test successful mind retrieval."""
        mock_session.exec.return_value.first.return_value = sample_mind

        result = await minds_service.get_mind("test-mind", mock_conversations_service)

        assert result.name == "test-mind"
        assert result.provider == "openai"
        mock_session.exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_mind_not_found(self, minds_service, mock_session, mock_conversations_service):
        """Test mind retrieval when mind doesn't exist."""
        mock_session.exec.return_value.first.return_value = None

        with pytest.raises(MindNotFoundError) as exc_info:
            await minds_service.get_mind("nonexistent-mind", mock_conversations_service)

        assert "Mind 'nonexistent-mind' not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_mind_database_error(self, minds_service, mock_session, mock_conversations_service):
        """Test mind retrieval with database error."""
        mock_session.exec.side_effect = Exception("Database error")

        with pytest.raises(MindsServiceError) as exc_info:
            await minds_service.get_mind("test-mind", mock_conversations_service)

        assert "Failed to get mind" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_mind_success(
        self, minds_service, mock_session, create_request, sample_mind, mock_data_catalog_loader
    ):
        """Test successful mind creation."""
        # Mock: No existing mind with same name
        mock_session.exec.return_value.first.return_value = None

        # Mock the created mind - use a valid UUID format
        def mock_refresh(mind):
            mind.id = str(uuid4())
            mind.created_at = datetime.now(timezone.utc)
            mind.modified_at = datetime.now(timezone.utc)

        mock_session.refresh.side_effect = mock_refresh

        result = await minds_service.create_mind(create_request, mock_data_catalog_loader)

        assert result.name == "new-mind"
        assert result.provider == "openai"
        mock_session.add.assert_called_once()
        # Commit is called once for mind creation (datasource addition is mocked)
        assert mock_session.commit.call_count == 1

    @pytest.mark.asyncio
    async def test_create_mind_already_exists(
        self, minds_service, mock_session, create_request, sample_mind, mock_data_catalog_loader
    ):
        """Test mind creation when mind already exists."""
        # Mock: Existing mind with same name
        mock_session.exec.return_value.first.return_value = sample_mind

        with pytest.raises(MindAlreadyExistsError) as exc_info:
            await minds_service.create_mind(create_request, mock_data_catalog_loader)

        assert "Mind 'new-mind' already exists" in str(exc_info.value)
        mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_mind_database_error(
        self, minds_service, mock_session, create_request, mock_data_catalog_loader
    ):
        """Test mind creation with database error."""
        # Mock: No existing mind
        mock_session.exec.return_value.first.return_value = None
        # Mock: Database error on commit
        mock_session.commit.side_effect = Exception("Database error")

        with pytest.raises(MindsServiceError) as exc_info:
            await minds_service.create_mind(create_request, mock_data_catalog_loader)

        assert "Failed to create mind" in str(exc_info.value)
        mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_mind_success(
        self, minds_service, mock_session, update_request, sample_mind, mock_data_catalog_loader
    ):
        """Test successful mind update."""
        # Mock: First call finds the original mind, second call finds no conflict
        mock_session.exec.return_value.first.side_effect = [
            sample_mind,  # Original mind found
            None,  # No name conflict
        ]

        result = await minds_service.update_mind("test-mind", update_request, mock_data_catalog_loader)

        assert result.name == "updated-mind"  # Should be updated
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_mind_not_found(self, minds_service, mock_session, update_request, mock_data_catalog_loader):
        """Test mind update when mind doesn't exist."""
        mock_session.exec.return_value.first.return_value = None

        with pytest.raises(MindNotFoundError) as exc_info:
            await minds_service.update_mind("nonexistent-mind", update_request, mock_data_catalog_loader)

        assert "Mind 'nonexistent-mind' not found" in str(exc_info.value)
        mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_mind_name_conflict(self, minds_service, mock_session, sample_mind, mock_data_catalog_loader):
        """Test mind update with name conflict."""
        # Mock: Original mind exists
        mock_session.exec.return_value.first.side_effect = [
            sample_mind,  # Original mind
            sample_mind,  # Conflicting mind with new name
        ]

        update_request = MindUpdateRequest(name="conflicting-name")

        with pytest.raises(MindAlreadyExistsError) as exc_info:
            await minds_service.update_mind("test-mind", update_request, mock_data_catalog_loader)

        assert "Mind with name 'conflicting-name' already exists" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_delete_mind_success(self, minds_service, mock_session, sample_mind):
        """Test successful mind deletion (soft delete)."""
        mock_session.exec.return_value.first.return_value = sample_mind

        result = await minds_service.delete_mind("test-mind")

        assert result is True
        assert sample_mind.deleted_at is not None  # Should be soft deleted
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_mind_not_found(self, minds_service, mock_session):
        """Test mind deletion when mind doesn't exist."""
        mock_session.exec.return_value.first.return_value = None

        with pytest.raises(MindNotFoundError) as exc_info:
            await minds_service.delete_mind("nonexistent-mind")

        assert "Mind 'nonexistent-mind' not found" in str(exc_info.value)
        # Note: The service doesn't call rollback for MindNotFoundError - it's raised directly

    @pytest.mark.asyncio
    async def test_delete_mind_database_error(self, minds_service, mock_session, sample_mind):
        """Test mind deletion with database error."""
        mock_session.exec.return_value.first.return_value = sample_mind
        mock_session.commit.side_effect = Exception("Database error")

        with pytest.raises(MindsServiceError) as exc_info:
            await minds_service.delete_mind("test-mind")

        assert "Failed to delete mind" in str(exc_info.value)
        mock_session.rollback.assert_called_once()

    async def test_mind_to_response_conversion(self, minds_service, sample_mind):
        """Test conversion of Mind model to MindResponse."""
        result = await minds_service._mind_to_response(sample_mind)

        assert result.name == "test-mind"
        assert result.provider == "openai"
        assert result.model_name == "gpt-4o"
        assert result.parameters == {"temperature": 0.7}
        assert result.datasources == []

    @pytest.mark.asyncio
    async def test_check_mind_exists_success(self, minds_service, sample_mind):
        """Test successful mind existence check."""
        minds_service._get_mind = AsyncMock(return_value=sample_mind)

        result = await minds_service.check_mind_exists("test-mind")

        assert result is None
        minds_service._get_mind.assert_called_once_with("test-mind")

    @pytest.mark.asyncio
    async def test_check_mind_exists_not_found(self, minds_service):
        """Test check mind exists when mind doesn't exist."""
        minds_service._get_mind = AsyncMock(return_value=None)

        with pytest.raises(MindNotFoundError, match="Mind 'test-mind' not found"):
            await minds_service.check_mind_exists("test-mind")

    @pytest.mark.asyncio
    async def test_check_mind_exists_lowercase(self, minds_service, sample_mind):
        """Test that mind name is converted to lowercase."""
        minds_service._get_mind = AsyncMock(return_value=sample_mind)

        result = await minds_service.check_mind_exists("TEST-MIND")

        assert result is None
        # Should be called with lowercase name
        minds_service._get_mind.assert_called_once_with("test-mind")

    @pytest.mark.asyncio
    async def test_check_mind_exists_database_error(self, minds_service):
        """Test check mind exists with database error."""
        minds_service._get_mind = AsyncMock(side_effect=Exception("Database error"))

        with pytest.raises(MindsServiceError, match="Failed to check mind existence"):
            await minds_service.check_mind_exists("test-mind")

    # -- count_minds tests --

    @pytest.mark.asyncio
    async def test_count_minds_success(self, minds_service, mock_session):
        """Test successful mind count."""
        mock_session.exec.return_value.one.return_value = 5

        result = await minds_service.count_minds()

        assert result == 5
        mock_session.exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_count_minds_with_is_sample_false(self, minds_service, mock_session):
        """Test count minds excluding sample minds."""
        mock_session.exec.return_value.one.return_value = 3

        result = await minds_service.count_minds(is_sample=False)

        assert result == 3
        mock_session.exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_count_minds_with_is_sample_true(self, minds_service, mock_session):
        """Test count minds including only sample minds."""
        mock_session.exec.return_value.one.return_value = 2

        result = await minds_service.count_minds(is_sample=True)

        assert result == 2
        mock_session.exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_count_minds_returns_zero(self, minds_service, mock_session):
        """Test count minds when there are none."""
        mock_session.exec.return_value.one.return_value = 0

        result = await minds_service.count_minds()

        assert result == 0

    @pytest.mark.asyncio
    async def test_count_minds_database_error(self, minds_service, mock_session):
        """Test count minds with database error."""
        mock_session.exec.side_effect = Exception("Database error")

        with pytest.raises(MindsServiceError, match="Failed to count minds"):
            await minds_service.count_minds()
