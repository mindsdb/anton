"""
Unit tests for Minds API endpoints.

Tests the FastAPI endpoints for minds management including:
- Request/response handling
- Error responses
- Authentication and authorization
- Input validation
"""

from unittest.mock import ANY, AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException

from minds.api.v1.endpoints.minds import (
    check_mind_exists,
    create_mind,
    delete_mind,
    get_mind,
    get_minds_service,
    list_minds,
    update_mind,
)
from minds.model.mind_datasource import DataCatalogStatus, DetailedDataCatalogStatus
from minds.schemas.minds import DatasourceConfig, MindCreateRequest, MindResponse, MindUpdateRequest
from minds.services.minds import MindAlreadyExistsError, MindNotFoundError, MindsService, MindsServiceError


class TestMindsAPI:
    """Test suite for Minds API endpoints."""

    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI request object."""
        request = Mock()
        request.headers = {"x-user-id": "test-user-123", "x-company-id": "test-company-456"}
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
    def mock_minds_service(self):
        """Mock MindsService instance."""
        service = Mock(spec=MindsService)
        service.user_id = "test-user-123"
        service.tenant_id = "test-tenant-456"
        return service

    @pytest.fixture
    def mock_data_catalog_loader(self):
        """Mock DataCatalogLoader instance."""
        loader = Mock()
        loader.load = AsyncMock()
        return loader

    @pytest.fixture
    def sample_mind_response(self):
        """Sample MindResponse for testing."""
        return MindResponse(
            name="test-mind",
            model_name="gpt-4o",
            provider="openai",
            parameters={"temperature": 0.7},
            datasources=[
                DatasourceConfig(
                    name="test-datasource",
                    tables=["test-table"],
                    status=DetailedDataCatalogStatus(tasks=[], progress=0.0, overall_status=DataCatalogStatus.PENDING),
                ),
            ],
            created_at="2024-01-01T00:00:00Z",
            modified_at="2024-01-01T00:00:00Z",
        )

    @pytest.fixture
    def create_request_data(self):
        """Sample mind creation request data."""
        return {
            "name": "new-mind",
            "provider": "openai",
            "model_name": "gpt-4o",
            "parameters": {"temperature": 0.8},
            "datasources": [
                DatasourceConfig(
                    name="datasource1",
                    tables=["table1"],
                    status=DetailedDataCatalogStatus(tasks=[], progress=0.0, overall_status=DataCatalogStatus.PENDING),
                )
            ],
        }

    @pytest.fixture
    def update_request_data(self):
        """Sample mind update request data."""
        return {"name": "updated-mind", "parameters": {"temperature": 0.9}}

    def test_get_minds_service_dependency(self, mock_request, mock_session):
        """Test the get_minds_service dependency function."""
        with (
            patch("minds.api.v1.endpoints.minds.extract_context_from_request") as mock_extract,
            patch("minds.api.v1.endpoints.minds.create_mindsdb_client_from_request") as mock_create_client,
        ):
            mock_extract.return_value.user_id = "test-user-123"
            mock_mindsdb_client = Mock()
            mock_create_client.return_value = mock_mindsdb_client

            service = get_minds_service(mock_request, mock_session)

            assert isinstance(service, MindsService)
            assert service.session == mock_session
            assert service.user_id == "test-user-123"
            assert service.mindsdb_client == mock_mindsdb_client

    @pytest.mark.asyncio
    async def test_list_minds_success(self, mock_minds_service, sample_mind_response):
        """Test successful minds listing."""
        mock_minds_service.list_minds = AsyncMock(return_value=[sample_mind_response])

        result = await list_minds(
            minds_service=mock_minds_service,
            provider="openai",
            include_deleted=False,
            limit=10,
            offset=0,
            with_detailed_data=False,
            include_total=False,
            sort_by=None,
            sort_order="desc",
        )

        assert len(result) == 1
        assert result[0].name == "test-mind"
        mock_minds_service.list_minds.assert_called_once_with(
            conversations_service=ANY,
            name=ANY,
            provider="openai",
            is_demo=ANY,
            include_deleted=False,
            limit=10,
            offset=0,
            with_detailed_data=False,
            include_total=False,
            sort_by=None,
            sort_order="desc",
        )

    @pytest.mark.asyncio
    async def test_list_minds_empty(self, mock_minds_service):
        """Test minds listing with empty result."""
        mock_minds_service.list_minds = AsyncMock(return_value=[])

        result = await list_minds(
            minds_service=mock_minds_service,
            include_total=False,
            sort_by=None,
            sort_order="desc",
        )

        assert result == []
        mock_minds_service.list_minds.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_minds_service_error(self, mock_minds_service):
        """Test minds listing with service error."""
        mock_minds_service.list_minds = AsyncMock(side_effect=MindsServiceError("Service error"))

        with pytest.raises(HTTPException) as exc_info:
            await list_minds(
                minds_service=mock_minds_service,
                include_total=False,
                sort_by=None,
                sort_order="desc",
            )

        assert exc_info.value.status_code == 400
        assert "Service error" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_list_minds_unexpected_error(self, mock_minds_service):
        """Test minds listing with unexpected error."""
        mock_minds_service.list_minds = AsyncMock(side_effect=Exception("Unexpected error"))

        with pytest.raises(HTTPException) as exc_info:
            await list_minds(
                minds_service=mock_minds_service,
                include_total=False,
                sort_by=None,
                sort_order="desc",
            )

        assert exc_info.value.status_code == 500
        assert "Internal server error" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_mind_success(self, mock_minds_service, sample_mind_response):
        """Test successful mind retrieval."""
        mock_minds_service.get_mind = AsyncMock(return_value=sample_mind_response)

        result = await get_mind(mind_name="test-mind", minds_service=mock_minds_service, with_detailed_data=True)

        assert result.name == "test-mind"
        mock_minds_service.get_mind.assert_called_once_with(
            mind_name="test-mind",
            conversations_service=ANY,
            with_detailed_data=True,
        )

    @pytest.mark.asyncio
    async def test_get_mind_not_found(self, mock_minds_service):
        """Test mind retrieval when mind doesn't exist."""
        mock_minds_service.get_mind = AsyncMock(side_effect=MindNotFoundError("Mind not found"))

        with pytest.raises(HTTPException) as exc_info:
            await get_mind(mind_name="nonexistent", minds_service=mock_minds_service)

        assert exc_info.value.status_code == 404
        assert "Mind not found" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_check_mind_exists_success(self, mock_minds_service):
        """Test successful mind existence check."""
        mock_minds_service.check_mind_exists = AsyncMock(return_value=None)

        result = await check_mind_exists(mind_name="test-mind", minds_service=mock_minds_service)

        assert result is None
        mock_minds_service.check_mind_exists.assert_called_once_with(mind_name="test-mind")

    @pytest.mark.asyncio
    async def test_check_mind_exists_not_found(self, mock_minds_service):
        """Test mind existence check when mind doesn't exist."""
        mock_minds_service.check_mind_exists = AsyncMock(side_effect=MindNotFoundError("Mind not found"))

        with pytest.raises(HTTPException) as exc_info:
            await check_mind_exists(mind_name="nonexistent", minds_service=mock_minds_service)

        assert exc_info.value.status_code == 404
        assert "Mind not found" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_check_mind_exists_service_error(self, mock_minds_service):
        """Test mind existence check with service error."""
        mock_minds_service.check_mind_exists = AsyncMock(side_effect=MindsServiceError("Database error"))

        with pytest.raises(HTTPException) as exc_info:
            await check_mind_exists(mind_name="test-mind", minds_service=mock_minds_service)

        assert exc_info.value.status_code == 500
        assert "Database error" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_check_mind_exists_unexpected_error(self, mock_minds_service):
        """Test mind existence check with unexpected error."""
        mock_minds_service.check_mind_exists = AsyncMock(side_effect=Exception("Unexpected error"))

        with pytest.raises(HTTPException) as exc_info:
            await check_mind_exists(mind_name="test-mind", minds_service=mock_minds_service)

        assert exc_info.value.status_code == 500
        assert "Internal server error" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_create_mind_success(
        self, mock_minds_service, mock_data_catalog_loader, create_request_data, sample_mind_response
    ):
        """Test successful mind creation."""
        mock_minds_service.create_mind = AsyncMock(return_value=sample_mind_response)
        request = MindCreateRequest(**create_request_data)

        result = await create_mind(
            mind_data=request, minds_service=mock_minds_service, data_catalog_loader=mock_data_catalog_loader
        )

        assert result.name == "test-mind"
        mock_minds_service.create_mind.assert_called_once_with(request, mock_data_catalog_loader)

    @pytest.mark.asyncio
    async def test_create_mind_already_exists(self, mock_minds_service, mock_data_catalog_loader, create_request_data):
        """Test mind creation when mind already exists."""
        mock_minds_service.create_mind = AsyncMock(side_effect=MindAlreadyExistsError("Mind already exists"))
        request = MindCreateRequest(**create_request_data)

        with pytest.raises(HTTPException) as exc_info:
            await create_mind(
                mind_data=request, minds_service=mock_minds_service, data_catalog_loader=mock_data_catalog_loader
            )

        assert exc_info.value.status_code == 409
        assert "Mind already exists" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_update_mind_success(
        self, mock_minds_service, mock_data_catalog_loader, update_request_data, sample_mind_response
    ):
        """Test successful mind update."""
        mock_minds_service.update_mind = AsyncMock(return_value=sample_mind_response)
        request = MindUpdateRequest(**update_request_data)

        result = await update_mind(
            mind_name="test-mind",
            mind_data=request,
            minds_service=mock_minds_service,
            data_catalog_loader=mock_data_catalog_loader,
        )

        assert result.name == "test-mind"
        mock_minds_service.update_mind.assert_called_once_with("test-mind", request, mock_data_catalog_loader)

    @pytest.mark.asyncio
    async def test_update_mind_not_found(self, mock_minds_service, mock_data_catalog_loader, update_request_data):
        """Test mind update when mind doesn't exist."""
        mock_minds_service.update_mind = AsyncMock(side_effect=MindNotFoundError("Mind not found"))
        request = MindUpdateRequest(**update_request_data)

        with pytest.raises(HTTPException) as exc_info:
            await update_mind(
                mind_name="nonexistent",
                mind_data=request,
                minds_service=mock_minds_service,
                data_catalog_loader=mock_data_catalog_loader,
            )

        assert exc_info.value.status_code == 404
        assert "Mind not found" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_delete_mind_success(self, mock_minds_service):
        """Test successful mind deletion."""
        mock_minds_service.delete_mind = AsyncMock(return_value=True)

        # Should return None for 204 No Content
        result = await delete_mind(mind_name="test-mind", minds_service=mock_minds_service)

        assert result is None
        mock_minds_service.delete_mind.assert_called_once_with("test-mind")

    @pytest.mark.asyncio
    async def test_delete_mind_not_found(self, mock_minds_service):
        """Test mind deletion when mind doesn't exist."""
        mock_minds_service.delete_mind = AsyncMock(side_effect=MindNotFoundError("Mind not found"))

        with pytest.raises(HTTPException) as exc_info:
            await delete_mind(mind_name="nonexistent", minds_service=mock_minds_service)

        assert exc_info.value.status_code == 404
        assert "Mind not found" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_api_error_handling_chain(self, mock_minds_service):
        """Test that all exception types are properly handled."""
        # Test MindsServiceError -> 400
        mock_minds_service.list_minds = AsyncMock(side_effect=MindsServiceError("Service error"))
        with pytest.raises(HTTPException) as exc_info:
            await list_minds(
                minds_service=mock_minds_service,
                include_total=False,
                sort_by=None,
                sort_order="desc",
            )
        assert exc_info.value.status_code == 400

        # Test MindNotFoundError -> 404
        mock_minds_service.get_mind = AsyncMock(side_effect=MindNotFoundError("Not found"))
        with pytest.raises(HTTPException) as exc_info:
            await get_mind(mind_name="test", minds_service=mock_minds_service)
        assert exc_info.value.status_code == 404

        # Test MindAlreadyExistsError -> 409
        mock_minds_service.create_mind = AsyncMock(side_effect=MindAlreadyExistsError("Already exists"))
        mock_data_catalog_loader = Mock()
        mock_data_catalog_loader.load = AsyncMock()
        with pytest.raises(HTTPException) as exc_info:
            await create_mind(
                mind_data=MindCreateRequest(
                    name="test",
                    provider="openai",
                    datasources=[],
                ),
                minds_service=mock_minds_service,
                data_catalog_loader=mock_data_catalog_loader,
            )
        assert exc_info.value.status_code == 409

        # Test unexpected Exception -> 500
        mock_minds_service.list_minds = AsyncMock(side_effect=Exception("Unexpected"))
        with pytest.raises(HTTPException) as exc_info:
            await list_minds(
                minds_service=mock_minds_service,
                include_total=False,
                sort_by=None,
                sort_order="desc",
            )
        assert exc_info.value.status_code == 500


class TestMindsAPIErrorHandling:
    """Test error handling paths that aren't covered by main test class."""

    @pytest.mark.asyncio
    async def test_get_mind_service_error(self):
        """Test get_mind with MindsServiceError (lines 112-114)."""
        mock_service = Mock(spec=MindsService)
        mock_service.user_id = "test-user"
        mock_service.tenant_id = "test-tenant"
        mock_service.get_mind = AsyncMock(side_effect=MindsServiceError("Database connection failed"))

        with pytest.raises(HTTPException) as exc_info:
            await get_mind(mind_name="test", minds_service=mock_service)

        assert exc_info.value.status_code == 400
        assert "Database connection failed" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_mind_unexpected_error(self):
        """Test get_mind with unexpected Exception (lines 115-117)."""
        mock_service = Mock(spec=MindsService)
        mock_service.user_id = "test-user"
        mock_service.tenant_id = "test-tenant"
        mock_service.get_mind = AsyncMock(side_effect=ValueError("Unexpected error"))

        with pytest.raises(HTTPException) as exc_info:
            await get_mind(mind_name="test", minds_service=mock_service)

        assert exc_info.value.status_code == 500
        assert "Internal server error" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_create_mind_service_error(self):
        """Test create_mind with MindsServiceError (lines 147-149)."""
        mock_service = Mock(spec=MindsService)
        mock_service.user_id = "test-user"
        mock_service.tenant_id = "test-tenant"
        mock_service.create_mind = AsyncMock(side_effect=MindsServiceError("Validation failed"))

        request = MindCreateRequest(name="test", provider="openai", datasources=[])
        mock_data_catalog_loader = Mock()
        mock_data_catalog_loader.load = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await create_mind(
                mind_data=request, minds_service=mock_service, data_catalog_loader=mock_data_catalog_loader
            )

        assert exc_info.value.status_code == 400
        assert "Validation failed" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_create_mind_unexpected_error(self):
        """Test create_mind with unexpected Exception (lines 150-152)."""
        mock_service = Mock(spec=MindsService)
        mock_service.user_id = "test-user"
        mock_service.tenant_id = "test-tenant"
        mock_service.create_mind = AsyncMock(side_effect=RuntimeError("Database error"))

        request = MindCreateRequest(name="test", provider="openai", datasources=[])
        mock_data_catalog_loader = Mock()
        mock_data_catalog_loader.load = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await create_mind(
                mind_data=request, minds_service=mock_service, data_catalog_loader=mock_data_catalog_loader
            )

        assert exc_info.value.status_code == 500
        assert "Internal server error" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_update_mind_service_error(self):
        """Test update_mind with MindsServiceError (lines 184-186)."""
        mock_service = Mock(spec=MindsService)
        mock_service.user_id = "test-user"
        mock_service.tenant_id = "test-tenant"
        mock_service.update_mind = AsyncMock(side_effect=MindsServiceError("Invalid parameters"))

        request = MindUpdateRequest(name="updated-test")
        mock_data_catalog_loader = Mock()
        mock_data_catalog_loader.load = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await update_mind(
                mind_name="test",
                mind_data=request,
                minds_service=mock_service,
                data_catalog_loader=mock_data_catalog_loader,
            )

        assert exc_info.value.status_code == 400
        assert "Invalid parameters" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_update_mind_unexpected_error(self):
        """Test update_mind with unexpected Exception (lines 187-189)."""
        mock_service = Mock(spec=MindsService)
        mock_service.user_id = "test-user"
        mock_service.tenant_id = "test-tenant"
        mock_service.update_mind = AsyncMock(side_effect=KeyError("Missing key"))

        request = MindUpdateRequest(name="updated-test")
        mock_data_catalog_loader = Mock()
        mock_data_catalog_loader.load = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await update_mind(
                mind_name="test",
                mind_data=request,
                minds_service=mock_service,
                data_catalog_loader=mock_data_catalog_loader,
            )

        assert exc_info.value.status_code == 500
        assert "Internal server error" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_delete_mind_service_error(self):
        """Test delete_mind with MindsServiceError (lines 219-221)."""
        mock_service = Mock(spec=MindsService)
        mock_service.user_id = "test-user"
        mock_service.tenant_id = "test-tenant"
        mock_service.delete_mind = AsyncMock(side_effect=MindsServiceError("Cannot delete mind"))

        with pytest.raises(HTTPException) as exc_info:
            await delete_mind(mind_name="test", minds_service=mock_service)

        assert exc_info.value.status_code == 400
        assert "Cannot delete mind" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_delete_mind_unexpected_error(self):
        """Test delete_mind with unexpected Exception (lines 222-224)."""
        mock_service = Mock(spec=MindsService)
        mock_service.user_id = "test-user"
        mock_service.tenant_id = "test-tenant"
        mock_service.delete_mind = AsyncMock(side_effect=OSError("File system error"))

        with pytest.raises(HTTPException) as exc_info:
            await delete_mind(mind_name="test", minds_service=mock_service)

        assert exc_info.value.status_code == 500
        assert "Internal server error" in exc_info.value.detail


class TestMindsAPIValidation:
    """Test input validation for minds API endpoints."""

    def test_mind_create_request_validation(self):
        """Test MindCreateRequest validation."""
        # Valid request
        valid_data = {"name": "test-mind", "provider": "openai", "model_name": "gpt-4o", "datasources": []}
        request = MindCreateRequest(**valid_data)
        assert request.name == "test-mind"
        assert request.provider == "openai"

        # Valid request - provider has default value, so it's not required
        request_minimal = MindCreateRequest(name="test", datasources=[])
        assert request_minimal.name == "test"
        assert request_minimal.provider == "openai"  # Default value

    def test_mind_update_request_validation(self):
        """Test MindUpdateRequest validation."""
        # All fields optional for updates
        request = MindUpdateRequest()
        assert request.name is None
        assert request.provider is None

        # Partial update
        request = MindUpdateRequest(name="new-name")
        assert request.name == "new-name"
        assert request.provider is None
