"""
Unit tests for datasources API endpoints.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from minds.api.v1.endpoints.datasources import (
    check_datasource_connection,
    create_datasource,
    delete_datasource,
    get_datasource,
    get_datasources_service,
    list_datasources,
    update_datasource,
)
from minds.schemas.datasources import (
    DatasourceConnectionStatus,
    DatasourceCreateRequest,
    DatasourceResponse,
    DatasourceUpdateRequest,
)
from minds.services.datasources import (
    DatasourceAlreadyExistsError,
    DatasourceNotFoundError,
    DatasourceServiceError,
    DatasourcesService,
)


class TestDatasourcesAPI:
    """Test datasources API endpoints."""

    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI request object."""
        request = Mock()
        request.headers = {
            "x-user-id": "test-user-123", 
            "x-company-id": "test-company-456"
        }
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
    def mock_datasources_service(self):
        """Mock DatasourcesService instance."""
        service = Mock(spec=DatasourcesService)
        service.user_id = "test-user-123"
        service.company_id = "test-company-456"
        return service

    @pytest.fixture
    def sample_datasource_response(self):
        """Sample DatasourceResponse for testing."""
        return DatasourceResponse(
            id="test-id-123",
            name="test_postgres",
            engine="postgres",
            connection_data={"host": "localhost", "port": 5432},
            created_at="2023-01-01T12:00:00",
            is_demo=False
        )

    @pytest.fixture
    def sample_connection_status(self):
        """Sample connection status for testing."""
        return DatasourceConnectionStatus(success=True, error_message=None)

    def test_get_datasources_service_dependency(self, mock_request, mock_session, mock_mindsdb_client):
        """Test the get_datasources_service dependency function."""
        with patch('minds.api.v1.endpoints.datasources.extract_context_from_request') as mock_extract, \
             patch('minds.api.v1.endpoints.datasources.create_mindsdb_client_from_request') as mock_create_client:
            
            mock_extract.return_value.user_id = "test-user-123"
            mock_extract.return_value.company_id = "test-company-456"
            mock_create_client.return_value = mock_mindsdb_client
            
            service = get_datasources_service(mock_request, mock_session)
            
            assert isinstance(service, DatasourcesService)
            assert service.session == mock_session
            assert service.user_id == "test-user-123"
            assert service.company_id == "test-company-456"

    @pytest.mark.asyncio
    async def test_list_datasources_success(self, mock_datasources_service, sample_datasource_response):
        """Test successful datasources listing."""
        mock_datasources_service.list_datasources = AsyncMock(return_value=[sample_datasource_response])
        
        result = await list_datasources(
            engine=None, limit=100, offset=0, with_detailed_data=False,
            datasources_service=mock_datasources_service
        )
        
        assert len(result) == 1
        assert result[0].name == "test_postgres"
        mock_datasources_service.list_datasources.assert_called_once_with(
            engine=None, limit=100, offset=0, with_detailed_data=False
        )

    @pytest.mark.asyncio
    async def test_list_datasources_empty(self, mock_datasources_service):
        """Test listing empty datasources."""
        mock_datasources_service.list_datasources = AsyncMock(return_value=[])
        
        result = await list_datasources(datasources_service=mock_datasources_service)

        assert result == []

    @pytest.mark.asyncio
    async def test_list_datasources_with_filters(self, mock_datasources_service, sample_datasource_response):
        """Test listing datasources with filters."""
        mock_datasources_service.list_datasources = AsyncMock(return_value=[sample_datasource_response])
        
        result = await list_datasources(
            engine="postgres", limit=10, offset=5, with_detailed_data=True,
            datasources_service=mock_datasources_service
        )
        
        assert len(result) == 1
        mock_datasources_service.list_datasources.assert_called_once_with(
            engine="postgres", limit=10, offset=5, with_detailed_data=True
        )

    @pytest.mark.asyncio
    async def test_list_datasources_service_error(self, mock_datasources_service):
        """Test list datasources with service error."""
        from fastapi import HTTPException
        
        mock_datasources_service.list_datasources = AsyncMock(
            side_effect=DatasourceServiceError("Service error")
        )
        
        with pytest.raises(HTTPException) as exc_info:
            await list_datasources(datasources_service=mock_datasources_service)
        
        assert exc_info.value.status_code == 400
        assert "Service error" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_datasource_success(self, mock_datasources_service, sample_datasource_response):
        """Test successful datasource retrieval."""
        mock_datasources_service.get_datasource = AsyncMock(return_value=sample_datasource_response)
        
        result = await get_datasource(
            datasource_name="test_postgres",
            with_detailed_data=False,
            datasources_service=mock_datasources_service
        )
        
        assert result.name == "test_postgres"
        mock_datasources_service.get_datasource.assert_called_once_with(
                datasource_name="test_postgres", with_detailed_data=False)

    @pytest.mark.asyncio
    async def test_get_datasource_not_found(self, mock_datasources_service):
        """Test get datasource when not found."""
        from fastapi import HTTPException
        
        mock_datasources_service.get_datasource = AsyncMock(
            side_effect=DatasourceNotFoundError("Datasource not found")
        )
        
        with pytest.raises(HTTPException) as exc_info:
            await get_datasource(
                datasource_name="nonexistent",
                datasources_service=mock_datasources_service
            )
        
        assert exc_info.value.status_code == 404
        assert "Datasource not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_create_datasource_success(self, mock_datasources_service, sample_datasource_response):
        """Test successful datasource creation."""
        mock_datasources_service.create_datasource = AsyncMock(return_value=sample_datasource_response)
        
        create_request = DatasourceCreateRequest(
            name="test_postgres",
            engine="postgres",
            connection_data={"host": "localhost", "port": 5432}
        )
        
        result = await create_datasource(
            datasource_data=create_request,
            datasources_service=mock_datasources_service
        )
        
        assert result.name == "test_postgres"
        mock_datasources_service.create_datasource.assert_called_once_with(create_request)

    @pytest.mark.asyncio
    async def test_create_datasource_already_exists(self, mock_datasources_service):
        """Test create datasource when it already exists."""
        from fastapi import HTTPException
        
        mock_datasources_service.create_datasource = AsyncMock(
            side_effect=DatasourceAlreadyExistsError("Already exists")
        )
        
        create_request = DatasourceCreateRequest(
            name="test_postgres",
            engine="postgres",
            connection_data={"host": "localhost", "port": 5432}
        )
        
        with pytest.raises(HTTPException) as exc_info:
            await create_datasource(
                datasource_data=create_request,
                datasources_service=mock_datasources_service
            )
        
        assert exc_info.value.status_code == 409
        assert "Already exists" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_update_datasource_success(self, mock_datasources_service, sample_datasource_response):
        """Test successful datasource update."""
        mock_datasources_service.update_datasource = AsyncMock(return_value=sample_datasource_response)
        
        update_request = DatasourceUpdateRequest(
            connection_data={"host": "newhost", "port": 5432}
        )
        
        result = await update_datasource(
            datasource_name="test_postgres",
            datasource_data=update_request,
            datasources_service=mock_datasources_service
        )
        
        assert result.name == "test_postgres"
        mock_datasources_service.update_datasource.assert_called_once_with("test_postgres", update_request)

    @pytest.mark.asyncio
    async def test_update_datasource_not_found(self, mock_datasources_service):
        """Test update datasource when not found."""
        from fastapi import HTTPException
        
        mock_datasources_service.update_datasource = AsyncMock(
            side_effect=DatasourceNotFoundError("Datasource not found")
        )
        
        update_request = DatasourceUpdateRequest(connection_data={"host": "newhost"})
        
        with pytest.raises(HTTPException) as exc_info:
            await update_datasource(
                datasource_name="nonexistent",
                datasource_data=update_request,
                datasources_service=mock_datasources_service
            )
        
        assert exc_info.value.status_code == 404
        assert "Datasource not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_delete_datasource_success(self, mock_datasources_service):
        """Test successful datasource deletion."""
        mock_datasources_service.delete_datasource = AsyncMock()
        
        result = await delete_datasource(
            datasource_name="test_postgres",
            cascade=False,
            datasources_service=mock_datasources_service
        )
        
        assert result is None
        mock_datasources_service.delete_datasource.assert_called_once_with("test_postgres", cascade=False)

    @pytest.mark.asyncio
    async def test_delete_datasource_not_found(self, mock_datasources_service):
        """Test delete datasource when not found."""
        from fastapi import HTTPException
        
        mock_datasources_service.delete_datasource = AsyncMock(
            side_effect=DatasourceNotFoundError("Datasource not found")
        )
        
        with pytest.raises(HTTPException) as exc_info:
            await delete_datasource(
                datasource_name="nonexistent",
                cascade=False,
                datasources_service=mock_datasources_service
            )
        
        assert exc_info.value.status_code == 404
        assert "Datasource not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_test_connection_success(self, mock_datasources_service, sample_connection_status):
        """Test successful connection testing."""
        mock_datasources_service.test_connection = AsyncMock(return_value=sample_connection_status)
        
        result = await check_datasource_connection(
            datasource_name="test_postgres",
            datasources_service=mock_datasources_service
        )
        
        assert result.success is True
        mock_datasources_service.test_connection.assert_called_once_with("test_postgres")

    @pytest.mark.asyncio
    async def test_test_connection_failure(self, mock_datasources_service):
        """Test connection testing failure."""
        failure_status = DatasourceConnectionStatus(success=False, error_message="Connection failed")
        mock_datasources_service.test_connection = AsyncMock(return_value=failure_status)
        
        result = await check_datasource_connection(
            datasource_name="test_postgres",
            datasources_service=mock_datasources_service
        )
        
        assert result.success is False
        assert "Connection failed" in result.error_message
