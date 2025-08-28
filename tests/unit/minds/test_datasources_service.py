"""
Unit tests for DatasourcesService.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from uuid import uuid4
from datetime import datetime

from sqlmodel import Session
from mindsdb_sdk.server import Server

from minds.services.datasources import (
    DatasourcesService, 
    DatasourceServiceError, 
    DatasourceNotFoundError, 
    DatasourceAlreadyExistsError
)
from minds.model.datasource import Datasource
from minds.schemas.datasources import (
    DatasourceCreateRequest, 
    DatasourceUpdateRequest,
    DatasourceResponse
)


class TestDatasourcesService:
    """Test cases for DatasourcesService."""

    @pytest.fixture
    def mock_session(self):
        """Mock database session."""
        session = Mock(spec=Session)
        session.add = Mock()
        session.commit = Mock()
        session.rollback = Mock()
        session.delete = Mock()
        session.refresh = Mock()
        return session

    @pytest.fixture
    def mock_mindsdb_client(self):
        """Mock MindsDB client."""
        client = Mock(spec=Server)
        client.databases = Mock()
        return client

    @pytest.fixture
    def service(self, mock_session, mock_mindsdb_client):
        """Create DatasourcesService instance."""
        return DatasourcesService(
            session=mock_session,
            user_id="test-user-123",
            company_id="test-company-456",
            mindsdb_client=mock_mindsdb_client
        )

    @pytest.fixture
    def sample_datasource(self):
        """Sample datasource for testing."""
        return Datasource(
            id=uuid4(),
            name="test_postgres",
            engine="postgres", 
            connection_data={"host": "localhost", "port": 5432, "user": "test"},
            user_id="test-user-123",
            company_id="test-company-456",
            created_on=datetime(2023, 1, 1, 12, 0, 0),
            modified_on=datetime(2023, 1, 1, 12, 0, 0)
        )

    @pytest.fixture  
    def sample_create_request(self):
        """Sample create request."""
        return DatasourceCreateRequest(
            name="test_postgres",
            engine="postgres",
            connection_data={"host": "localhost", "port": 5432, "user": "test"}
        )

    def test_service_initialization(self, mock_session, mock_mindsdb_client):
        """Test service initialization."""
        service = DatasourcesService(
            session=mock_session,
            user_id="test-user",
            company_id="test-company", 
            mindsdb_client=mock_mindsdb_client
        )
        
        assert service.session == mock_session
        assert service.user_id == "test-user"
        assert service.company_id == "test-company"
        assert service.mindsdb_client == mock_mindsdb_client

    def test_datasource_to_response(self, service, sample_datasource):
        """Test _datasource_to_response conversion."""
        result = service._datasource_to_response(sample_datasource)
        
        assert isinstance(result, DatasourceResponse)
        assert result.id == str(sample_datasource.id)
        assert result.name == "test_postgres"
        assert result.engine == "postgres"
        assert result.connection_data == {"host": "localhost", "port": 5432, "user": "test"}
        assert result.is_demo is False

    @pytest.mark.asyncio
    async def test_list_datasources_empty(self, service, mock_session):
        """Test listing datasources when none exist."""
        mock_result = Mock()
        mock_result.all.return_value = []
        mock_session.exec.return_value = mock_result
        
        result = await service.list_datasources()
        
        assert result == []
        mock_session.exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_datasources_success(self, service, mock_session, sample_datasource):
        """Test successful datasources listing."""
        mock_result = Mock()
        mock_result.all.return_value = [sample_datasource]
        mock_session.exec.return_value = mock_result
        
        result = await service.list_datasources()
        
        assert len(result) == 1
        assert isinstance(result[0], DatasourceResponse) 
        assert result[0].name == "test_postgres"
        assert result[0].engine == "postgres"

    @pytest.mark.asyncio
    async def test_list_datasources_with_filters(self, service, mock_session, sample_datasource):
        """Test listing datasources with filters."""
        mock_result = Mock()
        mock_result.all.return_value = [sample_datasource]
        mock_session.exec.return_value = mock_result
        
        result = await service.list_datasources(engine="postgres", limit=10, offset=5)
        
        assert len(result) == 1
        mock_session.exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_datasources_database_error(self, service, mock_session):
        """Test list datasources with database error.""" 
        mock_session.exec.side_effect = Exception("Database error")
        
        with pytest.raises(DatasourceServiceError, match="Failed to list datasources"):
            await service.list_datasources()

    @pytest.mark.asyncio
    async def test_get_datasource_success(self, service, mock_session, sample_datasource):
        """Test successful datasource retrieval."""
        mock_result = Mock()
        mock_result.first.return_value = sample_datasource
        mock_session.exec.return_value = mock_result
        
        result = await service.get_datasource("test_postgres")
        
        assert isinstance(result, DatasourceResponse)
        assert result.name == "test_postgres"

    @pytest.mark.asyncio
    async def test_get_datasource_not_found(self, service, mock_session):
        """Test get datasource when not found."""
        mock_result = Mock()
        mock_result.first.return_value = None
        mock_session.exec.return_value = mock_result
        
        with pytest.raises(DatasourceNotFoundError, match="Datasource 'nonexistent' not found"):
            await service.get_datasource("nonexistent")

    @pytest.mark.asyncio
    async def test_get_datasource_database_error(self, service, mock_session):
        """Test get datasource with database error."""
        mock_session.exec.side_effect = Exception("Database error")
        
        with pytest.raises(DatasourceServiceError, match="Failed to get datasource"):
            await service.get_datasource("test_postgres")

    @pytest.mark.asyncio
    async def test_create_datasource_success(self, service, mock_session, mock_mindsdb_client, sample_create_request):
        """Test successful datasource creation."""
        mock_result = Mock()
        mock_result.first.return_value = None   
        mock_session.exec.return_value = mock_result
        
        mock_databases = Mock()
        mock_databases.create = AsyncMock()
        mock_mindsdb_client.databases = mock_databases
        
        result = await service.create_datasource(sample_create_request)
        
        assert isinstance(result, DatasourceResponse)
        assert result.name == "test_postgres"
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_datasource_already_exists(self, service, mock_session, sample_create_request, sample_datasource):
        """Test create datasource when it already exists."""
        mock_result = Mock()
        mock_result.first.return_value = sample_datasource
        mock_session.exec.return_value = mock_result
        
        with pytest.raises(DatasourceAlreadyExistsError, match="Datasource with name 'test_postgres' already exists"):
            await service.create_datasource(sample_create_request)

    @pytest.mark.asyncio
    async def test_update_datasource_success(self, service, mock_session, mock_mindsdb_client, sample_datasource):
        """Test successful datasource update."""
        mock_result = Mock()
        mock_result.first.return_value = sample_datasource
        mock_session.exec.return_value = mock_result
        
        mock_databases = Mock()
        mock_databases.create = AsyncMock()
        mock_databases.drop = AsyncMock()
        mock_mindsdb_client.databases = mock_databases
        
        update_request = DatasourceUpdateRequest(
            connection_data={"host": "newhost", "port": 5432}
        )
        
        result = await service.update_datasource("test_postgres", update_request)
        
        assert isinstance(result, DatasourceResponse)
        assert result.name == "test_postgres"
        assert result.connection_data == {"host": "newhost", "port": 5432}
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_datasource_not_found(self, service, mock_session):
        """Test update datasource when not found."""
        mock_result = Mock()
        mock_result.first.return_value = None
        mock_session.exec.return_value = mock_result
        
        update_request = DatasourceUpdateRequest(connection_data={"host": "newhost"})
        
        with pytest.raises(DatasourceNotFoundError, match="Datasource 'nonexistent' not found"):
            await service.update_datasource("nonexistent", update_request)

    @pytest.mark.asyncio
    async def test_delete_datasource_success(self, service, mock_session, mock_mindsdb_client, sample_datasource):
        """Test successful datasource deletion."""
        mock_result = Mock()
        mock_result.first.return_value = sample_datasource
        mock_session.exec.return_value = mock_result
        
        mock_databases = Mock()
        mock_databases.drop = AsyncMock()
        mock_mindsdb_client.databases = mock_databases
        
        await service.delete_datasource("test_postgres")
        
        mock_session.delete.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_datasource_not_found(self, service, mock_session):
        """Test delete datasource when not found."""
        mock_result = Mock()
        mock_result.first.return_value = None
        mock_session.exec.return_value = mock_result
        
        with pytest.raises(DatasourceNotFoundError, match="Datasource 'nonexistent' not found"):
            await service.delete_datasource("nonexistent")
