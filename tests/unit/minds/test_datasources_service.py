"""
Unit tests for DatasourcesService.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pandas as pd
import pytest
from mindsdb_sdk.databases import Database
from mindsdb_sdk.server import Server
from sqlmodel import Session

from minds.model.datasource import Datasource
from minds.schemas.datasources import (
    ColumnResponse,
    DataCatalogResponse,
    DatasourceCreateRequest,
    DatasourceResponse,
    DatasourceTableSampleResponse,
    DatasourceUpdateRequest,
    TableResponse,
)
from minds.services.datasources import (
    DatasourceAlreadyExistsError,
    DatasourceNotFoundError,
    DatasourceServiceError,
    DatasourcesService,
    DatasourceTableColumnNotCatalogedError,
    DatasourceTableColumnNotFoundError,
    DatasourceTableNotCatalogedError,
    DatasourceTableNotFoundError,
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
            user_id=uuid4(),
            organization_id=uuid4(),
            mindsdb_client=mock_mindsdb_client,
        )

    @pytest.fixture
    def sample_datasource(self, user_id=uuid4(), organization_id=uuid4()):
        """Sample datasource for testing."""
        return Datasource(
            id=uuid4(),
            name="test_postgres",
            engine="postgres",
            connection_data={"host": "localhost", "port": 5432, "user": "test"},
            user_id=user_id,
            organization_id=organization_id,
            created_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
        )

    @pytest.fixture
    def sample_create_request(self):
        """Sample create request."""
        return DatasourceCreateRequest(
            name="test_postgres", engine="postgres", connection_data={"host": "localhost", "port": 5432, "user": "test"}
        )

    def test_service_initialization(self, mock_session, mock_mindsdb_client):
        """Test service initialization."""
        user_id = uuid4()
        organization_id = uuid4()
        service = DatasourcesService(
            session=mock_session, user_id=user_id, organization_id=organization_id, mindsdb_client=mock_mindsdb_client
        )

        assert service.session == mock_session
        assert service.user_id == user_id
        assert service.organization_id == organization_id
        assert service.mindsdb_client == mock_mindsdb_client

    def test_datasource_to_response(self, service, sample_datasource):
        """Test _datasource_to_response conversion."""
        result = service._datasource_to_response(sample_datasource)

        assert isinstance(result, DatasourceResponse)
        assert result.id == sample_datasource.id
        assert result.name == "test_postgres"
        assert result.engine == "postgres"

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

        # Mock the refresh method to set an id and datetime fields on the datasource
        def mock_refresh(datasource):
            datasource.id = uuid4()
            datasource.created_at = datetime.now(timezone.utc)
            datasource.modified_at = datetime.now(timezone.utc)

        mock_session.refresh.side_effect = mock_refresh

        mock_databases = Mock()
        mock_databases.create = AsyncMock()
        mock_mindsdb_client.databases = mock_databases

        result = await service.create_datasource(sample_create_request)

        assert isinstance(result, DatasourceResponse)
        assert result.name == "test_postgres"
        assert result.id is not None  # Should have an ID after creation
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_datasource_already_exists(
        self, service, mock_session, sample_create_request, sample_datasource
    ):
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
        mock_databases.update = AsyncMock()
        mock_mindsdb_client.databases = mock_databases

        update_request = DatasourceUpdateRequest(
            description="Updated description", connection_data={"host": "newhost", "port": 5432}
        )

        result = await service.update_datasource("test_postgres", update_request)

        assert isinstance(result, DatasourceResponse)
        assert result.name == "test_postgres"
        mock_session.commit.assert_called_once()
        mock_databases.update.assert_called_once()

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

        # Soft delete only.
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_datasource_not_found(self, service, mock_session):
        """Test delete datasource when not found."""
        mock_result = Mock()
        mock_result.first.return_value = None
        mock_session.exec.return_value = mock_result

        with pytest.raises(DatasourceNotFoundError, match="Datasource 'nonexistent' not found"):
            await service.delete_datasource("nonexistent")

    @pytest.mark.asyncio
    async def test_connection_success(self, service, mock_mindsdb_client):
        """Test successful connection testing."""
        service._get_datasource = AsyncMock(return_value=Mock())

        mock_database = Mock(spec=Database)
        mock_tables = Mock()
        mock_tables.list = Mock()
        mock_database.tables = mock_tables

        mock_databases = Mock()
        mock_databases.get.return_value = mock_database
        mock_mindsdb_client.databases = mock_databases

        result = await service.test_connection("test_postgres")

        assert result.success is True
        assert result.error_message is None
        mock_tables.list.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_datasource_not_found_in_mindsdb(self, service, mock_mindsdb_client):
        """Test connection when datasource not found in MindsDB."""
        service._get_datasource = AsyncMock(return_value=Mock())

        mock_databases = Mock()
        mock_databases.get.side_effect = AttributeError("Database not found")
        mock_mindsdb_client.databases = mock_databases

        result = await service.test_connection("test_postgres")

        assert result.success is False
        assert "Datasource not found in MindsDB" in result.error_message

    @pytest.mark.asyncio
    async def test_connection_table_list_failure(self, service, mock_mindsdb_client):
        """Test connection when table listing fails."""
        service._get_datasource = AsyncMock(return_value=Mock())

        mock_database = Mock(spec=Database)
        mock_tables = Mock()
        mock_tables.list.side_effect = Exception("Connection failed")
        mock_database.tables = mock_tables

        mock_databases = Mock()
        mock_databases.get.return_value = mock_database
        mock_mindsdb_client.databases = mock_databases

        result = await service.test_connection("test_postgres")

        assert result.success is False
        assert "Connection test failed: Connection failed" in result.error_message

    @pytest.mark.asyncio
    async def test_connection_datasource_not_found(self, service):
        """Test connection when datasource doesn't exist."""
        service._get_datasource = AsyncMock(side_effect=DatasourceNotFoundError("Not found"))

        result = await service.test_connection("nonexistent")

        assert result.success is False
        assert "Datasource not found" in result.error_message

    @pytest.mark.asyncio
    async def test_connection_unexpected_error(self, service):
        """Test connection with unexpected error."""
        service._get_datasource = AsyncMock(side_effect=Exception("Unexpected error"))

        result = await service.test_connection("test_postgres")

        assert result.success is False
        assert "Unexpected error" in result.error_message

    @pytest.mark.asyncio
    async def test_get_datasource_table_sample_success(self, service, mock_mindsdb_client):
        """Test getting sample data for a table from a datasource."""
        # Mock the _get_datasource method to return a datasource
        service._get_datasource = AsyncMock(return_value=Mock())

        sample_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        # Mock the query chain: databases.get().query().fetch()
        mock_query = Mock()
        mock_query.fetch.return_value = sample_df

        mock_database = Mock()
        mock_database.query.return_value = mock_query

        mock_databases = Mock()
        mock_databases.get.return_value = mock_database
        mock_mindsdb_client.databases = mock_databases

        result = await service.get_datasource_table_sample("test_postgres", "test_table")

        # Verify the response structure
        assert isinstance(result, DatasourceTableSampleResponse)
        assert hasattr(result, "data")
        assert hasattr(result, "column_names")
        assert result.data == [[1, "a"], [2, "b"], [3, "c"]]
        assert result.column_names == ["col1", "col2"]

        # Verify the query was called correctly
        mock_database.query.assert_called_once_with("SELECT * FROM test_table LIMIT 10")
        mock_query.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_datasource_table_sample_datasource_not_found(self, service):
        """Test getting sample data when datasource is not found."""
        service._get_datasource = AsyncMock(return_value=None)

        with pytest.raises(DatasourceNotFoundError, match="Datasource 'test_postgres' not found"):
            await service.get_datasource_table_sample("test_postgres", "test_table")

    @pytest.mark.asyncio
    async def test_get_datasource_table_sample_query_error(self, service, mock_mindsdb_client):
        """Test getting sample data when query fails."""
        service._get_datasource = AsyncMock(return_value=Mock())

        # Mock the query chain to raise an exception
        mock_query = Mock()
        mock_query.fetch.side_effect = Exception("Query failed")

        mock_database = Mock()
        mock_database.query.return_value = mock_query

        mock_databases = Mock()
        mock_databases.get.return_value = mock_database
        mock_mindsdb_client.databases = mock_databases

        with pytest.raises(DatasourceServiceError, match="Failed to get sample data"):
            await service.get_datasource_table_sample("test_postgres", "test_table")

    @pytest.mark.asyncio
    async def test_get_datasource_table_row_count_success(self, service, mock_mindsdb_client):
        """Test getting row count for a table from a datasource."""
        # Mock the _get_datasource method to return a datasource
        service._get_datasource = AsyncMock(return_value=Mock())

        # Mock the query result - COUNT(*) returns a DataFrame with one row and one column
        row_count_df = pd.DataFrame({"count": [42]})

        # Mock the query chain: databases.get().query().fetch()
        mock_query = Mock()
        mock_query.fetch.return_value = row_count_df

        mock_database = Mock()
        mock_database.query.return_value = mock_query

        mock_databases = Mock()
        mock_databases.get.return_value = mock_database
        mock_mindsdb_client.databases = mock_databases

        result = await service.get_datasource_table_row_count("test_postgres", "test_table")

        # Verify the result is the expected integer
        assert result == 42

        # Verify the query was called correctly
        mock_database.query.assert_called_once_with("SELECT COUNT(*) FROM test_table")
        mock_query.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_datasource_table_row_count_datasource_not_found(self, service):
        """Test getting row count when datasource is not found."""
        service._get_datasource = AsyncMock(return_value=None)

        with pytest.raises(DatasourceNotFoundError, match="Datasource 'test_postgres' not found"):
            await service.get_datasource_table_row_count("test_postgres", "test_table")

    @pytest.mark.asyncio
    async def test_get_datasource_table_row_count_query_error(self, service, mock_mindsdb_client):
        """Test getting row count when query fails."""
        service._get_datasource = AsyncMock(return_value=Mock())

        # Mock the query chain to raise an exception
        mock_query = Mock()
        mock_query.fetch.side_effect = Exception("Query failed")

        mock_database = Mock()
        mock_database.query.return_value = mock_query

        mock_databases = Mock()
        mock_databases.get.return_value = mock_database
        mock_mindsdb_client.databases = mock_databases

        with pytest.raises(DatasourceServiceError, match="Failed to get row count"):
            await service.get_datasource_table_row_count("test_postgres", "test_table")

    @pytest.mark.asyncio
    async def test_check_datasource_exists_success(self, service, sample_datasource):
        """Test successful datasource existence check."""
        service._get_datasource = AsyncMock(return_value=sample_datasource)

        result = await service.check_datasource_exists("test_postgres")

        assert result is None
        service._get_datasource.assert_called_once_with("test_postgres")

    @pytest.mark.asyncio
    async def test_check_datasource_exists_not_found(self, service):
        """Test check datasource exists when datasource doesn't exist."""
        service._get_datasource = AsyncMock(return_value=None)

        with pytest.raises(DatasourceNotFoundError, match="Datasource 'test_postgres' not found"):
            await service.check_datasource_exists("test_postgres")

    @pytest.mark.asyncio
    async def test_check_datasource_exists_lowercase(self, service, sample_datasource):
        """Test that datasource name is converted to lowercase."""
        service._get_datasource = AsyncMock(return_value=sample_datasource)

        result = await service.check_datasource_exists("TEST_POSTGRES")

        assert result is None
        # Should be called with lowercase name
        service._get_datasource.assert_called_once_with("test_postgres")

    @pytest.mark.asyncio
    async def test_check_datasource_exists_database_error(self, service):
        """Test check datasource exists with database error."""
        service._get_datasource = AsyncMock(side_effect=Exception("Database error"))

        with pytest.raises(DatasourceServiceError, match="Failed to check datasource existence"):
            await service.check_datasource_exists("test_postgres")

    @pytest.fixture
    def datasource_with_catalog(self):
        """Create a minimal datasource-like object with catalog tables."""
        now = datetime.now(timezone.utc)

        column = Mock()
        column.name = "id"
        column.data_type = "integer"
        column.description = "Primary key"
        column.default_value = None
        column.is_nullable = False
        column.statistics = None

        table = Mock()
        table.name = "test_table"
        table.schema = "public"
        table.description = "Test table"
        table.type = "BASE TABLE"
        table.row_count = 123
        table.columns = [column]
        table.primary_key_constraints = []
        table.foreign_key_constraints = []

        datasource = Mock()
        datasource.id = uuid4()
        datasource.name = "test_postgres"
        datasource.description = "Test PostgreSQL datasource"
        datasource.engine = "postgres"
        datasource.created_at = now
        datasource.modified_at = now
        datasource.tables = [table]
        datasource.is_sample = True

        return datasource

    @pytest.mark.asyncio
    async def test_get_datasource_catalog_success(self, service, mock_session, datasource_with_catalog):
        """Test successful catalog retrieval for a datasource."""
        mock_result = Mock()
        mock_result.first.return_value = datasource_with_catalog
        mock_session.exec.return_value = mock_result

        result = await service.get_datasource_catalog("test_postgres")

        assert isinstance(result, DataCatalogResponse)
        assert result.datasource.name == "test_postgres"
        assert len(result.tables) == 1
        assert result.tables[0].name == "test_table"
        assert len(result.tables[0].columns) == 1
        assert result.tables[0].columns[0].name == "id"

    @pytest.mark.asyncio
    async def test_get_datasource_catalog_not_found(self, service, mock_session):
        """Test catalog retrieval when datasource does not exist."""
        mock_result = Mock()
        mock_result.first.return_value = None
        mock_session.exec.return_value = mock_result

        with pytest.raises(DatasourceNotFoundError, match="Datasource 'test_postgres' not found"):
            await service.get_datasource_catalog("test_postgres")

    @pytest.mark.asyncio
    async def test_get_datasource_catalog_database_error(self, service, mock_session):
        """Test catalog retrieval wraps unexpected errors."""
        mock_session.exec.side_effect = Exception("DB exploded")

        with pytest.raises(DatasourceServiceError, match="Failed to get data catalog"):
            await service.get_datasource_catalog("test_postgres")

    @pytest.mark.asyncio
    async def test_get_datasource_table_catalog_success(self, service):
        """Test successful catalog retrieval for a table."""
        table = Mock()
        table.name = "test_table"
        table.schema = "public"
        table.description = "Test table"
        table.type = "BASE TABLE"
        table.row_count = 123
        table.columns = []
        table.primary_key_constraints = []
        table.foreign_key_constraints = []

        service._get_datasource_table_catalog = AsyncMock(return_value=table)

        result = await service.get_datasource_table_catalog("test_postgres", "test_table")

        assert isinstance(result, TableResponse)
        assert result.name == "test_table"
        service._get_datasource_table_catalog.assert_called_once_with("test_postgres", "test_table")

    @pytest.mark.asyncio
    async def test_get_datasource_table_catalog_table_not_found(self, service):
        """Test table catalog retrieval when table doesn't exist in datasource."""
        service._get_datasource_table_catalog = AsyncMock(return_value=None)
        service._check_table_exists = Mock(return_value=False)

        with pytest.raises(
            DatasourceTableNotFoundError, match="Table 'missing_table' not found in datasource 'test_postgres'"
        ):
            await service.get_datasource_table_catalog("test_postgres", "missing_table")

    @pytest.mark.asyncio
    async def test_get_datasource_table_catalog_table_not_cataloged(self, service):
        """Test table catalog retrieval when table exists but isn't cataloged."""
        service._get_datasource_table_catalog = AsyncMock(return_value=None)
        service._check_table_exists = Mock(return_value=True)

        with pytest.raises(
            DatasourceTableNotCatalogedError, match="Table 'test_table' not cataloged in datasource 'test_postgres'"
        ):
            await service.get_datasource_table_catalog("test_postgres", "test_table")

    @pytest.mark.asyncio
    async def test_update_table_catalog_description_success(self, service, mock_session):
        """Test updating a table description in the catalog."""
        table = Mock()
        table.name = "test_table"
        table.schema = "public"
        table.description = "Old"
        table.type = "BASE TABLE"
        table.row_count = 123
        table.columns = []
        table.primary_key_constraints = []
        table.foreign_key_constraints = []

        service._get_datasource_table_catalog = AsyncMock(return_value=table)

        result = await service.update_datasource_table_catalog_description("test_postgres", "test_table", "New")

        assert isinstance(result, TableResponse)
        assert result.description == "New"
        assert table.description == "New"
        mock_session.add.assert_called_once_with(table)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(table)

    @pytest.mark.asyncio
    async def test_update_table_catalog_description_table_not_found(self, service):
        """Test updating table description when the table doesn't exist."""
        service._get_datasource_table_catalog = AsyncMock(return_value=None)
        service._check_table_exists = Mock(return_value=False)

        with pytest.raises(DatasourceTableNotFoundError, match="Table 'missing_table' not found in datasource"):
            await service.update_datasource_table_catalog_description("test_postgres", "missing_table", "New")

    @pytest.mark.asyncio
    async def test_update_table_catalog_description_table_not_cataloged(self, service):
        """Test updating table description when the table exists but isn't cataloged."""
        service._get_datasource_table_catalog = AsyncMock(return_value=None)
        service._check_table_exists = Mock(return_value=True)

        with pytest.raises(DatasourceTableNotCatalogedError, match="Table 'test_table' not cataloged in datasource"):
            await service.update_datasource_table_catalog_description("test_postgres", "test_table", "New")

    @pytest.mark.asyncio
    async def test_update_table_catalog_column_description_success(self, service, mock_session):
        """Test updating a column description in the catalog."""
        datasource = Mock()
        datasource.id = uuid4()

        table = Mock()
        table.id = uuid4()

        column = Mock()
        column.name = "id"
        column.data_type = "integer"
        column.description = "Old"
        column.default_value = None
        column.is_nullable = False
        column.statistics = None

        service._get_datasource = AsyncMock(return_value=datasource)

        table_result = Mock()
        table_result.first.return_value = table
        column_result = Mock()
        column_result.first.return_value = column
        mock_session.exec.side_effect = [table_result, column_result]

        result = await service.update_datasource_table_catalog_column_description(
            datasource_name="test_postgres",
            table_name="test_table",
            column_name="id",
            description="New",
        )

        assert isinstance(result, ColumnResponse)
        assert result.name == "id"
        assert result.description == "New"
        assert column.description == "New"
        mock_session.add.assert_called_once_with(column)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(column)

    @pytest.mark.asyncio
    async def test_update_table_catalog_column_description_datasource_not_found(self, service):
        """Test updating column description when datasource doesn't exist."""
        service._get_datasource = AsyncMock(return_value=None)

        with pytest.raises(DatasourceNotFoundError, match="Datasource 'test_postgres' not found"):
            await service.update_datasource_table_catalog_column_description("test_postgres", "test_table", "id", "New")

    @pytest.mark.asyncio
    async def test_update_table_catalog_column_description_table_not_found(self, service, mock_session):
        """Test updating column description when table doesn't exist in datasource."""
        datasource = Mock()
        datasource.id = uuid4()
        service._get_datasource = AsyncMock(return_value=datasource)

        table_result = Mock()
        table_result.first.return_value = None
        mock_session.exec.side_effect = [table_result]

        service._check_table_exists = Mock(return_value=False)

        with pytest.raises(DatasourceTableNotFoundError, match="Table 'missing_table' not found in datasource"):
            await service.update_datasource_table_catalog_column_description(
                "test_postgres", "missing_table", "id", "New"
            )

    @pytest.mark.asyncio
    async def test_update_table_catalog_column_description_table_not_cataloged(self, service, mock_session):
        """Test updating column description when table exists but is not cataloged."""
        datasource = Mock()
        datasource.id = uuid4()
        service._get_datasource = AsyncMock(return_value=datasource)

        table_result = Mock()
        table_result.first.return_value = None
        mock_session.exec.side_effect = [table_result]

        service._check_table_exists = Mock(return_value=True)

        with pytest.raises(DatasourceTableNotCatalogedError, match="Table 'test_table' not cataloged in datasource"):
            await service.update_datasource_table_catalog_column_description("test_postgres", "test_table", "id", "New")

    @pytest.mark.asyncio
    async def test_update_table_catalog_column_description_column_not_found(self, service, mock_session):
        """Test updating column description when column doesn't exist in table."""
        datasource = Mock()
        datasource.id = uuid4()
        service._get_datasource = AsyncMock(return_value=datasource)

        table = Mock()
        table.id = uuid4()

        table_result = Mock()
        table_result.first.return_value = table
        column_result = Mock()
        column_result.first.return_value = None
        mock_session.exec.side_effect = [table_result, column_result]

        service._check_column_exists = Mock(return_value=False)

        with pytest.raises(DatasourceTableColumnNotFoundError, match="Column 'missing_col' not found in table"):
            await service.update_datasource_table_catalog_column_description(
                "test_postgres", "test_table", "missing_col", "New"
            )

    @pytest.mark.asyncio
    async def test_update_table_catalog_column_description_column_not_cataloged(self, service, mock_session):
        """Test updating column description when column exists but is not cataloged."""
        datasource = Mock()
        datasource.id = uuid4()
        service._get_datasource = AsyncMock(return_value=datasource)

        table = Mock()
        table.id = uuid4()

        table_result = Mock()
        table_result.first.return_value = table
        column_result = Mock()
        column_result.first.return_value = None
        mock_session.exec.side_effect = [table_result, column_result]

        service._check_column_exists = Mock(return_value=True)

        with pytest.raises(DatasourceTableColumnNotCatalogedError, match="Column 'id' not cataloged in table"):
            await service.update_datasource_table_catalog_column_description("test_postgres", "test_table", "id", "New")

    # -- count_datasources tests --

    @pytest.mark.asyncio
    async def test_count_datasources_success(self, service, mock_session):
        """Test successful datasource count."""
        mock_session.exec.return_value.one.return_value = 7

        result = await service.count_datasources()

        assert result == 7
        mock_session.exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_count_datasources_with_is_sample_false(self, service, mock_session):
        """Test count datasources excluding sample datasources."""
        mock_session.exec.return_value.one.return_value = 4

        result = await service.count_datasources(is_sample=False)

        assert result == 4
        mock_session.exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_count_datasources_with_is_sample_true(self, service, mock_session):
        """Test count datasources including only sample datasources."""
        mock_session.exec.return_value.one.return_value = 3

        result = await service.count_datasources(is_sample=True)

        assert result == 3
        mock_session.exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_count_datasources_returns_zero(self, service, mock_session):
        """Test count datasources when there are none."""
        mock_session.exec.return_value.one.return_value = 0

        result = await service.count_datasources()

        assert result == 0

    @pytest.mark.asyncio
    async def test_count_datasources_database_error(self, service, mock_session):
        """Test count datasources with database error."""
        mock_session.exec.side_effect = Exception("Database error")

        with pytest.raises(DatasourceServiceError, match="Failed to count datasources"):
            await service.count_datasources()
