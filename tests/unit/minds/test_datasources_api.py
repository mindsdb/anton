"""
Unit tests for datasources API endpoints.
"""

from unittest.mock import ANY, AsyncMock, Mock

import pytest
from fastapi import HTTPException

from minds.api.v1.endpoints.datasources import (
    check_datasource_connection,
    check_datasource_exists,
    create_datasource,
    delete_datasource,
    get_datasource,
    get_datasource_catalog,
    get_datasource_table_catalog,
    get_datasource_table_row_count,
    get_datasource_table_sample,
    get_datasources_service,
    list_datasources,
    update_column_description,
    update_datasource,
    update_table_description,
)
from minds.schemas.datasources import (
    ColumnResponse,
    DataCatalogResponse,
    DatasourceConnectionStatus,
    DatasourceCreateRequest,
    DatasourceResponse,
    DatasourceTableSampleResponse,
    DatasourceUpdateRequest,
    TableResponse,
    UpdateColumnDescriptionRequest,
    UpdateTableDescriptionRequest,
)
from minds.schemas.limits import LimitsConfig, MindLimitsConfig, ResourceUsageConfig, UsageConfig
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
from minds.services.limits import LimitsService


class TestDatasourcesAPI:
    """Test datasources API endpoints."""

    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI request object."""
        request = Mock()
        request.headers = {"X-User-Id": "test-user-123"}
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
        service.organization_id = "test-organization-456"
        return service

    @pytest.fixture
    def mock_limits_service(self):
        """Mock LimitsService that allows all operations (under limit)."""
        service = Mock(spec=LimitsService)
        service.get_mind_limits = AsyncMock(return_value=MindLimitsConfig())
        return service

    @pytest.fixture
    def sample_datasource_response(self, test_uuid):
        """Sample DatasourceResponse for testing."""
        return DatasourceResponse(
            id=test_uuid,
            is_sample=True,
            name="test_postgres",
            description="Test PostgreSQL datasource for API testing",
            engine="postgres",
            connection_data={"host": "localhost", "port": 5432},
            created_at="2023-01-01T12:00:00",
        )

    @pytest.fixture
    def sample_connection_status(self):
        """Sample connection status for testing."""
        return DatasourceConnectionStatus(success=True, error_message=None)

    def test_get_datasources_service_dependency(self, mock_request, mock_session, mock_mindsdb_client):
        """Test the get_datasources_service dependency function."""
        mock_context = Mock()
        mock_context.user_id = "test-user-123"
        mock_context.organization_id = "test-organization-456"

        service = get_datasources_service(
            context=mock_context, session=mock_session, mindsdb_client=mock_mindsdb_client
        )

        assert isinstance(service, DatasourcesService)
        assert service.session == mock_session
        assert service.user_id == "test-user-123"

    @pytest.mark.asyncio
    async def test_list_datasources_success(self, mock_datasources_service, sample_datasource_response):
        """Test successful datasources listing."""
        mock_datasources_service.list_datasources = AsyncMock(return_value=[sample_datasource_response])

        result = await list_datasources(
            engine=None,
            limit=100,
            offset=0,
            with_detailed_data=False,
            include_total=False,
            sort_by=None,
            sort_order="desc",
            datasources_service=mock_datasources_service,
        )

        assert len(result) == 1
        assert result[0].name == "test_postgres"
        mock_datasources_service.list_datasources.assert_called_once_with(
            name=ANY,
            engine=None,
            include_deleted=ANY,
            limit=100,
            offset=0,
            with_detailed_data=False,
            include_total=False,
            sort_by=None,
            sort_order="desc",
        )

    @pytest.mark.asyncio
    async def test_list_datasources_include_total_success(self, mock_datasources_service, sample_datasource_response):
        """Test successful datasources listing with total count included."""
        mock_datasources_service.list_datasources = AsyncMock(return_value=([sample_datasource_response], 1))

        result = await list_datasources(
            engine=None,
            limit=100,
            offset=0,
            with_detailed_data=False,
            include_total=True,
            sort_by=None,
            sort_order="desc",
            datasources_service=mock_datasources_service,
        )

        assert isinstance(result, dict)
        assert result["total"] == 1
        assert len(result["datasources"]) == 1
        assert result["datasources"][0].name == "test_postgres"
        mock_datasources_service.list_datasources.assert_called_once_with(
            name=ANY,
            engine=None,
            include_deleted=ANY,
            limit=100,
            offset=0,
            with_detailed_data=False,
            include_total=True,
            sort_by=None,
            sort_order="desc",
        )

    @pytest.mark.asyncio
    async def test_list_datasources_empty(self, mock_datasources_service):
        """Test listing empty datasources."""
        mock_datasources_service.list_datasources = AsyncMock(return_value=[])

        result = await list_datasources(
            datasources_service=mock_datasources_service,
            include_total=False,
            sort_by=None,
            sort_order="desc",
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_list_datasources_with_filters(self, mock_datasources_service, sample_datasource_response):
        """Test listing datasources with filters."""
        mock_datasources_service.list_datasources = AsyncMock(return_value=[sample_datasource_response])

        result = await list_datasources(
            engine="postgres",
            limit=10,
            offset=5,
            with_detailed_data=True,
            include_total=False,
            sort_by=None,
            sort_order="desc",
            datasources_service=mock_datasources_service,
        )

        assert len(result) == 1
        mock_datasources_service.list_datasources.assert_called_once_with(
            name=ANY,
            engine="postgres",
            include_deleted=ANY,
            limit=10,
            offset=5,
            with_detailed_data=True,
            include_total=False,
            sort_by=None,
            sort_order="desc",
        )

    @pytest.mark.asyncio
    async def test_list_datasources_service_error(self, mock_datasources_service):
        """Test list datasources with service error."""
        from fastapi import HTTPException

        mock_datasources_service.list_datasources = AsyncMock(side_effect=DatasourceServiceError("Service error"))

        with pytest.raises(HTTPException) as exc_info:
            await list_datasources(
                datasources_service=mock_datasources_service,
                include_total=False,
                sort_by=None,
                sort_order="desc",
            )

        assert exc_info.value.status_code == 400
        assert "Service error" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_datasource_success(self, mock_datasources_service, sample_datasource_response):
        """Test successful datasource retrieval."""
        mock_datasources_service.get_datasource = AsyncMock(return_value=sample_datasource_response)

        result = await get_datasource(
            datasource_name="test_postgres", with_detailed_data=False, datasources_service=mock_datasources_service
        )

        assert result.name == "test_postgres"
        mock_datasources_service.get_datasource.assert_called_once_with(
            datasource_name="test_postgres", with_detailed_data=False
        )

    @pytest.mark.asyncio
    async def test_get_datasource_not_found(self, mock_datasources_service):
        """Test get datasource when not found."""
        from fastapi import HTTPException

        mock_datasources_service.get_datasource = AsyncMock(side_effect=DatasourceNotFoundError("Datasource not found"))

        with pytest.raises(HTTPException) as exc_info:
            await get_datasource(datasource_name="nonexistent", datasources_service=mock_datasources_service)

        assert exc_info.value.status_code == 404
        assert "Datasource not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_create_datasource_success(
        self, mock_datasources_service, mock_limits_service, sample_datasource_response
    ):
        """Test successful datasource creation."""
        mock_datasources_service.create_datasource = AsyncMock(return_value=sample_datasource_response)

        create_request = DatasourceCreateRequest(
            name="test_postgres",
            description="New test PostgreSQL datasource",
            engine="postgres",
            connection_data={"host": "localhost", "port": 5432},
        )

        result = await create_datasource(
            datasource_data=create_request,
            datasources_service=mock_datasources_service,
            limits_service=mock_limits_service,
        )

        assert result.name == "test_postgres"
        mock_datasources_service.create_datasource.assert_called_once_with(create_request)

    @pytest.mark.asyncio
    async def test_create_datasource_already_exists(self, mock_datasources_service, mock_limits_service):
        """Test create datasource when it already exists."""
        mock_datasources_service.create_datasource = AsyncMock(
            side_effect=DatasourceAlreadyExistsError("Already exists")
        )

        create_request = DatasourceCreateRequest(
            name="test_postgres",
            description="Duplicate test PostgreSQL datasource",
            engine="postgres",
            connection_data={"host": "localhost", "port": 5432},
        )

        with pytest.raises(HTTPException) as exc_info:
            await create_datasource(
                datasource_data=create_request,
                datasources_service=mock_datasources_service,
                limits_service=mock_limits_service,
            )

        assert exc_info.value.status_code == 409
        assert "Already exists" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_create_datasource_rejects_when_limit_exceeded(self, mock_datasources_service):
        """Test that create_datasource returns 429 when datasources limit is exceeded."""
        exceeded_limits = MindLimitsConfig(
            datasources=ResourceUsageConfig(
                limit=LimitsConfig(lifetime=3, monthly=3),
                usage=UsageConfig(lifetime=3, billing_cycle=3),
            ),
        )
        limits_service = Mock(spec=LimitsService)
        limits_service.get_mind_limits = AsyncMock(return_value=exceeded_limits)

        create_request = DatasourceCreateRequest(
            name="test_postgres",
            description="Should be rejected",
            engine="postgres",
            connection_data={"host": "localhost", "port": 5432},
        )

        with pytest.raises(HTTPException) as exc_info:
            await create_datasource(
                datasource_data=create_request,
                datasources_service=mock_datasources_service,
                limits_service=limits_service,
            )

        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_update_datasource_success(self, mock_datasources_service, sample_datasource_response):
        """Test successful datasource update."""
        mock_datasources_service.update_datasource = AsyncMock(return_value=sample_datasource_response)

        update_request = DatasourceUpdateRequest(
            description="Updated test PostgreSQL datasource", connection_data={"host": "newhost", "port": 5432}
        )

        result = await update_datasource(
            datasource_name="test_postgres",
            datasource_data=update_request,
            datasources_service=mock_datasources_service,
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

        update_request = DatasourceUpdateRequest(
            description="Updated description for non-existent datasource", connection_data={"host": "newhost"}
        )

        with pytest.raises(HTTPException) as exc_info:
            await update_datasource(
                datasource_name="nonexistent",
                datasource_data=update_request,
                datasources_service=mock_datasources_service,
            )

        assert exc_info.value.status_code == 404
        assert "Datasource not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_delete_datasource_success(self, mock_datasources_service):
        """Test successful datasource deletion."""
        mock_datasources_service.delete_datasource = AsyncMock()

        result = await delete_datasource(
            datasource_name="test_postgres", cascade=False, datasources_service=mock_datasources_service
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
                datasource_name="nonexistent", cascade=False, datasources_service=mock_datasources_service
            )

        assert exc_info.value.status_code == 404
        assert "Datasource not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_test_connection_success(self, mock_datasources_service, sample_connection_status):
        """Test successful connection testing."""
        mock_datasources_service.test_connection = AsyncMock(return_value=sample_connection_status)

        result = await check_datasource_connection(
            datasource_name="test_postgres", datasources_service=mock_datasources_service
        )

        assert result.success is True
        mock_datasources_service.test_connection.assert_called_once_with("test_postgres")

    @pytest.mark.asyncio
    async def test_test_connection_failure(self, mock_datasources_service):
        """Test connection testing failure."""
        failure_status = DatasourceConnectionStatus(success=False, error_message="Connection failed")
        mock_datasources_service.test_connection = AsyncMock(return_value=failure_status)

        result = await check_datasource_connection(
            datasource_name="test_postgres", datasources_service=mock_datasources_service
        )

        assert result.success is False
        assert "Connection failed" in result.error_message

    @pytest.mark.asyncio
    async def test_test_connection_exception_returns_failure_status(self, mock_datasources_service):
        """Test connection testing when the service raises an exception."""
        mock_datasources_service.test_connection = AsyncMock(side_effect=Exception("boom"))

        result = await check_datasource_connection(
            datasource_name="test_postgres", datasources_service=mock_datasources_service
        )

        assert result.success is False
        assert result.error_message is not None
        assert "Connection test failed" in result.error_message

    @pytest.mark.asyncio
    async def test_check_datasource_exists_success(self, mock_datasources_service):
        """Test successful datasource existence check."""
        mock_datasources_service.check_datasource_exists = AsyncMock(return_value=None)

        result = await check_datasource_exists(
            datasource_name="test_postgres", datasources_service=mock_datasources_service
        )

        assert result is None
        mock_datasources_service.check_datasource_exists.assert_called_once_with(datasource_name="test_postgres")

    @pytest.mark.asyncio
    async def test_check_datasource_exists_not_found(self, mock_datasources_service):
        """Test check datasource exists when not found."""
        from fastapi import HTTPException

        mock_datasources_service.check_datasource_exists = AsyncMock(
            side_effect=DatasourceNotFoundError("Datasource 'test_postgres' not found")
        )

        with pytest.raises(HTTPException) as exc_info:
            await check_datasource_exists(datasource_name="test_postgres", datasources_service=mock_datasources_service)

        assert exc_info.value.status_code == 404
        assert "Datasource 'test_postgres' not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_check_datasource_exists_service_error(self, mock_datasources_service):
        """Test check datasource exists with service error."""
        from fastapi import HTTPException

        mock_datasources_service.check_datasource_exists = AsyncMock(
            side_effect=DatasourceServiceError("Failed to check datasource existence")
        )

        with pytest.raises(HTTPException) as exc_info:
            await check_datasource_exists(datasource_name="test_postgres", datasources_service=mock_datasources_service)

        assert exc_info.value.status_code == 500
        assert "Failed to check datasource existence" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_check_datasource_exists_unexpected_error(self, mock_datasources_service):
        """Test check datasource exists with unexpected error."""
        from fastapi import HTTPException

        mock_datasources_service.check_datasource_exists = AsyncMock(side_effect=Exception("Unexpected error"))

        with pytest.raises(HTTPException) as exc_info:
            await check_datasource_exists(datasource_name="test_postgres", datasources_service=mock_datasources_service)

        assert exc_info.value.status_code == 500
        assert "Internal server error" in str(exc_info.value.detail)

    @pytest.fixture
    def sample_table_sample_response(self):
        """Sample table sample response for testing."""
        return DatasourceTableSampleResponse(
            data=[["30/09/2007", 441854, "house", 2], ["31/12/2007", 441854, "house", 2]],
            column_names=["saledate", "MA", "type", "bedrooms"],
        )

    @pytest.mark.asyncio
    async def test_get_datasource_table_sample_success(self, mock_datasources_service, sample_table_sample_response):
        """Test successful table sample retrieval."""
        mock_datasources_service.get_datasource_table_sample = AsyncMock(return_value=sample_table_sample_response)

        result = await get_datasource_table_sample(
            datasource_name="test_postgres",
            table_name="test_table",
            limit=10,
            datasources_service=mock_datasources_service,
        )

        assert result.data == [["30/09/2007", 441854, "house", 2], ["31/12/2007", 441854, "house", 2]]
        assert result.column_names == ["saledate", "MA", "type", "bedrooms"]
        mock_datasources_service.get_datasource_table_sample.assert_called_once_with("test_postgres", "test_table", 10)

    @pytest.mark.asyncio
    async def test_get_datasource_table_sample_not_found(self, mock_datasources_service):
        """Test get table sample when datasource is not found."""
        from fastapi import HTTPException

        mock_datasources_service.get_datasource_table_sample = AsyncMock(
            side_effect=DatasourceNotFoundError("Datasource 'test_postgres' not found")
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_datasource_table_sample(
                datasource_name="test_postgres", table_name="test_table", datasources_service=mock_datasources_service
            )

        assert exc_info.value.status_code == 404
        assert "Datasource 'test_postgres' not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_datasource_table_sample_service_error(self, mock_datasources_service):
        """Test get table sample with service error."""
        from fastapi import HTTPException

        mock_datasources_service.get_datasource_table_sample = AsyncMock(
            side_effect=DatasourceServiceError("Failed to get sample data")
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_datasource_table_sample(
                datasource_name="test_postgres", table_name="test_table", datasources_service=mock_datasources_service
            )

        assert exc_info.value.status_code == 400
        assert "Failed to get sample data" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_datasource_table_sample_unexpected_error(self, mock_datasources_service):
        """Test get table sample with unexpected error."""
        from fastapi import HTTPException

        mock_datasources_service.get_datasource_table_sample = AsyncMock(side_effect=Exception("Unexpected error"))

        with pytest.raises(HTTPException) as exc_info:
            await get_datasource_table_sample(
                datasource_name="test_postgres", table_name="test_table", datasources_service=mock_datasources_service
            )

        assert exc_info.value.status_code == 500
        assert "Internal server error" in str(exc_info.value.detail)

    @pytest.fixture
    def sample_column_response(self):
        """Sample ColumnResponse for catalog testing."""
        return ColumnResponse(
            name="id",
            data_type="integer",
            description="Primary key",
            default_value=None,
            is_nullable=False,
            statistics=None,
        )

    @pytest.fixture
    def sample_table_response(self, sample_column_response):
        """Sample TableResponse for catalog testing."""
        return TableResponse(
            name="test_table",
            schema="public",
            description="Test table",
            type="BASE TABLE",
            row_count=123,
            columns=[sample_column_response],
            primary_key_constraints=[],
            foreign_key_constraints=[],
        )

    @pytest.fixture
    def sample_data_catalog_response(self, sample_datasource_response, sample_table_response):
        """Sample DataCatalogResponse for catalog testing."""
        return DataCatalogResponse(datasource=sample_datasource_response, tables=[sample_table_response])

    @pytest.mark.asyncio
    async def test_get_datasource_catalog_success(self, mock_datasources_service, sample_data_catalog_response):
        """Test successful datasource catalog retrieval."""
        from fastapi import HTTPException

        mock_datasources_service.get_datasource_catalog = AsyncMock(return_value=sample_data_catalog_response)

        result = await get_datasource_catalog(
            datasource_name="test_postgres",
            datasources_service=mock_datasources_service,
        )

        assert isinstance(result, DataCatalogResponse)
        assert result.datasource.name == "test_postgres"
        assert len(result.tables) == 1
        mock_datasources_service.get_datasource_catalog.assert_called_once_with(datasource_name="test_postgres")

        # Ensure we did not accidentally raise
        assert not isinstance(result, HTTPException)

    @pytest.mark.asyncio
    async def test_get_datasource_catalog_not_found(self, mock_datasources_service):
        """Test datasource catalog retrieval when datasource is not found."""
        from fastapi import HTTPException

        mock_datasources_service.get_datasource_catalog = AsyncMock(
            side_effect=DatasourceNotFoundError("Datasource 'test_postgres' not found")
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_datasource_catalog(datasource_name="test_postgres", datasources_service=mock_datasources_service)

        assert exc_info.value.status_code == 404
        assert "not found" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_get_datasource_catalog_service_error(self, mock_datasources_service):
        """Test datasource catalog retrieval with service error."""
        from fastapi import HTTPException

        mock_datasources_service.get_datasource_catalog = AsyncMock(side_effect=DatasourceServiceError("Bad request"))

        with pytest.raises(HTTPException) as exc_info:
            await get_datasource_catalog(datasource_name="test_postgres", datasources_service=mock_datasources_service)

        assert exc_info.value.status_code == 400
        assert "Bad request" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_datasource_table_catalog_success(self, mock_datasources_service, sample_table_response):
        """Test successful datasource table catalog retrieval."""
        mock_datasources_service.get_datasource_table_catalog = AsyncMock(return_value=sample_table_response)

        result = await get_datasource_table_catalog(
            datasource_name="test_postgres",
            table_name="test_table",
            datasources_service=mock_datasources_service,
        )

        assert isinstance(result, TableResponse)
        assert result.name == "test_table"
        mock_datasources_service.get_datasource_table_catalog.assert_called_once_with("test_postgres", "test_table")

    @pytest.mark.asyncio
    async def test_get_datasource_table_catalog_not_cataloged(self, mock_datasources_service):
        """Test datasource table catalog retrieval when table is not cataloged."""
        from fastapi import HTTPException

        mock_datasources_service.get_datasource_table_catalog = AsyncMock(
            side_effect=DatasourceTableNotCatalogedError("Table not cataloged")
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_datasource_table_catalog(
                datasource_name="test_postgres",
                table_name="test_table",
                datasources_service=mock_datasources_service,
            )

        assert exc_info.value.status_code == 409
        assert "not cataloged" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_get_datasource_table_catalog_table_not_found_returns_400(self, mock_datasources_service):
        """
        get_datasource_table_catalog currently maps table-not-found to 400
        (it does not catch DatasourceTableNotFoundError explicitly).
        """
        from fastapi import HTTPException

        mock_datasources_service.get_datasource_table_catalog = AsyncMock(
            side_effect=DatasourceTableNotFoundError("Table not found")
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_datasource_table_catalog(
                datasource_name="test_postgres",
                table_name="missing_table",
                datasources_service=mock_datasources_service,
            )

        assert exc_info.value.status_code == 400
        assert "Table not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_update_table_description_success(self, mock_datasources_service, sample_table_response):
        """Test successful table description update."""
        mock_datasources_service.update_datasource_table_catalog_description = AsyncMock(
            return_value=sample_table_response
        )

        update_request = UpdateTableDescriptionRequest(description="New description")

        result = await update_table_description(
            datasource_name="test_postgres",
            table_name="test_table",
            update_request=update_request,
            datasources_service=mock_datasources_service,
        )

        assert isinstance(result, TableResponse)
        assert result.name == "test_table"
        mock_datasources_service.update_datasource_table_catalog_description.assert_called_once_with(
            datasource_name="test_postgres",
            table_name="test_table",
            description="New description",
        )

    @pytest.mark.asyncio
    async def test_update_table_description_table_not_found(self, mock_datasources_service):
        """Test update table description when table is not found."""
        from fastapi import HTTPException

        mock_datasources_service.update_datasource_table_catalog_description = AsyncMock(
            side_effect=DatasourceTableNotFoundError("Table not found")
        )

        update_request = UpdateTableDescriptionRequest(description="New description")

        with pytest.raises(HTTPException) as exc_info:
            await update_table_description(
                datasource_name="test_postgres",
                table_name="missing_table",
                update_request=update_request,
                datasources_service=mock_datasources_service,
            )

        assert exc_info.value.status_code == 404
        assert "Table not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_update_table_description_table_not_cataloged(self, mock_datasources_service):
        """Test update table description when table is not cataloged."""
        from fastapi import HTTPException

        mock_datasources_service.update_datasource_table_catalog_description = AsyncMock(
            side_effect=DatasourceTableNotCatalogedError("Table not cataloged")
        )

        update_request = UpdateTableDescriptionRequest(description="New description")

        with pytest.raises(HTTPException) as exc_info:
            await update_table_description(
                datasource_name="test_postgres",
                table_name="test_table",
                update_request=update_request,
                datasources_service=mock_datasources_service,
            )

        assert exc_info.value.status_code == 409
        assert "not cataloged" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_update_column_description_success(self, mock_datasources_service, sample_column_response):
        """Test successful column description update."""
        mock_datasources_service.update_datasource_table_catalog_column_description = AsyncMock(
            return_value=sample_column_response
        )

        update_request = UpdateColumnDescriptionRequest(description="New column description")

        result = await update_column_description(
            datasource_name="test_postgres",
            table_name="test_table",
            column_name="id",
            update_request=update_request,
            datasources_service=mock_datasources_service,
        )

        assert isinstance(result, ColumnResponse)
        assert result.name == "id"
        mock_datasources_service.update_datasource_table_catalog_column_description.assert_called_once_with(
            datasource_name="test_postgres",
            table_name="test_table",
            column_name="id",
            description="New column description",
        )

    @pytest.mark.asyncio
    async def test_update_column_description_column_not_found(self, mock_datasources_service):
        """Test update column description when column is not found."""
        from fastapi import HTTPException

        mock_datasources_service.update_datasource_table_catalog_column_description = AsyncMock(
            side_effect=DatasourceTableColumnNotFoundError("Column not found")
        )

        update_request = UpdateColumnDescriptionRequest(description="New column description")

        with pytest.raises(HTTPException) as exc_info:
            await update_column_description(
                datasource_name="test_postgres",
                table_name="test_table",
                column_name="missing_col",
                update_request=update_request,
                datasources_service=mock_datasources_service,
            )

        assert exc_info.value.status_code == 404
        assert "Column not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_update_column_description_column_not_cataloged(self, mock_datasources_service):
        """Test update column description when column is not cataloged."""
        from fastapi import HTTPException

        mock_datasources_service.update_datasource_table_catalog_column_description = AsyncMock(
            side_effect=DatasourceTableColumnNotCatalogedError("Column not cataloged")
        )

        update_request = UpdateColumnDescriptionRequest(description="New column description")

        with pytest.raises(HTTPException) as exc_info:
            await update_column_description(
                datasource_name="test_postgres",
                table_name="test_table",
                column_name="id",
                update_request=update_request,
                datasources_service=mock_datasources_service,
            )

        assert exc_info.value.status_code == 409
        assert "not cataloged" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_get_datasource_table_row_count_success(self, mock_datasources_service):
        """Test successful table row count retrieval."""
        mock_datasources_service.get_datasource_table_row_count = AsyncMock(return_value=42)

        result = await get_datasource_table_row_count(
            datasource_name="test_postgres", table_name="test_table", datasources_service=mock_datasources_service
        )

        assert result == 42
        mock_datasources_service.get_datasource_table_row_count.assert_called_once_with("test_postgres", "test_table")

    @pytest.mark.asyncio
    async def test_get_datasource_table_row_count_not_found(self, mock_datasources_service):
        """Test get table row count when datasource is not found."""
        from fastapi import HTTPException

        mock_datasources_service.get_datasource_table_row_count = AsyncMock(
            side_effect=DatasourceNotFoundError("Datasource 'test_postgres' not found")
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_datasource_table_row_count(
                datasource_name="test_postgres", table_name="test_table", datasources_service=mock_datasources_service
            )

        assert exc_info.value.status_code == 404
        assert "Datasource 'test_postgres' not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_datasource_table_row_count_service_error(self, mock_datasources_service):
        """Test get table row count with service error."""
        from fastapi import HTTPException

        mock_datasources_service.get_datasource_table_row_count = AsyncMock(
            side_effect=DatasourceServiceError("Failed to get row count")
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_datasource_table_row_count(
                datasource_name="test_postgres", table_name="test_table", datasources_service=mock_datasources_service
            )

        assert exc_info.value.status_code == 400
        assert "Failed to get row count" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_datasource_table_row_count_unexpected_error(self, mock_datasources_service):
        """Test get table row count with unexpected error."""
        from fastapi import HTTPException

        mock_datasources_service.get_datasource_table_row_count = AsyncMock(side_effect=Exception("Unexpected error"))

        with pytest.raises(HTTPException) as exc_info:
            await get_datasource_table_row_count(
                datasource_name="test_postgres", table_name="test_table", datasources_service=mock_datasources_service
            )

        assert exc_info.value.status_code == 500
        assert "Internal server error" in str(exc_info.value.detail)
