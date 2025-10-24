from unittest.mock import Mock, patch
from uuid import UUID

import pytest
from sqlmodel import Session

from minds.jobs.data_catalog_loader_flow import DataCatalogLoaderError
from minds.model.datasource import Datasource
from minds.model.mind_datasource import MindDatasource
from minds.services.data_catalog.data_catalog_loader import DataCatalogExecutionMode, DataCatalogLoader


class TestDataCatalogLoader:
    """Test cases for DataCatalogLoader class."""

    @pytest.fixture
    def mock_session(self):
        """Mock database session."""
        session = Mock(spec=Session)
        session.add_all = Mock()
        session.flush = Mock()
        session.commit = Mock()
        session.rollback = Mock()
        session.exec = Mock()
        session.add = Mock()
        return session

    @pytest.fixture
    def mock_mind_datasource(self):
        """Mock mind datasource."""
        mind_datasource = Mock(spec=MindDatasource)
        mind_datasource.id = UUID("12345678-1234-5678-1234-567812345678")
        mind_datasource.tenant_id = "test_tenant_456"
        mind_datasource.datasource_id = UUID("87654321-4321-8765-4321-876543218765")
        mind_datasource.flow_run_id = None

        # Mock datasource
        datasource = Mock(spec=Datasource)
        datasource.id = UUID("87654321-4321-8765-4321-876543218765")
        datasource.name = "test_datasource"
        mind_datasource.datasource = datasource

        return mind_datasource

    @pytest.fixture
    def data_catalog_loader(self, mock_session):
        """Create DataCatalogLoader instance with mocked dependencies."""
        return DataCatalogLoader(
            session=mock_session,
            tenant_id="test_tenant_456",
            user_id="test_user_123",
        )

    def test_initialization(self, mock_session):
        """Test DataCatalogLoader initialization."""
        loader = DataCatalogLoader(
            session=mock_session,
            tenant_id="test_tenant_456",
            user_id="test_user_123",
        )

        assert loader.session == mock_session
        assert loader.tenant_id == "test_tenant_456"
        assert loader.user_id == "test_user_123"

    @pytest.mark.asyncio
    async def test_load_sync_mode(self, data_catalog_loader, mock_mind_datasource):
        """Test load operation in synchronous mode."""
        with (
            patch("minds.services.data_catalog.data_catalog_loader.DATA_CATALOG_EXECUTION_MODE", "synchronous"),
            patch("minds.services.data_catalog.data_catalog_loader.load_data_catalog") as mock_load_flow,
        ):
            # Execute load
            await data_catalog_loader.load(mock_mind_datasource, ["table1", "table2"])

            # Verify the flow was called with correct parameters
            mock_load_flow.assert_called_once_with(
                mind_datasource_id=mock_mind_datasource.id,
                tenant_id="test_tenant_456",
                user_id="test_user_123",
                table_names=["table1", "table2"],
            )

    @pytest.mark.asyncio
    async def test_load_async_mode(self, data_catalog_loader, mock_mind_datasource):
        """Test load operation in asynchronous mode."""
        mock_flow_run = Mock()
        mock_flow_run.id = "flow-run-123"

        async def mock_run_deployment(*args, **kwargs):
            return mock_flow_run

        with (
            patch("minds.services.data_catalog.data_catalog_loader.DATA_CATALOG_EXECUTION_MODE", "asynchronous"),
            patch(
                "minds.services.data_catalog.data_catalog_loader.run_deployment", side_effect=mock_run_deployment
            ) as mock_run_deployment_patch,
        ):
            # Execute load
            await data_catalog_loader.load(mock_mind_datasource, ["table1", "table2"])

            # Verify the deployment was called with correct parameters
            mock_run_deployment_patch.assert_called_once_with(
                name="load-data-catalog/dev",
                timeout=0,
                parameters={
                    "mind_datasource_id": mock_mind_datasource.id,
                    "tenant_id": "test_tenant_456",
                    "user_id": "test_user_123",
                    "table_names": ["table1", "table2"],
                },
            )

            # Verify flow_run_id was set and session was committed
            assert mock_mind_datasource.flow_run_id == "flow-run-123"
            data_catalog_loader.session.add.assert_called_once_with(mock_mind_datasource)
            data_catalog_loader.session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_invalid_execution_mode(self, data_catalog_loader, mock_mind_datasource):
        """Test load operation with invalid execution mode."""
        with (
            patch("minds.services.data_catalog.data_catalog_loader.DATA_CATALOG_EXECUTION_MODE", "invalid_mode"),
            pytest.raises(ValueError, match="Invalid data catalog execution mode: invalid_mode"),
        ):
            await data_catalog_loader.load(mock_mind_datasource, ["table1", "table2"])

    @pytest.mark.asyncio
    async def test_load_with_none_table_names(self, data_catalog_loader, mock_mind_datasource):
        """Test load operation with None table_names."""
        with (
            patch("minds.services.data_catalog.data_catalog_loader.DATA_CATALOG_EXECUTION_MODE", "synchronous"),
            patch("minds.services.data_catalog.data_catalog_loader.load_data_catalog") as mock_load_flow,
        ):
            # Execute load with None table_names
            await data_catalog_loader.load(mock_mind_datasource, None)

            # Verify the flow was called with None table_names
            mock_load_flow.assert_called_once_with(
                mind_datasource_id=mock_mind_datasource.id,
                tenant_id="test_tenant_456",
                user_id="test_user_123",
                table_names=None,
            )

    @pytest.mark.asyncio
    async def test_load_async_mode_deployment_failure(self, data_catalog_loader, mock_mind_datasource):
        """Test load operation in asynchronous mode when deployment fails."""

        async def mock_run_deployment(*args, **kwargs):
            raise Exception("Deployment failed")

        with (
            patch("minds.services.data_catalog.data_catalog_loader.DATA_CATALOG_EXECUTION_MODE", "asynchronous"),
            patch("minds.services.data_catalog.data_catalog_loader.run_deployment", side_effect=mock_run_deployment),
            pytest.raises(Exception, match="Deployment failed"),
        ):
            # Execute load and expect exception
            await data_catalog_loader.load(mock_mind_datasource, ["table1", "table2"])

    @pytest.mark.asyncio
    async def test_load_sync_mode_flow_failure(self, data_catalog_loader, mock_mind_datasource):
        """Test load operation in synchronous mode when flow fails."""
        with (
            patch("minds.services.data_catalog.data_catalog_loader.DATA_CATALOG_EXECUTION_MODE", "synchronous"),
            patch("minds.services.data_catalog.data_catalog_loader.load_data_catalog") as mock_load_flow,
            pytest.raises(DataCatalogLoaderError, match="Flow execution failed"),
        ):
            mock_load_flow.side_effect = DataCatalogLoaderError("Flow execution failed")

            # Execute load and expect exception
            await data_catalog_loader.load(mock_mind_datasource, ["table1", "table2"])

    def test_data_catalog_execution_mode_enum(self):
        """Test DataCatalogExecutionMode enum values."""
        assert DataCatalogExecutionMode.ASYNC.value == "asynchronous"
        assert DataCatalogExecutionMode.SYNC.value == "synchronous"
