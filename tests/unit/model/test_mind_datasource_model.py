"""
Unit tests for MindDatasource model and related status models.

Tests the SQLModel classes including:
- DataCatalogStatus enum
- DataCatalogTaskStatus model
- DetailedDataCatalogStatus model
- MindDatasource model (creation, status computation, state conversion)
"""

from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

import pytest
from prefect import states

from minds.model.data_catalog import Table  # noqa

# Import related models to ensure SQLAlchemy can resolve all relationships
from minds.model.datasource import Datasource  # noqa
from minds.model.mind import Mind  # noqa
from minds.model.mind_datasource import (
    DataCatalogStatus,
    DataCatalogTaskStatus,
    DetailedDataCatalogStatus,
    MindDatasource,
)
from minds.model.mind_datasource_table import MindDatasourceTable  # noqa


class TestDataCatalogStatus:
    """Test suite for DataCatalogStatus enum."""

    def test_enum_values(self):
        """Test that all expected enum values exist."""
        assert DataCatalogStatus.PENDING == "PENDING"
        assert DataCatalogStatus.LOADING == "LOADING"
        assert DataCatalogStatus.COMPLETED == "COMPLETED"
        assert DataCatalogStatus.FAILED == "FAILED"
        assert DataCatalogStatus.CANCELLED == "CANCELLED"

    def test_enum_string_representation(self):
        """Test enum string representation."""
        # Enum string representation includes the enum class name
        assert "PENDING" in str(DataCatalogStatus.PENDING)
        assert "COMPLETED" in str(DataCatalogStatus.COMPLETED)
        # The value property gives just the string value
        assert DataCatalogStatus.PENDING.value == "PENDING"
        assert DataCatalogStatus.COMPLETED.value == "COMPLETED"


class TestDataCatalogTaskStatus:
    """Test suite for DataCatalogTaskStatus model."""

    def test_task_status_creation(self):
        """Test creating a DataCatalogTaskStatus with name and status."""
        task_status = DataCatalogTaskStatus(name="load_tables", status=DataCatalogStatus.LOADING)

        assert task_status.name == "load_tables"
        assert task_status.status == DataCatalogStatus.LOADING

    def test_task_status_with_different_statuses(self):
        """Test creating task statuses with different status values."""
        statuses = [
            DataCatalogTaskStatus(name="task1", status=DataCatalogStatus.PENDING),
            DataCatalogTaskStatus(name="task2", status=DataCatalogStatus.LOADING),
            DataCatalogTaskStatus(name="task3", status=DataCatalogStatus.COMPLETED),
            DataCatalogTaskStatus(name="task4", status=DataCatalogStatus.FAILED),
            DataCatalogTaskStatus(name="task5", status=DataCatalogStatus.CANCELLED),
        ]

        for task_status in statuses:
            assert task_status.name.startswith("task")
            assert task_status.status in DataCatalogStatus


class TestDetailedDataCatalogStatus:
    """Test suite for DetailedDataCatalogStatus model."""

    def test_detailed_status_creation_with_defaults(self):
        """Test creating DetailedDataCatalogStatus with default values."""
        detailed_status = DetailedDataCatalogStatus()

        assert detailed_status.tasks == []
        assert detailed_status.progress == 0.0
        assert detailed_status.overall_status == DataCatalogStatus.PENDING

    def test_detailed_status_creation_with_all_fields(self):
        """Test creating DetailedDataCatalogStatus with all fields specified."""
        tasks = [
            DataCatalogTaskStatus(name="task1", status=DataCatalogStatus.COMPLETED),
            DataCatalogTaskStatus(name="task2", status=DataCatalogStatus.LOADING),
        ]

        detailed_status = DetailedDataCatalogStatus(tasks=tasks, progress=0.5, overall_status=DataCatalogStatus.LOADING)

        assert len(detailed_status.tasks) == 2
        assert detailed_status.tasks[0].name == "task1"
        assert detailed_status.tasks[1].name == "task2"
        assert detailed_status.progress == 0.5
        assert detailed_status.overall_status == DataCatalogStatus.LOADING

    def test_detailed_status_with_completed_status(self):
        """Test DetailedDataCatalogStatus with completed status."""
        tasks = [
            DataCatalogTaskStatus(name="task1", status=DataCatalogStatus.COMPLETED),
            DataCatalogTaskStatus(name="task2", status=DataCatalogStatus.COMPLETED),
        ]

        detailed_status = DetailedDataCatalogStatus(
            tasks=tasks, progress=1.0, overall_status=DataCatalogStatus.COMPLETED
        )

        assert detailed_status.progress == 1.0
        assert detailed_status.overall_status == DataCatalogStatus.COMPLETED
        assert all(task.status == DataCatalogStatus.COMPLETED for task in detailed_status.tasks)


class TestMindDatasource:
    """Test suite for MindDatasource model."""

    @pytest.fixture
    def sample_mind_datasource_data(self):
        """Sample mind datasource data for testing."""
        return {
            "mind_id": UUID("00000000-0000-0000-0000-000000000001"),
            "datasource_id": UUID("00000000-0000-0000-0000-000000000002"),
            "flow_run_id": UUID("00000000-0000-0000-0000-000000000003"),
        }

    def test_mind_datasource_creation_with_all_fields(self, sample_mind_datasource_data):
        """Test creating a MindDatasource with all fields."""
        mind_datasource = MindDatasource(**sample_mind_datasource_data)

        assert mind_datasource.mind_id == UUID("00000000-0000-0000-0000-000000000001")
        assert mind_datasource.datasource_id == UUID("00000000-0000-0000-0000-000000000002")
        assert mind_datasource.flow_run_id == UUID("00000000-0000-0000-0000-000000000003")

    def test_mind_datasource_creation_without_flow_run_id(self):
        """Test creating a MindDatasource without flow_run_id."""
        mind_datasource = MindDatasource(
            mind_id=UUID("00000000-0000-0000-0000-000000000001"),
            datasource_id=UUID("00000000-0000-0000-0000-000000000002"),
        )

        assert mind_datasource.mind_id == UUID("00000000-0000-0000-0000-000000000001")
        assert mind_datasource.datasource_id == UUID("00000000-0000-0000-0000-000000000002")
        assert mind_datasource.flow_run_id is None

    def test_mind_datasource_repr(self, sample_mind_datasource_data):
        """Test the string representation of MindDatasource."""
        mind_datasource = MindDatasource(**sample_mind_datasource_data)
        repr_str = repr(mind_datasource)

        assert "MindDatasource" in repr_str
        assert str(sample_mind_datasource_data["mind_id"]) in repr_str
        assert str(sample_mind_datasource_data["datasource_id"]) in repr_str

    def test_prefect_state_to_data_catalog_status_running(self):
        """Test conversion of running Prefect state to LOADING."""
        mock_state = Mock(spec=states.State)
        mock_state.is_running = Mock(return_value=True)
        mock_state.is_completed = Mock(return_value=False)
        mock_state.is_failed = Mock(return_value=False)
        mock_state.is_crashed = Mock(return_value=False)
        mock_state.is_cancelling = Mock(return_value=False)
        mock_state.is_cancelled = Mock(return_value=False)

        mind_datasource = MindDatasource(
            mind_id=UUID("00000000-0000-0000-0000-000000000001"),
            datasource_id=UUID("00000000-0000-0000-0000-000000000002"),
        )

        result = mind_datasource._prefect_state_to_data_catalog_status(mock_state)
        assert result == DataCatalogStatus.LOADING

    def test_prefect_state_to_data_catalog_status_completed(self):
        """Test conversion of completed Prefect state to COMPLETED."""
        mock_state = Mock(spec=states.State)
        mock_state.is_running = Mock(return_value=False)
        mock_state.is_completed = Mock(return_value=True)
        mock_state.is_failed = Mock(return_value=False)
        mock_state.is_crashed = Mock(return_value=False)
        mock_state.is_cancelling = Mock(return_value=False)
        mock_state.is_cancelled = Mock(return_value=False)

        mind_datasource = MindDatasource(
            mind_id=UUID("00000000-0000-0000-0000-000000000001"),
            datasource_id=UUID("00000000-0000-0000-0000-000000000002"),
        )

        result = mind_datasource._prefect_state_to_data_catalog_status(mock_state)
        assert result == DataCatalogStatus.COMPLETED

    def test_prefect_state_to_data_catalog_status_failed(self):
        """Test conversion of failed Prefect state to FAILED."""
        mock_state = Mock(spec=states.State)
        mock_state.is_running = Mock(return_value=False)
        mock_state.is_completed = Mock(return_value=False)
        mock_state.is_failed = Mock(return_value=True)
        mock_state.is_crashed = Mock(return_value=False)
        mock_state.is_cancelling = Mock(return_value=False)
        mock_state.is_cancelled = Mock(return_value=False)

        mind_datasource = MindDatasource(
            mind_id=UUID("00000000-0000-0000-0000-000000000001"),
            datasource_id=UUID("00000000-0000-0000-0000-000000000002"),
        )

        result = mind_datasource._prefect_state_to_data_catalog_status(mock_state)
        assert result == DataCatalogStatus.FAILED

    def test_prefect_state_to_data_catalog_status_crashed(self):
        """Test conversion of crashed Prefect state to FAILED."""
        mock_state = Mock(spec=states.State)
        mock_state.is_running = Mock(return_value=False)
        mock_state.is_completed = Mock(return_value=False)
        mock_state.is_failed = Mock(return_value=False)
        mock_state.is_crashed = Mock(return_value=True)
        mock_state.is_cancelling = Mock(return_value=False)
        mock_state.is_cancelled = Mock(return_value=False)

        mind_datasource = MindDatasource(
            mind_id=UUID("00000000-0000-0000-0000-000000000001"),
            datasource_id=UUID("00000000-0000-0000-0000-000000000002"),
        )

        result = mind_datasource._prefect_state_to_data_catalog_status(mock_state)
        assert result == DataCatalogStatus.FAILED

    def test_prefect_state_to_data_catalog_status_cancelling(self):
        """Test conversion of cancelling Prefect state to CANCELLED."""
        mock_state = Mock(spec=states.State)
        mock_state.is_running = Mock(return_value=False)
        mock_state.is_completed = Mock(return_value=False)
        mock_state.is_failed = Mock(return_value=False)
        mock_state.is_crashed = Mock(return_value=False)
        mock_state.is_cancelling = Mock(return_value=True)
        mock_state.is_cancelled = Mock(return_value=False)

        mind_datasource = MindDatasource(
            mind_id=UUID("00000000-0000-0000-0000-000000000001"),
            datasource_id=UUID("00000000-0000-0000-0000-000000000002"),
        )

        result = mind_datasource._prefect_state_to_data_catalog_status(mock_state)
        assert result == DataCatalogStatus.CANCELLED

    def test_prefect_state_to_data_catalog_status_cancelled(self):
        """Test conversion of cancelled Prefect state to CANCELLED."""
        mock_state = Mock(spec=states.State)
        mock_state.is_running = Mock(return_value=False)
        mock_state.is_completed = Mock(return_value=False)
        mock_state.is_failed = Mock(return_value=False)
        mock_state.is_crashed = Mock(return_value=False)
        mock_state.is_cancelling = Mock(return_value=False)
        mock_state.is_cancelled = Mock(return_value=True)

        mind_datasource = MindDatasource(
            mind_id=UUID("00000000-0000-0000-0000-000000000001"),
            datasource_id=UUID("00000000-0000-0000-0000-000000000002"),
        )

        result = mind_datasource._prefect_state_to_data_catalog_status(mock_state)
        assert result == DataCatalogStatus.CANCELLED

    def test_prefect_state_to_data_catalog_status_pending(self):
        """Test conversion of unknown Prefect state to PENDING (default)."""
        mock_state = Mock(spec=states.State)
        mock_state.is_running = Mock(return_value=False)
        mock_state.is_completed = Mock(return_value=False)
        mock_state.is_failed = Mock(return_value=False)
        mock_state.is_crashed = Mock(return_value=False)
        mock_state.is_cancelling = Mock(return_value=False)
        mock_state.is_cancelled = Mock(return_value=False)

        mind_datasource = MindDatasource(
            mind_id=UUID("00000000-0000-0000-0000-000000000001"),
            datasource_id=UUID("00000000-0000-0000-0000-000000000002"),
        )

        result = mind_datasource._prefect_state_to_data_catalog_status(mock_state)
        assert result == DataCatalogStatus.PENDING

    @pytest.mark.asyncio
    async def test_status_property_with_flow_run_id(self, sample_mind_datasource_data):
        """Test status property when flow_run_id is present."""
        flow_run_id = sample_mind_datasource_data["flow_run_id"]

        # Create mock task states
        mock_task_state1 = Mock(spec=states.State)
        mock_task_state1.is_completed = Mock(return_value=True)
        mock_task_state2 = Mock(spec=states.State)
        mock_task_state2.is_completed = Mock(return_value=False)

        task_states = {"task1": mock_task_state1, "task2": mock_task_state2}

        # Create mock flow run state (list of states, we use the last one)
        mock_flow_state = Mock(spec=states.State)
        mock_flow_state.is_running = Mock(return_value=True)
        mock_flow_state.is_completed = Mock(return_value=False)
        mock_flow_state.is_failed = Mock(return_value=False)
        mock_flow_state.is_crashed = Mock(return_value=False)
        mock_flow_state.is_cancelling = Mock(return_value=False)
        mock_flow_state.is_cancelled = Mock(return_value=False)
        flow_run_states = [mock_flow_state]

        # Mock PrefectClient
        mock_prefect_client = Mock()
        mock_prefect_client.get_flow_run_task_states = AsyncMock(return_value=task_states)
        mock_prefect_client.get_flow_run_state = AsyncMock(return_value=flow_run_states)

        mind_datasource = MindDatasource(**sample_mind_datasource_data)

        with patch("minds.client.prefect.PrefectClient", return_value=mock_prefect_client):
            status = await mind_datasource.status

        assert isinstance(status, DetailedDataCatalogStatus)
        assert status.overall_status == DataCatalogStatus.LOADING
        assert status.progress == 0.5  # 1 out of 2 tasks completed
        assert len(status.tasks) == 2
        assert status.tasks[0].name == "task1"
        assert status.tasks[1].name == "task2"

        # Verify PrefectClient methods were called
        mock_prefect_client.get_flow_run_task_states.assert_awaited_once_with(flow_run_id)
        mock_prefect_client.get_flow_run_state.assert_awaited_once_with(str(flow_run_id))

    @pytest.mark.asyncio
    async def test_status_property_without_flow_run_id(self):
        """Test status property when flow_run_id is None."""
        mind_datasource = MindDatasource(
            mind_id=UUID("00000000-0000-0000-0000-000000000001"),
            datasource_id=UUID("00000000-0000-0000-0000-000000000002"),
            flow_run_id=None,
        )

        # Mock PrefectClient (should not be called)
        mock_prefect_client = Mock()
        mock_prefect_client.get_flow_run_task_states = AsyncMock()
        mock_prefect_client.get_flow_run_state = AsyncMock()

        with patch("minds.client.prefect.PrefectClient", return_value=mock_prefect_client):
            # When flow_run_id is None, the method doesn't enter the if block and returns None
            status = await mind_datasource.status

        # The method returns None when flow_run_id is None (no return statement in that branch)
        assert status is None

        # The PrefectClient should not be called when flow_run_id is None
        mock_prefect_client.get_flow_run_task_states.assert_not_awaited()
        mock_prefect_client.get_flow_run_state.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_status_property_with_all_completed_tasks(self, sample_mind_datasource_data):
        """Test status property when all tasks are completed."""
        # Create mock task states - all completed
        mock_task_state1 = Mock(spec=states.State)
        mock_task_state1.is_completed = Mock(return_value=True)
        mock_task_state2 = Mock(spec=states.State)
        mock_task_state2.is_completed = Mock(return_value=True)

        task_states = {"task1": mock_task_state1, "task2": mock_task_state2}

        # Create mock flow run state - completed
        mock_flow_state = Mock(spec=states.State)
        mock_flow_state.is_running = Mock(return_value=False)
        mock_flow_state.is_completed = Mock(return_value=True)
        mock_flow_state.is_failed = Mock(return_value=False)
        mock_flow_state.is_crashed = Mock(return_value=False)
        mock_flow_state.is_cancelling = Mock(return_value=False)
        mock_flow_state.is_cancelled = Mock(return_value=False)
        flow_run_states = [mock_flow_state]

        # Mock PrefectClient
        mock_prefect_client = Mock()
        mock_prefect_client.get_flow_run_task_states = AsyncMock(return_value=task_states)
        mock_prefect_client.get_flow_run_state = AsyncMock(return_value=flow_run_states)

        mind_datasource = MindDatasource(**sample_mind_datasource_data)

        with patch("minds.client.prefect.PrefectClient", return_value=mock_prefect_client):
            status = await mind_datasource.status

        assert isinstance(status, DetailedDataCatalogStatus)
        assert status.overall_status == DataCatalogStatus.COMPLETED
        assert status.progress == 1.0  # All tasks completed
        assert len(status.tasks) == 2

    @pytest.mark.asyncio
    async def test_status_property_with_failed_flow(self, sample_mind_datasource_data):
        """Test status property when flow run has failed."""
        # Create mock task states
        mock_task_state = Mock(spec=states.State)
        mock_task_state.is_completed = Mock(return_value=False)
        task_states = {"task1": mock_task_state}

        # Create mock flow run state - failed
        mock_flow_state = Mock(spec=states.State)
        mock_flow_state.is_running = Mock(return_value=False)
        mock_flow_state.is_completed = Mock(return_value=False)
        mock_flow_state.is_failed = Mock(return_value=True)
        mock_flow_state.is_crashed = Mock(return_value=False)
        mock_flow_state.is_cancelling = Mock(return_value=False)
        mock_flow_state.is_cancelled = Mock(return_value=False)
        flow_run_states = [mock_flow_state]

        # Mock PrefectClient
        mock_prefect_client = Mock()
        mock_prefect_client.get_flow_run_task_states = AsyncMock(return_value=task_states)
        mock_prefect_client.get_flow_run_state = AsyncMock(return_value=flow_run_states)

        mind_datasource = MindDatasource(**sample_mind_datasource_data)

        with patch("minds.client.prefect.PrefectClient", return_value=mock_prefect_client):
            status = await mind_datasource.status

        assert isinstance(status, DetailedDataCatalogStatus)
        assert status.overall_status == DataCatalogStatus.FAILED
        assert status.progress == 0.0  # No tasks completed
        assert len(status.tasks) == 1

    @pytest.mark.asyncio
    async def test_status_property_with_empty_task_states(self, sample_mind_datasource_data):
        """Test status property when there are no task states."""
        # Empty task states
        task_states = {}

        # Create mock flow run state
        mock_flow_state = Mock(spec=states.State)
        mock_flow_state.is_running = Mock(return_value=True)
        mock_flow_state.is_completed = Mock(return_value=False)
        mock_flow_state.is_failed = Mock(return_value=False)
        mock_flow_state.is_crashed = Mock(return_value=False)
        mock_flow_state.is_cancelling = Mock(return_value=False)
        mock_flow_state.is_cancelled = Mock(return_value=False)
        flow_run_states = [mock_flow_state]

        # Mock PrefectClient
        mock_prefect_client = Mock()
        mock_prefect_client.get_flow_run_task_states = AsyncMock(return_value=task_states)
        mock_prefect_client.get_flow_run_state = AsyncMock(return_value=flow_run_states)

        mind_datasource = MindDatasource(**sample_mind_datasource_data)

        with patch("minds.client.prefect.PrefectClient", return_value=mock_prefect_client):
            status = await mind_datasource.status

        assert isinstance(status, DetailedDataCatalogStatus)
        assert status.overall_status == DataCatalogStatus.LOADING
        assert status.progress == 0.0  # No tasks, so progress is 0
        assert len(status.tasks) == 0
