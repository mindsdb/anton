from unittest.mock import AsyncMock, Mock, patch

import pytest

from minds.client.prefect import PrefectClient


class _DummyAsyncContextManager:
    def __init__(self, obj):
        self._obj = obj

    async def __aenter__(self):
        return self._obj

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_get_flow_run_state_uses_prefect_client():
    mock_state = Mock()
    mock_client = Mock()
    mock_client.read_flow_run_states = AsyncMock(return_value=mock_state)

    with patch("minds.client.prefect.get_client", return_value=_DummyAsyncContextManager(mock_client)):
        pc = PrefectClient()
        result = await pc.get_flow_run_state("flow-123")

    assert result is mock_state
    mock_client.read_flow_run_states.assert_awaited_once_with(flow_run_id="flow-123")


@pytest.mark.asyncio
async def test_get_flow_run_task_states_returns_task_name_to_state_mapping():
    # Create mock task runs with name and state attributes
    mock_task1 = Mock()
    mock_task1.name = "task-1"
    mock_state1 = Mock()
    mock_task1.state = mock_state1

    mock_task2 = Mock()
    mock_task2.name = "task-2"
    mock_state2 = Mock()
    mock_task2.state = mock_state2

    mock_task_runs = [mock_task1, mock_task2]

    mock_client = Mock()
    mock_client.read_task_runs = AsyncMock(return_value=mock_task_runs)

    # Use a valid UUID format for the flow_run_id
    flow_run_id = "12345678-1234-5678-1234-567812345678"

    with patch("minds.client.prefect.get_client", return_value=_DummyAsyncContextManager(mock_client)):
        pc = PrefectClient()
        result = await pc.get_flow_run_task_states(flow_run_id)

    # Verify the result is a dictionary mapping task names to states
    assert isinstance(result, dict)
    assert len(result) == 2
    assert result["task-1"] is mock_state1
    assert result["task-2"] is mock_state2

    # Verify the client method was called with the correct filter
    mock_client.read_task_runs.assert_awaited_once()
    called_args, called_kwargs = mock_client.read_task_runs.await_args
    flow_run_filter = called_kwargs.get("flow_run_filter")
    assert flow_run_filter is not None
    # Verify the filter has the correct flow_run_id (may be stored as UUID object)
    filter_flow_run_id = flow_run_filter.id.any_[0]
    assert str(filter_flow_run_id) == flow_run_id


@pytest.mark.asyncio
async def test_cancel_flow_run_sets_cancelled_state():
    mock_client = Mock()
    mock_client.set_flow_run_state = AsyncMock(return_value=None)

    with patch("minds.client.prefect.get_client", return_value=_DummyAsyncContextManager(mock_client)):
        pc = PrefectClient()
        await pc.cancel_flow_run("flow-xyz")

    # Assert the client was asked to set the flow run state and that a Cancelled state was provided
    mock_client.set_flow_run_state.assert_awaited_once()
    called_args, called_kwargs = mock_client.set_flow_run_state.await_args
    # first positional arg should be flow_run_id keyword in our implementation; check kwargs
    assert called_kwargs.get("flow_run_id") == "flow-xyz"
    state_arg = called_kwargs.get("state")
    assert state_arg is not None
    # Ensure the provided state represents a Prefect Cancelled state.
    # Different Prefect versions may represent Cancelled either as a subclass
    # named 'Cancelled' or as a State instance with name == 'Cancelled',
    # so accept either form for robustness.
    clsname = state_arg.__class__.__name__
    name = getattr(state_arg, "name", None)
    assert clsname == "Cancelled" or name == "Cancelled"
