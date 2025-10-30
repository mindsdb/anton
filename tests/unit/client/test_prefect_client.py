import pytest
from unittest.mock import AsyncMock, Mock, patch

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
