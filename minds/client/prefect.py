from prefect import states
from prefect.client.orchestration import get_client

from minds.common.logger import setup_logging

logger = setup_logging()


class PrefectClient:
    """
    Client for Prefect.
    """

    async def get_flow_run_state(self, flow_run_id: str) -> states.State:
        """
        Get the state of a flow run.
        """
        logger.debug(f"Getting state for flow run {flow_run_id}")
        async with get_client() as client:
            state = await client.read_flow_run_states(flow_run_id=flow_run_id)
        logger.debug(f"Flow run {flow_run_id} is in state {state}")
        return state

    async def cancel_flow_run(self, flow_run_id: str) -> None:
        """
        Cancel a flow run.
        """
        logger.debug(f"Cancelling flow run {flow_run_id}")
        async with get_client() as client:
            await client.set_flow_run_state(flow_run_id=flow_run_id, state=states.Cancelled())
        logger.debug(f"Flow run {flow_run_id} cancelled successfully")
