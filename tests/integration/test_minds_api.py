import logging
import time
import pytest
from requests import Session
from .config import MINDS_API_BASE_URL
# States we expect a mind to pass through before completion
IN_PROGRESS_STATUSES = ["PENDING", "LOADING"]
MAX_RETRIES = 60   # e.g., 60 attempts
POLL_INTERVAL = 2  # seconds between polls

def poll_mind_transitions(api_client: Session, mind_name: str) -> dict:
    """
    Polls a MindsDB mind until it reaches a final state (COMPLETED or FAILED).
    Verifies that the mind passes through all expected intermediate states.

    Args:
        api_client (Session): requests.Session instance or compatible client.
        mind_name (str): name of the mind to poll.

    Returns:
        dict: JSON response of the mind in its final state.

    Raises:
        pytest.fail: if mind fails or does not pass through expected states.
    """
    observed_states = set()

    for attempt in range(1, MAX_RETRIES + 1):
        resp = api_client.get(f"{MINDS_API_BASE_URL}/api/v1/minds/{mind_name}")
        data = resp.json()
        status = data.get("status")

        if status in IN_PROGRESS_STATUSES:
            observed_states.add(status)

        logging.info(f"Attempt {attempt}: mind '{mind_name}' status={status}")
        print(f"Attempt {attempt}: mind '{mind_name}' status={status}")

        if status == "COMPLETED":
            missing = [s for s in IN_PROGRESS_STATUSES if s not in observed_states]
            if missing:
                pytest.fail(
                    f"Mind '{mind_name}' reached COMPLETED but did not pass through states: {missing}"
                )
            logging.info(f"SUCCESS: Mind '{mind_name}' reached COMPLETED with all transitions")
            return data

        if status == "FAILED":
            pytest.fail(
                f"Mind '{mind_name}' FAILED after {attempt} attempts. "
                f"Error: {data.get('error_message', 'N/A')}"
            )

        time.sleep(POLL_INTERVAL)

    pytest.fail(f"Mind '{mind_name}' did not reach COMPLETED within {MAX_RETRIES * POLL_INTERVAL} seconds")

@pytest.mark.happy_path
class TestMindsAPI:
    def test_mind_crud_workflow(self, api_client, temporary_mind):
        """
        Tests the full create, read, update, and delete (CRUD) lifecycle of a Mind.
        """
        mind_name = temporary_mind

        # 1. READ (GET) - Verify creation
        logging.info(f"TEST: Verifying creation of mind '{mind_name}'")
        get_resp = api_client.get(f"{MINDS_API_BASE_URL}/api/v1/minds/{mind_name}")
        assert get_resp.status_code == 200, f"Failed to get mind. Response: {get_resp.text}"
        assert get_resp.json()["name"] == mind_name
        logging.info(f"VERIFIED: Fetched mind '{mind_name}'")

        # 1.1 READ (GET) - Wait for mind to in final state
        mind_data = poll_mind_transitions(api_client, mind_name)
        assert mind_data["status"] == "COMPLETED"

        # 2. UPDATE (PUT) - Modify the mind's parameters
        logging.info(f"TEST: Updating mind '{mind_name}'")
        update_payload = {"parameters": {"temperature": 0.5}}
        update_resp = api_client.put(f"{MINDS_API_BASE_URL}/api/v1/minds/{mind_name}", json=update_payload)
        assert update_resp.status_code == 200, f"Failed to update mind. Response: {update_resp.text}"
        assert update_resp.json()["parameters"]["temperature"] == 0.5
        logging.info(f"VERIFIED: Updated mind '{mind_name}' successfully.")

    def test_list_minds(self, api_client, temporary_mind):
        """Tests the GET /api/v1/minds/ endpoint, with retries to handle eventual consistency."""
        logging.info("TEST: Listing all minds and searching for the new mind.")

        max_retries = 5
        delay_seconds = 2
        for attempt in range(max_retries):
            list_resp = api_client.get(f"{MINDS_API_BASE_URL}/api/v1/minds/")
            assert list_resp.status_code == 200, (
                f"Failed to list minds on attempt {attempt + 1}. Response: {list_resp.text}"
            )

            minds = list_resp.json()
            assert isinstance(minds, list)

            if temporary_mind in [mind["name"] for mind in minds]:
                logging.info(f"VERIFIED: Found '{temporary_mind}' in the list on attempt {attempt + 1}.")
                return  # Test passes
            message = (
                f"Attempt {attempt + 1}/{max_retries}: Mind '{temporary_mind}' not found. "
                f"Retrying in {delay_seconds}s..."
            )
            logging.warning(message)
            time.sleep(delay_seconds)

        pytest.fail(f"Mind '{temporary_mind}' was not found in the list after {max_retries} attempts.")
