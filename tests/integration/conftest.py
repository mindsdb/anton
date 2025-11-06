# minds/tests/integration/conftest.py
import logging
import os
import time
import uuid

import pytest
import requests
from requests import Session

from .config import DATASOURCE_CONFIGS, MINDS_API_BASE_URL, MINDS_API_KEY


# ----------------------------------
# API client fixture
# ----------------------------------
@pytest.fixture(scope="session")
def api_client():
    """Returns a requests.Session configured for MindsDB API calls."""
    session = requests.Session()
    headers = {"Content-Type": "application/json"}
    if MINDS_API_KEY:
        headers["Authorization"] = f"Bearer {MINDS_API_KEY}"
    session.headers.update(headers)
    yield session
    session.close()


# ----------------------------------
# Polling function
# ----------------------------------
IN_PROGRESS_STATUSES = ["PENDING", "LOADING"]
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 60))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 2))  # seconds


def poll_mind_transitions(api_client: Session, mind_name: str) -> dict:
    """
    Polls a MindsDB mind until it reaches a final state (COMPLETED or FAILED).
    Observes intermediate states, but does not fail if skipped (fast completion).
    """
    observed_states = set()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = api_client.get(f"{MINDS_API_BASE_URL}/api/v1/minds/{mind_name}")
            resp.raise_for_status()
        except Exception as e:
            logging.warning(f"Attempt {attempt}: HTTP error for mind '{mind_name}': {e}")
            time.sleep(POLL_INTERVAL)
            continue

        data = resp.json()
        status = data.get("status")
        if status in IN_PROGRESS_STATUSES:
            observed_states.add(status)

        logging.info(f"Attempt {attempt}: mind '{mind_name}' status={status}")

        if status == "COMPLETED":
            missing = [s for s in IN_PROGRESS_STATUSES if s not in observed_states]
            if missing:
                logging.warning(f"Mind '{mind_name}' reached COMPLETED but skipped states: {missing}")
            logging.info(f"SUCCESS: Mind '{mind_name}' reached COMPLETED with states: {observed_states}")
            return data

        if status == "FAILED":
            pytest.fail(
                f"Mind '{mind_name}' FAILED after {attempt} attempts. Error: {data.get('error_message', 'N/A')}"
            )

        time.sleep(POLL_INTERVAL)

    pytest.fail(f"Mind '{mind_name}' did not reach COMPLETED within {MAX_RETRIES * POLL_INTERVAL} seconds")


# ----------------------------------
# Datasource fixture
# ----------------------------------
@pytest.fixture(scope="function", params=DATASOURCE_CONFIGS, ids=[c["engine"] for c in DATASOURCE_CONFIGS])
def temporary_datasource(request, api_client: Session):
    """
    Create and cleanup a temporary datasource for each configured engine.
    Returns: (datasource_name, config)
    """
    config = request.param
    unique_name = f"{config['name_prefix']}-{uuid.uuid4()}"
    logging.info(f"SETUP: Creating {config['engine']} datasource '{unique_name}'...")

    payload = {"name": unique_name, "engine": config["engine"], "connection_data": config["connection_data"]}
    create_resp = api_client.post(f"{MINDS_API_BASE_URL}/api/v1/datasources/", json=payload)
    assert create_resp.status_code == 201, f"FIXTURE FAILED: Could not create datasource. Response: {create_resp.text}"

    yield unique_name, config

    logging.info(f"TEARDOWN: Deleting datasource '{unique_name}'...")
    delete_resp = api_client.delete(f"{MINDS_API_BASE_URL}/api/v1/datasources/{unique_name}")
    assert delete_resp.status_code in [204, 404], "FIXTURE FAILED: Could not delete datasource."


# ----------------------------------
# Mind fixture
# ----------------------------------
@pytest.fixture(scope="function")
def temporary_mind(api_client: Session, temporary_datasource):
    ds_name, config = temporary_datasource
    unique_name = f"test-mind-fixture-{uuid.uuid4()}"
    logging.info(f"SETUP: Creating mind '{unique_name}' linked to datasource '{ds_name}'...")

    payload = {
        "name": unique_name,
        "provider": "openai",
        "model_name": "gpt-4",
        "parameters": {},
        "datasources": [
            {
                "name": ds_name,
                "tables": [config["sample_table"]],  # Must exist in the datasource
            }
        ],
    }

    create_resp = api_client.post(f"{MINDS_API_BASE_URL}/api/v1/minds/", json=payload)
    assert create_resp.status_code == 201, f"FIXTURE FAILED: Could not create mind. Response: {create_resp.text}"

    poll_mind_transitions(api_client, unique_name)

    # Yield both mind_name and datasource info
    yield unique_name, temporary_datasource

    logging.info(f"TEARDOWN: Deleting mind '{unique_name}'...")
    delete_resp = api_client.delete(f"{MINDS_API_BASE_URL}/api/v1/minds/{unique_name}")
    assert delete_resp.status_code in [204, 404], "FIXTURE FAILED: Could not delete mind."
