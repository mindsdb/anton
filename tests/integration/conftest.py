# minds/tests/integration/conftest.py
import logging
import os
import time
import uuid

import pytest
import requests
from requests import Session

# ===================================================================
# EXPLICIT TEST CONFIGURATION
# ===================================================================

MINDS_API_BASE_URL = os.getenv("MINDS_API_BASE_URL")
MINDS_API_KEY = os.getenv("MINDS_API_KEY")

# --- DATASOURCE CONFIGURATIONS ---
DATASOURCE_CONFIGS = []

# Postgres Defaults (standardized for testing)
POSTGRES_CONFIG = {
    "host": os.getenv("PG_HOST", "samples.mindsdb.com"),
    "port": int(os.getenv("PG_PORT", 5432)),
    "user": os.getenv("PG_USER", "demo_user"),
    "password": os.getenv("PG_PASSWORD", "demo_password"),
    "database": os.getenv("PG_DB_NAME", "demo"),
    "schema": os.getenv("PG_SCHEMA", "demo"),
}
# Only use Postgres tests if we have a complete config
if all(POSTGRES_CONFIG.values()):
    DATASOURCE_CONFIGS.append(
        {
            "engine": "postgres",
            "name_prefix": "test-pg-ds",
            "connection_data": POSTGRES_CONFIG,
            "sample_table": "house_sales",
        }
    )


# ----------------------------------
# API client fixture
# ----------------------------------

def _create_test_user_api_key():
    """Uses the local endpoint to get a test user from a test env"""
    gateway_internal_url = os.environ.get("GATEWAY_INTERNAL_URL")
    if not gateway_internal_url:
        pytest.fail("GATEWAY_INTERNAL_URL is not set, can't create a test user")

    response = requests.post(f"http://{gateway_internal_url}/cloud/create_test_user")
    if response.status_code != 200:
        pytest.fail(f"Failed to create test user: {response.text}")

    return response.json()["api_key"]

@pytest.fixture(scope="session")
def api_client():
    """Returns a requests.Session configured for MindsDB API calls."""
    session = requests.Session()
    headers = {"Content-Type": "application/json"}
    if MINDS_API_KEY:
        headers["Authorization"] = f"Bearer {MINDS_API_KEY}"
    else:
        headers["Authorization"] = f"Bearer {_create_test_user_api_key()}"
    session.headers.update(headers)
    yield session
    session.close()


# ----------------------------------
# Polling function
# ----------------------------------
IN_PROGRESS_STATUSES = ["PENDING", "LOADING"]
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 60))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 2))


def poll_mind_transitions(api_client: Session, mind_name: str) -> dict:
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
            return data

        if status == "FAILED":
            pytest.fail(f"Mind '{mind_name}' FAILED. Error: {data.get('error_message')}")

        time.sleep(POLL_INTERVAL)

    pytest.fail(f"Mind '{mind_name}' did not complete within {MAX_RETRIES * POLL_INTERVAL}s")


# ----------------------------------
# Datasource fixture
# ----------------------------------
@pytest.fixture(scope="function", params=DATASOURCE_CONFIGS, ids=[c["engine"] for c in DATASOURCE_CONFIGS])
def temporary_datasource(request, api_client: Session):
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
        "datasources": [{"name": ds_name, "tables": [config["sample_table"]]}],
    }

    create_resp = api_client.post(f"{MINDS_API_BASE_URL}/api/v1/minds/", json=payload)
    assert create_resp.status_code == 201, f"FIXTURE FAILED: Could not create mind. {create_resp.text}"

    poll_mind_transitions(api_client, unique_name)
    yield unique_name, temporary_datasource

    api_client.delete(f"{MINDS_API_BASE_URL}/api/v1/minds/{unique_name}")
