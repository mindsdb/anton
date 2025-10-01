import logging
import time
import uuid

import pytest
import requests

from .config import AUTH_TOKEN, DATASOURCE_CONFIGS, MINDS_API_BASE_URL


@pytest.fixture(scope="session")
def api_client():
    """Creates a configured requests session that is reused across all tests."""
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    if AUTH_TOKEN:
        session.headers.update({"Authorization": f"Bearer {AUTH_TOKEN}"})
    return session


@pytest.fixture(scope="function", params=DATASOURCE_CONFIGS, ids=[c["engine"] for c in DATASOURCE_CONFIGS])
def temporary_datasource(request, api_client):
    """
    Parameterized fixture to create and clean up a temporary datasource for each configured engine.
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


@pytest.fixture(scope="function")
def temporary_mind(api_client, temporary_datasource):
    """
    Fixture to create a temporary mind LINKED to a temporary datasource.
    """
    ds_name, ds_config = temporary_datasource
    unique_name = f"test-mind-fixture-{uuid.uuid4()}"
    logging.info(f"SETUP: Creating mind '{unique_name}' linked to datasource '{ds_name}'...")

    payload = {"name": unique_name, "provider": "openai", "model_name": "gpt-4", "datasources": [{"name": ds_name}]}

    create_resp = api_client.post(f"{MINDS_API_BASE_URL}/api/v1/minds/", json=payload)
    assert create_resp.status_code == 201, f"FIXTURE FAILED: Could not create mind. Response: {create_resp.text}"

    # more time to fully initialize the data-aware mind before tests run.
    # logging.info("Pausing for 10 seconds to allow for mind initialization...")
    # time.sleep(10)

    yield unique_name

    logging.info(f"TEARDOWN: Deleting mind '{unique_name}'...")
    delete_resp = api_client.delete(f"{MINDS_API_BASE_URL}/api/v1/minds/{unique_name}")
    assert delete_resp.status_code in [204, 404], "FIXTURE FAILED: Could not delete mind."
