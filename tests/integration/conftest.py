"""Shared fixtures for inference-only integration tests."""

import os

import pytest
import requests

MINDS_API_BASE_URL = os.getenv("MINDS_API_BASE_URL")
MINDS_API_PREFIX = os.getenv("MINDS_API_PREFIX", "/v1")
MINDS_API_KEY = os.getenv("MINDS_API_KEY")


def _create_test_user_api_key():
    """Mint a test-user API key via the gateway's create_test_user endpoint."""
    gateway_internal_url = os.environ.get("GATEWAY_INTERNAL_URL")
    if not gateway_internal_url:
        pytest.fail("GATEWAY_INTERNAL_URL is not set, can't create a test user")

    response = requests.post(f"http://{gateway_internal_url}/cloud/create_test_user")
    if response.status_code != 200:
        pytest.fail(f"Failed to create test user: {response.text}")

    return response.json()["api_key"]


@pytest.fixture(scope="session")
def api_client():
    """requests.Session pre-configured with bearer auth for the minds API."""
    session = requests.Session()
    headers = {"Content-Type": "application/json"}
    api_key = MINDS_API_KEY or _create_test_user_api_key()
    headers["Authorization"] = f"Bearer {api_key}"
    session.headers.update(headers)
    yield session
    session.close()


@pytest.fixture(scope="session")
def passthrough_model(api_client):
    """Pick a configured passthrough alias from /v1/models.

    Skips the test if no providers are wired up in the target environment.
    """
    resp = api_client.get(f"{MINDS_API_BASE_URL}{MINDS_API_PREFIX}/models/", timeout=30)
    assert resp.status_code == 200, f"GET /models failed: {resp.status_code} {resp.text}"
    data = resp.json().get("data", [])
    if not data:
        pytest.skip("No passthrough models configured in target environment")
    # Prefer a cheap/fast model when available; otherwise take whatever is first.
    preferred = ("latest:gpt-mini", "latest:gpt-nano", "latest:haiku", "latest:gemini-flash")
    ids = [m["id"] for m in data]
    for p in preferred:
        if p in ids:
            return p
    return ids[0]
