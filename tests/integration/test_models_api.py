"""Integration tests for /v1/models (OpenAI-compatible listing)."""

import pytest

from .conftest import MINDS_API_BASE_URL, MINDS_API_PREFIX


@pytest.mark.happy_path
def test_list_models_openai_shape(api_client):
    """GET /v1/models returns an OpenAI ListObject of configured aliases."""
    resp = api_client.get(f"{MINDS_API_BASE_URL}{MINDS_API_PREFIX}/models/", timeout=30)
    assert resp.status_code == 200, f"{resp.status_code}: {resp.text}"

    body = resp.json()
    assert body.get("object") == "list"
    assert isinstance(body.get("data"), list)

    for entry in body["data"]:
        assert entry["object"] == "model"
        assert entry["id"].startswith("latest:"), f"unexpected id: {entry['id']}"
        assert entry.get("owned_by"), "owned_by missing"
