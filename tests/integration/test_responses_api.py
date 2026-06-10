"""Integration tests for /v1/responses (stateful conversations)."""

import pytest

from .conftest import MINDS_API_BASE_URL, MINDS_API_PREFIX


@pytest.mark.happy_path
def test_responses_create_new_conversation(api_client, passthrough_model):
    """First POST creates a conversation and returns an assistant response."""
    resp = api_client.post(
        f"{MINDS_API_BASE_URL}{MINDS_API_PREFIX}/responses/",
        json={
            "model": passthrough_model,
            "input": "Reply with the single word: pong",
            "stream": False,
        },
        timeout=60,
    )
    assert resp.status_code == 200, f"{resp.status_code}: {resp.text}"
    data = resp.json()
    assert data["choices"][0]["message"].get("content"), "no assistant content"
