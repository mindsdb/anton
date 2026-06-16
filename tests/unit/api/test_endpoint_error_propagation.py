"""Regression tests: client-error HTTPExceptions must not be masked as 500.

The /chat/completions and /responses endpoints wrap their handler in a
catch-all ``except Exception`` that re-raises as HTTP 500. Intended client
errors (e.g. the 400 from reasoning-effort validation or an unknown
passthrough alias raised in ``ModelResolver.resolve``) are themselves
``HTTPException``s, so without an explicit re-raise they were being
swallowed and surfaced to the client as 500 with a ``"400: ..."`` detail.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from minds.api.v1.router import api_router
from minds.common.constants import HEADER_ORGANIZATION_ID, HEADER_USER_ID


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(api_router)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def auth_headers():
    return {
        HEADER_USER_ID: "00000000-0000-0000-0000-000000000001",
        HEADER_ORGANIZATION_ID: "00000000-0000-0000-0000-000000000002",
    }


def test_chat_completions_preserves_handler_400(client, auth_headers):
    """A 400 raised inside the chat handler reaches the client as 400."""
    with (
        patch("minds.api.v1.endpoints.chat.require_usage_available", new=AsyncMock()),
        patch("minds.api.v1.endpoints.chat.is_langfuse_enabled", return_value=True),
        patch(
            "minds.api.v1.endpoints.chat.chat_completions_request_handler",
            new=AsyncMock(side_effect=HTTPException(status_code=400, detail="Invalid reasoning effort 'banana'")),
        ),
    ):
        response = client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={"model": "latest:opus", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 400
    # And the detail is the original message, not a double-prefixed "400: ..."
    assert response.json()["detail"] == "Invalid reasoning effort 'banana'"


def test_chat_completions_still_maps_unexpected_errors_to_500(client, auth_headers):
    """A non-HTTPException is still wrapped as a 500 (catch-all preserved)."""
    with (
        patch("minds.api.v1.endpoints.chat.require_usage_available", new=AsyncMock()),
        patch("minds.api.v1.endpoints.chat.is_langfuse_enabled", return_value=True),
        patch(
            "minds.api.v1.endpoints.chat.chat_completions_request_handler",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ),
    ):
        response = client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={"model": "latest:opus", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "boom"


def test_responses_preserves_handler_400(client, auth_headers):
    """A 400 raised inside the responses handler reaches the client as 400."""
    with (
        patch("minds.api.v1.endpoints.responses.require_usage_available", new=AsyncMock()),
        patch("minds.api.v1.endpoints.responses.is_langfuse_enabled", return_value=True),
        patch(
            "minds.api.v1.endpoints.responses.responses_request_handler",
            new=AsyncMock(side_effect=HTTPException(status_code=400, detail="Invalid reasoning effort 'banana'")),
        ),
    ):
        response = client.post(
            "/v1/responses/",
            headers=auth_headers,
            json={"model": "latest:opus", "input": "hi", "reasoning": {"effort": "banana"}},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid reasoning effort 'banana'"
