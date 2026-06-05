"""Tests for the /v1/models endpoint."""

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from minds.api.v1.router import api_router
from minds.common.constants import HEADER_ORGANIZATION_ID, HEADER_USER_ID
from minds.inference.types import ApiKind, PassthroughModelConfig, WebSearchMode


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(api_router)
    return TestClient(app)


@pytest.fixture
def auth_headers():
    return {
        HEADER_USER_ID: "00000000-0000-0000-0000-000000000001",
        HEADER_ORGANIZATION_ID: "00000000-0000-0000-0000-000000000002",
    }


@pytest.fixture
def stub_available_models():
    """Patch the resolver so the endpoint doesn't depend on env-configured provider keys."""
    configs = [
        PassthroughModelConfig(
            api_kind=ApiKind.ANTHROPIC_MESSAGES,
            model_name="claude-sonnet-test",
            api_key="k",
            web_search_mode=WebSearchMode.ANTHROPIC_NATIVE,
            label="anthropic",
            alias="sonnet",
        ),
        PassthroughModelConfig(
            api_kind=ApiKind.OPENAI_RESPONSES,
            model_name="gpt-test",
            api_key="k",
            web_search_mode=WebSearchMode.OPENAI_NATIVE,
            label="openai",
            alias="gpt-low",
            reasoning_effort="low",
        ),
    ]
    with patch(
        "minds.inference.model_resolver.ModelResolver.list_available",
        return_value=configs,
    ):
        yield configs


def test_list_models_returns_openai_list_shape(client, auth_headers, stub_available_models):
    response = client.get("/v1/models/", headers=auth_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert isinstance(body["data"], list)
    assert len(body["data"]) == 2

    sonnet = next(m for m in body["data"] if m["id"] == "latest:sonnet")
    assert sonnet["object"] == "model"
    assert sonnet["owned_by"] == "anthropic"
    assert sonnet["created"] == 0

    gpt = next(m for m in body["data"] if m["id"] == "latest:gpt-low")
    assert gpt["owned_by"] == "openai"


def test_list_models_excludes_deprecated_aliases(client, auth_headers, stub_available_models):
    # The endpoint relies on list_available_passthrough_models to filter; this
    # confirms no deprecated spelling leaks through the endpoint's id format.
    response = client.get("/v1/models/", headers=auth_headers)
    ids = [m["id"] for m in response.json()["data"]]
    assert "_reason_" not in ids
    assert "_code_" not in ids
    # And every id uses the canonical latest:* form.
    assert all(model_id.startswith("latest:") for model_id in ids)


def test_list_models_requires_auth(client, stub_available_models):
    response = client.get("/v1/models/")
    assert response.status_code == 401
