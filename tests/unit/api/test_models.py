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
    from minds.schemas.passthrough import PassthroughModelStatsigConfig

    with (
        patch(
            "minds.api.v1.endpoints.models.get_passthrough_model_config",
            return_value=PassthroughModelStatsigConfig(),
        ),
        patch(
            "minds.inference.model_resolver.ModelResolver.list_available",
            return_value=configs,
        ),
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


@pytest.mark.parametrize("path", ["/v1/models", "/v1/models/"])
def test_list_models_serves_both_slash_variants_without_redirect(client, auth_headers, stub_available_models, path):
    """Both /v1/models and /v1/models/ return 200 directly (no 307 redirect).

    The OpenAI SDK's client.models.list() hits /v1/models with no trailing
    slash; relying on a redirect can drop auth headers behind some proxies.
    """
    response = client.get(path, headers=auth_headers, follow_redirects=False)
    assert response.status_code == 200
    assert response.json()["object"] == "list"


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


def test_list_models_threads_statsig_policy_into_listing(client, auth_headers):
    """The endpoint passes the per-user overrides/allow-list to the resolver."""
    from minds.schemas.passthrough import PassthroughModelStatsigConfig

    statsig_config = PassthroughModelStatsigConfig(
        alias_overrides={"opus": "claude-opus-4-8"}, allowed_aliases=["opus"]
    )
    with (
        patch(
            "minds.api.v1.endpoints.models.get_passthrough_model_config",
            return_value=statsig_config,
        ),
        patch(
            "minds.inference.model_resolver.ModelResolver.list_available",
            return_value=[],
        ) as mock_list,
    ):
        response = client.get("/v1/models/", headers=auth_headers)

    assert response.status_code == 200
    assert mock_list.call_args.kwargs["policy"] is statsig_config


def _config(alias, model_name, label, api_kind, reasoning_effort=None):
    return PassthroughModelConfig(
        api_kind=api_kind,
        model_name=model_name,
        api_key="k",
        label=label,
        alias=alias,
        reasoning_effort=reasoning_effort,
    )


def test_list_models_advertises_reasoning_efforts(client, auth_headers):
    """Effort-capable models carry reasoning_efforts + default; others omit both."""
    from minds.schemas.passthrough import PassthroughModelStatsigConfig

    configs = [
        _config("opus", "claude-opus-4-8", "anthropic", ApiKind.ANTHROPIC_MESSAGES),
        _config("haiku", "claude-haiku-4-5", "anthropic", ApiKind.ANTHROPIC_MESSAGES),
        _config("gpt", "gpt-5.5", "openai", ApiKind.OPENAI_RESPONSES, reasoning_effort="low"),
        _config("kimi", "accounts/fireworks/models/kimi-k2p6", "fireworks", ApiKind.FIREWORKS),
    ]
    with (
        patch(
            "minds.api.v1.endpoints.models.get_passthrough_model_config",
            return_value=PassthroughModelStatsigConfig(),
        ),
        patch("minds.inference.model_resolver.ModelResolver.list_available", return_value=configs),
    ):
        response = client.get("/v1/models/", headers=auth_headers)

    assert response.status_code == 200
    by_id = {m["id"]: m for m in response.json()["data"]}

    opus = by_id["latest:opus"]
    assert opus["reasoning_efforts"] == ["low", "medium", "high", "xhigh", "max"]
    assert opus["default_reasoning_effort"] == "high"

    # Alias-pinned effort wins over the model's provider-side default.
    gpt = by_id["latest:gpt"]
    assert gpt["reasoning_efforts"] == ["none", "low", "medium", "high", "xhigh"]
    assert gpt["default_reasoning_effort"] == "low"

    # No effort support → both keys omitted (UI shows picker iff present).
    for model_id in ("latest:haiku", "latest:kimi"):
        assert "reasoning_efforts" not in by_id[model_id]
        assert "default_reasoning_effort" not in by_id[model_id]


def test_list_models_applies_statsig_effort_overrides(client, auth_headers):
    """A Statsig effort_overrides entry reshapes the advertised levels — no deploy."""
    from minds.schemas.passthrough import PassthroughModelStatsigConfig

    configs = [_config("opus", "claude-opus-4-9", "anthropic", ApiKind.ANTHROPIC_MESSAGES)]
    statsig_config = PassthroughModelStatsigConfig(
        effort_overrides={"claude-opus-4-9": {"levels": ["low", "high", "ultra"], "default": "ultra"}}
    )
    with (
        patch(
            "minds.api.v1.endpoints.models.get_passthrough_model_config",
            return_value=statsig_config,
        ),
        patch("minds.inference.model_resolver.ModelResolver.list_available", return_value=configs),
    ):
        response = client.get("/v1/models/", headers=auth_headers)

    opus = next(m for m in response.json()["data"] if m["id"] == "latest:opus")
    assert opus["reasoning_efforts"] == ["low", "high", "ultra"]
    assert opus["default_reasoning_effort"] == "ultra"
