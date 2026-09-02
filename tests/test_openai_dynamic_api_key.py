from __future__ import annotations

from unittest.mock import patch

import httpx
import openai
import pytest

from anton.core.llm.client import LLMClient
from anton.core.llm.openai import OpenAIProvider
from anton.core.llm.provider import EndpointConfigurationError


async def test_live_requests_reread_api_key_without_rebuilding_provider():
    current_token = "token-a"
    authorization_headers: list[str] = []

    async def api_key_provider() -> str:
        return current_token

    def handler(request: httpx.Request) -> httpx.Response:
        authorization_headers.append(request.headers["authorization"])
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    real_client = openai.AsyncOpenAI
    transport = httpx.MockTransport(handler)

    def build_client(**kwargs):
        kwargs["http_client"] = httpx.AsyncClient(transport=transport)
        return real_client(**kwargs)

    with patch(
        "anton.core.llm.openai.openai.AsyncOpenAI", side_effect=build_client
    ):
        provider = OpenAIProvider(
            api_key="token-a",
            api_key_provider=api_key_provider,
            base_url="https://gateway.test/v1",
        )

    try:
        await provider.complete(model="test-model", system="sys", messages=[])
        current_token = "token-b"
        await provider.complete(model="test-model", system="sys", messages=[])
    finally:
        await provider._client.close()

    assert authorization_headers == ["Bearer token-a", "Bearer token-b"]
    assert provider.export_connection_info().api_key == "token-a"


def _completion_body() -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "done"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


async def test_a_401_on_the_stale_token_is_confirmed_with_the_rotated_one():
    """The two halves of ENG-2116 are only useful together.

    ``test_live_requests_reread_api_key_without_rebuilding_provider`` proves the
    per-request re-read but never returns a 401, and ``tests/test_client.py``
    proves the one bounded retry against mocks that carry no credential. Neither
    can assert that the confirmation attempt used the ROTATED token, which is
    the whole mechanism. Refactoring either half apart from the other would keep
    both green while the stale-JWT symptom returned.

    The 401 body is the production shape: Keycloak answered ``invalid_token``
    with reason "Token is not active" for all seven named ENG-2116 failures.
    """
    live = {"credential": "token-a"}
    authorization_headers: list[str] = []

    async def api_key_provider() -> str:
        return live["credential"]

    def handler(request: httpx.Request) -> httpx.Response:
        authorization = request.headers["authorization"]
        authorization_headers.append(authorization)
        if authorization == "Bearer token-a":
            # The desktop refresh lands between the refusal and the retry.
            live["credential"] = "token-b"
            return httpx.Response(
                401,
                json={
                    "error": {
                        "message": "Token is not active",
                        "code": "invalid_token",
                    }
                },
            )
        return httpx.Response(200, json=_completion_body())

    real_client = openai.AsyncOpenAI
    transport = httpx.MockTransport(handler)

    def build_client(**kwargs):
        kwargs["http_client"] = httpx.AsyncClient(transport=transport)
        return real_client(**kwargs)

    with patch(
        "anton.core.llm.openai.openai.AsyncOpenAI", side_effect=build_client
    ):
        provider = OpenAIProvider(
            api_key="token-a",
            api_key_provider=api_key_provider,
            base_url="https://gateway.test/v1",
        )

    client = LLMClient(
        planning_provider=provider,
        planning_model="test-model",
        coding_provider=provider,
        coding_model="test-model",
    )
    try:
        response = await client.plan(system="sys", messages=[])
    finally:
        await provider._client.close()

    assert response.content == "done"
    # The header sequence is what separates "retried and got lucky" from
    # "retried with the rotated credential".
    assert authorization_headers == ["Bearer token-a", "Bearer token-b"]


async def test_azure_refuses_a_credential_supplier_it_cannot_await():
    """AsyncAzureOpenAI never awaits a callable api_key.

    Its ``_prepare_options`` override does not chain to super(), so
    ``AsyncOpenAI._refresh_api_key`` never runs, and the base ``__init__`` has
    already replaced the callable with "". Constructing such a client would send
    an empty ``api-key`` header on every request and 401 forever, with no typed
    error to explain it.
    """

    async def api_key_provider() -> str:  # pragma: no cover - never awaited
        return "token-a"

    with pytest.raises(EndpointConfigurationError, match="callable api_key"):
        OpenAIProvider(
            api_key="static-key",
            api_key_provider=api_key_provider,
            api_version="2024-06-01",
            base_url="https://example.openai.azure.com",
        )
