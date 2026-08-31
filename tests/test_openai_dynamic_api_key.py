from __future__ import annotations

from unittest.mock import patch

import httpx
import openai

from anton.core.llm.openai import OpenAIProvider


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
