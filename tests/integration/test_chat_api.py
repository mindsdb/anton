"""Integration tests for /v1/chat/completions (passthrough inference)."""

import json

import pytest

from .conftest import MINDS_API_BASE_URL, MINDS_API_PREFIX


@pytest.mark.happy_path
class TestChatCompletions:
    def test_non_streaming(self, api_client, passthrough_model):
        """Non-streaming passthrough returns an OpenAI-shaped completion."""
        payload = {
            "model": passthrough_model,
            "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
            "stream": False,
            "max_tokens": 16,
        }

        resp = api_client.post(
            f"{MINDS_API_BASE_URL}{MINDS_API_PREFIX}/chat/completions",
            json=payload,
            timeout=60,
        )

        assert resp.status_code == 200, f"{resp.status_code}: {resp.text}"
        data = resp.json()
        assert data.get("object") == "chat.completion"
        assert data["choices"], "missing choices"
        message = data["choices"][0]["message"]
        assert message["role"] == "assistant"
        assert message.get("content"), "assistant content is empty"
        assert "usage" in data and data["usage"].get("total_tokens", 0) > 0

    def test_streaming(self, api_client, passthrough_model):
        """Streaming passthrough emits valid SSE chunks and a [DONE] marker."""
        payload = {
            "model": passthrough_model,
            "messages": [{"role": "user", "content": "Count: 1, 2, 3."}],
            "stream": True,
            "max_tokens": 16,
        }

        resp = api_client.post(
            f"{MINDS_API_BASE_URL}{MINDS_API_PREFIX}/chat/completions",
            json=payload,
            stream=True,
            timeout=60,
        )
        assert resp.status_code == 200, f"{resp.status_code}: {resp.text}"

        chunks = 0
        saw_assistant_delta = False
        saw_done = False
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            chunks += 1
            payload_str = line[len("data: ") :]
            if payload_str.strip() == "[DONE]":
                saw_done = True
                break
            try:
                event = json.loads(payload_str)
            except json.JSONDecodeError:
                pytest.fail(f"Non-JSON SSE chunk: {payload_str!r}")
            delta = event.get("choices", [{}])[0].get("delta", {})
            if delta.get("role") == "assistant" or delta.get("content"):
                saw_assistant_delta = True

        assert chunks > 0, "no SSE chunks received"
        assert saw_assistant_delta, "stream had no assistant content"
        assert saw_done, "stream did not terminate with [DONE]"


def test_invalid_model_returns_error(api_client):
    """Unknown alias should be rejected upstream (not silently routed)."""
    payload = {
        "model": "latest:definitely-not-a-real-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }
    resp = api_client.post(
        f"{MINDS_API_BASE_URL}{MINDS_API_PREFIX}/chat/completions",
        json=payload,
        timeout=30,
    )
    assert resp.status_code >= 400, f"expected error, got {resp.status_code}: {resp.text}"
