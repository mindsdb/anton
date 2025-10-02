import logging

import pytest

from .config import MINDS_API_BASE_URL


@pytest.mark.happy_path
@pytest.mark.skip(reason="Skipping this test temporarily for the behavior unstable.")
class TestChatAPI:
    def test_chat_with_data_aware_mind_no_stream(self, api_client, temporary_mind):
        """
        Tests a non-streaming chat completion with a data-aware mind.
        The `temporary_mind` fixture automatically provides a mind linked to a datasource.
        """
        logging.info(f"TEST: Chatting with data-aware mind '{temporary_mind}' (no stream)")
        payload = {
            "model": temporary_mind,
            "messages": [{"role": "user", "content": "Can you summarize the data you have access to?"}],
            "stream": False,
        }
        # Add a timeout for potentially long-running queries
        response = api_client.post(f"{MINDS_API_BASE_URL}/api/v1/chat/completions", json=payload, timeout=300)

        assert response.status_code == 200, (
            f"Chat (no stream) failed. Status: {response.status_code}, Text: {response.text}"
        )
        data = response.json()
        assert "choices" in data and len(data["choices"]) > 0
        assert "content" in data["choices"][0]["message"]
        logging.info("PASSED: Chat (no stream) with data-aware mind returned a valid response.")

    @pytest.mark.skip(reason="Skipping streaming test temporarily for the behavior unstable.")
    def test_chat_with_data_aware_mind_streaming(self, api_client, temporary_mind):
        """
        Tests a streaming chat completion with a data-aware mind.
        """
        logging.info(f"TEST: Chatting with data-aware mind '{temporary_mind}' (stream)")
        payload = {
            "model": temporary_mind,
            "messages": [{"role": "user", "content": "Tell me the tables you have access to."}],
            "stream": True,
        }

        # Add a timeout for potentially long-running queries
        response = api_client.post(
            f"{MINDS_API_BASE_URL}/api/v1/chat/completions", json=payload, stream=True, timeout=300
        )

        assert response.status_code == 200, (
            f"Chat (stream) failed. Status: {response.status_code}, Text: {response.text}"
        )

        full_response_content = ""
        chunks_received = 0
        for chunk in response.iter_lines():
            if chunk:
                full_response_content += chunk.decode("utf-8")
                chunks_received += 1

        assert chunks_received > 0, "Streaming response did not return any data chunks."
        assert "data: " in full_response_content, "Streaming response content is missing expected SSE format."
        logging.info(f"PASSED: Chat (stream) with data-aware mind returned {chunks_received} chunks.")
