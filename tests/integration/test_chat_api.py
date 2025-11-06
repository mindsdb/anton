import logging

import pytest

from .conftest import MINDS_API_BASE_URL


@pytest.mark.happy_path
class TestChatAPI:
    @pytest.mark.skip(reason="Skipping streaming test temporarily for the behavior unstable.")
    def test_chat_with_data_aware_mind_no_stream(self, api_client, temporary_mind):
        """
        Tests a non-streaming chat completion with a data-aware mind.
        Asserts a specific, known answer to a specific question.
        """
        # Unpack the mind_name from the fixture
        mind_name, (ds_name, config) = temporary_mind

        logging.info(f"TEST: Chatting with data-aware mind '{mind_name}' (no stream)")

        payload = {
            "model": mind_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "what is the top most expensive used cars available"},
            ],
            "stream": False,
        }

        response = api_client.post(f"{MINDS_API_BASE_URL}/api/v1/chat/completions", json=payload, timeout=300)

        assert response.status_code == 200, (
            f"Chat (no stream) failed. Status: {response.status_code}, Text: {response.text}"
        )

        data = response.json()

        assert "choices" in data and len(data["choices"]) > 0, "Response has no 'choices' field or it's empty."

        last_choice = data["choices"][-1]
        assert "message" in last_choice, "The last choice has no 'message' field."

        message = last_choice["message"]
        assert message.get("role") == "assistant", f"Last message role was not 'assistant'. Got: {message.get('role')}"

        assert "content" in message and message["content"], "Last message has no 'content' or content is empty."

        content = message["content"]
        assert "R8" in content, "Expected 'R8' in the response content"
        assert "145,000" in content, "Expected '145,000' in the response content"

        logging.info("PASSED: Chat (no stream) returned the correct assistant response.")

    @pytest.mark.skip(reason="Skipping streaming test temporarily for the behavior unstable.")
    def test_chat_with_data_aware_mind_streaming(self, api_client, temporary_mind):
        """
        Tests a streaming chat completion with a data-aware mind.
        """
        mind_name, (ds_name, config) = temporary_mind

        logging.info(f"TEST: Chatting with data-aware mind '{mind_name}' (stream)")

        payload = {
            "model": mind_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me the tables you have access to."},
            ],
            "stream": True,
        }

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
                decoded_chunk = chunk.decode("utf-8")
                full_response_content += decoded_chunk
                chunks_received += 1

        assert chunks_received > 0, "Streaming response did not return any data chunks."

        # Check that we received Server-Sent Events (SSE)
        assert "data: " in full_response_content, "Streaming response content is missing expected SSE 'data: ' format."
        # Check that we received the final answer, not just "thinking"
        assert '"role": "assistant"' in full_response_content, (
            "Streaming response did not contain a final 'assistant' message."
        )

        logging.info(f"PASSED: Chat (stream) with data-aware mind returned {chunks_received} chunks.")
