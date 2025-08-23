from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from minds.client.openai_client import OpenAIClient
from minds.common.vars import (
    OPEN_AI_API_KEY,
    OPEN_AI_API_URL,
    OPEN_AI_MAX_TOKENS,
    OPEN_AI_MODEL_NAME,
)
from minds.requests.schemas import Message, Role


@pytest.fixture
def sample_messages():
    return [
        Message(role=Role.user, content="Hello"),
        Message(role=Role.assistant, content="Hi there"),
    ]


@pytest.fixture
def client():
    return OpenAIClient(
        api_url=OPEN_AI_API_URL,
        api_key=OPEN_AI_API_KEY,
        chat_completions_model=OPEN_AI_MODEL_NAME,
        max_tokens=OPEN_AI_MAX_TOKENS,
    )


class TestOpenAIClient:
    @pytest.mark.asyncio
    async def test_chat_completions_non_streaming(self, client, sample_messages):
        # Mock response object
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Mocked response"
        mock_response.choices = [mock_choice]

        with patch.object(
            client.client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_response),
        ) as mock_create:
            results = []
            async for content in client.chat_completions(messages=sample_messages, stream=False):
                results.append(content)

            mock_create.assert_awaited_once_with(
                model=OPEN_AI_MODEL_NAME,
                messages=sample_messages,
                stream=False,
                temperature=None,
                max_tokens=None,
            )

            assert results == ["Mocked response"]

    @pytest.mark.asyncio
    async def test_chat_completions_streaming(self, client, sample_messages):
        # Create mock async generator for streaming
        async def mock_stream():
            class Delta:
                content = "chunk1"

            class Choice:
                delta = Delta()

            yield MagicMock(choices=[Choice()])

            class Delta2:
                content = "chunk2"

            class Choice2:
                delta = Delta2()

            yield MagicMock(choices=[Choice2()])

        with patch.object(
            client.client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_stream()),
        ) as mock_create:
            results = []
            async for content in client.chat_completions(messages=sample_messages, stream=True):
                results.append(content)

            mock_create.assert_awaited_once()
            assert results == ["chunk1", "chunk2"]

    @pytest.mark.asyncio
    async def test_chat_completions_handles_exception(self, client, sample_messages):
        with (
            patch.object(
                client.client.chat.completions,
                "create",
                new=AsyncMock(side_effect=RuntimeError("API Error")),
            ),
            pytest.raises(RuntimeError, match="API Error"),
        ):
            async for _ in client.chat_completions(messages=sample_messages):
                pass
