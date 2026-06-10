"""Tests for Gemini provider adapter."""

from unittest.mock import MagicMock, patch

from google.genai import types as genai_types

from minds.inference.providers.gemini import (
    _chat_messages_to_gemini,
    _gemini_finish_reason_to_openai,
    _gemini_response_to_openai,
    _get_gemini_client,
    _translate_tools_for_gemini,
)
from minds.inference.types import PassthroughModelConfig


class TestMessageConversion:
    """Tests for message format conversion to Gemini."""

    def test_system_message_extraction(self):
        """Test that system messages are extracted."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        system_instruction, contents = _chat_messages_to_gemini(messages)
        assert system_instruction == "You are helpful"
        assert len(contents) > 0

    def test_multiple_system_messages_joined(self):
        """Test that multiple system messages are joined with blank lines."""
        messages = [
            {"role": "system", "content": "First system"},
            {"role": "system", "content": "Second system"},
        ]
        system_instruction, _ = _chat_messages_to_gemini(messages)
        assert "First system" in system_instruction
        assert "Second system" in system_instruction

    def test_user_message_conversion(self):
        """Test user message conversion."""
        messages = [{"role": "user", "content": "Hello"}]
        _, contents = _chat_messages_to_gemini(messages)
        assert len(contents) > 0
        # Verify user message is in contents with role="user"
        assert any(content.role == "user" for content in contents)

    def test_assistant_message_conversion(self):
        """Test assistant message conversion."""
        messages = [{"role": "assistant", "content": "Hi there"}]
        _, contents = _chat_messages_to_gemini(messages)
        assert len(contents) > 0
        # Verify assistant message maps to "model" role
        assert any(content.role == "model" for content in contents)

    def test_tool_result_message_conversion(self):
        """Test tool result message conversion."""
        messages = [
            {
                "role": "assistant",
                "content": "I'll search",
                "tool_calls": [{"id": "call_1", "function": {"name": "search"}}],
            },
            {"role": "tool", "content": '{"result": "found"}', "tool_call_id": "call_1"},
        ]
        _, contents = _chat_messages_to_gemini(messages)
        # Tool results should be mapped to user role with function_response
        assert len(contents) > 0

    def test_empty_system_message_ignored(self):
        """Test that empty system messages are ignored."""
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Hello"},
        ]
        system_instruction, _ = _chat_messages_to_gemini(messages)
        assert system_instruction is None or system_instruction == ""


class TestToolConversion:
    """Tests for tool format conversion to Gemini."""

    def test_translate_tools_for_gemini(self):
        """Test tool translation to Gemini format."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        gemini_tools = _translate_tools_for_gemini(tools)
        assert gemini_tools is not None
        assert len(gemini_tools) > 0


class TestFinishReasonConversion:
    """Tests for finish reason conversion."""

    def test_finish_reason_stop(self):
        """Test stop finish reason conversion."""
        result = _gemini_finish_reason_to_openai(None, has_tool_calls=False)
        assert result == "stop"

    def test_finish_reason_max_tokens(self):
        """Test max tokens finish reason conversion with has_tool_calls."""
        result = _gemini_finish_reason_to_openai(genai_types.FinishReason.MAX_TOKENS, has_tool_calls=False)
        assert result == "length"

    def test_finish_reason_tool_calls(self):
        """Test tool calls finish reason conversion."""
        result = _gemini_finish_reason_to_openai(None, has_tool_calls=True)
        assert result == "tool_calls"

    def test_finish_reason_tool_calls_takes_precedence(self):
        """Test that has_tool_calls takes precedence over finish_reason."""
        result = _gemini_finish_reason_to_openai(genai_types.FinishReason.MAX_TOKENS, has_tool_calls=True)
        assert result == "tool_calls"


class TestResponseConversion:
    """Tests for response conversion from Gemini format."""

    def test_gemini_text_response_conversion(self):
        """Test converting Gemini text response to OpenAI format."""
        # Mock Gemini response with text
        mock_part = MagicMock()
        mock_part.text = "Hello, world!"
        mock_part.function_call = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]
        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = genai_types.FinishReason.STOP

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5
        result = _gemini_response_to_openai(mock_response, "gemini-2.0-flash")
        assert result is not None
        assert result["object"] == "chat.completion"
        assert result["model"] == "gemini-2.0-flash"
        assert result["choices"][0]["message"]["content"] == "Hello, world!"

    def test_gemini_tool_call_response_conversion(self):
        """Test converting Gemini tool call response to OpenAI format."""
        # Mock tool call part
        mock_part = MagicMock()
        mock_part.text = None
        mock_part.function_call = MagicMock()
        mock_part.function_call.name = "search"
        mock_part.function_call.args = {"query": "test"}
        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = None
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5

        result = _gemini_response_to_openai(mock_response, "gemini-2.0-flash")
        assert result is not None
        assert "tool_calls" in result["choices"][0]["message"]
        assert result["choices"][0]["finish_reason"] == "tool_calls"


class TestClientInitialization:
    """Tests for Gemini client initialization."""

    @patch("minds.inference.providers.gemini.genai.Client")
    def test_get_gemini_client(self, mock_client_class):
        """Test Gemini client is initialized with API key."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        config = PassthroughModelConfig(
            api_kind="gemini",
            model_name="gemini-2.0-flash",
            api_key="test-key",
        )
        client = _get_gemini_client(config)
        assert client is not None
        # Verify client was created with API key
        mock_client_class.assert_called_once_with(api_key="test-key")
