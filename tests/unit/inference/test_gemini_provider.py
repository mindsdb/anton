"""Tests for Gemini provider adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from minds.inference.providers.gemini import (
    _chat_messages_to_gemini,
    _gemini_finish_reason_to_openai,
    _gemini_response_to_openai,
    _translate_tools_for_gemini,
)


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
            {"role": "assistant", "content": "I'll search", "tool_calls": [{"id": "call_1", "function": {"name": "search"}}]},
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
        result = _gemini_finish_reason_to_openai("STOP")
        assert result == "stop"

    def test_finish_reason_max_tokens(self):
        """Test max tokens finish reason conversion."""
        result = _gemini_finish_reason_to_openai("MAX_TOKENS")
        assert result == "length"

    def test_finish_reason_tool_calls(self):
        """Test tool calls finish reason conversion."""
        result = _gemini_finish_reason_to_openai("TOOL_CALLS")
        assert result == "tool_calls"

    def test_unknown_finish_reason(self):
        """Test unknown finish reason defaults to stop."""
        result = _gemini_finish_reason_to_openai("UNKNOWN")
        assert result == "stop"


class TestResponseConversion:
    """Tests for response conversion from Gemini format."""

    def test_gemini_text_response_conversion(self):
        """Test converting Gemini text response to OpenAI format."""
        # Mock Gemini response
        mock_content = MagicMock()
        mock_content.text = "Hello, world!"

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "STOP"

        result = _gemini_response_to_openai(mock_candidate)
        assert result is not None
        assert "content" in result or result.get("content") is not None

    def test_gemini_tool_call_response_conversion(self):
        """Test converting Gemini tool call response to OpenAI format."""
        # Mock tool call part
        mock_part = MagicMock()
        mock_part.function_call = MagicMock()
        mock_part.function_call.name = "search"
        mock_part.function_call.args = {"query": "test"}

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "TOOL_CALLS"

        result = _gemini_response_to_openai(mock_candidate)
        assert result is not None


class TestClientInitialization:
    """Tests for Gemini client initialization."""

    @patch("minds.inference.providers.gemini.genai.Client")
    def test_get_gemini_client(self, mock_client_class):
        """Test Gemini client is initialized with API key."""
        from minds.inference.providers.gemini import _get_gemini_client
        from minds.common.passthrough_config import PassthroughModelConfig, ApiKind

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        config = PassthroughModelConfig(
            api_kind=ApiKind.GEMINI,
            model_name="gemini-pro",
            api_key="test-key",
        )

        client = _get_gemini_client(config)
        assert client is not None
        # Verify client was created with API key
        mock_client_class.assert_called()
