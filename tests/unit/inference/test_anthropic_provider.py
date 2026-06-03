"""Tests for Anthropic provider adapter."""

from unittest.mock import MagicMock, patch

from minds.common.passthrough_config import PassthroughModelConfig
from minds.inference.providers.anthropic import (
    _anthropic_response_to_openai,
    _collect_anthropic_server_artifacts,
    _get_anthropic_client,
    _openai_messages_to_anthropic,
    _openai_tool_choice_to_anthropic,
    _translate_tools_for_anthropic,
)


class TestMessageConversion:
    """Tests for message format conversion to Anthropic."""

    def test_system_message_extraction(self):
        """Test that system messages are extracted."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        system_prompt, anthropic_messages = _openai_messages_to_anthropic(messages)
        assert system_prompt == "You are helpful"
        assert len(anthropic_messages) > 0

    def test_single_system_message(self):
        """Test that single system message is extracted."""
        messages = [
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Hello"},
        ]
        system_prompt, _ = _openai_messages_to_anthropic(messages)
        assert system_prompt == "Be concise"

    def test_system_message_is_replaced_by_latest(self):
        """Test that multiple system messages keep only the last one."""
        messages = [
            {"role": "system", "content": "First"},
            {"role": "system", "content": "Second"},
            {"role": "user", "content": "Hello"},
        ]
        system_prompt, _ = _openai_messages_to_anthropic(messages)
        # Anthropic only supports one system prompt, so the last one wins
        assert system_prompt == "Second"

    def test_user_message_conversion(self):
        """Test user message conversion."""
        messages = [{"role": "user", "content": "Hello"}]
        _, anthropic_messages = _openai_messages_to_anthropic(messages)
        assert len(anthropic_messages) > 0
        assert any(msg.get("role") == "user" for msg in anthropic_messages)

    def test_assistant_message_conversion(self):
        """Test assistant message conversion."""
        messages = [{"role": "assistant", "content": "Hi there"}]
        _, anthropic_messages = _openai_messages_to_anthropic(messages)
        assert len(anthropic_messages) > 0
        assert any(msg.get("role") == "assistant" for msg in anthropic_messages)

    def test_tool_result_message_conversion(self):
        """Test tool result message conversion to tool_result content block."""
        messages = [
            {
                "role": "assistant",
                "content": "I'll search",
                "tool_calls": [{"id": "call_1", "function": {"name": "search"}}],
            },
            {"role": "tool", "content": '{"result": "found"}', "tool_call_id": "call_1"},
        ]
        _, anthropic_messages = _openai_messages_to_anthropic(messages)
        # Tool results should be mapped to user role with tool_result content blocks
        assert any(
            msg.get("role") == "user" and any(block.get("type") == "tool_result" for block in msg.get("content", []))
            for msg in anthropic_messages
        )

    def test_assistant_with_tool_calls(self):
        """Test assistant message with tool calls."""
        messages = [
            {
                "role": "assistant",
                "content": "Let me search",
                "tool_calls": [{"id": "call_1", "function": {"name": "search", "arguments": '{"q": "test"}'}}],
            }
        ]
        _, anthropic_messages = _openai_messages_to_anthropic(messages)
        # Should have assistant message with tool_use content blocks
        assert any(
            msg.get("role") == "assistant" and any(block.get("type") == "tool_use" for block in msg.get("content", []))
            for msg in anthropic_messages
        )

    def test_assistant_with_text_and_tool_calls(self):
        """Test assistant message with both text and tool calls."""
        messages = [
            {
                "role": "assistant",
                "content": "Let me search for that",
                "tool_calls": [{"id": "call_1", "function": {"name": "search"}}],
            }
        ]
        _, anthropic_messages = _openai_messages_to_anthropic(messages)
        # Should have both text and tool_use blocks
        assistant_msg = next(msg for msg in anthropic_messages if msg.get("role") == "assistant")
        has_text = any(block.get("type") == "text" for block in assistant_msg.get("content", []))
        has_tool_use = any(block.get("type") == "tool_use" for block in assistant_msg.get("content", []))
        assert has_text and has_tool_use


class TestToolConversion:
    """Tests for tool format conversion to Anthropic."""

    def test_translate_tools_for_anthropic(self):
        """Test tool translation to Anthropic format."""
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
        anthropic_tools = _translate_tools_for_anthropic(tools)
        assert anthropic_tools is not None
        assert hasattr(anthropic_tools, "tools")
        assert len(anthropic_tools.tools) > 0

    def test_empty_tools_list(self):
        """Test that empty tools list is handled correctly."""
        anthropic_tools = _translate_tools_for_anthropic([])
        assert anthropic_tools.tools == []

    def test_none_tools(self):
        """Test that None tools is handled correctly."""
        anthropic_tools = _translate_tools_for_anthropic(None)
        assert anthropic_tools.tools == []

    def test_multiple_tools(self):
        """Test translation of multiple tools."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch",
                    "description": "Fetch URL content",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
        anthropic_tools = _translate_tools_for_anthropic(tools)
        assert len(anthropic_tools.tools) >= 2


class TestToolChoice:
    """Tests for tool_choice conversion."""

    def test_tool_choice_auto(self):
        """Test that auto tool_choice is converted correctly."""
        result = _openai_tool_choice_to_anthropic("auto")
        assert result == {"type": "auto"}

    def test_tool_choice_required(self):
        """Test that required tool_choice is converted to any."""
        result = _openai_tool_choice_to_anthropic("required")
        assert result == {"type": "any"}

    def test_tool_choice_none(self):
        """Test that none tool_choice is converted to None."""
        result = _openai_tool_choice_to_anthropic("none")
        assert result is None

    def test_tool_choice_none_passthrough(self):
        """Test that None tool_choice returns None."""
        result = _openai_tool_choice_to_anthropic(None)
        assert result is None

    def test_tool_choice_function_specific(self):
        """Test that specific function tool_choice is converted correctly."""
        tool_choice = {"type": "function", "function": {"name": "search"}}
        result = _openai_tool_choice_to_anthropic(tool_choice)
        assert result == {"type": "tool", "name": "search"}

    def test_tool_choice_function_without_name(self):
        """Test that function tool_choice without name defaults to auto."""
        tool_choice = {"type": "function", "function": {}}
        result = _openai_tool_choice_to_anthropic(tool_choice)
        assert result == {"type": "auto"}


class TestResponseConversion:
    """Tests for response conversion from Anthropic format."""

    def test_anthropic_text_response_conversion(self):
        """Test converting Anthropic text response to ChatCompletion format."""
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Hello, world!"

        mock_response = MagicMock()
        mock_response.content = [mock_text_block]
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5

        result = _anthropic_response_to_openai(mock_response, "claude-sonnet-4")

        assert result is not None
        assert result["object"] == "chat.completion"
        assert result["model"] == "claude-sonnet-4"
        assert result["choices"][0]["message"]["role"] == "assistant"
        assert "Hello, world!" in result["choices"][0]["message"]["content"]

    def test_anthropic_tool_use_response_conversion(self):
        """Test converting Anthropic tool_use response to ChatCompletion format."""
        mock_tool_use = MagicMock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.id = "tool_1"
        mock_tool_use.name = "search"
        mock_tool_use.input = {"query": "test"}

        mock_response = MagicMock()
        mock_response.content = [mock_tool_use]
        mock_response.stop_reason = "tool_use"
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5

        result = _anthropic_response_to_openai(mock_response, "claude-sonnet-4")

        assert result is not None
        assert "tool_calls" in result["choices"][0]["message"]
        assert result["choices"][0]["finish_reason"] == "tool_calls"

    def test_anthropic_mixed_content_response(self):
        """Test Anthropic response with both text and tool_use."""
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "I'll search for that"

        mock_tool_use = MagicMock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.id = "tool_1"
        mock_tool_use.name = "search"
        mock_tool_use.input = {"query": "test"}

        mock_response = MagicMock()
        mock_response.content = [mock_text_block, mock_tool_use]
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5

        result = _anthropic_response_to_openai(mock_response, "claude-sonnet-4")

        assert result is not None
        message = result["choices"][0]["message"]
        assert message["content"] is not None
        assert "tool_calls" in message


class TestServerArtifacts:
    """Tests for server artifact collection."""

    def test_collect_anthropic_artifacts_empty(self):
        """Test collecting artifacts when none are present."""
        mock_response = MagicMock()
        mock_response.content = []

        artifacts = _collect_anthropic_server_artifacts(mock_response)

        assert artifacts == []

    def test_collect_anthropic_artifacts_with_text_only(self):
        """Test collecting artifacts with only text content."""
        mock_text = MagicMock()
        mock_text.type = "text"

        mock_response = MagicMock()
        mock_response.content = [mock_text]

        artifacts = _collect_anthropic_server_artifacts(mock_response)

        assert artifacts == []


class TestClientInitialization:
    """Tests for Anthropic client initialization."""

    @patch("minds.inference.providers.anthropic.AsyncAnthropic")
    def test_get_anthropic_client_basic(self, mock_client_class):
        """Test Anthropic client is initialized with API key."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        config = PassthroughModelConfig(
            api_kind="anthropic_messages",
            model_name="claude-sonnet-4",
            api_key="test-key",
        )

        client = _get_anthropic_client(config)
        assert client is not None
        mock_client_class.assert_called_once()
        call_kwargs = mock_client_class.call_args[1]
        assert call_kwargs["api_key"] == "test-key"

    @patch("minds.inference.providers.anthropic.AsyncAnthropic")
    def test_get_anthropic_client_with_base_url(self, mock_client_class):
        """Test Anthropic client with custom base_url (Fireworks)."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        config = PassthroughModelConfig(
            api_kind="anthropic_messages",
            model_name="claude-sonnet-4",
            api_key="test-key",
            base_url="https://api.fireworks.ai/anthropic/v1",
        )

        client = _get_anthropic_client(config)
        assert client is not None
        call_kwargs = mock_client_class.call_args[1]
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["base_url"] == "https://api.fireworks.ai/anthropic/v1"
