"""Tests for OpenAI provider adapter."""

from unittest.mock import MagicMock, patch

from minds.inference.providers.openai import (
    _chat_messages_to_responses_input,
    _chat_tool_choice_to_responses,
    _collect_responses_server_artifacts,
    _flatten_reasoning_summary,
    _get_openai_client,
    _responses_response_to_chat_completion,
    _translate_tools_for_openai,
)
from minds.inference.types import PassthroughModelConfig


class TestMessageConversion:
    """Tests for message format conversion to OpenAI Responses API."""

    def test_system_message_extraction(self):
        """Test that system messages are extracted as instructions."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        instructions, input_items = _chat_messages_to_responses_input(messages)
        assert instructions == "You are helpful"
        assert len(input_items) > 0
        assert any(item.get("role") == "user" for item in input_items)

    def test_multiple_system_messages_joined(self):
        """Test that multiple system messages are joined with blank lines."""
        messages = [
            {"role": "system", "content": "First system"},
            {"role": "system", "content": "Second system"},
        ]
        instructions, _ = _chat_messages_to_responses_input(messages)
        assert "First system" in instructions
        assert "Second system" in instructions
        assert "\n\n" in instructions

    def test_empty_system_message_ignored(self):
        """Test that empty system messages are ignored."""
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Hello"},
        ]
        instructions, _ = _chat_messages_to_responses_input(messages)
        assert instructions is None or instructions == ""

    def test_user_message_conversion(self):
        """Test user message conversion."""
        messages = [{"role": "user", "content": "Hello"}]
        _, input_items = _chat_messages_to_responses_input(messages)
        assert len(input_items) > 0
        assert any(item.get("role") == "user" for item in input_items)
        assert any(item.get("content") == "Hello" for item in input_items)

    def test_assistant_message_conversion(self):
        """Test assistant message conversion."""
        messages = [{"role": "assistant", "content": "Hi there"}]
        _, input_items = _chat_messages_to_responses_input(messages)
        assert len(input_items) > 0
        assert any(item.get("role") == "assistant" for item in input_items)

    def test_tool_result_message_conversion(self):
        """Test tool result message conversion to function_call_output."""
        messages = [
            {
                "role": "assistant",
                "content": "I'll search",
                "tool_calls": [{"id": "call_1", "function": {"name": "search"}}],
            },
            {"role": "tool", "content": '{"result": "found"}', "tool_call_id": "call_1"},
        ]
        _, input_items = _chat_messages_to_responses_input(messages)
        # Tool results should be mapped to function_call_output
        assert any(
            item.get("type") == "function_call_output" and item.get("call_id") == "call_1" for item in input_items
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
        _, input_items = _chat_messages_to_responses_input(messages)
        # Should have both assistant message and function_call items
        assert any(item.get("type") == "function_call" for item in input_items)
        assert any(item.get("role") == "assistant" for item in input_items)

    def test_assistant_without_text_with_tool_calls(self):
        """Test assistant message with only tool calls (no text)."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "function": {"name": "search"}}],
            }
        ]
        _, input_items = _chat_messages_to_responses_input(messages)
        # Should only have function_call, not empty assistant message
        assert any(item.get("type") == "function_call" for item in input_items)


class TestToolConversion:
    """Tests for tool format conversion to OpenAI Responses API."""

    def test_translate_tools_for_openai(self):
        """Test tool translation to OpenAI Responses format."""
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
        openai_tools = _translate_tools_for_openai(tools)
        assert openai_tools is not None
        assert len(openai_tools) > 0
        # Responses API flattens the structure
        assert openai_tools[0]["type"] == "function"
        assert openai_tools[0]["name"] == "search"

    def test_empty_tools_list(self):
        """Test that empty tools list is handled correctly."""
        openai_tools = _translate_tools_for_openai([])
        assert openai_tools == []

    def test_none_tools(self):
        """Test that None tools is handled correctly."""
        openai_tools = _translate_tools_for_openai(None)
        assert openai_tools == []

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
        openai_tools = _translate_tools_for_openai(tools)
        assert len(openai_tools) == 2
        assert openai_tools[0]["name"] == "search"
        assert openai_tools[1]["name"] == "fetch"

    def test_tool_without_description(self):
        """Test tool conversion without description."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        openai_tools = _translate_tools_for_openai(tools)
        assert len(openai_tools) > 0
        assert openai_tools[0]["name"] == "search"
        # Description should not be present if not provided
        assert "description" not in openai_tools[0] or openai_tools[0].get("description") is None


class TestToolChoice:
    """Tests for tool_choice conversion."""

    def test_tool_choice_string_passthrough(self):
        """Test that string tool_choice values pass through unchanged."""
        result = _chat_tool_choice_to_responses("auto")
        assert result == "auto"

        result = _chat_tool_choice_to_responses("required")
        assert result == "required"

        result = _chat_tool_choice_to_responses("none")
        assert result == "none"

    def test_tool_choice_none_passthrough(self):
        """Test that None tool_choice passes through."""
        result = _chat_tool_choice_to_responses(None)
        assert result is None

    def test_tool_choice_function_flattening(self):
        """Test that function tool_choice is flattened."""
        tool_choice = {"type": "function", "function": {"name": "search"}}
        result = _chat_tool_choice_to_responses(tool_choice)
        assert result == {"type": "function", "name": "search"}

    def test_tool_choice_unrecognized_passthrough(self):
        """Test that unrecognized tool_choice passes through."""
        tool_choice = {"some": "value"}
        result = _chat_tool_choice_to_responses(tool_choice)
        assert result == {"some": "value"}


class TestResponseConversion:
    """Tests for response conversion from OpenAI Responses API to ChatCompletion format."""

    def test_text_response_conversion(self):
        """Test converting OpenAI Responses text response to ChatCompletion format."""
        # Mock Responses API response
        mock_part = MagicMock()
        mock_part.type = "output_text"
        mock_part.text = "Hello, world!"

        mock_content_item = MagicMock()
        mock_content_item.type = "message"
        mock_content_item.content = [mock_part]

        mock_response = MagicMock()
        mock_response.output = [mock_content_item]
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5

        result = _responses_response_to_chat_completion(mock_response, "gpt-4o")

        assert result is not None
        assert result["object"] == "chat.completion"
        assert result["model"] == "gpt-4o"
        assert result["choices"][0]["message"]["content"] == "Hello, world!"
        assert result["choices"][0]["message"]["role"] == "assistant"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5

    def test_function_call_response_conversion(self):
        """Test converting function_call response to ChatCompletion tool_calls format."""
        mock_function_call = MagicMock()
        mock_function_call.type = "function_call"
        mock_function_call.call_id = "call_1"
        mock_function_call.name = "search"
        mock_function_call.arguments = '{"query": "test"}'

        mock_response = MagicMock()
        mock_response.output = [mock_function_call]
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5

        result = _responses_response_to_chat_completion(mock_response, "gpt-4o")

        assert result is not None
        assert "tool_calls" in result["choices"][0]["message"]
        assert len(result["choices"][0]["message"]["tool_calls"]) > 0
        assert result["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "search"
        assert result["choices"][0]["finish_reason"] == "tool_calls"

    def test_response_with_text_and_tool_calls(self):
        """Test response with both text and tool calls."""
        mock_text_part = MagicMock()
        mock_text_part.type = "output_text"
        mock_text_part.text = "Let me search for that"

        mock_message = MagicMock()
        mock_message.type = "message"
        mock_message.content = [mock_text_part]

        mock_function_call = MagicMock()
        mock_function_call.type = "function_call"
        mock_function_call.call_id = "call_1"
        mock_function_call.name = "search"
        mock_function_call.arguments = '{"query": "test"}'

        mock_response = MagicMock()
        mock_response.output = [mock_message, mock_function_call]
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5

        result = _responses_response_to_chat_completion(mock_response, "gpt-4o")

        assert result["choices"][0]["message"]["content"] == "Let me search for that"
        assert "tool_calls" in result["choices"][0]["message"]
        assert result["choices"][0]["finish_reason"] == "tool_calls"


class TestServerArtifacts:
    """Tests for server artifact collection."""

    def test_collect_web_search_artifacts(self):
        """Test collecting web_search_call artifacts."""
        mock_web_search = MagicMock()
        mock_web_search.type = "web_search_call"
        mock_web_search.id = "ws_1"
        mock_web_search.status = "completed"
        mock_web_search.action = MagicMock()
        mock_web_search.action.query = "test query"

        mock_response = MagicMock()
        mock_response.output = [mock_web_search]

        artifacts = _collect_responses_server_artifacts(mock_response)

        assert len(artifacts) > 0
        assert artifacts[0]["type"] == "web_search_call"
        assert artifacts[0]["query"] == "test query"

    def test_collect_reasoning_artifacts(self):
        """Test collecting reasoning artifacts."""
        mock_reasoning_entry = MagicMock()
        mock_reasoning_entry.text = "Thinking step 1"

        mock_reasoning = MagicMock()
        mock_reasoning.type = "reasoning"
        mock_reasoning.id = "reasoning_1"
        mock_reasoning.summary = [mock_reasoning_entry]

        mock_response = MagicMock()
        mock_response.output = [mock_reasoning]

        artifacts = _collect_responses_server_artifacts(mock_response)

        assert len(artifacts) > 0
        assert artifacts[0]["type"] == "reasoning"
        assert "step 1" in artifacts[0]["summary"]

    def test_flatten_reasoning_summary_empty(self):
        """Test flattening empty reasoning summary."""
        result = _flatten_reasoning_summary(None)
        assert result is None

        result = _flatten_reasoning_summary([])
        assert result is None

    def test_flatten_reasoning_summary_with_entries(self):
        """Test flattening reasoning summary with entries."""
        entry1 = MagicMock()
        entry1.text = "Step 1"
        entry2 = MagicMock()
        entry2.text = "Step 2"

        result = _flatten_reasoning_summary([entry1, entry2])

        assert result is not None
        assert "Step 1" in result
        assert "Step 2" in result


class TestClientInitialization:
    """Tests for OpenAI client initialization."""

    @patch("minds.inference.providers.openai.AsyncOpenAI")
    def test_get_openai_client_basic(self, mock_client_class):
        """Test OpenAI client is initialized with API key."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        config = PassthroughModelConfig(
            api_kind="openai",
            model_name="gpt-4o",
            api_key="test-key",
        )

        client = _get_openai_client(config)
        assert client is not None
        mock_client_class.assert_called_once_with(api_key="test-key")

    @patch("minds.inference.providers.openai.AsyncOpenAI")
    def test_get_openai_client_with_base_url(self, mock_client_class):
        """Test OpenAI client with custom base_url."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        config = PassthroughModelConfig(
            api_kind="openai",
            model_name="gpt-4o",
            api_key="test-key",
            base_url="https://custom.openai.com",
        )

        client = _get_openai_client(config)
        assert client is not None
        mock_client_class.assert_called_once()
        call_kwargs = mock_client_class.call_args[1]
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["base_url"] == "https://custom.openai.com"

    @patch("minds.inference.providers.openai.AsyncOpenAI")
    def test_get_openai_client_without_base_url(self, mock_client_class):
        """Test OpenAI client without base_url doesn't set it."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        config = PassthroughModelConfig(
            api_kind="openai",
            model_name="gpt-4o",
            api_key="test-key",
            base_url=None,
        )

        client = _get_openai_client(config)
        assert client is not None
        call_kwargs = mock_client_class.call_args[1]
        assert "base_url" not in call_kwargs
