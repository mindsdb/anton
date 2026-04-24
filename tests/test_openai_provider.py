from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anton.config.settings import AntonSettings
from anton.core.llm.client import LLMClient
from anton.core.llm.openai import (
    OpenAIProvider,
    build_chat_completion_kwargs,
    _translate_messages,
    _translate_tools,
)
from anton.core.llm.provider import LLMProvider


def _make_mock_response(*, content="Hello", tool_calls=None, prompt_tokens=10, completion_tokens=20, finish_reason="stop"):
    """Build a mock OpenAI ChatCompletion response."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


class TestOpenAIProvider:
    async def test_complete_text_response(self):
        with patch("anton.core.llm.openai.openai") as mock_openai:
            mock_client = AsyncMock()
            mock_openai.AsyncOpenAI.return_value = mock_client

            mock_client.chat.completions.create = AsyncMock(
                return_value=_make_mock_response(content="Hello world", prompt_tokens=5, completion_tokens=10)
            )

            provider = OpenAIProvider(api_key="test-key")
            result = await provider.complete(
                model="gpt-4.1",
                system="be helpful",
                messages=[{"role": "user", "content": "hi"}],
            )

            assert result.content == "Hello world"
            assert result.tool_calls == []
            assert result.usage.input_tokens == 5
            assert result.usage.output_tokens == 10
            assert result.stop_reason == "stop"

    async def test_complete_tool_use_response(self):
        with patch("anton.core.llm.openai.openai") as mock_openai:
            mock_client = AsyncMock()
            mock_openai.AsyncOpenAI.return_value = mock_client

            tc = MagicMock()
            tc.id = "call_abc123"
            tc.function.name = "create_plan"
            tc.function.arguments = json.dumps({"reasoning": "test"})

            mock_client.chat.completions.create = AsyncMock(
                return_value=_make_mock_response(content=None, tool_calls=[tc], finish_reason="tool_calls")
            )

            provider = OpenAIProvider(api_key="test-key")
            result = await provider.complete(
                model="gpt-4.1",
                system="plan",
                messages=[{"role": "user", "content": "do something"}],
                tools=[{"name": "create_plan", "description": "plan", "input_schema": {}}],
            )

            assert result.content == ""
            assert len(result.tool_calls) == 1
            assert result.tool_calls[0].name == "create_plan"
            assert result.tool_calls[0].input == {"reasoning": "test"}
            assert result.stop_reason == "tool_calls"

    async def test_complete_passes_tool_choice(self):
        with patch("anton.core.llm.openai.openai") as mock_openai:
            mock_client = AsyncMock()
            mock_openai.AsyncOpenAI.return_value = mock_client

            mock_client.chat.completions.create = AsyncMock(
                return_value=_make_mock_response()
            )

            provider = OpenAIProvider(api_key="test-key")
            tool_choice = {"type": "tool", "name": "my_tool"}
            tools = [{"name": "my_tool", "description": "d", "input_schema": {"type": "object"}}]
            await provider.complete(
                model="gpt-4.1",
                system="sys",
                messages=[{"role": "user", "content": "hi"}],
                tools=tools,
                tool_choice=tool_choice,
            )

            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["tool_choice"] == {"type": "function", "function": {"name": "my_tool"}}
            assert call_kwargs["max_completion_tokens"] == 4096
            assert "max_tokens" not in call_kwargs


class TestBuildChatCompletionKwargs:
    def test_uses_modern_max_completion_tokens_field(self):
        kwargs = build_chat_completion_kwargs(
            model="gpt-5.4",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )

        assert kwargs["model"] == "gpt-5.4"
        assert kwargs["messages"] == [{"role": "user", "content": "ping"}]
        assert kwargs["max_completion_tokens"] == 1
        assert "max_tokens" not in kwargs

    def test_adds_stream_options_for_streaming_requests(self):
        kwargs = build_chat_completion_kwargs(
            model="gpt-5.4",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            stream=True,
        )

        assert kwargs["stream"] is True
        assert kwargs["stream_options"] == {"include_usage": True}


class TestTranslateTools:
    def test_translate_tools(self):
        anthropic_tools = [
            {
                "name": "read_file",
                "description": "Read a file",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ]
        result = _translate_tools(anthropic_tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "read_file"
        assert result[0]["function"]["description"] == "Read a file"
        assert result[0]["function"]["parameters"]["type"] == "object"
        assert "path" in result[0]["function"]["parameters"]["properties"]


class TestTranslateMessages:
    def test_plain_text_messages(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        result = _translate_messages("system prompt", msgs)
        assert result[0] == {"role": "system", "content": "system prompt"}
        assert result[1] == {"role": "user", "content": "hello"}
        assert result[2] == {"role": "assistant", "content": "hi there"}

    def test_translate_messages_with_tool_use(self):
        msgs = [
            {"role": "user", "content": "do something"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll use a tool"},
                    {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "read_file",
                        "input": {"path": "/tmp/test.txt"},
                    },
                ],
            },
        ]
        result = _translate_messages("sys", msgs)
        # system + user + assistant
        assert len(result) == 3
        assistant_msg = result[2]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"] == "I'll use a tool"
        assert len(assistant_msg["tool_calls"]) == 1
        tc = assistant_msg["tool_calls"][0]
        assert tc["id"] == "tool_1"
        assert tc["function"]["name"] == "read_file"
        assert json.loads(tc["function"]["arguments"]) == {"path": "/tmp/test.txt"}

    def test_translate_messages_with_tool_result(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_1",
                        "content": "file contents here",
                    }
                ],
            },
        ]
        result = _translate_messages("sys", msgs)
        # system + tool message
        assert len(result) == 2
        tool_msg = result[1]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "tool_1"
        assert tool_msg["content"] == "file contents here"


class TestFromSettingsOpenAI:
    def test_from_settings_openai(self):
        with patch("anton.core.llm.openai.openai"):
            settings = AntonSettings(
                planning_provider="openai",
                coding_provider="openai",
                planning_model="gpt-4.1",
                coding_model="gpt-4.1",
                openai_api_key="test-key",
                _env_file=None,
            )
            client = LLMClient.from_settings(settings)
            assert isinstance(client, LLMClient)
            assert isinstance(client._planning_provider, OpenAIProvider)
            assert isinstance(client._coding_provider, OpenAIProvider)


class TestAzureOpenAIProvider:
    def test_uses_async_azure_openai_when_api_version_set(self):
        """When api_version is provided, AsyncAzureOpenAI must be used."""
        mock_azure_client = MagicMock()
        with patch("anton.core.llm.openai.openai"), \
             patch("anton.core.llm.openai.AsyncAzureOpenAI", return_value=mock_azure_client) as mock_cls:
            provider = OpenAIProvider(
                api_key="azure-key",
                base_url="https://myresource.cognitiveservices.azure.com",
                api_version="2024-12-01-preview",
            )
            mock_cls.assert_called_once()
            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["api_version"] == "2024-12-01-preview"
            assert call_kwargs["api_key"] == "azure-key"
            assert call_kwargs["azure_endpoint"] == "https://myresource.cognitiveservices.azure.com"
            assert provider._client is mock_azure_client

    def test_uses_async_openai_when_no_api_version(self):
        """Without api_version, the standard AsyncOpenAI client must be used."""
        mock_std_client = MagicMock()
        with patch("anton.core.llm.openai.openai") as mock_openai:
            mock_openai.AsyncOpenAI.return_value = mock_std_client
            provider = OpenAIProvider(api_key="sk-test", base_url="http://localhost:11434/v1")
            mock_openai.AsyncOpenAI.assert_called_once()
            assert provider._client is mock_std_client

    def test_export_connection_info_includes_api_version(self):
        with patch("anton.core.llm.openai.openai"), \
             patch("anton.core.llm.openai.AsyncAzureOpenAI"):
            provider = OpenAIProvider(
                api_key="key",
                base_url="https://res.openai.azure.com",
                api_version="2024-12-01-preview",
            )
            info = provider.export_connection_info()
            assert info.api_version == "2024-12-01-preview"
            assert info.base_url == "https://res.openai.azure.com"

    def test_from_settings_passes_api_version_to_provider(self):
        """LLMClient.from_settings propagates openai_api_version to OpenAIProvider."""
        with patch("anton.core.llm.openai.openai"), \
             patch("anton.core.llm.openai.AsyncAzureOpenAI") as mock_azure_cls:
            settings = AntonSettings(
                planning_provider="openai-compatible",
                coding_provider="openai-compatible",
                planning_model="gpt-4.1-mini",
                coding_model="gpt-4.1-mini",
                openai_api_key="azure-key",
                openai_base_url="https://myresource.cognitiveservices.azure.com",
                openai_api_version="2024-12-01-preview",
                _env_file=None,
            )
            client = LLMClient.from_settings(settings)
            assert mock_azure_cls.called
            call_kwargs = mock_azure_cls.call_args.kwargs
            assert call_kwargs["api_version"] == "2024-12-01-preview"
            assert isinstance(client._planning_provider, OpenAIProvider)

    async def test_azure_provider_complete_calls_chat_completions(self):
        """Azure provider routes complete() through chat.completions just like standard."""
        mock_azure_client = AsyncMock()
        mock_azure_client.chat.completions.create = AsyncMock(
            return_value=_make_mock_response(content="azure response", prompt_tokens=8, completion_tokens=12)
        )
        with patch("anton.core.llm.openai.openai"), \
             patch("anton.core.llm.openai.AsyncAzureOpenAI", return_value=mock_azure_client):
            provider = OpenAIProvider(
                api_key="azure-key",
                base_url="https://myresource.cognitiveservices.azure.com",
                api_version="2024-12-01-preview",
            )
            result = await provider.complete(
                model="gpt-4.1-mini",
                system="be helpful",
                messages=[{"role": "user", "content": "hello"}],
            )
            assert result.content == "azure response"
            assert result.usage.input_tokens == 8
            assert result.usage.output_tokens == 12
            mock_azure_client.chat.completions.create.assert_awaited_once()
