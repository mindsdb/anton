from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from anton.llm.ollama import (
    OllamaProvider,
    normalize_ollama_base_url,
    translate_messages_to_ollama,
)
from anton.llm.provider import (
    StreamComplete,
    StreamTaskProgress,
    StreamTextDelta,
    StreamToolUseDelta,
    StreamToolUseEnd,
    StreamToolUseStart,
)


class _AsyncIter:
    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


def _ollama_response(*, content="", tool_calls=None, prompt_tokens=5, completion_tokens=10, done_reason="stop"):
    return SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=tool_calls or [], thinking=None),
        prompt_eval_count=prompt_tokens,
        eval_count=completion_tokens,
        done_reason=done_reason,
    )


class TestOllamaProvider:
    async def test_complete_text_response(self):
        with patch("anton.llm.ollama.ollama") as mock_ollama:
            mock_client = AsyncMock()
            mock_ollama.AsyncClient.return_value = mock_client
            mock_client.chat = AsyncMock(return_value=_ollama_response(content="Hello world"))

            provider = OllamaProvider(base_url="http://localhost:11434/v1")
            result = await provider.complete(
                model="qwen3.5:4b",
                system="be helpful",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=123,
                request_options={"think": False},
            )

            call_kwargs = mock_client.chat.call_args.kwargs
            assert call_kwargs["think"] is False
            assert call_kwargs["options"] == {"num_predict": 123}
            assert result.content == "Hello world"
            assert result.tool_calls == []
            assert result.usage.input_tokens == 5
            assert result.usage.output_tokens == 10
            assert result.stop_reason == "stop"

    async def test_complete_tool_use_response(self):
        with patch("anton.llm.ollama.ollama") as mock_ollama:
            mock_client = AsyncMock()
            mock_ollama.AsyncClient.return_value = mock_client
            tool_call = SimpleNamespace(
                function=SimpleNamespace(name="lookup_weather", arguments={"city": "London"})
            )
            mock_client.chat = AsyncMock(
                return_value=_ollama_response(content="", tool_calls=[tool_call], done_reason="tool_calls")
            )

            provider = OllamaProvider(base_url="http://localhost:11434")
            result = await provider.complete(
                model="qwen3.5:4b",
                system="plan",
                messages=[{"role": "user", "content": "weather?"}],
                tools=[{"name": "lookup_weather", "description": "d", "input_schema": {"type": "object"}}],
            )

            assert result.content == ""
            assert len(result.tool_calls) == 1
            assert result.tool_calls[0].id == "ollama_tool_1"
            assert result.tool_calls[0].name == "lookup_weather"
            assert result.tool_calls[0].input == {"city": "London"}
            assert result.stop_reason == "tool_calls"

    async def test_stream_emits_reasoning_progress_and_tool_events(self):
        with patch("anton.llm.ollama.ollama") as mock_ollama:
            mock_client = AsyncMock()
            mock_ollama.AsyncClient.return_value = mock_client
            tool_call = SimpleNamespace(
                function=SimpleNamespace(name="lookup_weather", arguments={"city": "London"})
            )
            mock_client.chat = AsyncMock(
                return_value=_AsyncIter([
                    SimpleNamespace(
                        message=SimpleNamespace(content="", thinking="Working", tool_calls=[]),
                        prompt_eval_count=5,
                        eval_count=0,
                        done_reason=None,
                    ),
                    SimpleNamespace(
                        message=SimpleNamespace(content="Done.", thinking=None, tool_calls=[tool_call]),
                        prompt_eval_count=None,
                        eval_count=6,
                        done_reason="stop",
                    ),
                ])
            )

            provider = OllamaProvider(base_url="http://localhost:11434")
            events = [event async for event in provider.stream(
                model="qwen3.5:4b",
                system="sys",
                messages=[{"role": "user", "content": "hi"}],
            )]

            assert any(isinstance(event, StreamTaskProgress) and event.phase == "reasoning" for event in events)
            assert any(isinstance(event, StreamTextDelta) and event.text == "Done." for event in events)
            assert any(isinstance(event, StreamToolUseStart) and event.name == "lookup_weather" for event in events)
            assert any(isinstance(event, StreamToolUseDelta) for event in events)
            assert any(isinstance(event, StreamToolUseEnd) for event in events)
            complete = next(event for event in events if isinstance(event, StreamComplete))
            assert complete.response.content == "Done."
            assert complete.response.tool_calls[0].input == {"city": "London"}


class TestOllamaHelpers:
    def test_normalize_ollama_base_url_strips_v1(self):
        assert normalize_ollama_base_url("localhost:11434/v1") == "http://localhost:11434"

    def test_translate_messages_to_ollama_handles_tool_results(self):
        messages = [
            {"role": "user", "content": "Find the file"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I will check."},
                    {"type": "tool_use", "id": "tool_1", "name": "read_file", "input": {"path": "/tmp/test.txt"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tool_1", "content": "hello"},
                ],
            },
        ]

        translated = translate_messages_to_ollama("system", messages)

        assert translated[0] == {"role": "system", "content": "system"}
        assert translated[2]["role"] == "assistant"
        assert translated[2]["tool_calls"][0]["function"]["name"] == "read_file"
        assert translated[3] == {"role": "tool", "tool_name": "read_file", "content": "hello"}
