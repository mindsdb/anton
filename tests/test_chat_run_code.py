from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

import anton.chat as chat_module
from anton.chat import ChatSession, RUN_CODE_TOOL
from anton.llm.provider import LLMResponse, StreamComplete, ToolCall, Usage


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="end_turn",
    )


def _run_code_response(
    text: str, code: str, tool_id: str = "tc_rc_1"
) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[
            ToolCall(
                id=tool_id,
                name="run_code",
                input={"code": code},
            ),
        ],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="tool_use",
    )


class TestRunCodeToolDefinition:
    def test_tool_definition_structure(self):
        assert RUN_CODE_TOOL["name"] == "run_code"
        props = RUN_CODE_TOOL["input_schema"]["properties"]
        assert "code" in props
        assert RUN_CODE_TOOL["input_schema"]["required"] == ["code"]

    async def test_run_code_tool_always_in_tools(self):
        """run_code should always be in _build_tools() output."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hi!"))
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run)
        await session.turn("hello")

        call_kwargs = mock_llm.plan.call_args
        tools = call_kwargs.kwargs.get("tools", [])
        tool_names = [t["name"] for t in tools]
        assert "run_code" in tool_names


class TestRunCodeExecution:
    async def test_run_code_basic_output(self):
        """print(2 + 2) should return '4'."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _run_code_response("Let me compute that.", "print(2 + 2)"),
                _text_response("The answer is 4."),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run)
        reply = await session.turn("what is 2 + 2?")

        # Check tool result in history
        tool_result_msgs = [
            m for m in session.history
            if m["role"] == "user" and isinstance(m["content"], list)
        ]
        assert len(tool_result_msgs) == 1
        result_content = tool_result_msgs[0]["content"][0]["content"]
        assert "4" in result_content

    async def test_run_code_stderr_included(self):
        """stderr output should appear with [stderr] label."""
        code = "import sys; sys.stderr.write('warning here')"
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _run_code_response("Running.", code),
                _text_response("Done."),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run)
        await session.turn("run it")

        tool_result_msgs = [
            m for m in session.history
            if m["role"] == "user" and isinstance(m["content"], list)
        ]
        result_content = tool_result_msgs[0]["content"][0]["content"]
        assert "[stderr]" in result_content
        assert "warning here" in result_content

    async def test_run_code_error_returns_traceback(self):
        """A script that raises should return the traceback."""
        code = 'raise ValueError("boom")'
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _run_code_response("Running.", code),
                _text_response("It failed."),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run)
        await session.turn("run it")

        tool_result_msgs = [
            m for m in session.history
            if m["role"] == "user" and isinstance(m["content"], list)
        ]
        result_content = tool_result_msgs[0]["content"][0]["content"]
        assert "ValueError" in result_content
        assert "boom" in result_content

    async def test_run_code_empty_code(self):
        """Empty code string returns 'No code provided.'"""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _run_code_response("Running.", ""),
                _text_response("No code."),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run)
        await session.turn("run nothing")

        tool_result_msgs = [
            m for m in session.history
            if m["role"] == "user" and isinstance(m["content"], list)
        ]
        result_content = tool_result_msgs[0]["content"][0]["content"]
        assert result_content == "No code provided."

    async def test_run_code_no_output(self):
        """Code with no print returns success message."""
        code = "x = 42"
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _run_code_response("Running.", code),
                _text_response("Done."),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run)
        await session.turn("run it")

        tool_result_msgs = [
            m for m in session.history
            if m["role"] == "user" and isinstance(m["content"], list)
        ]
        result_content = tool_result_msgs[0]["content"][0]["content"]
        assert result_content == "Code executed successfully (no output)."

    async def test_run_code_timeout(self, monkeypatch):
        """Code that runs too long should be killed and return timeout message."""
        monkeypatch.setattr(chat_module, "_RUN_CODE_TIMEOUT", 1)
        code = "import time; time.sleep(60)"
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _run_code_response("Running.", code),
                _text_response("Timed out."),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run)
        await session.turn("run it")

        tool_result_msgs = [
            m for m in session.history
            if m["role"] == "user" and isinstance(m["content"], list)
        ]
        result_content = tool_result_msgs[0]["content"][0]["content"]
        assert "timed out" in result_content.lower()

    async def test_run_code_output_truncation(self, monkeypatch):
        """Output longer than 10k chars is truncated."""
        code = "print('x' * 20000)"
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _run_code_response("Running.", code),
                _text_response("Done."),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run)
        await session.turn("run it")

        tool_result_msgs = [
            m for m in session.history
            if m["role"] == "user" and isinstance(m["content"], list)
        ]
        result_content = tool_result_msgs[0]["content"][0]["content"]
        assert "truncated" in result_content
        assert len(result_content) < 20000


class _FakeAsyncIter:
    """Wraps items into an async iterator for mocking plan_stream."""

    def __init__(self, items):
        self._items = items

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


class TestRunCodeStreaming:
    async def test_run_code_in_streaming_path(self):
        """run_code should work in turn_stream() too."""
        code = "print(3 * 7)"

        tool_response = _run_code_response("Computing.", code)
        final_response = _text_response("The answer is 21.")

        mock_llm = AsyncMock()

        call_count = 0

        def fake_plan_stream(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _FakeAsyncIter([StreamComplete(response=tool_response)])
            return _FakeAsyncIter([StreamComplete(response=final_response)])

        mock_llm.plan_stream = fake_plan_stream
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run)

        events = []
        async for event in session.turn_stream("what is 3 * 7?"):
            events.append(event)

        # Should have gotten StreamComplete events
        assert any(isinstance(e, StreamComplete) for e in events)

        # Check tool result in history
        tool_result_msgs = [
            m for m in session.history
            if m["role"] == "user" and isinstance(m["content"], list)
        ]
        assert len(tool_result_msgs) == 1
        result_content = tool_result_msgs[0]["content"][0]["content"]
        assert "21" in result_content
