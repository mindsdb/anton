from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from anton.chat import ChatSession
from anton.llm.provider import (
    LLMResponse,
    StreamComplete,
    StreamTaskProgress,
    StreamTextDelta,
    ToolCall,
    Usage,
)


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="end_turn",
    )


def _tool_response(text: str, task: str, tool_id: str = "tc_1") -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[
            ToolCall(id=tool_id, name="execute_task", input={"task": task}),
        ],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="tool_use",
    )


class TestChatSession:
    async def test_conversational_turn(self):
        """Text-only response for casual conversation."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hey! How can I help?"))
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run)
        reply = await session.turn("hi")

        assert reply == "Hey! How can I help?"
        mock_run.assert_not_awaited()
        assert len(session.history) == 2  # user + assistant

    async def test_tool_call_delegates_to_agent(self):
        """When LLM calls execute_task, it delegates to the run callback."""
        mock_llm = AsyncMock()
        # First call returns tool_use, second call (after tool result) returns text
        mock_llm.plan = AsyncMock(
            side_effect=[
                _tool_response("Let me do that.", "list all Python files"),
                _text_response("Done! Found 12 Python files."),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run)
        reply = await session.turn("list all python files")

        assert reply == "Done! Found 12 Python files."
        mock_run.assert_awaited_once_with("list all Python files")
        # user, assistant(tool_use), user(tool_result), assistant(text)
        assert len(session.history) == 4

    async def test_tool_call_failure_reported(self):
        """When the task raises an exception, the error is fed back to the LLM."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _tool_response("On it.", "do something risky"),
                _text_response("That didn't work. Want me to try a different approach?"),
            ]
        )
        mock_run = AsyncMock(side_effect=RuntimeError("skill not found"))

        session = ChatSession(mock_llm, mock_run)
        reply = await session.turn("do something risky")

        assert "different approach" in reply
        mock_run.assert_awaited_once()

    async def test_history_grows_across_turns(self):
        """Multiple turns accumulate in history."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _text_response("Hi there!"),
                _text_response("Sure, what repo?"),
                _text_response("Got it, I'll look into that."),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run)
        await session.turn("hello")
        await session.turn("can you check something")
        await session.turn("the anton repo")

        # 3 user messages + 3 assistant messages
        assert len(session.history) == 6
        assert session.history[0]["role"] == "user"
        assert session.history[1]["role"] == "assistant"

    async def test_tool_result_format(self):
        """Tool result messages follow the Anthropic protocol format."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _tool_response("Running.", "test task", tool_id="tc_abc"),
                _text_response("All done."),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run)
        await session.turn("do it")

        # Check the tool_result message
        tool_result_msg = session.history[2]
        assert tool_result_msg["role"] == "user"
        assert isinstance(tool_result_msg["content"], list)
        assert tool_result_msg["content"][0]["type"] == "tool_result"
        assert tool_result_msg["content"][0]["tool_use_id"] == "tc_abc"

    async def test_assistant_tool_use_message_format(self):
        """Assistant messages with tool calls use the content-blocks format."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _tool_response("Thinking...", "analyze code", tool_id="tc_99"),
                _text_response("Analysis complete."),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run)
        await session.turn("analyze the code")

        # Check the assistant tool_use message
        assistant_msg = session.history[1]
        assert assistant_msg["role"] == "assistant"
        assert isinstance(assistant_msg["content"], list)
        blocks = assistant_msg["content"]
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "tool_use"
        assert blocks[1]["id"] == "tc_99"
        assert blocks[1]["name"] == "execute_task"

    async def test_empty_content_with_tool_call(self):
        """Tool call with no accompanying text still works."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="tc_1", name="execute_task", input={"task": "run tests"}),
                    ],
                    usage=Usage(),
                    stop_reason="tool_use",
                ),
                _text_response("Tests passed!"),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run)
        reply = await session.turn("run the tests")

        assert reply == "Tests passed!"
        mock_run.assert_awaited_once_with("run tests")
        # assistant content blocks should only have tool_use (no empty text block)
        assistant_msg = session.history[1]
        assert len(assistant_msg["content"]) == 1
        assert assistant_msg["content"][0]["type"] == "tool_use"


# --- Helpers for streaming tests ---

async def _fake_plan_stream(events):
    """Return an async generator factory that yields events from a list of event sequences."""
    call_count = 0

    async def _gen(**kwargs):
        nonlocal call_count
        for ev in events[call_count]:
            yield ev
        call_count += 1

    return _gen


class TestChatSessionStreaming:
    async def test_turn_stream_yields_text_deltas(self):
        """Streaming turn yields text deltas and updates history."""
        mock_llm = AsyncMock()

        async def _stream(**kwargs):
            yield StreamTextDelta(text="Hello ")
            yield StreamTextDelta(text="world!")
            yield StreamComplete(response=_text_response("Hello world!"))

        mock_llm.plan_stream = _stream
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run)
        events = []
        async for event in session.turn_stream("hi"):
            events.append(event)

        # Should have 2 text deltas + 1 complete
        text_deltas = [e for e in events if isinstance(e, StreamTextDelta)]
        completes = [e for e in events if isinstance(e, StreamComplete)]
        assert len(text_deltas) == 2
        assert text_deltas[0].text == "Hello "
        assert text_deltas[1].text == "world!"
        assert len(completes) == 1

        # History: user + assistant
        assert len(session.history) == 2
        assert session.history[1]["content"] == "Hello world!"

    async def test_turn_stream_handles_tool_calls(self):
        """Streaming turn handles tool-call loop across two LLM calls."""
        mock_llm = AsyncMock()
        call_count = 0

        async def _stream(**kwargs):
            nonlocal call_count
            if call_count == 0:
                # First call: tool use
                yield StreamTextDelta(text="Let me do that.")
                yield StreamComplete(
                    response=_tool_response("Let me do that.", "list files")
                )
            else:
                # Second call: text-only follow-up
                yield StreamTextDelta(text="Done!")
                yield StreamComplete(response=_text_response("Done!"))
            call_count += 1

        mock_llm.plan_stream = _stream
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run)
        events = []
        async for event in session.turn_stream("list files"):
            events.append(event)

        mock_run.assert_awaited_once_with("list files")

        # Should have deltas from both calls
        text_deltas = [e for e in events if isinstance(e, StreamTextDelta)]
        assert len(text_deltas) == 2

        # History: user, assistant(tool_use), user(tool_result), assistant(text)
        assert len(session.history) == 4

    async def test_turn_stream_yields_task_progress(self):
        """Streaming turn with run_task_stream yields StreamTaskProgress events."""
        mock_llm = AsyncMock()
        call_count = 0

        async def _stream(**kwargs):
            nonlocal call_count
            if call_count == 0:
                yield StreamComplete(
                    response=_tool_response("", "build it")
                )
            else:
                yield StreamTextDelta(text="Done!")
                yield StreamComplete(response=_text_response("Done!"))
            call_count += 1

        mock_llm.plan_stream = _stream

        async def _fake_run_task_stream(task: str):
            yield StreamTaskProgress(phase="planning", message="Analyzing task...")
            yield StreamTaskProgress(phase="executing", message="Step 1/2: read", eta_seconds=5.0)
            yield StreamTaskProgress(phase="executing", message="Step 2/2: write", eta_seconds=2.0)

        mock_run = AsyncMock()
        session = ChatSession(mock_llm, mock_run, run_task_stream=_fake_run_task_stream)
        events = []
        async for event in session.turn_stream("build it"):
            events.append(event)

        progress = [e for e in events if isinstance(e, StreamTaskProgress)]
        assert len(progress) == 3
        assert progress[0].phase == "planning"
        assert progress[1].eta_seconds == 5.0
        # run_task (non-streaming) should NOT have been called
        mock_run.assert_not_awaited()
