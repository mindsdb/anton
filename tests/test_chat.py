from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from anton.chat import ChatSession
from anton.core.session import ChatSessionConfig
from tests.conftest import make_mock_llm
from anton.core.llm.provider import (
    ContextOverflowError,
    LLMResponse,
    StreamComplete,
    StreamContextCompacted,
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


class TestChatSession:
    async def test_conversational_turn(self):
        """Text-only response for casual conversation."""
        mock_llm = make_mock_llm()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hey! How can I help?"))

        session = ChatSession(ChatSessionConfig(llm_client=mock_llm))
        reply = await session.turn("hi")

        assert reply == "Hey! How can I help?"
        assert len(session.history) == 2  # user + assistant

    async def test_history_grows_across_turns(self):
        """Multiple turns accumulate in history."""
        mock_llm = make_mock_llm()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _text_response("Hi there!"),
                _text_response("Sure, what repo?"),
                _text_response("Got it, I'll look into that."),
            ]
        )

        session = ChatSession(ChatSessionConfig(llm_client=mock_llm))
        await session.turn("hello")
        await session.turn("can you check something")
        await session.turn("the anton repo")

        # 3 user messages + 3 assistant messages
        assert len(session.history) == 6
        assert session.history[0]["role"] == "user"
        assert session.history[1]["role"] == "assistant"


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
        mock_llm = make_mock_llm()

        async def _stream(**kwargs):
            yield StreamTextDelta(text="Hello ")
            yield StreamTextDelta(text="world!")
            yield StreamComplete(response=_text_response("Hello world!"))

        mock_llm.plan_stream = _stream

        session = ChatSession(ChatSessionConfig(llm_client=mock_llm))
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


class TestContextCompaction:
    async def test_overflow_then_high_pressure_summarizes_once(self):
        """If the first LLM call overflows and the retry comes back with high
        context pressure, _summarize_history must only be called once — not twice."""
        call_count = 0

        async def _plan_stream(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ContextOverflowError("overflow")
            else:
                yield StreamComplete(
                    response=LLMResponse(
                        content="Done",
                        usage=Usage(context_pressure=0.9),
                    )
                )

        session = ChatSession(ChatSessionConfig(llm_client=make_mock_llm()))
        session._llm.plan_stream = _plan_stream
        session._llm.plan = AsyncMock(return_value=_text_response("STATUS: COMPLETE — done"))
        session._summarize_history = AsyncMock()

        events = [e async for e in session.turn_stream("hello")]

        assert session._summarize_history.call_count == 1
        compacted = [e for e in events if isinstance(e, StreamContextCompacted)]
        assert len(compacted) == 1

    async def test_high_pressure_alone_summarizes_once(self):
        """A single response above the pressure threshold triggers exactly one compaction."""
        async def _plan_stream(**kwargs):
            yield StreamComplete(
                response=LLMResponse(
                    content="Done",
                    usage=Usage(context_pressure=0.9),
                )
            )

        session = ChatSession(ChatSessionConfig(llm_client=make_mock_llm()))
        session._llm.plan_stream = _plan_stream
        session._llm.plan = AsyncMock(return_value=_text_response("STATUS: COMPLETE — done"))
        session._summarize_history = AsyncMock()

        events = [e async for e in session.turn_stream("hello")]

        assert session._summarize_history.call_count == 1
        compacted = [e for e in events if isinstance(e, StreamContextCompacted)]
        assert len(compacted) == 1

    async def test_normal_turn_does_not_summarize(self):
        """A normal turn with no overflow and low pressure never triggers compaction."""
        async def _plan_stream(**kwargs):
            yield StreamComplete(
                response=LLMResponse(
                    content="Hello!",
                    usage=Usage(context_pressure=0.1),
                )
            )

        session = ChatSession(ChatSessionConfig(llm_client=make_mock_llm()))
        session._llm.plan_stream = _plan_stream
        session._llm.plan = AsyncMock(return_value=_text_response("STATUS: COMPLETE — done"))
        session._summarize_history = AsyncMock()

        events = [e async for e in session.turn_stream("hello")]

        session._summarize_history.assert_not_called()
        compacted = [e for e in events if isinstance(e, StreamContextCompacted)]
        assert len(compacted) == 0
