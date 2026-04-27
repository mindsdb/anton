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


class TestHardTruncateHistory:
    def _make_session(self) -> ChatSession:
        return ChatSession(ChatSessionConfig(llm_client=make_mock_llm()))

    def test_noop_when_history_short(self):
        session = self._make_session()
        session._history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        before = list(session._history)
        session.hard_truncate_history(keep=4)
        assert session._history == before

    def test_preserves_pair_boundaries(self):
        session = self._make_session()
        session._history = [
            {"role": "user", "content": "old 1"},
            {"role": "assistant", "content": "old reply"},
            {"role": "user", "content": "old 2"},
            {"role": "assistant", "content": "another old reply"},
            {"role": "user", "content": "recent"},
            {"role": "assistant", "content": "recent reply"},
        ]
        session.hard_truncate_history(keep=2)
        # placeholder + separator + tail
        assert len(session._history) == 4
        assert session._history[0]["role"] == "user"
        assert "truncated" in session._history[0]["content"]
        assert session._history[1]["role"] == "assistant"
        assert session._history[-2] == {"role": "user", "content": "recent"}
        assert session._history[-1] == {"role": "assistant", "content": "recent reply"}

    def test_drops_orphaned_tool_result_and_exposed_assistant(self):
        """Regression: when the tail starts with assistant → user(tool_result only)
        → assistant → user, dropping the orphaned tool_result must not leave
        two consecutive assistant messages at the head of the final history.
        """
        session = self._make_session()
        session._history = [
            {"role": "user", "content": "very old"},
            {"role": "assistant", "content": "very old reply"},
            # These four are the tail (keep=4):
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "x", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
            ]},
            {"role": "assistant", "content": "I analyzed the tool result"},
            {"role": "user", "content": "thanks"},
        ]
        session.hard_truncate_history(keep=4)

        # No two consecutive same-role messages anywhere in the result.
        roles = [m["role"] for m in session._history]
        for i in range(len(roles) - 1):
            assert roles[i] != roles[i + 1], (
                f"consecutive same-role at {i}: {roles}"
            )
        # First message must be user (API rule).
        assert roles[0] == "user"
        # The final real user message should still be present.
        assert session._history[-1] == {"role": "user", "content": "thanks"}

    def test_filters_tool_result_from_mixed_head(self):
        """A user message with mixed text + tool_result content at the
        head keeps its text blocks; only the orphaned tool_result is stripped.
        """
        session = self._make_session()
        session._history = [
            {"role": "user", "content": "very old"},
            {"role": "assistant", "content": "very old reply"},
            # Tail starts here (keep=3):
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "x", "content": "data"},
                {"type": "text", "text": "plus my follow-up question"},
            ]},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "ok"},
        ]
        session.hard_truncate_history(keep=3)

        # First non-placeholder user message retains its text block only.
        tail_head = session._history[2]
        assert tail_head["role"] == "user"
        assert tail_head["content"] == [
            {"type": "text", "text": "plus my follow-up question"},
        ]
