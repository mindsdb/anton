from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from anton.chat import ChatSession
from anton.core.session import ChatSessionConfig
from tests.conftest import make_mock_llm, run_turn
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
        reply = await run_turn(session, "hi")

        assert reply == "Hey! How can I help?"
        assert len(session.history) == 2  # user + assistant

    async def test_non_streaming_turn_wraps_turn_stream(self):
        """`turn()` is a compatibility wrapper for external anton-agent
        consumers: it must return the final reply and route through the
        streaming path (there is no parallel non-streaming loop anymore)."""
        mock_llm = make_mock_llm()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hey! How can I help?"))

        session = ChatSession(ChatSessionConfig(llm_client=mock_llm))
        reply = await session.turn("hi")

        assert reply == "Hey! How can I help?"
        assert len(session.history) == 2  # user + assistant
        # Routed through the streaming path, not a separate implementation.
        assert mock_llm.plan.await_count == 1

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
        await run_turn(session, "hello")
        await run_turn(session, "can you check something")
        await run_turn(session, "the anton repo")

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

    async def test_failed_compaction_emits_no_event_and_closes_gate(self):
        """A failed/no-op compaction (returns False) must not emit
        StreamContextCompacted and must leave _compacted_this_turn False —
        reverting the `if compacted:` guards to unconditional breaks this. It
        also sets _compaction_failed_this_turn so the proactive check won't
        re-fire for the rest of the turn (ENG-1274 #1)."""
        async def _plan_stream(**kwargs):
            yield StreamComplete(
                response=LLMResponse(content="Done", usage=Usage(context_pressure=0.9))
            )

        session = ChatSession(ChatSessionConfig(llm_client=make_mock_llm()))
        session._llm.plan_stream = _plan_stream
        session._llm.plan = AsyncMock(return_value=_text_response("STATUS: COMPLETE — done"))
        session._summarize_history = AsyncMock(return_value=False)

        events = [e async for e in session.turn_stream("hello")]

        assert session._summarize_history.call_count == 1
        assert not any(isinstance(e, StreamContextCompacted) for e in events)
        assert session._compacted_this_turn is False
        assert session._compaction_failed_this_turn is True

    async def test_failed_compaction_overflow_plus_proactive_bounded(self):
        """Overflow (reactive path) followed by a high-pressure retry
        (proactive path) is the ceiling for a non-looping turn: two summarize
        calls, and no StreamContextCompacted on a failed attempt (ENG-1274 #1).
        The across-tool-rounds test below is what pins the proactive guard."""
        call_count = 0

        async def _plan_stream(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ContextOverflowError("overflow")
            yield StreamComplete(
                response=LLMResponse(content="Done", usage=Usage(context_pressure=0.9))
            )

        session = ChatSession(ChatSessionConfig(llm_client=make_mock_llm()))
        session._llm.plan_stream = _plan_stream
        session._llm.plan = AsyncMock(return_value=_text_response("STATUS: COMPLETE — done"))
        session._summarize_history = AsyncMock(return_value=False)

        events = [e async for e in session.turn_stream("hello")]

        assert session._summarize_history.call_count == 2
        assert not any(isinstance(e, StreamContextCompacted) for e in events)
        assert session._compaction_failed_this_turn is True

    # The unknown-tool dispatch path touches the AsyncMock llm client and
    # leaks one never-awaited coroutine per turn — a test-double artifact (the
    # real provider awaits/schedules it), not product behaviour.
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    async def test_failed_compaction_not_retried_across_tool_rounds(self):
        """Several tool rounds all above the pressure threshold must still
        yield at most one proactive compaction attempt. This is what actually
        pins the `and not self._compaction_failed_this_turn` guard: remove it
        and the count climbs to one summarize call per round (ENG-1274 #1).
        `<=` rather than `==` so it isn't a tripwire on unrelated recovery-path
        changes."""
        rounds = 6

        async def _plan_stream(**kwargs):
            nonlocal rounds
            rounds -= 1
            if rounds > 0:
                yield StreamComplete(response=LLMResponse(
                    content="",
                    tool_calls=[ToolCall(id=f"t{rounds}", name="no_such_tool", input={})],
                    usage=Usage(context_pressure=0.9),
                    stop_reason="tool_use",
                ))
            else:
                yield StreamComplete(response=LLMResponse(
                    content="done",
                    usage=Usage(context_pressure=0.9),
                    stop_reason="end_turn",
                ))

        session = ChatSession(ChatSessionConfig(llm_client=make_mock_llm()))
        session._llm.plan_stream = _plan_stream
        session._llm.plan = AsyncMock(return_value=_text_response("STATUS: COMPLETE — done"))
        session._summarize_history = AsyncMock(return_value=False)

        events = [e async for e in session.turn_stream("hello")]

        assert session._summarize_history.call_count <= 2


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
