from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from anton.chat import ChatSession
from anton.llm.provider import (
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
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hey! How can I help?"))

        session = ChatSession(mock_llm)
        reply = await session.turn("hi")

        assert reply == "Hey! How can I help?"
        assert len(session.history) == 2  # user + assistant

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

        session = ChatSession(mock_llm)
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
        mock_llm = AsyncMock()

        async def _stream(**kwargs):
            yield StreamTextDelta(text="Hello ")
            yield StreamTextDelta(text="world!")
            yield StreamComplete(response=_text_response("Hello world!"))

        mock_llm.plan_stream = _stream

        session = ChatSession(mock_llm)
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
                yield  # make it an async generator
            else:
                yield StreamComplete(
                    response=LLMResponse(
                        content="Done",
                        usage=Usage(context_pressure=0.9),
                    )
                )

        session = ChatSession(AsyncMock())
        session._llm.plan_stream = _plan_stream
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

        session = ChatSession(AsyncMock())
        session._llm.plan_stream = _plan_stream
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

        session = ChatSession(AsyncMock())
        session._llm.plan_stream = _plan_stream
        session._summarize_history = AsyncMock()

        events = [e async for e in session.turn_stream("hello")]

        session._summarize_history.assert_not_called()
        compacted = [e for e in events if isinstance(e, StreamContextCompacted)]
        assert len(compacted) == 0


class _FakeAsyncIter:
    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


class TestOllamaVerifier:
    async def test_ollama_verifier_disables_thinking(self, monkeypatch):
        mock_llm = AsyncMock()
        mock_llm.planning_provider_name = "ollama"
        mock_llm.plan = AsyncMock(return_value=_text_response("STATUS: COMPLETE — done"))

        calls = 0

        def fake_plan_stream(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return _FakeAsyncIter([
                    StreamComplete(
                        response=LLMResponse(
                            content="",
                            tool_calls=[ToolCall(id="tool_1", name="scratchpad", input={"action": "view", "name": "main"})],
                            usage=Usage(),
                            stop_reason="tool_use",
                        )
                    )
                ])
            return _FakeAsyncIter([
                StreamComplete(response=_text_response("All done."))
            ])

        mock_llm.plan_stream = fake_plan_stream

        async def fake_dispatch_tool(session, name, input):
            return "tool ok"

        monkeypatch.setattr("anton.chat.dispatch_tool", fake_dispatch_tool)

        session = ChatSession(mock_llm)
        events = [event async for event in session.turn_stream("help")]

        assert any(isinstance(event, StreamComplete) for event in events)
        assert mock_llm.plan.await_count == 1
        assert mock_llm.plan.call_args.kwargs["request_options"] == {"think": False}

    async def test_ollama_unparseable_verifier_stops_retry_loop(self, monkeypatch):
        mock_llm = AsyncMock()
        mock_llm.planning_provider_name = "ollama"
        mock_llm.plan = AsyncMock(return_value=_text_response(""))

        calls = 0

        def fake_plan_stream(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return _FakeAsyncIter([
                    StreamComplete(
                        response=LLMResponse(
                            content="",
                            tool_calls=[ToolCall(id="tool_1", name="scratchpad", input={"action": "view", "name": "main"})],
                            usage=Usage(),
                            stop_reason="tool_use",
                        )
                    )
                ])
            return _FakeAsyncIter([
                StreamComplete(response=_text_response("All done."))
            ])

        mock_llm.plan_stream = fake_plan_stream

        async def fake_dispatch_tool(session, name, input):
            return "tool ok"

        monkeypatch.setattr("anton.chat.dispatch_tool", fake_dispatch_tool)

        session = ChatSession(mock_llm)
        events = [event async for event in session.turn_stream("help")]

        assert any(isinstance(event, StreamComplete) for event in events)
        assert calls == 2

    async def test_non_ollama_unparseable_verifier_continues_working(self, monkeypatch):
        mock_llm = AsyncMock()
        mock_llm.planning_provider_name = "anthropic"
        mock_llm.plan = AsyncMock(return_value=_text_response(""))

        calls = 0

        def fake_plan_stream(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return _FakeAsyncIter([
                    StreamComplete(
                        response=LLMResponse(
                            content="",
                            tool_calls=[ToolCall(id="tool_1", name="scratchpad", input={"action": "view", "name": "main"})],
                            usage=Usage(),
                            stop_reason="tool_use",
                        )
                    )
                ])
            return _FakeAsyncIter([
                StreamComplete(response=_text_response("All done."))
            ])

        mock_llm.plan_stream = fake_plan_stream

        async def fake_dispatch_tool(session, name, input):
            return "tool ok"

        monkeypatch.setattr("anton.chat.dispatch_tool", fake_dispatch_tool)

        session = ChatSession(mock_llm)
        events = [event async for event in session.turn_stream("help")]

        assert any(isinstance(event, StreamComplete) for event in events)
        assert calls == 3
        assert mock_llm.plan.call_args.kwargs["request_options"] is None
