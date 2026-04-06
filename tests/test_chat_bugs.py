"""Tests that reproduce known bugs in chat.py.

Each test documents a specific bug and will fail once the bug is fixed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anton.chat import ChatSession, _apply_error_tracking
from anton.llm.provider import (
    LLMResponse,
    StreamComplete,
    StreamTextDelta,
    ToolCall,
    Usage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_response(text: str, *, context_pressure: float = 0.0) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=20, context_pressure=context_pressure),
        stop_reason="end_turn",
    )


def _tool_response(
    text: str,
    tool_calls: list[ToolCall],
    *,
    context_pressure: float = 0.0,
) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=tool_calls,
        usage=Usage(input_tokens=10, output_tokens=20, context_pressure=context_pressure),
        stop_reason="tool_use",
    )


def _make_tool_call(name: str = "memorize", tc_id: str = "tc_1", **inputs) -> ToolCall:
    return ToolCall(id=tc_id, name=name, input=inputs or {"entries": []})


# ---------------------------------------------------------------------------
# Bug 1: turn() never increments _turn_count, persists history, or logs
#         episodic memory — unlike turn_stream() which does all three.
# ---------------------------------------------------------------------------

class TestBug1_TurnMissingBookkeeping:
    """The non-streaming turn() is missing turn_count increment, history
    persistence, and episodic memory logging that turn_stream() has."""

    async def test_turn_does_not_increment_turn_count(self):
        """BUG: turn() never increments _turn_count."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hi"))

        session = ChatSession(mock_llm)
        assert session._turn_count == 0

        await session.turn("hello")

        # BUG: _turn_count is still 0 after a turn
        assert session._turn_count == 0, (
            "If this fails, Bug 1 (turn_count) has been fixed!"
        )

    async def test_turn_does_not_persist_history(self):
        """BUG: turn() never calls _persist_history()."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hi"))

        mock_store = MagicMock()
        session = ChatSession(mock_llm, history_store=mock_store, session_id="s1")

        await session.turn("hello")

        # BUG: history store was never written to
        mock_store.save.assert_not_called()

    async def test_turn_does_not_log_episodic_memory(self):
        """BUG: turn() never logs to episodic memory."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hi"))

        mock_episodic = MagicMock()
        mock_episodic.enabled = True
        session = ChatSession(mock_llm, episodic=mock_episodic)

        await session.turn("hello")

        # BUG: episodic memory was never written to
        mock_episodic.log_turn.assert_not_called()

    async def test_turn_stream_does_increment_turn_count(self):
        """Contrast: turn_stream() correctly increments _turn_count."""
        mock_llm = AsyncMock()

        async def _stream(**kwargs):
            yield StreamTextDelta(text="Hi")
            yield StreamComplete(response=_text_response("Hi"))

        mock_llm.plan_stream = _stream

        session = ChatSession(mock_llm)
        assert session._turn_count == 0

        async for _ in session.turn_stream("hello"):
            pass

        assert session._turn_count == 1


# ---------------------------------------------------------------------------
# Bug 2: _apply_error_tracking matches bare "failed" substring,
#         causing false positives on successful results.
# ---------------------------------------------------------------------------

class TestBug2_ErrorTrackingFalsePositives:
    """The bare 'failed' substring in _apply_error_tracking triggers on
    successful results that merely mention the word 'failed'."""

    def test_success_mentioning_failed_triggers_error_streak(self):
        """BUG: A successful result that mentions 'failed' is counted as error."""
        error_streak: dict[str, int] = {}
        resilience_nudged: set[str] = set()

        # This is a SUCCESS message — the previous approach failed but we
        # recovered and succeeded.
        success_text = (
            "The previous approach failed, so I tried a different library. "
            "Result: data loaded successfully with 1000 rows."
        )

        result = _apply_error_tracking(
            success_text, "scratchpad", error_streak, resilience_nudged
        )

        # BUG: error_streak is 1 even though the tool succeeded
        assert error_streak["scratchpad"] == 1, (
            "If this fails, Bug 2 has been fixed!"
        )

    def test_two_success_mentions_trigger_resilience_nudge(self):
        """BUG: Two successful results mentioning 'failed' trigger resilience nudge."""
        error_streak: dict[str, int] = {}
        resilience_nudged: set[str] = set()

        msg1 = "The old method failed but the new one works perfectly."
        msg2 = "Previous URL failed to load; used a mirror instead. Success."

        _apply_error_tracking(msg1, "scratchpad", error_streak, resilience_nudged)
        result = _apply_error_tracking(msg2, "scratchpad", error_streak, resilience_nudged)

        # BUG: resilience nudge is appended even though both calls succeeded
        assert "SYSTEM: This tool has failed twice" in result, (
            "If this fails, Bug 2 has been fixed!"
        )

    def test_actual_error_correctly_detected(self):
        """Sanity check: real errors are still detected."""
        error_streak: dict[str, int] = {}
        resilience_nudged: set[str] = set()

        error_text = "[error] ModuleNotFoundError: No module named 'pandas'"
        _apply_error_tracking(error_text, "scratchpad", error_streak, resilience_nudged)

        assert error_streak["scratchpad"] == 1


# ---------------------------------------------------------------------------
# Bug 3: _summarize_history walks split backward through tool_result pairs
#         until split < 2, making tool-heavy sessions unsummarizable.
# ---------------------------------------------------------------------------

class TestBug3_UnsummarizableToolHistory:
    """Sessions dominated by tool_use/tool_result pairs can never be
    summarized because the backward walk pushes split below 2."""

    async def test_tool_heavy_history_cannot_be_summarized(self):
        """BUG: History full of tool pairs is never summarized.

        With 8 messages the split calculation yields index 4, which lands
        on a user message with tool_result content. The backward walk then
        pulls split down through every tool_result/assistant pair until
        split < 2, causing _summarize_history to return without doing
        anything.

        History layout (8 messages, split initially = 4):
            0: user "Analyze"                     <-- only plain user msg
            1: assistant [tool_use tc_0]
            2: user [tool_result tc_0]
            3: assistant [tool_use tc_1]
            4: user [tool_result tc_1]             <-- split lands here
            5: assistant [tool_use tc_2]
            6: user [tool_result tc_2]
            7: assistant "Done"

        Backward walk: 4→3→2→1, split becomes 1, which is < 2 → return.
        """
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("done"))
        mock_llm.code = AsyncMock(return_value=MagicMock(content="Summary"))

        session = ChatSession(mock_llm)

        # 8 messages: 1 user + 3*(assistant+tool_result_user) + 1 assistant
        session._history = [{"role": "user", "content": "Analyze this data"}]
        for i in range(3):
            session._history.append({
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": f"tc_{i}", "name": "scratchpad", "input": {}},
                ],
            })
            session._history.append({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": f"tc_{i}", "content": f"result {i}"},
                ],
            })
        session._history.append({"role": "assistant", "content": "Done."})

        original_len = len(session._history)
        assert original_len == 8

        await session._summarize_history()

        # BUG: History length is unchanged — summarization was a no-op
        # because the backward walk pushed split below 2.
        assert len(session._history) == original_len, (
            "If this fails, Bug 3 has been fixed!"
        )

        # The code model was never called for summarization
        mock_llm.code.assert_not_called()


# ---------------------------------------------------------------------------
# Bug 4: asyncio.create_task in _maybe_consolidate_scratchpads and
#         cortex.compact_all swallow exceptions silently.
# ---------------------------------------------------------------------------

class TestBug4_FireAndForgetExceptions:
    """Fire-and-forget tasks created with asyncio.create_task silently
    swallow exceptions."""

    async def test_consolidate_exception_is_silently_lost(self):
        """BUG: Exception in _consolidate is never surfaced."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("done"))

        mock_cortex = MagicMock()
        mock_cortex.mode = "autopilot"

        session = ChatSession(mock_llm, cortex=mock_cortex)

        # Make _consolidate raise an exception
        error = RuntimeError("consolidation exploded")
        session._consolidate = AsyncMock(side_effect=error)

        # Create a mock pad with cells that trigger consolidation
        mock_pad = MagicMock()
        mock_cell = MagicMock()
        mock_cell.error = True  # triggers should_replay
        mock_pad.cells = [mock_cell, mock_cell, mock_cell]
        session._scratchpads._pads = {"test": mock_pad}

        # Patch at the import location inside _maybe_consolidate_scratchpads
        with patch("anton.memory.consolidator.Consolidator") as MockConsolidator:
            mock_consolidator = MockConsolidator.return_value
            mock_consolidator.should_replay.return_value = True

            # This fires _consolidate as a background task
            session._maybe_consolidate_scratchpads()

            # Give the event loop a chance to run the task
            await asyncio.sleep(0.05)

        # The _consolidate method was called and raised...
        session._consolidate.assert_called_once()

        # ...but there's no error handler, no done_callback, no logging.
        # Python will emit "Task exception was never retrieved" as a warning,
        # but the error is otherwise lost. This test documents the behavior.


# ---------------------------------------------------------------------------
# Bug 5: In _stream_and_handle_tools(), when tool_round > _MAX_TOOL_ROUNDS,
#         the final LLM response after "stop retrying" is yielded as events
#         but never appended to history. The `return` bypasses the
#         unconditional append after the while loop.
# ---------------------------------------------------------------------------

class TestBug5_StreamingMissingHistoryAtRoundLimit:
    """When the tool round limit is hit in _stream_and_handle_tools(),
    the final "stop retrying" response is streamed to the user but
    never saved to history because `return` skips the post-loop append."""

    async def test_streaming_round_limit_final_response_missing_from_history(self):
        """BUG: Final response after round limit is not in history (streaming)."""
        mock_llm = AsyncMock()

        # Build 26 tool responses to exhaust the round limit, then a final text.
        tool_responses = []
        for i in range(26):
            tool_responses.append(
                _tool_response(
                    f"Trying approach {i}",
                    [_make_tool_call("memorize", f"tc_{i}", entries=[])],
                )
            )
        final_resp = _text_response("I was unable to complete the task.")

        stream_call_count = 0

        async def _fake_plan_stream(**kwargs):
            nonlocal stream_call_count
            resp = ([*tool_responses, final_resp])[stream_call_count]
            stream_call_count += 1
            if resp.content:
                yield StreamTextDelta(text=resp.content)
            yield StreamComplete(response=resp)

        mock_llm.plan_stream = _fake_plan_stream

        session = ChatSession(mock_llm)

        # Collect all streamed events
        events = []
        with patch("anton.chat.dispatch_tool", new_callable=AsyncMock, return_value="ok"):
            async for event in session.turn_stream("do something complex"):
                events.append(event)

        # The final "stop retrying" text WAS streamed to the user
        text_events = [e for e in events if isinstance(e, StreamTextDelta)]
        final_texts = [e.text for e in text_events if "unable" in e.text]
        assert len(final_texts) >= 1, "Final response should have been streamed"

        # BUG: But the final response is NOT in history.
        # The `return` in the round-limit branch of _stream_and_handle_tools
        # skips the post-loop `self._history.append(...)` that would save it.
        assistant_msgs = [
            msg for msg in session.history
            if msg.get("role") == "assistant"
            and isinstance(msg.get("content"), str)
            and "unable" in msg["content"]
        ]
        assert len(assistant_msgs) == 0, (
            "If this fails, Bug 5 has been fixed!"
        )
