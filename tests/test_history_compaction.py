"""Tests for the persisted-summary compaction result (ENG-664).

Covers `ChatSession.last_compaction` (what a host reads to persist the
summary + cutoff) and the "skip if there's little new material" guard in
`_summarize_history`. The once-per-turn guard (`_compacted_this_turn`) is
already covered by `TestContextCompaction` in test_chat.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from anton.core.session import ChatSession, ChatSessionConfig, _COMPACTED_MARKER
from anton.core.llm.provider import LLMResponse, Usage
from tests.conftest import make_mock_llm


def _msg(role: str, text: str) -> dict:
    return {"role": role, "content": text}


def _summarize_response(text: str) -> LLMResponse:
    return LLMResponse(content=text, usage=Usage(input_tokens=10, output_tokens=20))


def _alternating_history(n: int, body: str) -> list[dict]:
    return [_msg("user" if i % 2 == 0 else "assistant", body) for i in range(n)]


class TestLastCompaction:
    async def test_none_before_any_compaction(self):
        session = ChatSession(ChatSessionConfig(
            llm_client=make_mock_llm(), initial_history=_alternating_history(10, "hi"),
        ))
        assert session.last_compaction is None

    async def test_populated_after_compaction(self):
        original = _alternating_history(10, "x" * 50)
        mock_llm = make_mock_llm()
        mock_llm.summarize = AsyncMock(return_value=_summarize_response("## Goal\nTest goal"))

        session = ChatSession(ChatSessionConfig(llm_client=mock_llm, initial_history=original))
        await session._summarize_history()

        # split = min(int(10*0.6), 10-4) = 6 — matches the plain 60/40 cut since
        # none of these messages carry tool_result blocks.
        compaction = session.last_compaction
        assert compaction is not None
        assert compaction["covered_through"] == 6
        assert compaction["summary"].startswith(_COMPACTED_MARKER)
        assert "## Goal\nTest goal" in compaction["summary"]
        # history[0] is exactly the summary reported by last_compaction.
        assert session.history[0]["content"] == compaction["summary"]
        # The last 4 (uncompacted) turns survive verbatim, in order.
        assert session.history[-4:] == original[6:]

    async def test_failed_summarize_reports_no_compaction(self):
        """A transient summarize error must NOT be reported as a compaction —
        the host would otherwise persist the fact-free placeholder as the
        durable summary, and it can never self-heal (it carries the marker)."""
        mock_llm = make_mock_llm()
        mock_llm.summarize = AsyncMock(side_effect=RuntimeError("blip"))

        session = ChatSession(ChatSessionConfig(
            llm_client=mock_llm, initial_history=_alternating_history(10, "x" * 50),
        ))
        await session._summarize_history()

        assert session.last_compaction is None

    async def test_hard_truncate_clears_compaction_record(self):
        """compact then hard_truncate in one turn (the recovery ladder's
        double-overflow path): history[0] is now the truncation placeholder,
        so last_compaction must report None, not the placeholder."""
        mock_llm = make_mock_llm()
        mock_llm.summarize = AsyncMock(return_value=_summarize_response("## Goal\nx"))

        session = ChatSession(ChatSessionConfig(
            llm_client=mock_llm, initial_history=_alternating_history(10, "x" * 50),
        ))
        await session._summarize_history()
        assert session.last_compaction is not None  # precondition
        session.hard_truncate_history()

        assert session.last_compaction is None


class TestSkipWhenLittleNewMaterial:
    async def test_skips_llm_call_when_old_turns_are_negligible(self):
        """Old turns tiny, recent turns huge — folding them in wouldn't
        meaningfully shrink history, so don't pay for the LLM round-trip."""
        history = _alternating_history(6, "ok") + _alternating_history(4, "y" * 5000)
        mock_llm = make_mock_llm()
        mock_llm.summarize = AsyncMock(return_value=_summarize_response("summary"))

        session = ChatSession(ChatSessionConfig(llm_client=mock_llm, initial_history=history))
        await session._summarize_history()

        mock_llm.summarize.assert_not_called()
        assert session.last_compaction is None
        assert session.history == history

    async def test_prior_summary_excluded_from_new_material_measurement(self):
        """A leading prior-summary entry doesn't count as "new" — if the
        actually-new old turns are negligible next to what stays verbatim,
        skip even though the prior summary itself is large."""
        prior_summary = _msg("user", f"{_COMPACTED_MARKER}\n" + ("z" * 5000))
        new_old_turns = _alternating_history(4, "ok")[1:]  # tiny "new" material
        recent_turns = _alternating_history(4, "y" * 5000)
        history = [prior_summary, *new_old_turns, *recent_turns]
        mock_llm = make_mock_llm()
        mock_llm.summarize = AsyncMock(return_value=_summarize_response("summary"))

        session = ChatSession(ChatSessionConfig(llm_client=mock_llm, initial_history=history))
        await session._summarize_history()

        mock_llm.summarize.assert_not_called()
        assert session.last_compaction is None

    async def test_runs_when_new_material_is_substantial(self):
        """Sanity check for the guard's threshold: same shape as the skip
        test above, but the new old turns are large enough to be worth it."""
        prior_summary = _msg("user", f"{_COMPACTED_MARKER}\nprior")
        new_old_turns = _alternating_history(4, "x" * 5000)[1:]
        recent_turns = _alternating_history(4, "y" * 100)
        history = [prior_summary, *new_old_turns, *recent_turns]
        mock_llm = make_mock_llm()
        mock_llm.summarize = AsyncMock(return_value=_summarize_response("updated summary"))

        session = ChatSession(ChatSessionConfig(llm_client=mock_llm, initial_history=history))
        await session._summarize_history()

        mock_llm.summarize.assert_called_once()
        assert session.last_compaction is not None
