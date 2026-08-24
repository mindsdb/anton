"""Tests for the persisted-summary compaction result (ENG-664).

Covers `ChatSession.last_compaction` (what a host reads to persist the
summary + cutoff) and the "skip if there's little new material" guard in
`_summarize_history`. The once-per-turn guard (`_compacted_this_turn`) is
already covered by `TestContextCompaction` in test_chat.py.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock

from anton.core.session import (
    ChatSession,
    ChatSessionConfig,
    _COMPACTED_MARKER,
    _summarizer_input_budget,
)
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
        assert await session._summarize_history() is True  # actually compacted

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

        original = _alternating_history(10, "x" * 50)
        session = ChatSession(ChatSessionConfig(
            llm_client=mock_llm, initial_history=list(original),
        ))
        # Returns False so callers skip `_compacted_this_turn` and the
        # StreamContextCompacted event for a compaction that didn't happen.
        assert await session._summarize_history() is False

        assert session.last_compaction is None
        # A failed summarization must NOT mutate history — the earlier turns
        # stay intact rather than being replaced by a fact-free placeholder
        # (ENG-1274).
        assert session.history == original

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


class TestSummarizerInputBudget:
    """ENG-1291: the summariser's input budget and which end survives it."""

    async def test_newest_folded_turn_reaches_the_summarizer(self):
        """A conversation this size fits the summariser's window whole, so no
        folded turn may be dropped — the old 8,000-char head-cut kept turn 0 and
        discarded turn 23, which is where current state lives."""
        history = [
            _msg("user" if i % 2 == 0 else "assistant", f"turn-{i} " + "x" * 1200)
            for i in range(40)
        ]
        mock_llm = make_mock_llm()
        mock_llm.summarize = AsyncMock(return_value=_summarize_response("## Goal\nx"))

        session = ChatSession(ChatSessionConfig(llm_client=mock_llm, initial_history=history))
        assert await session._summarize_history() is True

        sent = mock_llm.summarize.await_args.kwargs["messages"][0]["content"]
        # compacted_count = min(int(40*0.6), 40-4) = 24 → turns 0..23 are folded.
        assert "turn-0 " in sent
        assert "turn-23 " in sent

    def test_budget_scales_with_the_window(self):
        # 200k-token window vs the 128k default for an unknown id.
        assert _summarizer_input_budget("claude-sonnet-4-6") > _summarizer_input_budget("mystery-1")

    def test_carried_forward_summary_shares_the_budget(self):
        full = _summarizer_input_budget("claude-sonnet-4-6")
        assert _summarizer_input_budget("claude-sonnet-4-6", reserved=5000) == full - 5000


class TestTruncatedRecord:
    async def test_partial_record_is_kept_and_logged_not_retried(self, caplog):
        """ENG-1291: hitting a cap 4x the requested length means the model ignored
        the length instruction, so more room only buys more of the same. Keep the
        partial record — it still beats the reactive path, which discards history —
        but leave a log line so it's identifiable."""
        mock_llm = make_mock_llm()
        mock_llm.summarize = AsyncMock(return_value=LLMResponse(
            content="## Goal\nhalf-writ",
            usage=Usage(input_tokens=10, output_tokens=8192),
            stop_reason="max_tokens",
        ))

        session = ChatSession(ChatSessionConfig(
            llm_client=mock_llm, initial_history=_alternating_history(10, "x" * 50),
        ))
        with caplog.at_level(logging.WARNING, logger="anton.core.session"):
            assert await session._summarize_history() is True

        mock_llm.summarize.assert_awaited_once()
        assert mock_llm.summarize.await_args.kwargs["max_tokens"] == 8192
        assert "half-writ" in session.history[0]["content"]
        assert "truncated" in caplog.text

    async def test_last_resort_folds_behind_a_marker_instead_of_declining(self, caplog):
        """The reactive callers run after the provider already rejected the
        request. Declining there sends the retry into `hard_truncate_history`
        (4 messages kept) — folding behind a marker keeps the newest ~40%."""
        original = _alternating_history(10, "x" * 50)
        mock_llm = make_mock_llm()
        mock_llm.summarize = AsyncMock(return_value=LLMResponse(
            content="   \n  ",  # whitespace-only counts as empty
            usage=Usage(input_tokens=10, output_tokens=8192),
            stop_reason="max_tokens",
        ))

        session = ChatSession(ChatSessionConfig(
            llm_client=mock_llm, initial_history=list(original),
        ))
        with caplog.at_level(logging.WARNING, logger="anton.core.session"):
            assert await session._summarize_history(last_resort=True) is True

        assert session.history[-4:] == original[6:]  # newest turns survive
        assert "No summary of them is available" in session.history[0]["content"]
        # The host must not replay a marker as the conversation's summary.
        assert session.last_compaction is None

    async def test_empty_record_leaves_history_intact(self, caplog):
        """A generation cut off before any text is a "success" carrying no
        record — `raise_on_empty_response` returns early on any stop_reason.
        Substituting a placeholder would replace real turns with a fact-free
        line that can never self-heal (ENG-1274)."""
        original = _alternating_history(10, "x" * 50)
        mock_llm = make_mock_llm()
        mock_llm.summarize = AsyncMock(return_value=LLMResponse(
            content="",
            usage=Usage(input_tokens=10, output_tokens=8192),
            stop_reason="max_tokens",
        ))

        session = ChatSession(ChatSessionConfig(
            llm_client=mock_llm, initial_history=list(original),
        ))
        with caplog.at_level(logging.WARNING, logger="anton.core.session"):
            assert await session._summarize_history() is False

        assert session.history == original
        assert session.last_compaction is None
        assert "empty record" in caplog.text

    async def test_budget_never_exceeds_the_client_ceiling(self):
        """A host configuring a smaller `max_tokens` would otherwise get a 400 on
        every compaction, logged only as an exception type name."""
        mock_llm = make_mock_llm()
        mock_llm.max_tokens = 4096
        mock_llm.summarize = AsyncMock(return_value=_summarize_response("## Goal\nx"))

        session = ChatSession(ChatSessionConfig(
            llm_client=mock_llm, initial_history=_alternating_history(10, "x" * 50),
        ))
        assert await session._summarize_history() is True

        assert mock_llm.summarize.await_args.kwargs["max_tokens"] == 4096


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
