"""Tests for the persisted-summary compaction result (ENG-664).

Covers `ChatSession.last_compaction` (what a host reads to persist the
summary + cutoff) and the "skip if there's little new material" guard in
`_summarize_history`. The once-per-turn guard (`_compacted_this_turn`) is
already covered by `TestContextCompaction` in test_chat.py.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

from anton.core import session as session_mod
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

    async def test_budget_ignores_the_larger_stream_budget(self):
        """Compaction is a non-streamed call: it stays under the client's own
        max_tokens even when streamed calls get more room."""
        mock_llm = make_mock_llm()
        mock_llm.max_tokens = 4096
        mock_llm.stream_budget = MagicMock(return_value=16384)
        mock_llm.summarize = AsyncMock(return_value=_summarize_response("## Goal\nx"))

        session = ChatSession(ChatSessionConfig(
            llm_client=mock_llm, initial_history=_alternating_history(10, "x" * 50),
        ))
        assert await session._summarize_history() is True

        assert mock_llm.summarize.await_args.kwargs["max_tokens"] == 4096

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

    async def test_clip_keeps_the_tail_when_the_budget_binds(self, caplog, monkeypatch):
        """The regime the fix is about: material larger than the budget. The old
        head-cut kept turn 0 and dropped everything after it — the tail-biased
        clip has to do the opposite, and say so in the log."""
        history = [
            _msg("user" if i % 2 == 0 else "assistant", f"turn-{i} " + "x" * 1200)
            for i in range(40)
        ]
        mock_llm = make_mock_llm()
        mock_llm.summarize = AsyncMock(return_value=_summarize_response("## Goal\nx"))
        monkeypatch.setattr(session_mod, "_summarizer_input_budget", lambda *a, **kw: 6000)

        session = ChatSession(ChatSessionConfig(llm_client=mock_llm, initial_history=history))
        with caplog.at_level(logging.WARNING, logger="anton.core.session"):
            assert await session._summarize_history() is True

        sent = mock_llm.summarize.await_args.kwargs["messages"][0]["content"]
        assert "turn-23 " in sent  # newest folded turn — what the head-cut lost
        assert "turn-12 " not in sent  # the middle is what gets elided
        assert "chars elided" in sent  # the drop is marked, not silent
        assert "input clipped" in caplog.text

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

    async def test_marker_does_not_overwrite_an_earlier_compactions_summary(self):
        """Compact successfully, then hit the last-resort marker later in the
        same session. The marker replaces history[0], so reporting the summary
        by reading that slot would hand the host the marker text under the
        earlier compaction's count — and the messages it covers would be
        replaced by "no summary is available" in every later turn."""
        mock_llm = make_mock_llm()
        mock_llm.summarize = AsyncMock(return_value=_summarize_response("## Goal\nreal"))

        session = ChatSession(ChatSessionConfig(
            llm_client=mock_llm, initial_history=_alternating_history(10, "x" * 50),
        ))
        assert await session._summarize_history() is True
        first = session.last_compaction

        session._history.extend(_alternating_history(10, "y" * 50))
        mock_llm.summarize = AsyncMock(return_value=_summarize_response("   "))
        assert await session._summarize_history(last_resort=True) is True

        assert "No summary of them is available" in session.history[0]["content"]
        assert session.last_compaction == first

    async def test_nothing_is_recorded_after_a_marker_folded_turns_away(self):
        """The marker drops turns instead of summarizing them. A later
        compaction summarizes what's left — including the marker line, not the
        turns behind it — so recording it would tell the host those turns are
        covered and stop them ever being replayed."""
        mock_llm = make_mock_llm()
        mock_llm.summarize = AsyncMock(return_value=_summarize_response("  "))

        session = ChatSession(ChatSessionConfig(
            llm_client=mock_llm, initial_history=_alternating_history(20, "x" * 50),
        ))
        assert await session._summarize_history(last_resort=True) is True
        assert session.last_compaction is None

        session._history.extend(_alternating_history(10, "y" * 50))
        mock_llm.summarize = AsyncMock(return_value=_summarize_response("## Goal\nreal"))
        assert await session._summarize_history() is True

        assert session.last_compaction is None


class TestSecondCompactionInOneTurn:
    """The reactive overflow path compacts without the once-per-turn guard, so
    a turn that compacts proactively and then overflows compacts twice."""

    async def test_covered_through_stays_in_seed_coordinates(self):
        """The second compaction indexes a history the first one already
        rewrote as [summary, separator, rest]. Reporting that raw index would
        undercount what's folded, and the host would keep replaying turns the
        summary already covers — the heavy conversations this is for would
        never stop being re-summarized."""
        mock_llm = make_mock_llm()
        mock_llm.summarize = AsyncMock(return_value=_summarize_response("## Goal\nfirst"))

        session = ChatSession(ChatSessionConfig(
            llm_client=mock_llm, initial_history=_alternating_history(20, "x" * 50),
        ))
        assert await session._summarize_history() is True
        assert session.last_compaction["covered_through"] == 12
        # [summary, separator, seed12..seed19]
        assert len(session.history) == 10

        session._history.extend(_alternating_history(4, "y" * 50))
        mock_llm.summarize = AsyncMock(return_value=_summarize_response("## Goal\nsecond"))
        assert await session._summarize_history() is True

        # Folds history[:8] = summary + separator + seed12..seed17, i.e. the
        # seed through index 17. The raw count would have said 8.
        assert session.last_compaction["covered_through"] == 18
        assert "second" in session.last_compaction["summary"]

    async def test_covered_through_never_runs_past_the_seed(self):
        """A second compaction can fold this turn's own messages too; those are
        the host's current input, not history it can put behind a cutoff."""
        mock_llm = make_mock_llm()
        mock_llm.summarize = AsyncMock(return_value=_summarize_response("## Goal\nfirst"))

        session = ChatSession(ChatSessionConfig(
            llm_client=mock_llm, initial_history=_alternating_history(20, "x" * 50),
        ))
        assert await session._summarize_history() is True

        session._history.extend(_alternating_history(10, "y" * 50))
        mock_llm.summarize = AsyncMock(return_value=_summarize_response("## Goal\nsecond"))
        assert await session._summarize_history() is True

        assert session.last_compaction["covered_through"] == 20  # the whole seed

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


class TestProactiveAbsoluteTokenCeiling:
    """Proactive compaction must also trigger on an absolute token ceiling so
    1M-window models do not wait until ~700k tokens before compacting."""

    def test_triggers_at_absolute_ceiling_even_with_low_pressure(self):
        from anton.core.settings import CoreSettings

        session = ChatSession(ChatSessionConfig(
            llm_client=make_mock_llm(),
            settings=CoreSettings(context_compaction_max_tokens=100_000),
        ))
        # 1M-window model: 100k tokens is only ~9.5% pressure, well below 0.7
        below = Usage(input_tokens=5_000, cache_read_tokens=94_999, context_pressure=0.095)
        at_cap = Usage(input_tokens=5_000, cache_read_tokens=95_000, context_pressure=0.095)
        assert session._should_proactively_compact(below) is False
        assert session._should_proactively_compact(at_cap) is True

    def test_zero_ceiling_disables_absolute_trigger(self):
        from anton.core.settings import CoreSettings

        session = ChatSession(ChatSessionConfig(
            llm_client=make_mock_llm(),
            settings=CoreSettings(context_compaction_max_tokens=0),
        ))
        large = Usage(input_tokens=5_000, cache_read_tokens=200_000, context_pressure=0.20)
        high_pressure = Usage(input_tokens=5_000, cache_read_tokens=5_000, context_pressure=0.80)
        assert session._should_proactively_compact(large) is False
        assert session._should_proactively_compact(high_pressure) is True

    def test_cooldown_blocks_immediate_recompaction_and_allows_long_turn_recompaction(self):
        from anton.core.session import _PROACTIVE_COMPACTION_COOLDOWN_ROUNDS
        from anton.core.settings import CoreSettings

        session = ChatSession(ChatSessionConfig(
            llm_client=make_mock_llm(),
            settings=CoreSettings(context_compaction_max_tokens=100_000),
        ))
        heavy = Usage(input_tokens=2_000, cache_read_tokens=105_000, context_pressure=0.10)
        session._turn_tool_rounds = 18
        assert session._should_proactively_compact(heavy) is True

        # Simulate successful compaction at round 18
        session._compacted_this_turn = True
        session._last_compaction_round = 18
        session._turn_tool_rounds = 18 + _PROACTIVE_COMPACTION_COOLDOWN_ROUNDS - 1
        assert session._should_proactively_compact(heavy) is False

        # Once full cooldown has elapsed, a long turn can compact again
        session._turn_tool_rounds = 18 + _PROACTIVE_COMPACTION_COOLDOWN_ROUNDS
        assert session._should_proactively_compact(heavy) is True

        # Failed compaction still latches off for the rest of the turn
        session._compaction_failed_this_turn = True
        assert session._should_proactively_compact(heavy) is False

    async def test_turn_stream_proactively_compacts_on_absolute_ceiling(self):
        from anton.core.llm.provider import StreamComplete, StreamContextCompacted
        from anton.core.settings import CoreSettings

        async def _plan_stream(**kwargs):
            yield StreamComplete(
                response=LLMResponse(
                    content="Done",
                    usage=Usage(
                        input_tokens=4_000,
                        cache_read_tokens=100_000,
                        context_pressure=0.10,
                    ),
                )
            )

        session = ChatSession(ChatSessionConfig(
            llm_client=make_mock_llm(),
            settings=CoreSettings(context_compaction_max_tokens=100_000),
        ))
        session._llm.plan_stream = _plan_stream
        session._llm.plan = AsyncMock(
            return_value=LLMResponse(content="STATUS: COMPLETE — done")
        )
        session._summarize_history = AsyncMock(return_value=True)

        events = [e async for e in session.turn_stream("hello")]

        assert session._summarize_history.await_count == 1
        assert sum(isinstance(e, StreamContextCompacted) for e in events) == 1
        assert session._compacted_this_turn is True
        assert session._last_compaction_round == 0

    def test_non_int_compaction_max_tokens_falls_back_to_default(self):
        from anton.core.settings import CoreSettings

        assert CoreSettings().context_compaction_max_tokens == 65_000

        mock_settings = MagicMock()
        mock_settings.context_pressure_threshold = 0.7
        mock_settings.max_consecutive_errors = 5
        mock_settings.resilience_nudge_at = 2
        mock_settings.model_call_idle_timeout_s = 120.0
        mock_settings.max_images_per_turn = 8
        mock_settings.max_tool_rounds = 100
        mock_settings.max_continuations = 3
        mock_settings.max_turn_tokens = 25_000_000
        mock_settings.max_turn_seconds = 0
        mock_settings.cell_timeout_default = 120
        mock_settings.cell_inactivity_timeout = 30
        mock_settings.cell_inactivity_after_progress = 60
        mock_settings.cell_inactivity_max = 60
        mock_settings.cell_total_max = 3600
        mock_settings.cell_install_timeout = 120
        mock_settings.cell_keep_recent = 5
        # MagicMock attribute (unconfigured) must fall back to 65_000
        session = ChatSession(ChatSessionConfig(
            llm_client=make_mock_llm(),
            settings=mock_settings,
        ))
        assert session._context_compaction_max_tokens == 65_000

        # Bool must also be rejected (isinstance(True, int) is True in Python)
        mock_settings.context_compaction_max_tokens = True
        session_bool = ChatSession(ChatSessionConfig(
            llm_client=make_mock_llm(),
            settings=mock_settings,
        ))
        assert session_bool._context_compaction_max_tokens == 65_000

    async def test_benign_early_turn_skip_does_not_disable_later_compaction(self):
        """A benign skip on round 0 (len(history) < 6) at >=100k tokens must
        NOT latch _compaction_failed_this_turn, so later rounds in the same
        turn still compact once enough history has accumulated."""
        from anton.core.llm.provider import (
            StreamComplete,
            StreamContextCompacted,
            ToolCall,
        )
        from anton.core.settings import CoreSettings

        heavy = Usage(input_tokens=4_000, cache_read_tokens=105_000, context_pressure=0.10)
        call_idx = 0

        async def _plan_stream(**kwargs):
            nonlocal call_idx
            call_idx += 1
            if call_idx <= 4:
                yield StreamComplete(
                    response=LLMResponse(
                        content="x" * 200,
                        tool_calls=[ToolCall(id=f"tc_{call_idx}", name="no_such_tool", input={})],
                        usage=heavy,
                        stop_reason="tool_use",
                    )
                )
            else:
                yield StreamComplete(
                    response=LLMResponse(
                        content="Finished",
                        usage=heavy,
                        stop_reason="end_turn",
                    )
                )

        mock_llm = make_mock_llm()
        mock_llm.plan_stream = _plan_stream
        mock_llm.plan = AsyncMock(return_value=_summarize_response("STATUS: COMPLETE — done"))
        mock_llm.summarize = AsyncMock(return_value=_summarize_response("## Goal\nCompacted"))

        session = ChatSession(ChatSessionConfig(
            llm_client=mock_llm,
            settings=CoreSettings(context_compaction_max_tokens=100_000),
        ))

        events = [e async for e in session.turn_stream("x" * 400)]

        # Round 0 had len(history) == 1 (< 6) and skipped benignly; later rounds
        # grew past 6 messages and compacted cleanly.
        assert mock_llm.summarize.await_count == 1
        assert sum(isinstance(e, StreamContextCompacted) for e in events) == 1
        assert session._compacted_this_turn is True
        assert session._compaction_failed_this_turn is False

    async def test_stream_drop_retry_preserves_compaction_cooldown_and_failure_latch(self):
        """Re-entering `_stream_and_handle_tools` on a mid-turn stream drop must
        preserve `_compacted_this_turn` / `_last_compaction_round` and
        `_compaction_failed_this_turn` instead of resetting them."""
        from anton.core.llm.provider import (
            StreamComplete,
            StreamContextCompacted,
            ToolCall,
            TransientProviderError,
        )
        from anton.core.settings import CoreSettings

        heavy = Usage(input_tokens=4_000, cache_read_tokens=105_000, context_pressure=0.10)
        drop = TransientProviderError(
            "Lost connection mid-response",
            provider="OpenAI",
            code="connection_error",
            session_backoff=True,
        )

        # Case 1: Compaction succeeded on round 0, then round 1 drops and retries.
        # The retry must NOT compact a second time inside the 15-round cooldown.
        step = 0

        async def _plan_stream_cooldown(**kwargs):
            nonlocal step
            step += 1
            if step == 1:
                yield StreamComplete(
                    response=LLMResponse(
                        content="step 1",
                        tool_calls=[ToolCall(id="tc_1", name="no_such_tool", input={})],
                        usage=heavy,
                        stop_reason="tool_use",
                    )
                )
            elif step == 2:
                raise drop
            else:
                yield StreamComplete(
                    response=LLMResponse(content="recovered", usage=heavy, stop_reason="end_turn")
                )

        mock_llm = make_mock_llm()
        mock_llm.plan_stream = _plan_stream_cooldown
        mock_llm.plan = AsyncMock(return_value=_summarize_response("STATUS: COMPLETE — done"))
        mock_llm.summarize = AsyncMock(return_value=_summarize_response("## Goal\nCompacted"))

        session = ChatSession(ChatSessionConfig(
            llm_client=mock_llm,
            initial_history=_alternating_history(10, "x" * 100),
            settings=CoreSettings(context_compaction_max_tokens=100_000),
        ))
        session._backoff_sleep = AsyncMock(return_value=False)

        events = [e async for e in session.turn_stream("go")]
        assert mock_llm.summarize.await_count == 1
        assert sum(isinstance(e, StreamContextCompacted) for e in events) == 1
        assert session._compacted_this_turn is True
        assert session._last_compaction_round == 0

        # Case 2: Summarization failed on round 0, then round 1 drops and retries.
        # The retry must keep `_compaction_failed_this_turn` latched True.
        step2 = 0

        async def _plan_stream_failed(**kwargs):
            nonlocal step2
            step2 += 1
            if step2 == 1:
                yield StreamComplete(
                    response=LLMResponse(
                        content="step 1",
                        tool_calls=[ToolCall(id="tc_1", name="no_such_tool", input={})],
                        usage=heavy,
                        stop_reason="tool_use",
                    )
                )
            elif step2 == 2:
                raise drop
            else:
                yield StreamComplete(
                    response=LLMResponse(content="recovered", usage=heavy, stop_reason="end_turn")
                )

        mock_llm2 = make_mock_llm()
        mock_llm2.plan_stream = _plan_stream_failed
        mock_llm2.plan = AsyncMock(return_value=_summarize_response("STATUS: COMPLETE — done"))
        mock_llm2.summarize = AsyncMock(side_effect=RuntimeError("summarizer down"))

        session2 = ChatSession(ChatSessionConfig(
            llm_client=mock_llm2,
            initial_history=_alternating_history(10, "x" * 100),
            settings=CoreSettings(context_compaction_max_tokens=100_000),
        ))
        session2._backoff_sleep = AsyncMock(return_value=False)

        events2 = [e async for e in session2.turn_stream("go")]
        assert mock_llm2.summarize.await_count == 1
        assert not any(isinstance(e, StreamContextCompacted) for e in events2)
        assert session2._compaction_failed_this_turn is True

    async def test_compaction_summary_includes_anti_loop_directive(self):
        mock_llm = make_mock_llm()
        mock_llm.summarize = AsyncMock(return_value=_summarize_response("## Remaining\nFinish X"))
        session = ChatSession(ChatSessionConfig(
            llm_client=mock_llm,
            initial_history=_alternating_history(10, "x" * 100),
        ))
        ok = await session._summarize_history()
        assert ok is True
        first_msg = session._history[0]["content"]
        assert "Do not repeat approaches already recorded under Completed or Blocked" in first_msg

    def test_strategy_pivot_checkpoints_fire_once_and_respect_guards(self):
        session = ChatSession(ChatSessionConfig(llm_client=make_mock_llm()))

        # Non-milestone round does not fire.
        results: list[dict] = []
        session._turn_tool_rounds = 24
        assert session._maybe_append_strategy_pivot_checkpoint(results) is False
        assert results == []

        # Milestone rounds 25, 50, 80 fire exactly once each.
        for r in (25, 50, 80):
            results = []
            session._turn_tool_rounds = r
            assert session._maybe_append_strategy_pivot_checkpoint(results) is True
            assert len(results) == 1
            assert results[0]["type"] == "text"
            assert f"Checkpoint (round {r})" in results[0]["text"]
            # Second invocation at same round (e.g. after retry) must not duplicate.
            assert session._maybe_append_strategy_pivot_checkpoint(results) is False
            assert len(results) == 1

        # Skipped when an unanswered user question ends the turn.
        session._pivot_checkpoints_fired.clear()
        session._unanswered_question = "Which file?"
        session._turn_tool_rounds = 25
        results = []
        assert session._maybe_append_strategy_pivot_checkpoint(results) is False
        assert results == []

        # Skipped when round_cap is reached on the milestone round itself.
        session._unanswered_question = None
        session._max_tool_rounds = 25
        results = []
        assert session._maybe_append_strategy_pivot_checkpoint(results) is False
        assert results == []

    async def test_strategy_pivot_checkpoint_not_duplicated_after_stream_drop_retry(self):
        """A mid-turn stream drop after round 25 must keep `_pivot_checkpoints_fired`
        intact across `_stream_and_handle_tools` re-entry."""
        from anton.core.llm.provider import (
            StreamComplete,
            ToolCall,
            TransientProviderError,
        )

        light = Usage(input_tokens=1_000, output_tokens=50, context_pressure=0.01)
        drop = TransientProviderError(
            "Lost connection mid-response",
            provider="OpenAI",
            code="connection_error",
            session_backoff=True,
        )
        step = 0

        async def _plan_stream(**kwargs):
            nonlocal step
            step += 1
            if step <= 25:
                yield StreamComplete(
                    response=LLMResponse(
                        content="",
                        tool_calls=[ToolCall(id=f"tc_{step}", name="no_such_tool", input={})],
                        usage=light,
                        stop_reason="tool_use",
                    )
                )
            elif step == 26:
                raise drop
            elif step == 27:
                yield StreamComplete(
                    response=LLMResponse(
                        content="",
                        tool_calls=[ToolCall(id="tc_26", name="no_such_tool", input={})],
                        usage=light,
                        stop_reason="tool_use",
                    )
                )
            else:
                yield StreamComplete(
                    response=LLMResponse(content="done", usage=light, stop_reason="end_turn")
                )

        mock_llm = make_mock_llm()
        mock_llm.plan_stream = _plan_stream
        mock_llm.plan = AsyncMock(return_value=_summarize_response("STATUS: COMPLETE — done"))
        session = ChatSession(ChatSessionConfig(llm_client=mock_llm))
        session._backoff_sleep = AsyncMock(return_value=False)

        _ = [e async for e in session.turn_stream("go")]
        checkpoints = [
            b["text"]
            for m in session._history
            if m.get("role") == "user" and isinstance(m.get("content"), list)
            for b in m["content"]
            if b.get("type") == "text" and "Checkpoint (round 25)" in b.get("text", "")
        ]
        assert len(checkpoints) == 1
        assert session._pivot_checkpoints_fired == {25}

    async def test_empty_summary_retries_once_and_recovers(self):
        """A transient empty summary response retries once before failing
        compaction, doubling budget across both Anthropic (`max_tokens`) and
        OpenAI/MindsHub (`length`) truncation dialects, and skipping retry on
        `content_filter` refusals."""
        for reason in ("max_tokens", "length"):
            mock_llm = make_mock_llm()
            mock_llm.max_tokens = 12_000
            mock_llm.summarize = AsyncMock(
                side_effect=[
                    LLMResponse(
                        content="",
                        usage=Usage(input_tokens=100, output_tokens=8192),
                        stop_reason=reason,
                    ),
                    _summarize_response(f"## Goal\nRecovered from {reason}"),
                ]
            )
            session = ChatSession(ChatSessionConfig(
                llm_client=mock_llm,
                initial_history=_alternating_history(10, "x" * 100),
            ))
            session._stagnation_counts["abc"] = 2
            session._stagnation_nudged.add("abc")
            session._stagnation_escalation_sent = True
            assert await session._summarize_history() is True
            assert mock_llm.summarize.await_count == 2
            assert mock_llm.summarize.await_args_list[0].kwargs["max_tokens"] == 8192
            assert mock_llm.summarize.await_args_list[1].kwargs["max_tokens"] == 12_000
            assert f"Recovered from {reason}" in session._history[0]["content"]
            # Successful compaction clears stagnation counters so post-compaction
            # re-inspections aren't falsely nudged.
            assert session._stagnation_counts == {}
            assert session._stagnation_nudged == set()
            assert session._stagnation_escalation_sent is False

        # content_filter refusal must not trigger a futile duplicate retry.
        mock_cf = make_mock_llm()
        mock_cf.summarize = AsyncMock(
            return_value=LLMResponse(
                content="",
                usage=Usage(input_tokens=100, output_tokens=0),
                stop_reason="content_filter",
            )
        )
        session_cf = ChatSession(ChatSessionConfig(
            llm_client=mock_cf,
            initial_history=_alternating_history(10, "x" * 100),
        ))
        assert await session_cf._summarize_history() is False
        assert mock_cf.summarize.await_count == 1

    def test_drop_older_tool_images_batches_and_preserves_other_images(self):
        from anton.core.artifacts.render_check import (
            OLDER_TOOL_IMAGE_PLACEHOLDER,
            SCREENSHOT_CAPTION,
            drop_older_tool_images,
            tool_images_refused,
        )

        def _img_msg(i: int) -> dict:
            return {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": f"tc_{i}",
                    "content": [
                        {"type": "text", "text": f"Image {i}"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": f"DATA_{i}"}},
                    ],
                }],
            }

        user_attachment = {
            "role": "user",
            "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "USER_IMG"}}],
        }
        render_msg = {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "tc_render",
                "content": [
                    {"type": "text", "text": f"{SCREENSHOT_CAPTION} index.html"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "RENDER_IMG"}},
                ],
            }],
        }

        # 5 tool images (< trigger_at=6): no-op, zero history mutation.
        history = [user_attachment, render_msg] + [_img_msg(i) for i in range(5)]
        assert drop_older_tool_images(history) == 0
        assert all(m["content"][0]["content"][1]["type"] == "image" for m in history[2:])

        # 6th tool image triggers batch eviction: oldest 3 replaced, newest 3 kept.
        history.append(_img_msg(5))
        assert drop_older_tool_images(history) == 3
        for idx in range(3):
            block = history[2 + idx]["content"][0]["content"][1]
            assert block == {"type": "text", "text": OLDER_TOOL_IMAGE_PLACEHOLDER}
        for idx in range(3, 6):
            block = history[2 + idx]["content"][0]["content"][1]
            assert block["type"] == "image"
            assert block["source"]["data"] == f"DATA_{idx}"
        # User attachment and render screenshot are untouched, and refusal flag is false.
        assert user_attachment["content"][0]["type"] == "image"
        assert render_msg["content"][0]["content"][1]["type"] == "image"
        assert tool_images_refused(history) is False

    def test_stagnation_detector_fires_once_on_third_identical_cycle(self):
        from anton.core.llm.provider import ToolCall

        session = ChatSession(ChatSessionConfig(llm_client=make_mock_llm()))
        tc = ToolCall(id="tc_1", name="bash", input={"command": "pytest -q"})
        body = "FAILED tests/test_core.py::test_calc - AssertionError: 1 != 2"

        # 1. Varying `handle_bash` elapsed-time headers `(exit 1, X.Ys)` still
        # normalize to the same signature, AND passing pre-nudge outputs through
        # `_apply_error_tracking` fires on the 3rd identical failure (not 4th!).
        error_streak: dict[str, int] = {}
        resilience_nudged: set[str] = set()
        for attempt, elapsed in enumerate(("1.2s", "1.5s", "1.3s"), start=1):
            raw = f"(exit 1, {elapsed})\n{body}"
            tracked = session._apply_error_tracking(
                raw, "bash", error_streak, resilience_nudged, ok=False
            )
            tr = [{"type": "tool_result", "tool_use_id": "tc_1", "content": tracked}]
            fired = session._record_and_maybe_nudge_stagnation(
                [tc], tr, raw_outputs={"tc_1": raw}
            )
            if attempt < 3:
                assert fired is False
                assert len(tr) == 1
            else:
                assert fired is True
                assert len(tr) == 2
                assert "Stagnation check" in tr[1]["text"]

        # 4th repeat for same signature does not duplicate the nudge.
        raw4 = f"(exit 1, 2.7s)\n{body}"
        tr4 = [{"type": "tool_result", "tool_use_id": "tc_1", "content": raw4}]
        assert session._record_and_maybe_nudge_stagnation([tc], tr4, raw_outputs={"tc_1": raw4}) is False
        assert len(tr4) == 1

        # Different exit code does not collide with the exit-1 signature.
        raw_rc2 = f"(exit 2, 1.2s)\n{body}"
        tr_rc2 = [{"type": "tool_result", "tool_use_id": "tc_1", "content": raw_rc2}]
        assert session._record_and_maybe_nudge_stagnation([tc], tr_rc2, raw_outputs={"tc_1": raw_rc2}) is False

        # 2. `edit_file` batch `edits=[...]` mode canonicalizes edits cleanly:
        # distinct batches do not collide, while 3 identical failing batches fire.
        tc_e1 = ToolCall(id="e1", name="edit_file", input={"path": "a.py", "edits": [{"old_string": "foo", "new_string": "bar"}]})
        tc_e2 = ToolCall(id="e2", name="edit_file", input={"path": "a.py", "edits": [{"old_string": "baz", "new_string": "qux"}]})
        err_msg = "Error: edits[0].old_string not found in a.py. No changes were written."
        for call in (tc_e1, tc_e1, tc_e2):
            tr_e = [{"type": "tool_result", "tool_use_id": call.id, "content": err_msg}]
            assert session._record_and_maybe_nudge_stagnation([call], tr_e, raw_outputs={call.id: err_msg}) is False
        tr_e3 = [{"type": "tool_result", "tool_use_id": tc_e1.id, "content": err_msg}]
        assert session._record_and_maybe_nudge_stagnation([tc_e1], tr_e3, raw_outputs={tc_e1.id: err_msg}) is True
        assert "Stagnation check" in tr_e3[1]["text"]

        # 3. Guard: skipped when unanswered question is pending.
        session._stagnation_counts.clear()
        session._stagnation_nudged.clear()
        session._stagnation_escalation_sent = False
        session._unanswered_question = "Need input?"
        for _ in range(3):
            raw_q = f"(exit 1, 0.4s)\n{body}"
            tr_q = [{"type": "tool_result", "tool_use_id": "tc_1", "content": raw_q}]
            assert session._record_and_maybe_nudge_stagnation([tc], tr_q, raw_outputs={"tc_1": raw_q}) is False

    def test_stagnation_escalation_on_fifth_repeat_and_third_distinct_loop(self):
        from anton.core.llm.provider import ToolCall

        # Case A: Same stagnant call repeated 5 times escalates once (and stays quiet on 6th).
        session = ChatSession(ChatSessionConfig(llm_client=make_mock_llm()))
        tc = ToolCall(id="tc_1", name="bash", input={"command": "pytest -q"})
        body = "FAILED tests/test_core.py::test_calc - AssertionError: 1 != 2"
        outcomes = []
        notes = []
        for attempt in range(1, 7):
            raw = f"(exit 1, 0.{attempt}s)\n{body}"
            tr = [{"type": "tool_result", "tool_use_id": "tc_1", "content": raw}]
            fired = session._record_and_maybe_nudge_stagnation([tc], tr, raw_outputs={"tc_1": raw})
            outcomes.append(fired)
            notes.append(tr[1]["text"] if len(tr) > 1 else "")
        assert outcomes == [False, False, True, False, True, False]
        assert "Stagnation check" in notes[2]
        assert "Stagnation escalation" in notes[4]
        assert "5 times" in notes[4]

        # Case B: Three separate stagnant loops trigger escalation on the 3rd loop, once.
        session2 = ChatSession(ChatSessionConfig(llm_client=make_mock_llm()))
        loop_notes = []
        for idx in range(1, 5):
            call = ToolCall(id=f"c_{idx}", name="bash", input={"command": f"make target_{idx}"})
            raw = f"(exit 2, 0.5s)\nmake: *** No rule to make target 'target_{idx}'. Stop."
            for rep in range(3):
                tr = [{"type": "tool_result", "tool_use_id": call.id, "content": raw}]
                fired = session2._record_and_maybe_nudge_stagnation([call], tr, raw_outputs={call.id: raw})
                if rep == 2:
                    loop_notes.append((fired, tr[1]["text"] if len(tr) > 1 else ""))
        assert loop_notes[0][0] is True and "Stagnation check" in loop_notes[0][1]
        assert loop_notes[1][0] is True and "Stagnation check" in loop_notes[1][1]
        assert loop_notes[2][0] is True and "Stagnation escalation" in loop_notes[2][1]
        assert "3 separate" in loop_notes[2][1]
        # After escalation is already sent, a 4th distinct loop still gets a normal check (not a 2nd escalation).
        assert loop_notes[3][0] is True and "Stagnation check" in loop_notes[3][1]
