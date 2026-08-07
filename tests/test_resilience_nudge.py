"""Tests for ChatSession._select_resilience_nudge — failure-type-aware nudging.

The generic RESILIENCE_NUDGE is scrape/fetch advice and misdirects scratchpad
failures (a too-big or too-slow cell doesn't need a different data source). The
selector routes scratchpad size/timeout failures to specific guidance and keeps
the generic nudge for everything else.
"""

from __future__ import annotations

from anton.core.llm.prompts import (
    RESILIENCE_NUDGE,
    SCRATCHPAD_SIZE_NUDGE,
    SCRATCHPAD_STUCK_NUDGE,
    SCRATCHPAD_TIMEOUT_NUDGE,
)
from anton.core.session import ChatSession

_select = ChatSession._select_resilience_nudge


class TestSelectResilienceNudge:
    def test_non_scratchpad_tool_gets_generic_nudge(self):
        assert _select("web_fetch", "failed to fetch the page") == RESILIENCE_NUDGE

    def test_scratchpad_timeout_gets_timeout_nudge(self):
        assert _select("scratchpad", "Cell timed out after 180s total") == SCRATCHPAD_TIMEOUT_NUDGE

    def test_scratchpad_liveness_kill_gets_stuck_nudge(self):
        # A silence/liveness kill means the worker looked dead — "too heavy"
        # is inverted advice there (ENG-578). Both the new wording and the
        # legacy "inactivity" wording (remote/old workers) route to STUCK.
        new_msg = "Cell killed after 30s without a liveness signal from the scratchpad worker"
        legacy_msg = "Cell killed after 60s of inactivity (no output or progress() calls)"
        assert _select("scratchpad", new_msg) == SCRATCHPAD_STUCK_NUDGE
        assert _select("scratchpad", legacy_msg) == SCRATCHPAD_STUCK_NUDGE

    def test_scratchpad_empty_code_gets_size_nudge(self):
        msg = "Scratchpad exec failed: the `code` argument was empty. ..."
        assert _select("scratchpad", msg) == SCRATCHPAD_SIZE_NUDGE

    def test_scratchpad_generic_error_gets_generic_nudge(self):
        # A NameError-style failure is neither size nor timeout; it still gets
        # the generic "failed twice, change approach" nudge (only size/timeout
        # get specialised scratchpad advice).
        assert _select("scratchpad", "[error]\nNameError: name 'data' is not defined") == RESILIENCE_NUDGE

    def test_scratchpad_nudges_never_mention_scraping(self):
        for nudge in (SCRATCHPAD_SIZE_NUDGE, SCRATCHPAD_TIMEOUT_NUDGE, SCRATCHPAD_STUCK_NUDGE):
            assert "archive.org" not in nudge
            assert "data source" not in nudge
