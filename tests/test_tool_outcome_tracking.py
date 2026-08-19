"""Explicit tool-failure outcomes drive the error streak, not text search (ENG-1276).

The per-tool failure streak (resilience nudge at 2, circuit breaker at 5) used
to be derived by substring-matching five phrases against the result text. That
misclassified in both directions: a success whose output contained "failed"
incremented the streak, and a genuine failure using none of the phrases RESET
it — the mechanism that kept the breaker asleep through the ENG-836 runaway
(interleaved false "successes" cleared the counter for ~50 minutes).

Covers each ENG-1276 Done-when bullet:
- a successful result containing "failed" does not increment the streak
- a genuine failure whose text lacks all five markers does increment it
- unmigrated handlers (ok=None) keep the legacy substring behaviour, and the
  fallback logs when it classifies an error so call sites stay discoverable
- the ENG-350 empty-code rejection is counted by its explicit ok=False, not
  by the word "failed" in its message, and still trips the breaker
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from anton.core.session import ChatSession
from anton.core.tools.registry import ToolOutcome, ToolRegistry
from anton.core.tools.tool_defs import ToolDef
from anton.core.utils.scratchpad import prepare_scratchpad_exec


def _session() -> ChatSession:
    """A bare object carrying only what _apply_error_tracking reads."""
    s = ChatSession.__new__(ChatSession)
    s._resilience_nudge_at = 2
    s._max_consecutive_errors = 5
    return s


def _track(s, text, *, ok, streak, nudged=None, tool="scratchpad"):
    return ChatSession._apply_error_tracking(
        s, text, tool, streak, nudged if nudged is not None else set(), ok=ok
    )


BREAKER_MARK = "times in a row. Stop retrying this approach."


class TestExplicitOutcomeClassification:
    def test_success_containing_failed_does_not_increment(self):
        # The false positive: "[output]\n0 records failed validation" is a
        # SUCCESS. Under substring matching it incremented the streak; the
        # handler's ok=True verdict must win.
        s, streak = _session(), {"scratchpad": 3}
        _track(s, "[output]\n0 records failed validation", ok=True, streak=streak)
        assert streak["scratchpad"] == 0

    def test_failure_without_any_marker_increments(self):
        # The false negative behind ENG-836: a genuine failure whose text
        # uses none of the five phrases used to RESET the streak.
        s, streak = _session(), {}
        text = "HTTP 404: the requested .deb package does not exist"
        _track(s, text, ok=False, streak=streak)
        assert streak["scratchpad"] == 1

    def test_interleaved_false_success_no_longer_resets(self):
        # The Kiranam shape: error, "success" that achieved nothing but whose
        # handler still reports the truth, error... With explicit outcomes the
        # streak survives only through real successes.
        s, streak, nudged = _session(), {}, set()
        _track(s, "ImportError: libodbc.so.2 missing", ok=False, streak=streak, nudged=nudged)
        _track(s, "downloaded page saved", ok=False, streak=streak, nudged=nudged)
        _track(s, "E: Permission denied", ok=False, streak=streak, nudged=nudged)
        assert streak["scratchpad"] == 3

    def test_five_explicit_failures_trip_the_breaker(self):
        s, streak, nudged = _session(), {}, set()
        out = ""
        for i in range(5):
            # No marker phrase anywhere — classification is purely ok=False.
            out = _track(s, f"clean failure text {i}", ok=False, streak=streak, nudged=nudged)
        assert streak["scratchpad"] == 5
        assert BREAKER_MARK in out

    def test_explicit_success_resets_streak_and_nudge_latch(self):
        s, streak, nudged = _session(), {"scratchpad": 4}, {"scratchpad"}
        _track(s, "all good", ok=True, streak=streak, nudged=nudged)
        assert streak["scratchpad"] == 0
        assert "scratchpad" not in nudged


class TestLegacyFallback:
    def test_unmigrated_marker_text_still_counts(self):
        # ok=None → the five-phrase substring match still applies, unchanged.
        s, streak = _session(), {}
        _track(s, "[error]\nTraceback ...", ok=None, streak=streak)
        assert streak["scratchpad"] == 1

    def test_unmigrated_clean_text_still_resets(self):
        s, streak = _session(), {"scratchpad": 2}
        _track(s, "fetched 200 OK", ok=None, streak=streak)
        assert streak["scratchpad"] == 0

    def test_fallback_classification_is_logged(self, caplog):
        # The ticket's discoverability requirement: when the fallback (not an
        # explicit outcome) classifies an error, it logs — so remaining
        # unmigrated handlers can be found instead of assumed.
        s, streak = _session(), {}
        with caplog.at_level(logging.INFO, logger="anton.core.session"):
            _track(s, "Task failed: boom", ok=None, streak=streak)
        assert any("text fallback" in r.message for r in caplog.records)

    def test_explicit_outcome_does_not_log_fallback(self, caplog):
        s, streak = _session(), {}
        with caplog.at_level(logging.INFO, logger="anton.core.session"):
            _track(s, "clean failure text", ok=False, streak=streak)
        assert not any("text fallback" in r.message for r in caplog.records)


class TestRegistryEnvelope:
    @pytest.mark.asyncio
    async def test_plain_string_handler_wraps_as_unclassified(self):
        registry = ToolRegistry()
        registry.register_tool(
            ToolDef(name="t", description="", input_schema={}, handler=AsyncMock(return_value="hi"))
        )
        outcome = await registry.dispatch_tool(SimpleNamespace(), "t", {})
        assert isinstance(outcome, ToolOutcome)
        assert outcome.content == "hi"
        assert outcome.ok is None

    @pytest.mark.asyncio
    async def test_outcome_returning_handler_passes_through(self):
        declared = ToolOutcome(content="nope", ok=False, reason="boom")
        registry = ToolRegistry()
        registry.register_tool(
            ToolDef(name="t", description="", input_schema={}, handler=AsyncMock(return_value=declared))
        )
        outcome = await registry.dispatch_tool(SimpleNamespace(), "t", {})
        assert outcome is declared

    @pytest.mark.asyncio
    async def test_multimodal_list_handler_wraps_content_intact(self):
        blocks = [{"type": "image", "source": {}}]
        registry = ToolRegistry()
        registry.register_tool(
            ToolDef(name="t", description="", input_schema={}, handler=AsyncMock(return_value=blocks))
        )
        outcome = await registry.dispatch_tool(SimpleNamespace(), "t", {})
        assert outcome.content is blocks
        assert outcome.ok is None


def _exec_session() -> SimpleNamespace:
    """Session stub for prepare_scratchpad_exec (guard + ACC reads only)."""
    return SimpleNamespace(
        _acc_observe=None,
        _agent_scratchpad_names=set(),
        _scratchpads=MagicMock(agent_pads=MagicMock(return_value=set())),
        _scratchpad_challenged=False,
    )


class TestScratchpadOutcomes:
    @pytest.mark.asyncio
    async def test_empty_code_rejection_is_explicit_failure(self):
        # ENG-350's rejection used to need the word "failed" in its text for
        # the substring matcher to count it toward the breaker. The
        # classification is now the ok flag, not the wording.
        outcome = await prepare_scratchpad_exec(
            _exec_session(), {"action": "exec", "name": "main", "code": "  "}
        )
        assert isinstance(outcome, ToolOutcome)
        assert outcome.ok is False
        assert outcome.reason == "scratchpad_empty_code"

    def test_empty_code_rejection_trips_breaker_without_wording(self):
        # Belt over the flag: five rejections reach the breaker even if the
        # message were reworded to avoid every legacy marker phrase.
        s, streak, nudged = _session(), {}, set()
        out = ""
        for _ in range(5):
            out = _track(
                s,
                "the `code` argument was empty — write output in small steps",
                ok=False,
                streak=streak,
                nudged=nudged,
            )
        assert BREAKER_MARK in out

    @pytest.mark.asyncio
    async def test_single_scratchpad_challenge_is_not_a_failure(self):
        # The challenge is guidance; it must not count toward the streak.
        # Previously that depended on its wording avoiding the marker phrases.
        session = _exec_session()
        session._agent_scratchpad_names = {"dash"}
        outcome = await prepare_scratchpad_exec(
            session, {"action": "exec", "name": "report", "code": "print(1)"}
        )
        assert isinstance(outcome, ToolOutcome)
        assert outcome.ok is True


def _install_session(pad) -> SimpleNamespace:
    return SimpleNamespace(
        _acc_observe=None,
        _agent_scratchpad_names=set(),
        _scratchpads=MagicMock(
            agent_pads=MagicMock(return_value=set()),
            get_or_create=AsyncMock(return_value=pad),
        ),
        _scratchpad_challenged=False,
        _settings=SimpleNamespace(analytics_enabled=True),
    )


class TestPackageInstallTelemetry:
    @pytest.mark.asyncio
    async def test_successful_install_sends_package_name_only(self, monkeypatch):
        # Visibility into what gets installed — package name, never the cell.
        sent = []
        monkeypatch.setattr(
            "anton.analytics.send_event",
            lambda settings, action, **extra: sent.append((action, extra)),
        )
        pad = SimpleNamespace(
            install_packages=AsyncMock(return_value="Successfully installed numpy")
        )
        result = await prepare_scratchpad_exec(
            _install_session(pad),
            {
                "action": "exec",
                "name": "main",
                "code": "import numpy; print('secret sauce')",
                "packages": ["numpy"],
            },
        )
        assert not isinstance(result, ToolOutcome)
        assert sent == [("scratchpad_package_installed", {"package": "numpy"})]

    @pytest.mark.asyncio
    async def test_already_installed_sends_no_event(self, monkeypatch):
        sent = []
        monkeypatch.setattr(
            "anton.analytics.send_event",
            lambda settings, action, **extra: sent.append((action, extra)),
        )
        pad = SimpleNamespace(
            install_packages=AsyncMock(return_value="All packages already installed.")
        )
        await prepare_scratchpad_exec(
            _install_session(pad),
            {"action": "exec", "name": "main", "code": "import numpy", "packages": ["numpy"]},
        )
        assert sent == []

    @pytest.mark.asyncio
    async def test_install_action_also_sends_the_event(self, monkeypatch):
        from anton.core.tools.tool_handlers import handle_scratchpad

        sent = []
        monkeypatch.setattr(
            "anton.analytics.send_event",
            lambda settings, action, **extra: sent.append((action, extra)),
        )
        pad = SimpleNamespace(
            install_packages=AsyncMock(return_value="Successfully installed cowsay")
        )
        session = _install_session(pad)
        result = await handle_scratchpad(
            session, {"action": "install", "name": "main", "packages": ["cowsay"]}
        )
        assert result == "Successfully installed cowsay"
        assert sent == [("scratchpad_package_installed", {"package": "cowsay"})]
