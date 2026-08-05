"""Scratchpad watchdog contract tests: liveness heartbeat, partial-stdout salvage,
kill-feedback truth (ENG-578).

Timing knobs are shrunk via ANTON_* env vars (CoreSettings reads them at call
time; the boot subprocess reads them at spawn) so nothing here sleeps more
than a few seconds. Env must be set BEFORE pad.start().
"""
from __future__ import annotations

import asyncio

import pytest

from anton.core.backends.local import LocalScratchpadRuntime
from anton.core.llm.prompts import (
    RESILIENCE_NUDGE,
    SCRATCHPAD_STUCK_NUDGE,
    SCRATCHPAD_TIMEOUT_NUDGE,
)
from anton.core.session import ChatSession

_DEFAULTS = dict(
    coding_provider="anthropic",
    coding_model="",
    coding_api_key="",
    coding_base_url="",
)


def make_pad(name: str = "eng578") -> LocalScratchpadRuntime:
    return LocalScratchpadRuntime(name=name, **_DEFAULTS)


def shrink_timers(monkeypatch, *, heartbeat: str = "0.2") -> None:
    """1s silence window, 15s total budget, sub-second beats.

    Also shrinks the post-progress() grace window to 1s: left at its 60s
    default, a single "Installing {module}..." progress line would widen the
    silence window enough to survive a several-second silent auto-install on
    its own, making test_silent_auto_install_survives pass with or without
    the heartbeat and pin nothing.
    """
    monkeypatch.setenv("ANTON_CELL_INACTIVITY_TIMEOUT", "1")
    monkeypatch.setenv("ANTON_CELL_INACTIVITY_MAX", "1")
    monkeypatch.setenv("ANTON_CELL_INACTIVITY_AFTER_PROGRESS", "1")
    monkeypatch.setenv("ANTON_CELL_TIMEOUT_DEFAULT", "15")
    monkeypatch.setenv("ANTON_SCRATCHPAD_HEARTBEAT_INTERVAL", heartbeat)


class TestLivenessHeartbeat:
    async def test_silent_sleep_survives_inactivity_window(self, monkeypatch):
        """The ENG-578 core repro: a cell silent for > the inactivity window
        must complete, because the runtime heartbeats on its behalf."""
        shrink_timers(monkeypatch)
        pad = make_pad()
        await pad.start()
        try:
            cell = await pad.execute("import time; time.sleep(3); print('done')")
            assert cell.error is None
            assert cell.stdout.strip() == "done"
        finally:
            await pad.close()

    async def test_beatless_silent_cell_still_killed(self, monkeypatch):
        """Heartbeat disabled -> today's behavior: silence past the window kills.
        Proves the watchdog still catches genuinely dead/wedged workers."""
        shrink_timers(monkeypatch, heartbeat="0")
        pad = make_pad()
        await pad.start()
        try:
            cell = await pad.execute("import time; time.sleep(3); print('done')")
            assert cell.error is not None
            assert "liveness" in cell.error.lower()
        finally:
            await pad.close()

    async def test_blocking_call_survives(self, monkeypatch):
        """Blocking socket I/O (the SMTP shape) releases the GIL; beats must flow."""
        shrink_timers(monkeypatch)
        pad = make_pad()
        await pad.start()
        try:
            cell = await pad.execute(
                "import socket\n"
                "s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
                "s.settimeout(3)\n"
                "try:\n"
                "    s.recvfrom(1)\n"
                "except socket.timeout:\n"
                "    print('blocked-then-done')\n"
            )
            assert cell.error is None
            assert "blocked-then-done" in cell.stdout
        finally:
            await pad.close()

    async def test_progress_still_works_and_is_yielded(self, monkeypatch):
        """progress() behavior is unchanged: yielded to the caller, widens window."""
        shrink_timers(monkeypatch)
        pad = make_pad()
        await pad.start()
        try:
            progress_msgs: list[str] = []
            final = None
            async for item in pad.execute_streaming("progress('halfway'); print('ok')"):
                if isinstance(item, str):
                    progress_msgs.append(item)
                else:
                    final = item
            assert "halfway" in progress_msgs
            assert final is not None and final.error is None
        finally:
            await pad.close()

    async def test_no_stray_beats_corrupt_next_cell(self, monkeypatch):
        """Beats stop with the cell; a following cell parses cleanly."""
        shrink_timers(monkeypatch)
        pad = make_pad()
        await pad.start()
        try:
            await pad.execute("import time; time.sleep(2); print('one')")
            await asyncio.sleep(1)  # idle gap where a leaked thread would tick
            cell = await pad.execute("print('two')")
            assert cell.error is None
            assert cell.stdout.strip() == "two"
        finally:
            await pad.close()

    async def test_silent_auto_install_survives(self, monkeypatch, tmp_path):
        """The ENG-1275 shape: in-cell auto-install silent past the window.
        ANTON_UV_PATH is pointed at a stub that sleeps then fails, so the cell
        must survive the silent install attempt and report the install error
        (not an inactivity kill).

        _find_uv is patched to return None: LocalScratchpadRuntime.start()
        otherwise always overwrites ANTON_UV_PATH with a real `uv` found on
        PATH (near-universal on a dev/CI box that runs `uv run pytest`),
        silently discarding this test's stub and making it install-fail in
        well under a second instead of exercising the slow, silent path.
        """
        shrink_timers(monkeypatch)
        stub = tmp_path / "slow_uv.sh"
        stub.write_text("#!/bin/sh\nsleep 3\nexit 1\n")
        stub.chmod(0o755)
        monkeypatch.setenv("ANTON_UV_PATH", str(stub))
        monkeypatch.setattr(
            LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: None)
        )
        pad = make_pad()
        await pad.start()
        try:
            cell = await pad.execute("import definitely_not_a_real_module_xyz")
            assert cell.error is not None
            assert "auto-install failed" in cell.error.lower()
            assert "liveness" not in cell.error.lower()
        finally:
            await pad.close()


class TestPartialStdoutSalvage:
    async def test_killed_cell_reports_partial_stdout(self, monkeypatch):
        """Prints from completed iterations survive a kill. Uses a
        total-budget kill (heartbeat ON, tiny total budget) because that is
        the deterministic kill shape once liveness beats exist."""
        monkeypatch.setenv("ANTON_CELL_INACTIVITY_TIMEOUT", "1")
        monkeypatch.setenv("ANTON_CELL_INACTIVITY_MAX", "1")
        monkeypatch.setenv("ANTON_CELL_TIMEOUT_DEFAULT", "3")
        monkeypatch.setenv("ANTON_SCRATCHPAD_HEARTBEAT_INTERVAL", "0.2")
        pad = make_pad()
        await pad.start()
        try:
            cell = await pad.execute(
                "import time\n"
                "print('sent 1/3')\n"
                "print('sent 2/3')\n"
                "time.sleep(30)\n"   # runs into the 3s total budget
                "print('sent 3/3')\n"
            )
            assert cell.error is not None and "timed out" in cell.error.lower()
            assert "sent 2/3" in cell.stdout
            assert "sent 3/3" not in cell.stdout
            assert "partial" in cell.error.lower()
        finally:
            await pad.close()

    async def test_crash_reports_partial_stdout(self, monkeypatch):
        """Process death (EOF path) also attaches salvage."""
        shrink_timers(monkeypatch)
        pad = make_pad()
        await pad.start()
        try:
            cell = await pad.execute(
                "import os, time\n"
                "print('before-crash')\n"
                "time.sleep(0.5)\n"   # > one 0.2s tick so the chunk ships
                "os._exit(1)\n"
            )
            assert cell.error is not None
            assert "before-crash" in cell.stdout
        finally:
            await pad.close()

    async def test_successful_cell_output_appears_exactly_once(self, monkeypatch):
        """Salvage is discarded on success — no duplicated stdout."""
        shrink_timers(monkeypatch)
        pad = make_pad()
        await pad.start()
        try:
            cell = await pad.execute(
                "import time\n"
                "print('marker-once')\n"
                "time.sleep(0.5)\n"   # ensures the chunk shipped pre-result
                "print('tail')\n"
            )
            assert cell.error is None
            assert cell.stdout.count("marker-once") == 1
        finally:
            await pad.close()


class TestKillMessages:
    async def test_total_budget_kill_says_timed_out(self, monkeypatch):
        monkeypatch.setenv("ANTON_CELL_INACTIVITY_TIMEOUT", "1")
        monkeypatch.setenv("ANTON_CELL_INACTIVITY_MAX", "1")
        monkeypatch.setenv("ANTON_CELL_TIMEOUT_DEFAULT", "2")
        monkeypatch.setenv("ANTON_SCRATCHPAD_HEARTBEAT_INTERVAL", "0.2")
        pad = make_pad()
        await pad.start()
        try:
            cell = await pad.execute("import time; time.sleep(30)")
            assert cell.error is not None
            assert "timed out" in cell.error.lower()
            assert "liveness" not in cell.error.lower()
        finally:
            await pad.close()


class TestNudgeRouting:
    """Pure string routing, no LLM: the post-kill nudge must name the right
    cause for the timer that fired (ENG-578 — 'too heavy' taught per-item
    round-trips to a cell that was deliberately waiting)."""

    def test_liveness_kill_does_not_claim_too_heavy(self):
        nudge = ChatSession._select_resilience_nudge(
            "scratchpad",
            "Cell killed after 30s without a liveness signal from the scratchpad worker",
        )
        assert nudge == SCRATCHPAD_STUCK_NUDGE
        assert "too heavy" not in nudge

    def test_legacy_inactivity_text_routes_to_stuck(self):
        """Remote/old workers still emit the old wording; it must not get
        'too heavy' advice either."""
        nudge = ChatSession._select_resilience_nudge(
            "scratchpad", "Cell killed after 30s of inactivity (no output or progress() calls)"
        )
        assert nudge == SCRATCHPAD_STUCK_NUDGE

    def test_total_budget_kill_keeps_too_heavy(self):
        nudge = ChatSession._select_resilience_nudge(
            "scratchpad", "Cell timed out after 120s total"
        )
        assert nudge == SCRATCHPAD_TIMEOUT_NUDGE
        assert "too heavy" in nudge

    def test_non_scratchpad_unchanged(self):
        assert (
            ChatSession._select_resilience_nudge("web_search", "timed out")
            == RESILIENCE_NUDGE
        )


from anton.core.memory.acc import Event, detect_kill_loop


def _kill_event(reason: str, name: str = "pad", round_idx: int = 0) -> Event:
    return Event("scratchpad_killed", 6, {"name": name, "reason": reason}, round_idx)


class TestKillLoopLesson:
    """The durable ACC rule must be cause-aware: liveness kills teach
    reset-and-retry, never 'smaller batches' (ENG-578)."""

    def test_liveness_kills_do_not_teach_smaller_batches(self):
        events = [
            _kill_event(
                "Cell killed after 30s without a liveness signal from the scratchpad worker",
                round_idx=1,
            ),
            _kill_event(
                "Cell killed after 30s without a liveness signal from the scratchpad worker",
                round_idx=2,
            ),
        ]
        lesson = detect_kill_loop(events)
        assert lesson is not None
        low = lesson.rule.lower()
        assert "smaller" not in low
        assert "reset" in low

    def test_budget_kills_still_teach_smaller(self):
        events = [
            _kill_event("Cell timed out after 120s total", round_idx=1),
            _kill_event("Cell timed out after 120s total", round_idx=2),
        ]
        lesson = detect_kill_loop(events)
        assert lesson is not None
        assert "smaller" in lesson.rule.lower()

    def test_single_kill_still_below_threshold(self):
        assert detect_kill_loop([_kill_event("Cell timed out after 120s total")]) is None
