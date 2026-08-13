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
    SCRATCHPAD_INSTALL_NUDGE,
    SCRATCHPAD_SILENT_TIMEOUT_NUDGE,
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
            # Within the FIRST 120 chars: observe_scratchpad_cell records
            # reason=err[:120] for ACC, so the routing keyword must survive
            # that slice — a future prefix pushing it out would silently
            # return every kill to the "too heavy" lesson with tests green.
            assert "liveness" in cell.error[:120].lower()
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
        """Beats stop with the cell: the next cell parses cleanly AND sees no
        leaked heartbeat thread from its predecessor (its own is the only
        daemon thread alive in the worker)."""
        shrink_timers(monkeypatch)
        pad = make_pad()
        await pad.start()
        try:
            await pad.execute("import time; time.sleep(2); print('one')")
            await asyncio.sleep(1)  # idle gap where a leaked thread would tick
            cell = await pad.execute(
                "import threading\n"
                "alive = [t for t in threading.enumerate()\n"
                "         if t is not threading.main_thread() and t.daemon]\n"
                "print(f'daemon-threads={len(alive)}')\n"
            )
            assert cell.error is None
            assert "daemon-threads=1" in cell.stdout
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


class TestAutoInstallBudget:
    """ENG-1275: the in-cell auto-installer must run under a budget derived
    from CoreSettings.cell_install_timeout (one number, both sides), fail with
    an error naming the install as the cause, and never take the pad down."""

    async def test_install_past_install_timeout_fails_named_and_pad_survives(
        self, monkeypatch, tmp_path
    ):
        """An install running past cell_install_timeout must produce a cell
        error naming the auto-install and the module — and the worker must
        survive it (no crash, next cell runs)."""
        shrink_timers(monkeypatch)
        monkeypatch.setenv("ANTON_CELL_INSTALL_TIMEOUT", "1")
        stub = tmp_path / "hung_uv.sh"
        stub.write_text("#!/bin/sh\nexec sleep 5\n")
        stub.chmod(0o755)
        monkeypatch.setenv("ANTON_UV_PATH", str(stub))
        monkeypatch.setattr(
            LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: None)
        )
        pad = make_pad()
        await pad.start()
        try:
            cell = await pad.execute("import module_that_installs_slowly_eng1275")
            assert cell.error is not None
            low = cell.error.lower()
            assert "auto-install" in low
            assert "module_that_installs_slowly_eng1275" in cell.error
            assert "liveness" not in low
            follow_up = await pad.execute("print('pad-still-alive')")
            assert follow_up.error is None
            assert "pad-still-alive" in follow_up.stdout
        finally:
            await pad.close()

    async def test_install_outlasting_total_budget_completes(
        self, monkeypatch, tmp_path
    ):
        """The `import torch` shape: an install longer than the cell's total
        budget must not be killed — the budget defers to the install's own
        allowance while it runs, and the retried cell completes."""
        monkeypatch.setenv("ANTON_CELL_INACTIVITY_TIMEOUT", "1")
        monkeypatch.setenv("ANTON_CELL_INACTIVITY_MAX", "1")
        monkeypatch.setenv("ANTON_CELL_INACTIVITY_AFTER_PROGRESS", "1")
        monkeypatch.setenv("ANTON_CELL_TIMEOUT_DEFAULT", "2")
        monkeypatch.setenv("ANTON_SCRATCHPAD_HEARTBEAT_INTERVAL", "0.2")
        fake_mod = tmp_path / "eng1275_fake_mod.py"
        stub = tmp_path / "slow_ok_uv.sh"
        stub.write_text(
            f"#!/bin/sh\nsleep 3\necho 'VALUE = 42' > '{fake_mod}'\nexit 0\n"
        )
        stub.chmod(0o755)
        monkeypatch.setenv("ANTON_UV_PATH", str(stub))
        monkeypatch.setattr(
            LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: None)
        )
        pad = make_pad()
        await pad.start()
        try:
            cell = await pad.execute(
                f"import sys\nsys.path.insert(0, {str(tmp_path)!r})\n"
                "import eng1275_fake_mod\nprint(eng1275_fake_mod.VALUE)\n"
            )
            assert cell.error is None
            assert "42" in cell.stdout
        finally:
            await pad.close()

    async def test_worker_death_mid_install_names_install(
        self, monkeypatch, tmp_path
    ):
        """A worker that dies while installing must not report the generic
        'Process exited unexpectedly.' — the error names the install."""
        shrink_timers(monkeypatch)
        stub = tmp_path / "killer_uv.sh"
        stub.write_text("#!/bin/sh\nkill -9 $PPID\n")
        stub.chmod(0o755)
        monkeypatch.setenv("ANTON_UV_PATH", str(stub))
        monkeypatch.setattr(
            LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: None)
        )
        pad = make_pad()
        await pad.start()
        try:
            cell = await pad.execute("import module_whose_install_crashes_eng1275")
            assert cell.error is not None
            assert "auto-install" in cell.error.lower()
            assert "module_whose_install_crashes_eng1275" in cell.error
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
            # Silent budget kill: the message must say the ambiguity out loud
            # (stuck vs silently heavy), within the ACC's 120-char slice.
            assert "without producing any output" in cell.error[:120].lower()
        finally:
            await pad.close()

    async def test_producing_budget_kill_has_no_silent_marker(self, monkeypatch):
        """A budget kill that WAS producing output keeps the plain message —
        that is the one case where "too heavy" is genuinely right."""
        monkeypatch.setenv("ANTON_CELL_INACTIVITY_TIMEOUT", "1")
        monkeypatch.setenv("ANTON_CELL_INACTIVITY_MAX", "1")
        monkeypatch.setenv("ANTON_CELL_TIMEOUT_DEFAULT", "3")
        monkeypatch.setenv("ANTON_SCRATCHPAD_HEARTBEAT_INTERVAL", "0.2")
        pad = make_pad()
        await pad.start()
        try:
            cell = await pad.execute(
                "import time\nprint('working')\ntime.sleep(0.5)\ntime.sleep(30)\n"
            )
            assert cell.error is not None
            assert "timed out" in cell.error.lower()
            assert "without producing any output" not in cell.error.lower()
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

    def test_silent_budget_kill_gets_honest_nudge(self):
        """A budget kill with zero output is ambiguous (stuck vs silently
        heavy) — it must not claim "too heavy" and must not claim wedged."""
        nudge = ChatSession._select_resilience_nudge(
            "scratchpad",
            "Cell timed out after 120s total without producing any output — "
            "either a call is stuck or the work is heavier than estimated",
        )
        assert nudge == SCRATCHPAD_SILENT_TIMEOUT_NUDGE
        assert "too heavy" not in nudge

    def test_non_scratchpad_unchanged(self):
        assert (
            ChatSession._select_resilience_nudge("web_search", "timed out")
            == RESILIENCE_NUDGE
        )

    def test_install_timeout_error_routes_to_install_nudge(self):
        """An install that ran out of its budget is neither a size nor a
        liveness problem — 'make the cell smaller' cannot make a package
        install (ENG-1275)."""
        nudge = ChatSession._select_resilience_nudge(
            "scratchpad",
            "ModuleNotFoundError: No module named 'torch'\n"
            "Auto-install of 'torch' was killed after 120s (cell_install_timeout) "
            "without finishing — the package is not installed.",
        )
        assert nudge == SCRATCHPAD_INSTALL_NUDGE
        assert "too heavy" not in nudge
        assert "smaller" not in nudge

    def test_install_failure_routes_to_install_nudge(self):
        nudge = ChatSession._select_resilience_nudge(
            "scratchpad",
            "ModuleNotFoundError: No module named 'torhc'\n"
            "Auto-install failed:\nERROR: No matching distribution found for torhc",
        )
        assert nudge == SCRATCHPAD_INSTALL_NUDGE

    def test_kill_during_install_routes_to_install_nudge(self):
        """The kill wording also contains 'liveness' — the install cause must
        win over the stuck diagnosis."""
        nudge = ChatSession._select_resilience_nudge(
            "scratchpad",
            "Cell killed during auto-install of 'torch' — no liveness signal "
            "for 150s: the worker process died or the installer is wedged; "
            "the package is likely not installed",
        )
        assert nudge == SCRATCHPAD_INSTALL_NUDGE


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

    def test_mixed_turn_majority_liveness_wins(self):
        """2 budget + 3 liveness must NOT write "too heavy" (majority wins)."""
        events = [
            _kill_event("Cell timed out after 120s total", round_idx=1),
            _kill_event("Cell timed out after 120s total", round_idx=2),
            _kill_event("Cell killed after 30s without a liveness signal", round_idx=3),
            _kill_event("Cell killed after 30s without a liveness signal", round_idx=4),
            _kill_event("Cell killed after 30s without a liveness signal", round_idx=5),
        ]
        lesson = detect_kill_loop(events)
        assert lesson is not None
        assert "smaller" not in lesson.rule.lower()

    def test_heavy_majority_still_wins(self):
        events = [
            _kill_event("Cell timed out after 120s total", round_idx=1),
            _kill_event("Cell timed out after 120s total", round_idx=2),
            _kill_event("Cell killed after 30s without a liveness signal", round_idx=3),
        ]
        lesson = detect_kill_loop(events)
        assert lesson is not None
        assert "smaller" in lesson.rule.lower()

    def test_unreasoned_kills_write_no_rule(self):
        """A kill with no recorded reason is ambiguous — no durable rule
        beats a wrong one."""
        events = [_kill_event("", round_idx=1), _kill_event("", round_idx=2)]
        assert detect_kill_loop(events) is None

    def test_silent_budget_kills_write_no_rule(self):
        """Silent budget kills are ambiguous (stuck vs silently heavy)."""
        events = [
            _kill_event(
                "Cell timed out after 120s total without producing any output",
                round_idx=1,
            ),
            _kill_event(
                "Cell timed out after 120s total without producing any output",
                round_idx=2,
            ),
        ]
        assert detect_kill_loop(events) is None

    def test_install_kills_write_no_rule(self):
        """A kill during a package install says nothing about the cell's size
        or the worker's health — no durable rule beats a wrong one (ENG-1275)."""
        events = [
            _kill_event(
                "Cell killed during auto-install of 'torch' — the install ran "
                "past its 120s budget and grace window without reporting a result",
                round_idx=1,
            ),
            _kill_event(
                "Cell killed during auto-install of 'torch' — no liveness signal "
                "for 150s: the worker process died or the installer is wedged",
                round_idx=2,
            ),
        ]
        assert detect_kill_loop(events) is None


class TestToolContractText:
    def test_description_does_not_claim_progress_is_survival_critical(self):
        from anton.core.tools.tool_defs import SCRATCHPAD_TOOL

        desc = SCRATCHPAD_TOOL.description.lower()
        # The new contract: the runtime keeps working cells alive; progress()
        # is status, not survival; only the total budget (or a dead worker)
        # kills a cell.
        assert "kept alive automatically" in desc
        assert "inactivity timeout" not in desc
        assert "reset the timer" not in desc

    def test_system_prompt_does_not_mandate_splitting_cells(self):
        # The system prompt was a fourth author of the wrong lesson: "hard
        # timeout of 120 seconds" + "you MUST break the work into smaller
        # cells" taught the per-item pattern before the tool was even called.
        from anton.core.llm.prompts import CHAT_SYSTEM_PROMPT

        low = CHAT_SYSTEM_PROMPT.lower()
        assert "must break the work into smaller cells" not in low
        assert "hard timeout of 120 seconds" not in low
        assert "kept alive automatically" in low


class TestCwdNote:
    """ENG-578 fix #5, warn-only: a cell that changes the CWD says so — the
    change silently persists into later cells and used to cost debug rounds."""

    async def test_chdir_cell_gets_note_with_both_paths(self, monkeypatch, tmp_path):
        shrink_timers(monkeypatch)
        target = (tmp_path / "subdir").resolve()
        target.mkdir()
        pad = make_pad()
        await pad.start()
        try:
            cell = await pad.execute(
                f"import os\nos.chdir({str(target)!r})\nprint('moved')\n"
            )
            assert cell.error is None
            assert "changed the working directory" in cell.stdout
            assert str(target) in cell.stdout
            assert "persists for subsequent cells" in cell.stdout
        finally:
            await pad.close()

    async def test_next_cell_without_chdir_has_no_note(self, monkeypatch, tmp_path):
        shrink_timers(monkeypatch)
        target = (tmp_path / "subdir").resolve()
        target.mkdir()
        pad = make_pad()
        await pad.start()
        try:
            await pad.execute(f"import os; os.chdir({str(target)!r})")
            cell = await pad.execute("print('still-here')")
            assert cell.error is None
            assert "changed the working directory" not in cell.stdout
        finally:
            await pad.close()

    async def test_chdir_to_same_directory_no_note(self, monkeypatch):
        shrink_timers(monkeypatch)
        pad = make_pad()
        await pad.start()
        try:
            cell = await pad.execute("import os; os.chdir(os.getcwd()); print('noop')")
            assert cell.error is None
            assert "changed the working directory" not in cell.stdout
        finally:
            await pad.close()

    async def test_error_after_chdir_still_notes(self, monkeypatch, tmp_path):
        shrink_timers(monkeypatch)
        target = (tmp_path / "subdir").resolve()
        target.mkdir()
        pad = make_pad()
        await pad.start()
        try:
            cell = await pad.execute(
                f"import os\nos.chdir({str(target)!r})\nraise RuntimeError('boom')\n"
            )
            assert cell.error is not None and "boom" in cell.error
            assert "changed the working directory" in cell.stdout
        finally:
            await pad.close()
