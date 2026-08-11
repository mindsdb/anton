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

    async def test_kill_message_says_state_is_restored_not_lost(self, tmp_path, monkeypatch):
        """ENG-1273: the old message ("state lost. Use reset to restart.")
        pointed the agent at the one recovery path that destroys the state
        ENG-1124 saved. With persistence actually configured, it must now
        say the state is being restored."""
        monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "true")
        monkeypatch.setenv("ANTON_CELL_INACTIVITY_TIMEOUT", "1")
        monkeypatch.setenv("ANTON_CELL_INACTIVITY_MAX", "1")
        monkeypatch.setenv("ANTON_CELL_TIMEOUT_DEFAULT", "2")
        monkeypatch.setenv("ANTON_SCRATCHPAD_HEARTBEAT_INTERVAL", "0.2")
        pad = LocalScratchpadRuntime(
            name="kill-msg-restored", _venvs_base=tmp_path / "venvs", session_id="conv", **_DEFAULTS
        )
        await pad.start()
        try:
            cell = await pad.execute("import time; time.sleep(30)")
            assert cell.error is not None
            low = cell.error.lower()
            assert "state lost" not in low
            assert "restore" in low
            # Still names reset, but as the deliberate-wipe option, not the
            # only path back.
            assert "reset" in low
            # The routing-critical phrase must still be intact (Global
            # Constraints) — this is a regression guard for THIS test file,
            # duplicating (deliberately) the check TestNudgeRouting already
            # makes on the routing function itself.
            assert "timed out" in low
        finally:
            await pad.cleanup()

    async def test_kill_message_is_honest_without_persistence_configured(self, monkeypatch):
        """ENG-1273 final-review finding: without a session id (bare CLI /
        this test's default pad — no snapshot is even possible), the kill
        message must NOT promise a restoration that cannot happen."""
        monkeypatch.setenv("ANTON_CELL_INACTIVITY_TIMEOUT", "1")
        monkeypatch.setenv("ANTON_CELL_INACTIVITY_MAX", "1")
        monkeypatch.setenv("ANTON_CELL_TIMEOUT_DEFAULT", "2")
        monkeypatch.setenv("ANTON_SCRATCHPAD_HEARTBEAT_INTERVAL", "0.2")
        pad = make_pad()  # no session_id -> no snapshot possible
        await pad.start()
        try:
            cell = await pad.execute("import time; time.sleep(30)")
            assert cell.error is not None
            low = cell.error.lower()
            assert "already saved to disk" not in low
            assert "restores it automatically" not in low
            assert "timed out" in low
        finally:
            await pad.close()


class TestResumeAfterKill:
    """ENG-1273: a watchdog-killed cell must cost only itself, not the pad's
    accumulated state — resume() (not reset) is the recovery path."""

    async def test_watchdog_kill_auto_resumes_and_restores_last_snapshot(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "true")
        monkeypatch.setenv("ANTON_CELL_INACTIVITY_TIMEOUT", "1")
        monkeypatch.setenv("ANTON_CELL_INACTIVITY_MAX", "1")
        monkeypatch.setenv("ANTON_CELL_TIMEOUT_DEFAULT", "2")
        monkeypatch.setenv("ANTON_SCRATCHPAD_HEARTBEAT_INTERVAL", "0.2")
        pad = LocalScratchpadRuntime(
            name="eng1273", _venvs_base=tmp_path / "venvs", session_id="conv", **_DEFAULTS
        )
        await pad.start()
        try:
            cell1 = await pad.execute("x = 42\nprint('set')")
            assert cell1.error is None, cell1.error

            # Cell 2: the watchdog kills it (silent, past both the
            # inactivity and total budget).
            cell2 = await pad.execute("import time; time.sleep(30)")
            assert cell2.error is not None
            assert "timed out" in cell2.error.lower()

            # Cell 3: the pad is dead; execute() must auto-resume — reloading
            # the namespace as of cell 1 — and run normally, with no
            # explicit reset call in between.
            cell3 = await pad.execute("print(x)")
            assert cell3.error is None, cell3.error
            assert cell3.stdout.strip() == "42"
            # The counter proved itself and reset back to zero.
            assert pad._consecutive_deaths == 0
        finally:
            await pad.cleanup()

    async def test_repeated_kills_fall_back_to_reset_and_say_so(self, tmp_path, monkeypatch):
        """A pad that dies on every resume (e.g. a corrupt reloaded snapshot)
        must not loop — it falls back to a real reset and tells the agent."""
        monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "true")
        pad = LocalScratchpadRuntime(
            name="looping", _venvs_base=tmp_path / "venvs", session_id="conv", **_DEFAULTS
        )
        await pad.start()
        try:
            await pad.execute("kept = 'first turn'")

            real_resume = pad.resume

            async def _resume_then_die():
                await real_resume()
                pad._proc.kill()
                await pad._proc.wait()

            monkeypatch.setattr(pad, "resume", _resume_then_die)

            # Kill the process to seed "dead" state, then let it die on
            # every resume attempt for exactly `cap` calls — each of those
            # uses up one of the allowed plain-resume attempts (the check in
            # _auto_resume is `consecutive_deaths >= cap`, tested BEFORE
            # incrementing, so it takes `cap` failed resumes to reach it).
            pad._proc.kill()
            await pad._proc.wait()
            for _ in range(pad._MAX_CONSECUTIVE_AUTO_RESUMES):
                cell = await pad.execute("print('should not run')")
                assert cell.error is not None  # still dead going into the next call

            # This call crosses the cap: falls back to a real reset() and
            # actually runs, but 'kept' is gone and the agent is told why.
            cell = await pad.execute("print('kept' in dir())")
            assert cell.error is None, cell.error
            assert cell.stdout.strip() == "False"
            assert "fully reset" in (cell.logs or "").lower()
        finally:
            await pad.cleanup()

    async def test_legitimate_consecutive_kills_do_not_wipe_state(self, tmp_path, monkeypatch):
        """ENG-1273 final-review finding: resume() succeeding (proven by the
        process accepting and running a cell) must not count toward the
        death cap just because that cell itself later gets killed for
        running long — only resume() FAILING to produce a working process
        should count. Otherwise a batch of several genuinely slow cells in
        a row would trip the fallback and wipe state for a reason unrelated
        to resume() at all."""
        monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "true")
        monkeypatch.setenv("ANTON_CELL_INACTIVITY_TIMEOUT", "1")
        monkeypatch.setenv("ANTON_CELL_INACTIVITY_MAX", "1")
        monkeypatch.setenv("ANTON_CELL_TIMEOUT_DEFAULT", "2")
        monkeypatch.setenv("ANTON_SCRATCHPAD_HEARTBEAT_INTERVAL", "0.2")
        pad = LocalScratchpadRuntime(
            name="legit-kills", _venvs_base=tmp_path / "venvs", session_id="conv", **_DEFAULTS
        )
        await pad.start()
        try:
            cell1 = await pad.execute("precious = 'still here'\nprint('set')")
            assert cell1.error is None, cell1.error

            # Three cells in a row, each individually killed for running
            # over budget — resume() succeeds each time (the process comes
            # back and actually runs the next cell), so this must NOT trip
            # the fallback-reset cap.
            for _ in range(3):
                cell = await pad.execute("import time; time.sleep(30)")
                assert cell.error is not None
                assert "timed out" in cell.error.lower()

            cell_final = await pad.execute("print(precious)")
            assert cell_final.error is None, cell_final.error
            assert cell_final.stdout.strip() == "still here"
            assert "fully reset" not in (cell_final.logs or "").lower()
        finally:
            await pad.cleanup()


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
        # ENG-1273: a liveness kill auto-resumes now — the durable lesson
        # must not tell the agent to reset (that destroys the state
        # resume() would otherwise have kept).
        assert "reset" not in low
        assert "retry" in low

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

    def test_description_says_kill_recovery_is_automatic(self):
        """ENG-1273: the agent should not have to guess that reset is now
        optional after a kill — the tool contract says so directly."""
        from anton.core.tools.tool_defs import SCRATCHPAD_TOOL

        desc = SCRATCHPAD_TOOL.description.lower()
        assert "restarts and restores everything else automatically" in desc
        assert "you do not need to reset" in desc
        # Must not regress the ENG-578 contract text this paragraph already carries.
        assert "kept alive automatically" in desc

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
