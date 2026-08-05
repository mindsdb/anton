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

_DEFAULTS = dict(
    coding_provider="anthropic",
    coding_model="",
    coding_api_key="",
    coding_base_url="",
)


def make_pad(name: str = "eng578") -> LocalScratchpadRuntime:
    return LocalScratchpadRuntime(name=name, **_DEFAULTS)


def shrink_timers(monkeypatch, *, heartbeat: str = "0.2") -> None:
    """1s silence window, 15s total budget, sub-second beats."""
    monkeypatch.setenv("ANTON_CELL_INACTIVITY_TIMEOUT", "1")
    monkeypatch.setenv("ANTON_CELL_INACTIVITY_MAX", "1")
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
        (not an inactivity kill)."""
        shrink_timers(monkeypatch)
        stub = tmp_path / "slow_uv.sh"
        stub.write_text("#!/bin/sh\nsleep 3\nexit 1\n")
        stub.chmod(0o755)
        monkeypatch.setenv("ANTON_UV_PATH", str(stub))
        pad = make_pad()
        await pad.start()
        try:
            cell = await pad.execute("import definitely_not_a_real_module_xyz")
            assert cell.error is not None
            assert "auto-install failed" in cell.error.lower()
            assert "liveness" not in cell.error.lower()
        finally:
            await pad.close()
