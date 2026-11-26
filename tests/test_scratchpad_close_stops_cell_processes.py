"""Closing a scratchpad stops the processes its cells started, as the tool
description says; one in its own session is outside that and keeps running."""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys

import pytest

from anton.core.backends.local import LocalScratchpadRuntime

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="process groups are POSIX")

_START = """
import subprocess
plain = subprocess.Popen(["sleep", "300"])
own_session = subprocess.Popen(["sleep", "300"], start_new_session=True)
print(plain.pid, own_session.pid)
"""


def _running(pid: int) -> bool:
    # A killed process can linger as a zombie until its new parent reaps it.
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout
    return bool(state.strip()) and not state.strip().startswith("Z")


async def _stopped(pid: int) -> bool:
    for _ in range(100):
        if not _running(pid):
            return True
        await asyncio.sleep(0.05)
    return False


async def test_closing_the_scratchpad_stops_its_cells_processes():
    pad = LocalScratchpadRuntime(
        name="background", coding_provider="anthropic", coding_model="", coding_api_key="", coding_base_url="",
    )
    await pad.start()
    try:
        cell = await pad.execute(_START)
        plain, own_session = (int(pid) for pid in cell.stdout.split())
    finally:
        await pad.close()

    try:
        assert await _stopped(plain)
        assert _running(own_session)
    finally:
        for pid in (plain, own_session):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
