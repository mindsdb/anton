"""A cloud turn closes itself to the pod's other processes before it reads its request.

Cell code runs as the same user as the turn, and the request carries the turn's
credentials. Here the test process stands in for a process a cell left behind:
same user, reading the turn's environment through /proc while the turn waits
for its request line.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from anton.cloud_turn import launcher

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="/proc and prctl are Linux only")

#: Covers the turn's imports on a slow runner.
CLOSED_WITHIN_SECONDS = 60.0


def _closed_to_this_user(pid: int) -> bool:
    try:
        Path(f"/proc/{pid}/environ").read_bytes()
    except PermissionError:
        return True
    return False


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["-m", "anton.cloud_turn"], id="direct"),
        pytest.param(
            ["-I", "-S", launcher.__file__, "--socket", f"anton-test-{uuid.uuid4().hex}"],
            id="launcher-without-resident",
        ),
    ],
)
def test_the_turn_is_closed_before_its_request_arrives(args, tmp_path):
    env = os.environ.copy()
    env["ANTON_CLOUD_WORKSPACE_PATH"] = str(tmp_path)
    proc = subprocess.Popen(
        [sys.executable, *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
    )
    try:
        deadline = time.monotonic() + CLOSED_WITHIN_SECONDS
        closed = False
        while time.monotonic() < deadline and proc.poll() is None:
            if _closed_to_this_user(proc.pid):
                closed = True
                break
            time.sleep(0.05)
        # Still waiting for its request: nothing has been sent yet.
        assert proc.poll() is None, proc.stderr.read().decode(errors="replace")[-2000:]
        assert closed
    finally:
        proc.kill()
        proc.communicate(timeout=10)
