"""Resident process for tests: ``python tests/cloud_turn_resident_entry.py <socket-name>``.

Runs the REAL ``anton.cloud_turn.resident.serve()`` under a socket name of the
test's own, with the turn entry and the preload picked by env. NOT collected by
pytest (no ``test_`` prefix). ``--direct`` runs the entry once in this process
instead, for comparing a fork with a freshly exec'd interpreter.

Env:
* ``RESIDENT_TEST_ENTRY``: ``turn`` (the real cloud turn), ``fake``
  (``cloud_turn_fake_entry.main``, mode from ``CLOUD_TURN_FAKE_MODE``) or
  ``probe`` (reports the process state, see ``_probe``).
* ``RESIDENT_TEST_PRELOAD``: ``real`` (default), ``fail``, or ``slow:<seconds>``.
* ``RESIDENT_TEST_MARKER``: a file the entry creates when a fork runs it.
* ``RESIDENT_TEST_MODULES_REPORT``: a file the entry appends the modules to that
  the turn imported on top of the preload.
"""

from __future__ import annotations

import atexit
import ctypes
import json
import os
import sys
import threading
import time
from pathlib import Path

from anton.cloud_turn import resident

_REPORTED_PREFIXES = ("anton", "openai", "httpx2", "httpcore2")
_PR_GET_DUMPABLE = 3


def _probe() -> int:
    """Read one JSON line of instructions, report this process, then act on them."""
    orders = json.loads(sys.stdin.readline() or "{}")
    fds = sorted(int(fd) for fd in os.listdir("/proc/self/fd"))
    state = {
        "kind": "probe",
        "pid": os.getpid(),
        "env": dict(os.environ),
        "cwd": os.getcwd(),
        "sys_path": sys.path,
        "flags": list(sys.flags),
        "threads": threading.active_count(),
        "fds": fds,
        "dumpable": ctypes.CDLL(None).prctl(_PR_GET_DUMPABLE, 0, 0, 0, 0),
    }
    os.write(1, (json.dumps(state) + "\n").encode())
    if orders.get("atexit_file"):
        atexit.register(Path(orders["atexit_file"]).write_text, "ran")
    if orders.get("thread_file"):
        def finish_late(path: str = orders["thread_file"]) -> None:
            time.sleep(0.5)
            Path(path).write_text("ran")

        threading.Thread(target=finish_late).start()
    time.sleep(orders.get("sleep", 0))
    return orders.get("exit", 0)


def _turn() -> int | None:
    from anton.cloud_turn.__main__ import main

    return main()


def _fake() -> int:
    sys.path.insert(0, str(Path(__file__).parent))
    import cloud_turn_fake_entry

    return cloud_turn_fake_entry.main()


_ENTRIES = {"turn": _turn, "fake": _fake, "probe": _probe}


def _entry():
    target = _ENTRIES[os.environ.get("RESIDENT_TEST_ENTRY", "turn")]
    marker = os.environ.get("RESIDENT_TEST_MARKER")
    report = os.environ.get("RESIDENT_TEST_MODULES_REPORT")

    def run():
        if marker:
            Path(marker).touch()
        before = set(sys.modules)
        try:
            return target()
        finally:
            if report:
                added = sorted(m for m in set(sys.modules) - before if m.startswith(_REPORTED_PREFIXES))
                with open(report, "a") as f:
                    f.write("".join(f"{m}\n" for m in added))

    return run


def _preload():
    spec = os.environ.get("RESIDENT_TEST_PRELOAD", "real")
    if spec == "real":
        return resident.preload_turn_modules
    if spec == "fail":
        def fail() -> None:
            raise ImportError("simulated preload failure")

        return fail
    seconds = float(spec.removeprefix("slow:"))

    def slow() -> None:
        time.sleep(seconds)
        resident.preload_turn_modules()

    return slow


if __name__ == "__main__":
    if sys.argv[1] == "--direct":
        sys.exit(_entry()())
    sys.exit(resident.serve(_entry(), name=sys.argv[1], preload=_preload()))
