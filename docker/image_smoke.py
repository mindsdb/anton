"""Prove the built image can actually run a turn, before it is pushed.

Run as a Dockerfile layer AFTER `USER 1000`, so every check sees exactly what a
scratchpad pod sees: the same venv, the same interpreter, the same uid.

The point is the class of failure where the image is well-formed and the build is
green, but no pod that pulls it can serve a turn. On 2026-08-31 every `sp-live-*`
pod failed before anton ran, and because nothing had ever executed the entrypoint
outside a pod there was no earlier signal to read: the first evidence the image
was unusable was that prod stopped answering. Each check below is one way that
can happen.

Every check RUNS the real entrypoint rather than importing it. An import proves
the module resolves; only running it proves the whole graph loads and the wire
still speaks the contract the controller reads.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

# Both entrypoints read one bounded thing from stdin and stop at EOF, so an empty
# stdin drives a complete, harmless run: cloud_turn fails to parse the empty line
# and emits its terminal event, scratchpad_boot sees EOF and leaves its cell loop.
EMPTY_STDIN = b""
TIMEOUT_S = 120


def run_module(module: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, "-m", module],
        input=EMPTY_STDIN,
        capture_output=True,
        timeout=TIMEOUT_S,
    )


def check_runtime_user() -> list[str]:
    """The layer must sit below `USER 1000`, or none of this tests the pod.

    Running these checks as root would pass while the pod still fails, which is
    worse than not running them: it reports a guarantee the image does not hold.
    """
    if os.getuid() != 1000:
        return [f"runtime user: running as uid {os.getuid()}, expected 1000"]
    return []


def check_version() -> list[str]:
    """The version the image reports must be the one the workflow resolved.

    A wrong version here is invisible in the cluster: every pod reports it, it is
    well-formed, and it reads as a real cohort in any breakdown rather than as a
    missing value (ENG-1796).
    """
    import anton

    want = os.environ.get("SETUPTOOLS_SCM_PRETEND_VERSION", "")
    if not want:
        return ["version: SETUPTOOLS_SCM_PRETEND_VERSION is not set in the image"]

    found = []
    if anton.__version__ != want:
        found.append(
            f"version: anton reports {anton.__version__!r}, image was built as {want!r}"
        )
    if anton.__version__.startswith("2.0.0"):
        found.append(f"version: {anton.__version__!r} is the hatch-vcs fallback")
    return found


def check_cloud_turn() -> list[str]:
    """`python -m anton.cloud_turn` is what the controller execs for a whole turn.

    Empty stdin fails to parse, which is the shortest path that still loads the
    entire turn graph and writes to the protocol descriptor. Exactly one line, and
    it must be a terminal event: the controller reads the terminal off the EVENT,
    not off the exit code, so a process that exits 0 having printed nothing is a
    turn that hangs until the stall timer fires.
    """
    proc = run_module("anton.cloud_turn")
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="replace")[-2000:]
        return [f"cloud_turn: exited {proc.returncode}\n{tail}"]

    lines = [ln for ln in proc.stdout.decode(errors="replace").splitlines() if ln.strip()]
    if len(lines) != 1:
        return [
            f"cloud_turn: expected 1 protocol line on stdout, got {len(lines)}: {lines[:5]}"
        ]
    try:
        event = json.loads(lines[0])
    except ValueError as exc:
        return [f"cloud_turn: stdout line is not JSON ({exc}): {lines[0][:200]!r}"]
    if event.get("kind") != "turn_failed":
        return [f"cloud_turn: expected a turn_failed terminal, got {event.get('kind')!r}"]
    return []


def check_scratchpad_boot() -> list[str]:
    """`/usr/local/bin/scratchpad-boot.sh` is what the controller execs for one cell.

    It has no main guard: the module body IS the cell loop, so running it is the
    only way to load it. EOF on stdin ends that loop with no cell executed.
    """
    proc = run_module("anton.core.backends.scratchpad_boot")
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="replace")[-2000:]
        return [f"scratchpad_boot: exited {proc.returncode}\n{tail}"]
    return []


def check_runtime_uv() -> list[str]:
    """scratchpad_boot shells out to `uv pip install` for a package a cell imports.

    ANTON_UV_PATH points at this exact path, so a uv that moved or lost its
    execute bit turns every missing-package recovery into a failed cell.
    """
    uv = "/usr/local/bin/uv"
    if not os.access(uv, os.X_OK):
        return [f"runtime uv: {uv} is missing or not executable"]
    return []


def check_venv_writable() -> list[str]:
    """That same install writes into the venv as uid 1000.

    The Dockerfile chowns the venv for exactly this. A chown that stops covering
    it leaves an image that serves turns normally right up until a cell imports
    something that is not installed.
    """
    venv = os.environ.get("VIRTUAL_ENV", "")
    if not venv:
        return ["venv: VIRTUAL_ENV is not set in the image"]
    if not os.access(venv, os.W_OK):
        return [f"venv: {venv} is not writable by uid {os.getuid()}"]
    return []


CHECKS = (
    check_runtime_user,
    check_version,
    check_cloud_turn,
    check_scratchpad_boot,
    check_runtime_uv,
    check_venv_writable,
)


def run_checks(checks=CHECKS) -> list[str]:
    """Run every check and collect what failed.

    A check that raises counts as a failed check rather than a crashed script: one
    broken check must not hide the verdict of the other five, and a smoke that
    dies partway through is indistinguishable from one that never ran.
    """
    failures: list[str] = []
    for check in checks:
        try:
            failures.extend(check())
        except Exception as exc:
            failures.append(f"{check.__name__}: raised {type(exc).__name__}: {exc}")
    return failures


def main() -> int:
    failures = run_checks()
    if failures:
        print("image smoke FAILED:", file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        return 1
    print("image smoke passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
