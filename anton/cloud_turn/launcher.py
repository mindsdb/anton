"""Hand a cloud turn to the pod's resident process, or run it directly.

Started by the scratchpad controller in place of ``python -m anton.cloud_turn``
and keeps its contract: one JSON request line on stdin, JSONL events on stdout,
diagnostics on stderr. The resident process has the turn's modules loaded and
forks a fresh process per turn, which takes over this process's stdin, stdout
and stderr; this process only waits for that turn's exit code.

Standard library only, run as ``python -I -S <this file>``: importing the
``anton`` package here would spend the time the resident process saves.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import socket
import struct
import sys
from typing import NoReturn

#: Abstract unix socket name (no leading NUL) the resident holds for the pod's lifetime.
SOCKET_NAME = "anton-cloud-turn-resident"
#: Covers the resident's imports on a pod that has just started.
READY_TIMEOUT_SECONDS = 15.0
READY = b"R"
NOT_READY = b"N"
HANDOFF = b"T"

_PEERCRED = struct.Struct("3i")
_PR_SET_DUMPABLE = 4


def socket_address(name: str) -> str:
    return "\0" + name


def forbid_same_user_access() -> None:
    """Keep other processes of the pod's user out of this process through /proc.

    Cell code runs as the same user. Without this it can read the environment
    and open the descriptors of whoever holds the turn's streams, and write
    events the controller trusts. Inherited by forks, reset by exec.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))


def connect_to_resident(name: str, resident_pid: int, timeout: float) -> tuple[socket.socket | None, str]:
    """Return a connection the resident declared ready on, or None and the reason.

    The peer is checked before anything is read or sent: while the resident is
    not listening, any process in the pod could hold the name, and it must never
    receive the turn's stdin.
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(socket_address(name))
    except OSError as exc:
        sock.close()
        return None, f"no resident process ({exc.strerror})"
    pid, uid, _ = _PEERCRED.unpack(sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _PEERCRED.size))
    if pid != resident_pid or uid != os.getuid():
        sock.close()
        return None, f"socket is held by pid {pid} uid {uid}, not the resident process"
    sock.settimeout(timeout)
    try:
        answer = sock.recv(1)
    except TimeoutError:
        sock.close()
        return None, f"resident process not ready after {timeout:.0f}s"
    except OSError as exc:
        sock.close()
        return None, f"resident process connection failed ({exc.strerror})"
    if answer != READY:
        sock.close()
        if answer == NOT_READY:
            return None, "resident process is not ready"
        return None, "resident process closed the connection"
    sock.settimeout(None)
    return sock, ""


def hand_off(sock: socket.socket) -> int:
    """Pass stdin, stdout and stderr to the resident and wait for the turn's exit code.

    Once the descriptors are sent there is no fallback: the turn may already be
    reading stdin, and a second run would read the same stream.
    """
    socket.send_fds(sock, [HANDOFF], [0, 1, 2])
    status = b""
    while not status.endswith(b"\n"):
        try:
            chunk = sock.recv(16)
        except OSError:
            chunk = b""
        if not chunk:
            print("cloud-turn launcher: resident process closed the connection mid-turn", file=sys.stderr, flush=True)
            return 1
        status += chunk
    return int(status)


def run_turn_directly(reason: str) -> NoReturn:
    print(f"cloud-turn launcher: running the turn directly: {reason}", file=sys.stderr, flush=True)
    os.execv(sys.executable, [sys.executable, "-m", "anton.cloud_turn"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one cloud turn through the pod's resident process.")
    # Tests run a resident that is not PID 1, under a name of their own.
    parser.add_argument("--socket", default=SOCKET_NAME, help=argparse.SUPPRESS)
    parser.add_argument("--resident-pid", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    try:
        forbid_same_user_access()
    except OSError as exc:
        # A direct turn has no such protection either; losing it is not worth losing the turn.
        print(f"cloud-turn launcher: could not close /proc access: {exc}", file=sys.stderr, flush=True)
    sock, reason = connect_to_resident(args.socket, args.resident_pid, READY_TIMEOUT_SECONDS)
    if sock is None:
        run_turn_directly(reason)
    with sock:
        return hand_off(sock)


if __name__ == "__main__":
    sys.exit(main())
