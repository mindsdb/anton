"""The pod's resident process: preload the cloud turn once, fork a process per turn.

Runs as the pod's main process (PID 1). For every turn the controller execs the
launcher (``launcher.py``), which passes its stdin, stdout and stderr here. A
forked copy takes them over and runs the cloud turn entrypoint with the turn's
modules already imported. The resident never reads the request, so the turn
key and everything the turn builds live only in that copy, which exits when the
turn does.

Linux only: abstract unix sockets, ``SO_PEERCRED`` and ``prctl``.
"""

from __future__ import annotations

import atexit
import contextlib
import gc
import importlib
import logging
import os
import selectors
import signal
import socket
import sys
import threading
import time
import traceback
from collections.abc import Callable
from typing import NoReturn

from anton.cloud_turn.launcher import (
    HANDOFF,
    NOT_READY,
    READY,
    SOCKET_NAME,
    forbid_same_user_access,
    socket_address,
)

logger = logging.getLogger(__name__)

#: What a cloud turn imports, loaded once before the first fork. Kept complete by
#: tests/test_cloud_turn_resident.py, which runs real turns in a fork.
PRELOAD_MODULES = (
    # Building the session.
    "anton.cloud_turn.__main__",
    "anton.config.settings",
    "anton.core.backends.local",
    "anton.core.llm.client",
    "anton.core.llm.identity",
    "anton.core.llm.openai",
    "anton.core.session",
    "anton.workspace",
    # Running it: memory, tools, artifact checks, analytics.
    "anton.analytics",
    "anton.core.artifacts.html_lint",
    "anton.core.artifacts.office_open_check",
    "anton.core.artifacts.ooxml_lint",
    "anton.core.artifacts.snapshot",
    "anton.core.artifacts.store",
    "anton.core.artifacts.xlsx_lint",
    "anton.core.artifacts.xlsx_office_check",
    "anton.core.memory.cortex",
    "anton.core.memory.hippocampus",
    "anton.tools",
    # The provider's first streamed request.
    "httpcore2",
    "openai.lib.streaming.chat",
    "openai.pagination",
    "openai.resources.chat.chat",
    "openai.resources.chat.completions.completions",
    "openai.resources.chat.completions.messages",
    "openai.types.chat.completions",
)

#: A client that connected but has not handed off its descriptors by then is dropped.
HANDOFF_TIMEOUT_SECONDS = 5.0
#: How long a SIGTERM waits for running turns; under the pod's 30 s grace period.
SHUTDOWN_GRACE_SECONDS = 25.0


def preload_turn_modules() -> None:
    for name in PRELOAD_MODULES:
        importlib.import_module(name)


def _prepare(preload: Callable[[], None]) -> bool:
    started = time.monotonic()
    try:
        forbid_same_user_access()
        preload()
    except Exception:
        logger.exception("resident process setup failed; every turn runs directly")
        return False
    # Frozen objects are never touched by the collector, so a fork's first
    # collection does not copy every inherited page.
    gc.freeze()
    logger.info("resident process ready in %.2fs", time.monotonic() - started)
    return True


def _exit_code(status: object) -> int:
    if status is None:
        return 0
    if isinstance(status, int):
        return status
    print(status, file=sys.stderr)
    return 1


def _finish_like_interpreter_exit(code: int) -> NoReturn:
    """End a fork the way a fresh interpreter ends, without unwinding the resident's stack.

    ``os._exit`` alone skips what a normal exit runs first: joining non-daemon
    threads, then the atexit handlers. The turn relies on both, the analytics
    flush that delivers ``turn_completed`` is an atexit handler.
    """
    try:
        threading._shutdown()
        atexit._run_exitfuncs()
    except BaseException:
        traceback.print_exc()
    logging.shutdown()
    for stream in (sys.stdout, sys.stderr):
        # A closed pipe has nowhere to report to; the interpreter ignores it at exit too.
        with contextlib.suppress(OSError, ValueError):
            stream.flush()
    os._exit(code)


class _Resident:
    def __init__(self, listener: socket.socket, entry: Callable[[], int | None], ready: bool) -> None:
        self._listener = listener
        self._entry = entry
        self._ready = ready
        self._selector = selectors.DefaultSelector()
        self._wakeup_read, self._wakeup_write = os.pipe()
        os.set_blocking(self._wakeup_read, False)
        os.set_blocking(self._wakeup_write, False)
        #: Connections told "ready" that have not handed off yet, with their deadline.
        self._pending: dict[socket.socket, float] = {}
        #: Running turns: forked pid -> the launcher waiting for its exit code.
        self._turns: dict[int, socket.socket] = {}
        self._stopping_since: float | None = None

    def run(self) -> int:
        signal.set_wakeup_fd(self._wakeup_write)
        signal.signal(signal.SIGCHLD, lambda *_: None)
        signal.signal(signal.SIGTERM, self._on_sigterm)
        self._listener.setblocking(False)
        self._selector.register(self._listener, selectors.EVENT_READ)
        self._selector.register(self._wakeup_read, selectors.EVENT_READ)
        while not self._should_exit():
            for key, _ in self._selector.select(self._select_timeout()):
                if key.fileobj is self._listener:
                    self._accept()
                elif key.fileobj == self._wakeup_read:
                    self._drain_wakeup()
                else:
                    self._receive(key.fileobj)
            self._reap()
            self._drop_expired()
        logger.info("resident process stopping")
        return 0

    def _on_sigterm(self, signum: int, frame: object) -> None:
        # Turns are not signalled: like the exec'd turns of a `sleep` pod, they
        # get the grace period to finish, then die with the pod.
        if self._stopping_since is None:
            self._stopping_since = time.monotonic()

    def _should_exit(self) -> bool:
        if self._stopping_since is None:
            return False
        return not self._turns or time.monotonic() - self._stopping_since > SHUTDOWN_GRACE_SECONDS

    def _select_timeout(self) -> float | None:
        deadlines = list(self._pending.values())
        if self._stopping_since is not None:
            deadlines.append(self._stopping_since + SHUTDOWN_GRACE_SECONDS)
        if not deadlines:
            return None
        return max(0.0, min(deadlines) - time.monotonic())

    def _drain_wakeup(self) -> None:
        try:
            while os.read(self._wakeup_read, 512):
                pass
        except BlockingIOError:
            return

    def _accept(self) -> None:
        while True:
            try:
                conn, _ = self._listener.accept()
            except BlockingIOError:
                return
            accepting = self._ready and self._stopping_since is None
            try:
                conn.sendall(READY if accepting else NOT_READY)
            except OSError:
                conn.close()
                continue
            if not accepting:
                conn.close()
                continue
            conn.setblocking(False)
            self._pending[conn] = time.monotonic() + HANDOFF_TIMEOUT_SECONDS
            self._selector.register(conn, selectors.EVENT_READ)

    def _forget(self, conn: socket.socket) -> None:
        self._pending.pop(conn, None)
        self._selector.unregister(conn)

    def _receive(self, conn: socket.socket) -> None:
        try:
            message, fds, _, _ = socket.recv_fds(conn, 16, 3)
        except BlockingIOError:
            return
        except OSError:
            logger.warning("launcher connection failed before the hand-off", exc_info=True)
            self._forget(conn)
            conn.close()
            return
        self._forget(conn)
        # Told "ready" before a SIGTERM: a turn forked now would die with the pod,
        # so closing fails it at once instead of after the grace period.
        if message != HANDOFF or len(fds) != 3 or self._stopping_since is not None:
            for fd in fds:
                os.close(fd)
            conn.close()
            return
        self._fork(conn, fds)

    def _fork(self, conn: socket.socket, fds: list[int]) -> None:
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            pid = os.fork()
        except OSError:
            # The descriptors are already here, so the launcher must not run the
            # turn itself; closing tells it the turn is lost.
            logger.exception("fork failed; the turn is lost")
            for fd in fds:
                os.close(fd)
            conn.close()
            return
        if pid == 0:
            self._run_turn(conn, fds)
        for fd in fds:
            os.close(fd)
        self._turns[pid] = conn

    def _run_turn(self, conn: socket.socket, fds: list[int]) -> NoReturn:
        try:
            self._become_turn_process(conn, fds)
            code = _exit_code(self._entry())
        except SystemExit as exc:
            code = _exit_code(exc.code)
        except BaseException:
            traceback.print_exc()
            code = 1
        _finish_like_interpreter_exit(code)

    def _become_turn_process(self, conn: socket.socket, fds: list[int]) -> None:
        """Leave the fork in the state a freshly exec'd interpreter starts in."""
        signal.set_wakeup_fd(-1)
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.pthread_sigmask(signal.SIG_SETMASK, set())
        self._selector.close()
        self._listener.close()
        os.close(self._wakeup_read)
        os.close(self._wakeup_write)
        for other in [conn, *self._pending, *self._turns.values()]:
            other.close()
        for target, fd in enumerate(fds):
            os.dup2(fd, target)
            os.close(fd)

    def _reap(self) -> None:
        # PID 1 also inherits every orphan in the pod, so wait on any child.
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return
            if pid == 0:
                return
            conn = self._turns.pop(pid, None)
            if conn is None:
                continue
            code = os.waitstatus_to_exitcode(status)
            if code < 0:
                code = 128 - code
            try:
                conn.sendall(f"{code}\n".encode())
            except OSError:
                logger.info("launcher of turn %d left before its exit code %d", pid, code)
            conn.close()

    def _drop_expired(self) -> None:
        now = time.monotonic()
        for conn, deadline in list(self._pending.items()):
            if now > deadline:
                self._forget(conn)
                conn.close()


def serve(
    entry: Callable[[], int | None],
    *,
    name: str = SOCKET_NAME,
    preload: Callable[[], None] = preload_turn_modules,
) -> int:
    """Hold the socket, preload, then fork ``entry`` for every launcher that hands off.

    The name is bound before the slow preload so that it is never free while
    this process lives: a launcher that connects meanwhile waits in the listen
    queue instead of finding the name unclaimed.
    """
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(socket_address(name))
    listener.listen(64)
    ready = _prepare(preload)
    return _Resident(listener, entry, ready).run()


def _run_cloud_turn() -> int | None:
    return importlib.import_module("anton.cloud_turn.__main__").main()


def main() -> int:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    # Its own handler rather than the root logger's: a fork must start with the
    # root unconfigured, as the turn configures it.
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return serve(_run_cloud_turn)


if __name__ == "__main__":
    sys.exit(main())
