"""The cloud turn launcher: hand-off to the resident process, or a direct run.

The launcher runs as a real subprocess, the way the controller execs it, so the
descriptors it passes and the stdin it must leave unread are the real ones. A
thread in the test process stands in for the resident.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import uuid

import pytest

from anton.cloud_turn import launcher

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="abstract unix sockets are Linux only")

# Accepted by the parser, rejected by the turn, so its error proves the line arrived.
_REQUEST = json.dumps({"protocol_version": 99, "conversation_id": "c", "input": "hi"}) + "\n"
_REQUEST_REACHED_THE_TURN = "unsupported cloud turn protocol version"


def _socket_name() -> str:
    return f"anton-test-{uuid.uuid4().hex}"


def _run(args: list[str], workspace, stdin: str = _REQUEST) -> tuple[int, list[dict], str]:
    env = os.environ.copy()
    env["ANTON_CLOUD_WORKSPACE_PATH"] = str(workspace)
    proc = subprocess.run(
        [sys.executable, *args], input=stdin.encode(), capture_output=True, env=env, timeout=60,
    )
    events = [json.loads(line) for line in proc.stdout.decode().splitlines() if line.strip()]
    return proc.returncode, events, proc.stderr.decode()


def _run_launcher(workspace, name: str, resident_pid: int = 1) -> tuple[int, list[dict], str]:
    return _run(
        ["-I", "-S", launcher.__file__, "--socket", name, "--resident-pid", str(resident_pid)], workspace,
    )


class _FakeResident:
    """Listens on the launcher's socket and plays one side of the hand-off."""

    def __init__(self, name: str) -> None:
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(launcher.socket_address(name))
        self.server.listen(1)
        self.received_fds: list[int] | None = None
        self._thread: threading.Thread | None = None

    def start(self, play) -> None:
        def run() -> None:
            conn, _ = self.server.accept()
            with conn:
                play(conn)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def receive(self, conn: socket.socket) -> list[int]:
        _, fds, _, _ = socket.recv_fds(conn, 16, 3)
        self.received_fds = fds
        return fds

    def close(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout=10)
        self.server.close()


@pytest.fixture
def resident():
    name = _socket_name()
    fake = _FakeResident(name)
    yield name, fake
    fake.close()


def test_without_a_resident_the_turn_runs_directly_like_the_plain_entrypoint(tmp_path):
    direct_code, direct_events, _ = _run(["-m", "anton.cloud_turn"], tmp_path)

    code, events, stderr = _run_launcher(tmp_path, _socket_name())

    assert (code, events) == (direct_code, direct_events)
    assert events[-1]["kind"] == "turn_failed"
    assert _REQUEST_REACHED_THE_TURN in events[-1]["error"]
    assert "running the turn directly: no resident process" in stderr


def test_a_ready_resident_gets_the_streams_and_its_exit_code_is_returned(tmp_path, resident):
    name, fake = resident

    def play(conn):
        conn.sendall(launcher.READY)
        stdin_fd, stdout_fd, stderr_fd = fake.receive(conn)
        request = os.read(stdin_fd, 4096).decode()
        os.write(stdout_fd, (json.dumps({"kind": "turn_completed", "echo": request}) + "\n").encode())
        os.write(stderr_fd, b"resident stderr\n")
        for fd in (stdin_fd, stdout_fd, stderr_fd):
            os.close(fd)
        conn.sendall(b"3\n")

    fake.start(play)
    code, events, stderr = _run_launcher(tmp_path, name, resident_pid=os.getpid())

    assert code == 3
    assert events == [{"kind": "turn_completed", "echo": _REQUEST}]
    assert "resident stderr" in stderr
    assert "running the turn directly" not in stderr


def test_the_launcher_is_closed_to_other_processes_while_it_holds_the_streams(tmp_path, resident):
    """It holds the exec stdout for the whole turn: same-uid cell code must not
    open it through /proc and write events the controller would trust."""
    name, fake = resident
    opened: dict[str, object] = {}

    def play(conn):
        pid, _, _ = launcher._PEERCRED.unpack(
            conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, launcher._PEERCRED.size)
        )
        try:
            os.close(os.open(f"/proc/{pid}/fd/1", os.O_WRONLY))
            opened["stdout"] = "opened"
        except PermissionError:
            opened["stdout"] = "denied"
        conn.sendall(launcher.NOT_READY)
        fake.receive(conn)

    fake.start(play)
    _run_launcher(tmp_path, name, resident_pid=os.getpid())
    fake.close()

    assert opened["stdout"] == "denied"


def test_a_resident_that_is_not_ready_gets_no_streams(tmp_path, resident):
    name, fake = resident

    def play(conn):
        conn.sendall(launcher.NOT_READY)
        fake.receive(conn)

    fake.start(play)
    _, events, stderr = _run_launcher(tmp_path, name, resident_pid=os.getpid())
    fake.close()

    assert fake.received_fds == []
    assert _REQUEST_REACHED_THE_TURN in events[-1]["error"]
    assert "running the turn directly: resident process is not ready" in stderr


def test_a_socket_held_by_another_process_gets_no_streams(tmp_path, resident):
    name, fake = resident
    fake.start(fake.receive)

    # The holder is this test process, and the launcher expects the resident as PID 1.
    _, events, stderr = _run_launcher(tmp_path, name)
    fake.close()

    assert fake.received_fds == []
    assert _REQUEST_REACHED_THE_TURN in events[-1]["error"]
    assert f"socket is held by pid {os.getpid()}" in stderr


def test_the_resident_dying_mid_turn_is_reported_without_a_second_run(tmp_path, resident):
    name, fake = resident

    def play(conn):
        conn.sendall(launcher.READY)
        for fd in fake.receive(conn):
            os.close(fd)

    fake.start(play)
    code, events, stderr = _run_launcher(tmp_path, name, resident_pid=os.getpid())

    assert code == 1
    assert events == []
    assert "resident process closed the connection mid-turn" in stderr
    assert "running the turn directly" not in stderr


def test_a_resident_that_never_answers_times_out(resident):
    name, fake = resident
    answered = threading.Event()
    fake.start(lambda conn: answered.wait(timeout=10))

    sock, reason = launcher.connect_to_resident(name, os.getpid(), timeout=0.2)
    answered.set()

    assert sock is None
    assert reason == "resident process not ready after 0s"


def test_a_full_listen_queue_does_not_hang_the_connect():
    """A resident that stopped accepting fills its queue; the launcher must still fall back."""
    name = _socket_name()
    stuck = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stuck.bind(launcher.socket_address(name))
    stuck.listen(0)
    queued = []
    while True:
        waiting = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        waiting.setblocking(False)
        try:
            waiting.connect(launcher.socket_address(name))
        except BlockingIOError:
            waiting.close()
            break
        queued.append(waiting)
    result: list[tuple[socket.socket | None, str]] = []
    connecting = threading.Thread(
        target=lambda: result.append(launcher.connect_to_resident(name, os.getpid(), timeout=0.5)),
        daemon=True,
    )

    connecting.start()
    connecting.join(timeout=5)

    assert not connecting.is_alive(), "connect() blocked past the launcher's timeout"
    assert result[0] == (None, "resident process is not accepting connections")
    for sock in [*queued, stuck]:
        sock.close()


def test_a_resident_that_closes_before_answering_is_not_ready(resident):
    name, fake = resident
    fake.start(lambda conn: None)

    sock, reason = launcher.connect_to_resident(name, os.getpid(), timeout=5)

    assert sock is None
    assert reason == "resident process closed the connection"
