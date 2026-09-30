"""The resident process: turns forked from a preloaded process, reached through the launcher.

Every test runs a real resident (``cloud_turn_resident_entry.py``) under a socket
name of its own and drives it through the real launcher, the way the controller
execs it. The resident is not PID 1 here, so the launcher is told its pid.
"""

from __future__ import annotations

import errno
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from anton.cloud_turn import launcher

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="the resident process is Linux only")

_HARNESS = str(Path(__file__).parent / "cloud_turn_resident_entry.py")
# Accepted by the parser, rejected by the turn, so its error proves the line arrived.
_REJECTED_REQUEST = json.dumps({"protocol_version": 99, "conversation_id": "c", "input": "hi"})
_REQUEST_REACHED_THE_TURN = "unsupported cloud turn protocol version"


class _Resident:
    def __init__(self, tmp_path: Path, env: dict[str, str]) -> None:
        self.name = f"anton-test-{uuid.uuid4().hex}"
        self.env = env
        self.log = tmp_path / f"resident-{self.name}.log"
        with open(self.log, "wb") as log:
            self.proc = subprocess.Popen(
                [sys.executable, _HARNESS, self.name],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=env,
            )

    def wait_ready(self, timeout: float = 60) -> None:
        deadline = time.monotonic() + timeout
        while True:
            sock, reason = launcher.connect_to_resident(self.name, self.proc.pid, timeout)
            if sock is not None:
                sock.close()
                return
            # The interpreter is still starting and has not bound the name yet.
            assert reason.startswith("no resident process") and time.monotonic() < deadline, (
                f"{reason}\n{self.log.read_text()}"
            )
            time.sleep(0.05)

    def launch(self, stdin: str, *, wait: bool = True):
        args = [sys.executable, "-I", "-S", launcher.__file__,
                "--socket", self.name, "--resident-pid", str(self.proc.pid)]
        request = self.log.with_name(f"request-{uuid.uuid4().hex}")
        request.write_text(stdin + "\n")
        with open(request, "rb") as request_file:
            proc = subprocess.Popen(
                args, stdin=request_file, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self.env,
            )
        return _collect(proc) if wait else proc

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=30)


def _collect(proc: subprocess.Popen) -> tuple[int, list[dict], str]:
    out, err = proc.communicate(timeout=120)
    events = [json.loads(line) for line in out.decode().splitlines() if line.strip()]
    return proc.returncode, events, err.decode()


@pytest.fixture
def start_resident(tmp_path):
    started: list[_Resident] = []

    def start(entry: str = "turn", *, preload: str = "real", ready: bool = True, **extra_env: str) -> _Resident:
        env = os.environ.copy()
        env.update(
            ANTON_CLOUD_WORKSPACE_PATH=str(tmp_path),
            RESIDENT_TEST_ENTRY=entry,
            RESIDENT_TEST_PRELOAD=preload,
            **extra_env,
        )
        res = _Resident(tmp_path, env)
        started.append(res)
        if ready:
            res.wait_ready()
        return res

    yield start
    for res in started:
        res.stop()


def _probe(res: _Resident, **orders) -> tuple[int, dict, str]:
    code, events, stderr = res.launch(json.dumps(orders))
    assert events and events[0]["kind"] == "probe", stderr
    return code, events[0], stderr


def _direct_probe(env: dict[str, str]) -> dict:
    proc = subprocess.run(
        [sys.executable, _HARNESS, "--direct"], input=b"{}\n", capture_output=True, env=env, timeout=60,
    )
    return json.loads(proc.stdout.decode().splitlines()[0])


# ── the turn itself ──────────────────────────────────────────────────────────

def test_a_turn_through_the_resident_matches_a_direct_run(start_resident, tmp_path):
    res = start_resident()
    direct = subprocess.run(
        [sys.executable, "-m", "anton.cloud_turn"], input=(_REJECTED_REQUEST + "\n").encode(),
        capture_output=True, env=res.env, timeout=60,
    )
    direct_events = [json.loads(line) for line in direct.stdout.decode().splitlines() if line.strip()]

    code, events, stderr = res.launch(_REJECTED_REQUEST)

    assert (code, events) == (direct.returncode, direct_events)
    assert _REQUEST_REACHED_THE_TURN in events[-1]["error"]
    assert "running the turn directly" not in stderr


@pytest.mark.slow
def test_a_forked_turn_runs_a_real_scratchpad_cell(start_resident, tmp_path):
    cell = {"tool": {"name": "scratchpad", "input": {
        "action": "exec", "name": "main", "code": "open('cell_ran.txt', 'w').write('ok')\n",
        "one_line_description": "test cell",
    }}}
    res = start_resident(
        "fake", CLOUD_TURN_FAKE_MODE="model", CLOUD_TURN_FAKE_SCRIPT=json.dumps([cell, {"text": "done"}]),
    )

    code, events, stderr = res.launch(json.dumps({"protocol_version": 1, "conversation_id": "c", "input": "go"}))

    assert code == 0
    assert events[-1]["kind"] == "turn_completed", stderr
    assert (tmp_path / "cell_ran.txt").read_text() == "ok"


def test_stray_output_in_a_forked_turn_stays_off_the_protocol_stream(start_resident):
    res = start_resident("fake", CLOUD_TURN_FAKE_MODE="stray")

    _, events, stderr = res.launch(json.dumps({"protocol_version": 1, "conversation_id": "c", "input": "hi"}))

    assert [e["kind"] for e in events] == ["turn_completed"]
    assert "STRAY via os.write(1)" in stderr
    assert "STRAY via print()" in stderr


@pytest.mark.slow
def test_turns_import_nothing_the_resident_did_not_preload(start_resident, tmp_path):
    """A missing module still works, it is just imported per turn: keep the list complete."""
    report = tmp_path / "modules.txt"
    cell = {"tool": {"name": "scratchpad", "input": {
        "action": "exec", "name": "main", "code": "print(1)\n", "one_line_description": "test cell",
    }}}
    fake = start_resident(
        "fake", CLOUD_TURN_FAKE_MODE="model", CLOUD_TURN_FAKE_SCRIPT=json.dumps([cell, {"text": "done"}]),
        RESIDENT_TEST_MODULES_REPORT=str(report),
    )
    fake.launch(json.dumps({"protocol_version": 1, "conversation_id": "c", "input": "go"}))
    # The real provider path up to the network, which the fake model skips.
    minds = start_resident("turn", RESIDENT_TEST_MODULES_REPORT=str(report))
    minds.launch(json.dumps({
        "protocol_version": 1, "conversation_id": "c", "input": "hi",
        "llm": {"provider": "minds-cloud", "api_key": "mdb_k", "base_url": "http://127.0.0.1:9"},
    }))

    assert sorted(set(report.read_text().split())) == []


# ── the fork starts like a fresh interpreter ─────────────────────────────────

def test_a_fork_starts_like_a_freshly_exec_d_interpreter(start_resident):
    res = start_resident("probe")

    _, forked, _ = _probe(res)
    direct = _direct_probe(res.env)

    for key in ("env", "cwd", "sys_path", "flags", "threads", "fds"):
        assert forked[key] == direct[key], key
    assert forked["threads"] == 1
    assert forked["pid"] != res.proc.pid


def test_every_turn_runs_in_its_own_process(start_resident):
    res = start_resident("probe")

    _, first, _ = _probe(res)
    _, second, _ = _probe(res)

    assert first["pid"] != second["pid"]


def test_the_turn_exit_code_reaches_the_launcher(start_resident):
    res = start_resident("probe")

    code, _, _ = _probe(res, exit=5)

    assert code == 5


def test_atexit_handlers_registered_by_the_turn_run(start_resident, tmp_path):
    res = start_resident("probe")
    marker = tmp_path / "atexit.txt"

    _probe(res, atexit_file=str(marker))

    assert marker.read_text() == "ran"


def test_the_fork_waits_for_the_turns_threads_like_an_interpreter_exit(start_resident, tmp_path):
    res = start_resident("probe")
    marker = tmp_path / "thread.txt"

    _probe(res, thread_file=str(marker))

    assert marker.read_text() == "ran"


def test_the_turns_environment_is_closed_to_other_processes(start_resident):
    res = start_resident("probe")

    _, forked, _ = _probe(res)

    assert forked["dumpable"] == 0
    with pytest.raises(PermissionError):
        Path(f"/proc/{res.proc.pid}/environ").read_bytes()


# ── concurrency ──────────────────────────────────────────────────────────────

def test_overlapping_turns_run_side_by_side(start_resident):
    res = start_resident("probe")

    started = time.monotonic()
    first = res.launch(json.dumps({"sleep": 1.5}), wait=False)
    second = res.launch(json.dumps({"sleep": 1.5}), wait=False)
    results = [_collect(first), _collect(second)]
    elapsed = time.monotonic() - started

    assert [code for code, _, _ in results] == [0, 0]
    assert results[0][1][0]["pid"] != results[1][1][0]["pid"]
    assert elapsed < 2.9


def test_a_client_that_never_hands_off_does_not_block_other_turns(start_resident):
    res = start_resident("probe")
    stuck = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stuck.connect(launcher.socket_address(res.name))
    assert stuck.recv(1) == launcher.READY

    started = time.monotonic()
    code, _, _ = _probe(res)

    assert code == 0
    assert time.monotonic() - started < 3
    stuck.close()


# ── not ready ────────────────────────────────────────────────────────────────

def test_a_launcher_that_gives_up_during_the_preload_never_gets_a_fork(start_resident, tmp_path):
    marker = tmp_path / "forked"
    res = start_resident("probe", preload="slow:1.5", ready=False, RESIDENT_TEST_MARKER=str(marker))
    time.sleep(0.3)

    sock, reason = launcher.connect_to_resident(res.name, res.proc.pid, timeout=0.3)
    res.wait_ready()
    time.sleep(0.5)

    assert sock is None
    assert reason.startswith("resident process not ready after")
    assert not marker.exists()


def test_a_failed_preload_keeps_the_name_and_sends_turns_the_direct_way(start_resident):
    res = start_resident("turn", preload="fail", ready=False)
    time.sleep(1)

    code, events, stderr = res.launch(_REJECTED_REQUEST)

    assert "running the turn directly: resident process is not ready" in stderr
    assert _REQUEST_REACHED_THE_TURN in events[-1]["error"]
    assert res.proc.poll() is None
    squatter = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    with pytest.raises(OSError) as exc:
        squatter.bind(launcher.socket_address(res.name))
    assert exc.value.errno == errno.EADDRINUSE
    squatter.close()


# ── shutdown ─────────────────────────────────────────────────────────────────

def test_sigterm_lets_the_running_turn_finish_then_exits(start_resident):
    res = start_resident("probe")
    turn = res.launch(json.dumps({"sleep": 1.5}), wait=False)
    time.sleep(0.5)

    res.proc.send_signal(signal.SIGTERM)
    code, events, _ = _collect(turn)

    assert code == 0
    assert events[0]["kind"] == "probe"
    assert res.proc.wait(timeout=10) == 0


def test_after_sigterm_new_turns_go_the_direct_way(start_resident):
    res = start_resident("turn")
    busy = start_resident("probe")
    turn = busy.launch(json.dumps({"sleep": 2}), wait=False)
    time.sleep(0.5)
    busy.proc.send_signal(signal.SIGTERM)
    time.sleep(0.3)

    sock, reason = launcher.connect_to_resident(busy.name, busy.proc.pid, timeout=5)

    assert sock is None
    assert reason == "resident process is not ready"
    _collect(turn)
    res.stop()
