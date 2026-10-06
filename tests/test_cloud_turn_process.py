"""Real-subprocess tests for the cloud-turn process boundary.

Runs the actual entrypoint (via cloud_turn_fake_entry.py, which calls the real
main()) as a child process, so FD-level stdout isolation, fresh
process/scratchpad state, and the deterministic E2E are proven end to end.
Event kinds match the controller contract: delta / turn_completed / turn_failed.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

_HARNESS = str(Path(__file__).parent / "cloud_turn_fake_entry.py")


def _run_cli(request, *, workspace, mode="model", script=None, timeout=60):
    """Run the entrypoint as a subprocess; return (exit_code, events, stdout, stderr).
    Parsing stdout as JSONL is itself the assertion that stdout is clean."""
    env = os.environ.copy()
    env["ANTON_CLOUD_WORKSPACE_PATH"] = str(workspace)
    env["CLOUD_TURN_FAKE_MODE"] = mode
    if script is not None:
        env["CLOUD_TURN_FAKE_SCRIPT"] = json.dumps(script)

    stdin = request if isinstance(request, str) else json.dumps(request)
    proc = subprocess.run(
        [sys.executable, _HARNESS],
        input=stdin.encode("utf-8"),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=timeout,
    )
    lines = [ln for ln in proc.stdout.decode("utf-8").splitlines() if ln.strip()]
    events = [json.loads(ln) for ln in lines]  # raises if stdout isn't clean JSONL
    return proc.returncode, events, proc.stdout.decode(), proc.stderr.decode()


def _req(**over):
    body = {"protocol_version": 1, "conversation_id": "c", "input": "hi"}
    body.update(over)
    return body


# ── FD-level stdout isolation ────────────────────────────────────────────────

def test_stray_stdout_never_corrupts_protocol(tmp_path):
    """print(), sys.stdout.write, os.write(1, …) and library logging during the
    turn must all land on stderr, leaving stdout a clean protocol stream."""
    _, events, stdout, stderr = _run_cli(_req(), workspace=tmp_path, mode="stray")
    assert [e["kind"] for e in events] == ["turn_completed"]
    assert "STRAY" not in stdout                # nothing stray on the protocol channel
    assert "STRAY via os.write(1)" in stderr    # direct FD-1 write redirected to stderr
    assert "STRAY via print()" in stderr


def test_stray_then_failure_stays_clean(tmp_path):
    _, events, stdout, _ = _run_cli(_req(), workspace=tmp_path, mode="stray_fail")
    assert [e["kind"] for e in events] == ["turn_failed"]
    assert "STRAY" not in stdout
    assert "Traceback" not in stdout            # no traceback leaks onto the wire
    assert "boom" in events[-1]["error"]


def test_malformed_input_clean_protocol(tmp_path):
    _, events, stdout, _ = _run_cli("{not valid json", workspace=tmp_path)
    assert len(events) == 1 and events[0]["kind"] == "turn_failed"
    assert events[0]["error"]


# ── deterministic E2E (real CLI, fake model, no network) ─────────────────────

def test_e2e_text_only_turn(tmp_path):
    """Full path: JSON on stdin -> parse -> cloud-safe session -> streaming
    delta events -> turn_completed."""
    _, events, *_ = _run_cli(
        _req(input="What is 2 + 2?"),
        workspace=tmp_path, mode="model", script=[{"text": "The answer is 4."}],
    )
    kinds = [e["kind"] for e in events]
    assert kinds[-1] == "turn_completed"
    text = "".join(e["text"] for e in events if e["kind"] == "delta")
    assert text == "The answer is 4."


# ── fresh process + scratchpad state per invocation ──────────────────────────
# Tool output is not on the wire (his contract), so we observe via workspace
# files: Turn A sets a variable + writes a file; Turn B (new process, same
# workspace) reports whether the variable survived. Files persist, runtime does not.

_CELL_A = (
    "X_SENTINEL = 4242\n"
    "open('a_done.txt', 'w').write('a-ok')\n"
    "print('set')\n"
)
_CELL_B = (
    "open('b_result.txt', 'w').write('has_x=' + str('X_SENTINEL' in dir()))\n"
    "print('done')\n"
)


def _scratchpad_step(code):
    return {"tool": {"name": "scratchpad", "input": {
        "action": "exec", "name": "main", "code": code,
        "one_line_description": "test cell",
    }}}


@pytest.mark.slow
def test_fresh_scratchpad_state_across_processes(tmp_path):
    codeA, eventsA, *_ = _run_cli(
        _req(conversation_id="A", input="set state"),
        workspace=tmp_path, mode="model", timeout=180,
        script=[_scratchpad_step(_CELL_A), {"text": "did A"}],
    )
    assert eventsA[-1]["kind"] == "turn_completed", eventsA
    # Turn A's scratchpad ran and its file persists in the workspace.
    assert (tmp_path / "a_done.txt").read_text() == "a-ok"

    codeB, eventsB, *_ = _run_cli(
        _req(conversation_id="B", input="read state"),
        workspace=tmp_path, mode="model", timeout=180,
        script=[_scratchpad_step(_CELL_B), {"text": "did B"}],
    )
    assert eventsB[-1]["kind"] == "turn_completed", eventsB
    # New process => fresh scratchpad namespace: Turn A's variable is gone.
    assert (tmp_path / "b_result.txt").read_text() == "has_x=False"


# ── session.close() terminates the inner scratchpad (no orphans) ─────────────

async def _real_cloud_session(tmp_path, monkeypatch):
    import anton.core.llm.client as llm_client_mod
    from unittest.mock import AsyncMock, MagicMock

    from anton.core.llm.provider import ProviderConnectionInfo
    from anton.cloud_turn.contract import TurnRequestV1
    from anton.cloud_turn.session import build_cloud_chat_session

    monkeypatch.setenv("ANTON_CLOUD_WORKSPACE_PATH", str(tmp_path))

    def _mk(cls, settings):
        llm = AsyncMock()
        llm.coding_provider = MagicMock()
        llm.coding_provider.export_connection_info = MagicMock(
            return_value=ProviderConnectionInfo(provider="anthropic", api_key="test"))
        llm.coding_model = "m"
        llm.planning_provider = MagicMock()
        llm.planning_provider.native_web_tools = MagicMock(return_value=set())
        return llm

    monkeypatch.setattr(llm_client_mod.LLMClient, "from_settings", classmethod(_mk))
    req = TurnRequestV1(protocol_version=1, conversation_id="c", input="hi")
    return build_cloud_chat_session(req)


@pytest.mark.slow
async def test_session_close_terminates_scratchpad(tmp_path, monkeypatch):
    session = await _real_cloud_session(tmp_path, monkeypatch)
    pad = await session._scratchpads.get_or_create("main")
    await pad.execute("x = 1")
    proc = pad._proc
    assert proc is not None and proc.returncode is None  # alive

    await session.close()
    assert pad._proc is None                              # manager released it
    assert proc.returncode is not None                    # OS process terminated


# ── interactive turn: ask_user answered over stdin ───────────────────────────


def _run_interactive_turn(tmp_path, *, close_stdin_before_wait):
    """Drive one interactive turn through the real entrypoint: answer the
    ask_user question with "pg", read to the terminal event, wait for exit.
    Returns (events, stderr, returncode)."""
    env = os.environ.copy()
    # `sys.executable <script>` puts the script's own dir first on sys.path, not
    # this repo, so an editable install pointing at a sibling checkout (a
    # multi-worktree dev setup) would otherwise shadow this worktree's `anton`.
    # Same fix as tests/e2e/harness.py's `_env()`: put the repo root first.
    env["PYTHONPATH"] = str(Path(__file__).parent.parent)
    env["ANTON_CLOUD_WORKSPACE_PATH"] = str(tmp_path)
    env["CLOUD_TURN_FAKE_MODE"] = "model"
    env["ANTON_CLOUD_ASK_USER_TIMEOUT_SECONDS"] = "30"
    env["CLOUD_TURN_FAKE_SCRIPT"] = json.dumps([
        {"tool": {"id": "t1", "name": "ask_user", "input": {
            "question": "Which database?",
            "options": [{"value": "pg"}, {"value": "my"}],
        }}},
        {"text": "Using pg."},
    ])
    proc = subprocess.Popen(
        [sys.executable, _HARNESS],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
    )
    watchdog = threading.Timer(120, proc.kill)
    watchdog.start()
    events = []
    try:
        proc.stdin.write((json.dumps(_req(interactive=True)) + "\n").encode())
        proc.stdin.flush()
        for raw in proc.stdout:
            event = json.loads(raw)
            events.append(event)
            if event["kind"] == "ask_user":
                answer = {"kind": "answer", "question_id": event["id"], "answer_id": "a1",
                          "values": ["pg"], "text": "", "skipped": False}
                proc.stdin.write((json.dumps(answer) + "\n").encode())
                proc.stdin.flush()
            if event["kind"] in ("turn_completed", "turn_failed"):
                break
        if close_stdin_before_wait:
            proc.stdin.close()
        proc.wait(timeout=60)
    finally:
        watchdog.cancel()
        with contextlib.suppress(Exception):
            proc.stdin.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
    return events, proc.stderr.read().decode(), proc.returncode


def test_interactive_turn_takes_an_answer_from_stdin(tmp_path):
    """Real ChatSession + real ask_user registration: the question goes out
    as an event, the answer comes back as a stdin line, the turn finishes."""
    events, stderr, returncode = _run_interactive_turn(tmp_path, close_stdin_before_wait=True)

    kinds = [e["kind"] for e in events]
    assert kinds[-1] == "turn_completed", (kinds, stderr[-2000:])
    assert returncode == 0, (returncode, stderr[-2000:])
    question = next(e for e in events if e["kind"] == "ask_user")
    assert [o["value"] for o in question["options"]] == ["pg", "my"]
    answered = next(e for e in events if e["kind"] == "ask_user_answered")
    assert answered["status"] == "answered"
    assert answered["values"] == ["pg"]
    assert answered["answer_id"] == "a1"
    # ask_user answers via elicit(), which bypasses ChatSession's generic
    # per-tool-call dispatch loop entirely — its result never reaches the
    # wire as a `tool_result` event (that kind is scratchpad-`dump`-only;
    # see cloud_turn/contract.py). The answer does reach the model, in the
    # pre-terminal `history` event the pod emits so cowork can replay this
    # turn's tool_use -> tool_result pair next turn.
    history_events = [e for e in events if e["kind"] == "history"]
    assert len(history_events) == 1, history_events
    history_event = history_events[0]
    ask_user_call_id = next(
        block["id"]
        for row in history_event["rows"]
        for block in row["content"]
        if block.get("type") == "tool_use" and block.get("name") == "ask_user"
    )
    tool_result_block = next(
        block
        for row in history_event["rows"]
        for block in row["content"]
        if block.get("type") == "tool_result" and block.get("tool_use_id") == ask_user_call_id
    )
    assert '"answered"' in tool_result_block["content"] and '"pg"' in tool_result_block["content"]




def test_interactive_turn_exits_cleanly_with_stdin_left_open(tmp_path):
    """The live-pod exec never closes stdin after the turn, so the process must
    exit cleanly with the write end still open.

    Before the fix, the daemon answer-reader thread blocked in ``readline()``
    on ``sys.stdin``'s own buffered reader, and interpreter shutdown aborted
    trying to finalize it while the thread held its lock: rc -6, "Fatal Python
    error: _enter_buffered_busy ... at interpreter shutdown, possibly due to
    daemon threads".
    """
    events, stderr, returncode = _run_interactive_turn(tmp_path, close_stdin_before_wait=False)

    kinds = [e["kind"] for e in events]
    assert kinds[-1] == "turn_completed", (kinds, stderr[-2000:])
    assert returncode == 0, (returncode, stderr[-2000:])
    assert "Fatal Python error" not in stderr
