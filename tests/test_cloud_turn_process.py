"""Real-subprocess tests for the cloud-turn process boundary.

Runs the actual entrypoint (via cloud_turn_fake_entry.py, which calls the real
main()) as a child process, so FD-level stdout isolation, fresh
process/scratchpad state, and the deterministic E2E are proven end to end.
Event kinds match the controller contract: delta / turn_completed / turn_failed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
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
