"""Real-subprocess tests for the cloud-turn process boundary.

These run the actual entrypoint (via ``cloud_turn_fake_entry.py``, which calls
the real ``main()``) as a child process, so FD-level stdout isolation, fresh
process/scratchpad state, exit codes, and the deterministic E2E are proven end
to end — not simulated in-process.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_HARNESS = str(Path(__file__).parent / "cloud_turn_fake_entry.py")


def _run_cli(request, *, workspace, mode="model", script=None, timeout=60, env_extra=None):
    """Run the entrypoint as a subprocess; return (exit_code, events, stdout, stderr).

    ``events`` is the parsed JSONL from stdout — the assertion that stdout is a
    clean protocol stream is that every non-blank line is valid JSON."""
    import os

    env = os.environ.copy()
    env["ANTON_CLOUD_WORKSPACE_PATH"] = str(workspace)
    env["CLOUD_TURN_FAKE_MODE"] = mode
    if script is not None:
        env["CLOUD_TURN_FAKE_SCRIPT"] = json.dumps(script)
    if env_extra:
        env.update(env_extra)

    stdin = request if isinstance(request, str) else json.dumps(request)
    proc = subprocess.run(
        [sys.executable, _HARNESS],
        input=stdin.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=timeout,
    )
    lines = [ln for ln in proc.stdout.decode("utf-8").splitlines() if ln.strip()]
    events = [json.loads(ln) for ln in lines]  # raises if stdout isn't clean JSONL
    return proc.returncode, events, proc.stdout.decode(), proc.stderr.decode()


def _req(**over):
    body = {"run_id": "r", "attempt_id": "a", "conversation_id": "c", "input": "hi"}
    body.update(over)
    return body


# ── item 1: FD-level stdout isolation ────────────────────────────────────────

def test_stray_stdout_never_corrupts_protocol(tmp_path):
    """print(), sys.stdout.write, os.write(1, …) and library logging during the
    turn must all land on stderr, leaving stdout a clean protocol stream."""
    code, events, stdout, stderr = _run_cli(_req(), workspace=tmp_path, mode="stray")
    assert code == 0
    assert [e["kind"] for e in events] == ["turn.started", "turn.completed"]
    assert events[-1]["final_text"] == "clean answer"
    assert "STRAY" not in stdout               # nothing stray on the protocol channel
    assert "STRAY via os.write(1)" in stderr   # direct FD-1 write was redirected to stderr
    assert "STRAY via print()" in stderr


def test_stray_then_failure_stays_clean(tmp_path):
    code, events, stdout, stderr = _run_cli(_req(), workspace=tmp_path, mode="stray_fail")
    assert code == 3
    assert [e["kind"] for e in events] == ["turn.started", "turn.failed"]
    assert "STRAY" not in stdout
    # No traceback / internals leak onto the protocol event.
    assert "Traceback" not in stdout
    assert events[-1]["error"]["code"] == "internal_turn_failure"


def test_malformed_input_clean_protocol(tmp_path):
    code, events, stdout, stderr = _run_cli("{not valid json", workspace=tmp_path)
    assert code == 2
    assert len(events) == 1 and events[0]["kind"] == "turn.failed"
    assert events[0]["sequence"] == 1
    assert events[0]["error"]["code"] == "invalid_request"


# ── item 7: exit codes + terminal-event guarantees ───────────────────────────

def test_exit_zero_on_completion(tmp_path):
    code, events, *_ = _run_cli(
        _req(), workspace=tmp_path, mode="model", script=[{"text": "done"}]
    )
    assert code == 0
    terminals = [e for e in events if e["kind"] in ("turn.completed", "turn.failed")]
    assert len(terminals) == 1 and terminals[0]["kind"] == "turn.completed"


def test_exit_three_on_execution_failure(tmp_path):
    code, events, *_ = _run_cli(_req(), workspace=tmp_path, mode="stray_fail")
    assert code == 3
    terminals = [e for e in events if e["kind"] in ("turn.completed", "turn.failed")]
    assert len(terminals) == 1 and terminals[0]["kind"] == "turn.failed"


def test_oversized_request_rejected_at_process_level(tmp_path):
    from anton.cloud_turn.protocol import MAX_REQUEST_BYTES

    oversized = json.dumps(_req(input="x" * (MAX_REQUEST_BYTES + 10)))
    code, events, stdout, _ = _run_cli(oversized, workspace=tmp_path, timeout=60)
    assert code == 2
    assert len(events) == 1 and events[0]["kind"] == "turn.failed"
    assert events[0]["error"]["code"] == "invalid_request"


# ── item 3: deterministic local E2E (real CLI, fake model, no network) ────────

def test_e2e_text_only_turn(tmp_path):
    """Full path: JSON on stdin → parse → cloud-safe session → runner →
    turn.started + turn.completed with valid output_messages and final_text."""
    code, events, *_ = _run_cli(
        _req(input="What is 2 + 2?"),
        workspace=tmp_path, mode="model", script=[{"text": "The answer is 4."}],
    )
    assert code == 0
    assert [e["kind"] for e in events] == ["turn.started", "turn.completed"]
    completed = events[-1]
    assert completed["final_text"] == "The answer is 4."
    assert completed["output_messages"] == [
        {"role": "assistant", "content": "The answer is 4."}
    ]


# ── item 2: fresh process + scratchpad state per invocation ───────────────────

_CELL_A = (
    "E2E_SENTINEL = 4242\n"
    "import os as _os\n"
    "_os.environ['A_MUT'] = 'turnA'\n"
    "print('set', E2E_SENTINEL, _os.environ.get('A_MUT'))\n"
)
_CELL_B = (
    "import os as _os\n"
    "print('sentinel_present', 'E2E_SENTINEL' in dir())\n"
    "print('env_mut', _os.environ.get('A_MUT'))\n"
)


def _scratchpad_step(code):
    return {"tool": {"name": "scratchpad", "input": {
        "action": "exec", "name": "main", "code": code,
        "one_line_description": "test cell",
    }}}


@pytest.mark.slow
def test_fresh_scratchpad_state_across_processes(tmp_path):
    """Turn A defines a scratchpad variable + mutates its env; Turn B, a NEW
    process against the SAME workspace, must not observe either."""
    # Turn A — sets state, then finishes.
    codeA, eventsA, *_ = _run_cli(
        _req(run_id="A", input="set state"),
        workspace=tmp_path, mode="model", timeout=180,
        script=[_scratchpad_step(_CELL_A), {"text": "did A"}],
    )
    assert codeA == 0, eventsA
    # Sanity: Turn A's scratchpad actually ran and set the sentinel.
    a_tool_results = _tool_result_texts(eventsA[-1])
    assert any("set 4242 turnA" in t for t in a_tool_results)

    # Turn B — brand-new process, same workspace, tries to read Turn A's state.
    codeB, eventsB, *_ = _run_cli(
        _req(run_id="B", input="read state"),
        workspace=tmp_path, mode="model", timeout=180,
        script=[_scratchpad_step(_CELL_B), {"text": "did B"}],
    )
    assert codeB == 0, eventsB
    b_tool_results = _tool_result_texts(eventsB[-1])
    joined = "\n".join(b_tool_results)
    assert "sentinel_present False" in joined   # Python global gone
    assert "env_mut None" in joined             # env mutation gone


def _tool_result_texts(completed_event) -> list[str]:
    """Extract tool_result content strings from a turn.completed event."""
    out = []
    for msg in completed_event.get("output_messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    c = block.get("content")
                    out.append(c if isinstance(c, str) else json.dumps(c))
    return out


# ── item 2: session.close() terminates the inner scratchpad (no orphans) ──────

async def _real_cloud_session(tmp_path, monkeypatch):
    import anton.core.llm.client as llm_client_mod
    from unittest.mock import AsyncMock, MagicMock

    from anton.core.llm.provider import ProviderConnectionInfo
    from anton.cloud_turn.protocol import TurnRequestV1
    from anton.cloud_turn.session import build_cloud_chat_session

    monkeypatch.setenv("ANTON_CLOUD_WORKSPACE_PATH", str(tmp_path))

    def _mk(cls, settings):
        llm = AsyncMock()
        llm.coding_provider = MagicMock()
        llm.coding_provider.export_connection_info = MagicMock(
            return_value=ProviderConnectionInfo(provider="anthropic", api_key="test")
        )
        llm.coding_model = "m"
        llm.planning_provider = MagicMock()
        llm.planning_provider.native_web_tools = MagicMock(return_value=set())
        return llm

    monkeypatch.setattr(llm_client_mod.LLMClient, "from_settings", classmethod(_mk))
    req = TurnRequestV1(run_id="r", attempt_id="a", conversation_id="c", input="hi")
    return build_cloud_chat_session(req)


@pytest.mark.slow
async def test_session_close_terminates_scratchpad(tmp_path, monkeypatch):
    """A real scratchpad subprocess is started, then session.close() must kill it
    — no orphaned child survives the runner."""
    session = await _real_cloud_session(tmp_path, monkeypatch)
    pad = await session._scratchpads.get_or_create("main")
    await pad.execute("x = 1")
    proc = pad._proc
    assert proc is not None and proc.returncode is None  # alive

    await session.close()
    assert pad._proc is None                              # manager released it
    assert proc.returncode is not None                    # OS process terminated
