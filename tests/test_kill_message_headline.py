"""A killed cell's error must name the kill wherever only one line of it is read.

The runtime's kill message leads with the cause ("Cell timed out after …",
"Cell killed after …") and appends recovery notes after it. Three readers keep
a single line of a cell error — the root-cause classifier behind
`tool_completed.root_cause_class`, the memory consolidator's cell summary, and
the notebook view — and they took the LAST line, which is right for a
traceback and wrong for a kill. That line used to be fixed MySQL/Snowflake
cancel advice, appended to every kill whatever the cell ran: a PDF read killed
by the watchdog was recorded as `unclassified` and turned into a Snowflake
lesson.

Every kill here comes from a real LocalScratchpadRuntime, never a hand-written
error string: the old tests fed the classifier a bare "Cell timed out after
300s total" and passed while every production kill was misclassified.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tests.conftest import make_mock_llm

from anton.core.backends.local import LocalScratchpadRuntime
from anton.core.llm.provider import LLMResponse, StreamComplete, ToolCall, Usage
from anton.core.memory.consolidator import Consolidator, _ConsolidatedLessons
from anton.core.session import ChatSession, ChatSessionConfig, _VerifierVerdict, _tool_failure_cause
from anton.core.utils.scratchpad import (
    KILL_ERROR_PREFIXES,
    cell_error_headline,
    cell_failure_reason,
)

_DEFAULTS = dict(
    coding_provider="anthropic",
    coding_model="",
    coding_api_key="",
    coding_base_url="",
)

_DB_ADVICE = ("snowflake", "processlist", "database query", "running queries")

# (id, heartbeat interval, cell code). A silent cell with heartbeats on runs
# into the 2s total budget; with heartbeats off it trips the 1s liveness
# window; one that printed first is killed with salvaged output, which appends
# a second paragraph — the case where first and last line differ.
_KILLS = [
    ("total-budget", "0.2", "import time; time.sleep(30)"),
    ("liveness", "0", "import time; time.sleep(30)"),
    ("salvaged", "0.2", "import time\nprint('sent 1/3')\ntime.sleep(0.5)\ntime.sleep(30)\n"),
]


def _shrink(monkeypatch, heartbeat: str) -> None:
    monkeypatch.setenv("ANTON_CELL_INACTIVITY_TIMEOUT", "1")
    monkeypatch.setenv("ANTON_CELL_INACTIVITY_MAX", "1")
    monkeypatch.setenv("ANTON_CELL_TIMEOUT_DEFAULT", "2")
    monkeypatch.setenv("ANTON_SCRATCHPAD_HEARTBEAT_INTERVAL", heartbeat)


async def _killed_cell(monkeypatch, heartbeat: str, code: str):
    _shrink(monkeypatch, heartbeat)
    pad = LocalScratchpadRuntime(name="kill-headline", **_DEFAULTS)
    await pad.start()
    try:
        cell = await pad.execute(code)
    finally:
        await pad.close()
    assert cell.error, "the cell was supposed to be killed"
    return cell


@pytest.mark.parametrize("_id, heartbeat, code", _KILLS, ids=[k[0] for k in _KILLS])
async def test_kill_message_carries_no_database_advice(monkeypatch, _id, heartbeat, code):
    cell = await _killed_cell(monkeypatch, heartbeat, code)
    low = cell.error.lower()
    for phrase in _DB_ADVICE:
        assert phrase not in low, f"{phrase!r} in a kill of a cell that ran no query"


@pytest.mark.parametrize("_id, heartbeat, code", _KILLS, ids=[k[0] for k in _KILLS])
async def test_kill_is_classified_as_a_timeout(monkeypatch, _id, heartbeat, code):
    cell = await _killed_cell(monkeypatch, heartbeat, code)
    reason = cell_failure_reason(cell.error)
    assert reason.startswith(("Cell timed out", "Cell killed")), reason
    assert _tool_failure_cause(False, reason) == ("transient", "timeout")


async def test_salvaged_kill_headline_is_the_cause_not_the_salvage_note(monkeypatch):
    _, heartbeat, code = _KILLS[2]
    cell = await _killed_cell(monkeypatch, heartbeat, code)
    assert "sent 1/3" in cell.stdout
    assert "\n" in cell.error.strip(), "salvage must append a second paragraph"
    assert cell_error_headline(cell.error).startswith("Cell timed out")
    assert "Partial output" not in cell_error_headline(cell.error)


async def test_cancelled_cell_leads_with_the_cancellation(monkeypatch):
    """A cancelled exec (the desktop Stop) raises a CancelledError with no
    message, so the kill message used to begin with a bare ". " — and, not
    starting with a kill prefix, its headline fell to the partial-output note."""
    monkeypatch.setenv("ANTON_SCRATCHPAD_HEARTBEAT_INTERVAL", "0.2")
    pad = LocalScratchpadRuntime(name="kill-cancel", **_DEFAULTS)
    await pad.start()

    async def run():
        async for _ in pad.execute_streaming(
            "import time\nprint('sent 1/3')\ntime.sleep(30)\n", estimated_seconds=60
        ):
            pass

    task = asyncio.create_task(run())
    try:
        await asyncio.sleep(2)  # past the first heartbeat, so output is salvaged
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        cell = pad.cells[-1]
    finally:
        await pad.close()

    assert "sent 1/3" in cell.stdout
    assert cell.error.startswith("Cancelled. ")
    assert cell.error.startswith(KILL_ERROR_PREFIXES)
    assert cell_error_headline(cell.error).startswith("Cancelled")


def test_traceback_headline_is_still_its_last_line():
    error = (
        "Hint: install the package first.\n"
        "Traceback (most recent call last):\n"
        '  File "<scratchpad>", line 1, in <module>\n'
        "ModuleNotFoundError: No module named 'openpyxl'"
    )
    assert cell_error_headline(error) == "ModuleNotFoundError: No module named 'openpyxl'"


async def test_consolidator_summary_names_the_kill(monkeypatch):
    """The consolidator reads one line per error cell and auto-encodes what it
    concludes as a durable project lesson, so that line must be the cause."""
    _, heartbeat, code = _KILLS[2]
    cell = await _killed_cell(monkeypatch, heartbeat, code)
    cell.stdout = ""  # a stdout preview would be shown instead of the error
    llm = AsyncMock()
    llm.generate_object_code = AsyncMock(return_value=_ConsolidatedLessons(items=[]))

    await Consolidator().replay_and_extract([cell, cell], llm)

    prompt = str(llm.generate_object_code.call_args)
    assert "ERROR: Cell timed out" in prompt
    assert "Partial output" not in prompt


async def test_notebook_view_names_the_kill(monkeypatch):
    _, heartbeat, code = _KILLS[2]
    _shrink(monkeypatch, heartbeat)
    pad = LocalScratchpadRuntime(name="kill-notebook", **_DEFAULTS)
    await pad.start()
    try:
        await pad.execute(code)
        notebook = pad.render_notebook()
    finally:
        await pad.close()
    error_lines = [ln for ln in notebook.splitlines() if ln.startswith("**Error:**")]
    assert error_lines and "Cell timed out" in error_lines[0]


def _exec_response(code: str) -> LLMResponse:
    return LLMResponse(
        content="running",
        tool_calls=[ToolCall(id="tc_kill", name="scratchpad",
                             input={"action": "exec", "name": "main", "code": code})],
        usage=Usage(input_tokens=1, output_tokens=1),
        stop_reason="tool_use",
    )


async def test_streaming_exec_reports_a_kill_as_timeout(tmp_path, monkeypatch):
    """The inline exec in `turn_stream` is the path desktop and web run; its
    tool_completed row is where root_cause_class reaches PostHog."""
    _, heartbeat, code = _KILLS[2]
    _shrink(monkeypatch, heartbeat)
    workspace = MagicMock(base=tmp_path)
    workspace.artifacts_dir = tmp_path / "artifacts"
    llm = make_mock_llm()
    llm.generate_object_code = AsyncMock(
        return_value=_VerifierVerdict(status="COMPLETE", reason="done")
    )
    responses = iter([_exec_response(code)])

    def plan_stream(**kwargs):
        async def gen():
            yield StreamComplete(response=next(responses, LLMResponse(
                content="done", tool_calls=[], usage=Usage(input_tokens=1, output_tokens=1),
                stop_reason="end_turn",
            )))
        return gen()

    llm.plan_stream = plan_stream
    session = ChatSession(ChatSessionConfig(llm_client=llm, workspace=workspace))
    pad = LocalScratchpadRuntime(name="main", **_DEFAULTS)
    await pad.start()
    session._scratchpads.get_or_create = AsyncMock(return_value=pad)
    try:
        with patch("anton.analytics.send_event") as sent:
            async for _ in session.turn_stream("run it"):
                pass
    finally:
        await pad.close()
        await session.close()

    rows = [c.kwargs for c in sent.call_args_list if c.args[1] == "tool_completed"]
    assert len(rows) == 1
    assert rows[0]["ok"] == "false"
    assert rows[0]["root_cause_class"] == "timeout"
