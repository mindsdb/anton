"""A cancel that lands while a scratchpad cell runs.

A host's Stop, its idle watchdog and its shutdown all cancel the task that runs
the turn. When that cancel lands in a running cell, the runtime kills the
cell's process tree, records the cell with its salvaged output, and lets the
cancel go on up, so the turn ends instead of reading an error cell and making
its next model call. A cell that runs past its own time budget still yields its
error cell, and the turn goes on. Neither cooperative cancel cancels a task: a
host that sets _cancel_event and keeps draining makes the turn loop call
pad.cancel(), and anton's CLI raises KeyboardInterrupt in its consumer on
Escape and closes the pads. During the CLI's first-run demo cell, Ctrl+C
reports a killed cell and starts chat. Earlier demo cancellation still exits.

Timing knobs are shrunk through ANTON_* env vars before pad.start(), as in
test_scratchpad_watchdog_contracts.py.
"""
from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from anton.channel.theme import build_rich_theme
from anton.chat import _run_demo_cell
from anton.chat_session import get_runtime_factory
from anton.core.backends.base import Cell
from anton.core.backends.local import LocalScratchpadRuntime
from anton.core.llm.provider import LLMResponse, StreamComplete, ToolCall, Usage
from anton.core.session import ChatSession, ChatSessionConfig, _VerifierVerdict
from tests.conftest import make_mock_llm

_DEFAULTS = dict(
    coding_provider="anthropic",
    coding_model="",
    coding_api_key="",
    coding_base_url="",
)

_LONG_CELL = "import time\nprint('before-cancel')\ntime.sleep(30)\n"


def _short_timers(*, monkeypatch, total_seconds: str = "15") -> None:
    """Sub-second heartbeats, a 1 s silence window and a short total budget."""
    monkeypatch.setenv("ANTON_CELL_INACTIVITY_TIMEOUT", "1")
    monkeypatch.setenv("ANTON_CELL_INACTIVITY_MAX", "1")
    monkeypatch.setenv("ANTON_CELL_INACTIVITY_AFTER_PROGRESS", "1")
    monkeypatch.setenv("ANTON_CELL_TIMEOUT_DEFAULT", total_seconds)
    monkeypatch.setenv("ANTON_SCRATCHPAD_HEARTBEAT_INTERVAL", "0.2")


async def _wait_for_salvaged_stdout(*, pad: LocalScratchpadRuntime, text: str) -> None:
    """Wait until the runtime has received ``text`` from the running cell."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 10
    while text not in "".join(getattr(pad, "_salvage", [])):
        assert loop.time() < deadline, f"{text!r} never reached the runtime"
        await asyncio.sleep(0.05)


@pytest.mark.parametrize("consumer", ["execute_streaming", "execute"])
async def test_a_cancel_kills_the_cell_and_reaches_the_caller(monkeypatch, consumer):
    # Both ways a caller reads a cell: draining the stream, as the turn loop
    # does, and execute(), which stops at the first Cell.
    _short_timers(monkeypatch=monkeypatch)
    pad = LocalScratchpadRuntime(name="cancel-in-cell", **_DEFAULTS)
    await pad.start()
    caller_went_on = False

    async def caller():
        nonlocal caller_went_on
        if consumer == "execute_streaming":
            async for _item in pad.execute_streaming(_LONG_CELL):
                pass
        else:
            await pad.execute(_LONG_CELL)
        caller_went_on = True

    try:
        task = asyncio.create_task(caller())
        await _wait_for_salvaged_stdout(pad=pad, text="before-cancel")
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=10)

        assert task in done, "the cancel did not reach the caller"
        assert task.cancelled()
        assert not caller_went_on
        assert pad._proc.returncode is not None
        assert pad.cells[-1].error.startswith("Cancelled")
        assert "before-cancel" in pad.cells[-1].stdout
    finally:
        await pad.close()


async def test_a_cell_past_its_budget_yields_its_error_and_the_turn_goes_on(monkeypatch):
    _short_timers(monkeypatch=monkeypatch, total_seconds="2")
    pad = LocalScratchpadRuntime(name="budget-in-cell", **_DEFAULTS)
    await pad.start()
    try:
        items = [item async for item in pad.execute_streaming("import time; time.sleep(30)")]
        next_cell = await pad.execute("print('the turn goes on')")
    finally:
        await pad.close()

    cells = [item for item in items if isinstance(item, Cell)]
    assert len(cells) == 1
    assert "timed out" in cells[0].error.lower()
    assert next_cell.error is None, next_cell.error
    assert next_cell.stdout.strip() == "the turn goes on"


async def test_a_cancelled_error_not_aimed_at_the_task_yields_the_cell(monkeypatch):
    # Only a cancel of the task running the cell ends the turn. A
    # CancelledError that leaks up from something the cell awaited, with no
    # cancel() on this task, is a kill like any other.
    _short_timers(monkeypatch=monkeypatch)
    pad = LocalScratchpadRuntime(name="stray-cancel", **_DEFAULTS)
    await pad.start()

    async def stray_cancel(**_kwargs):
        raise asyncio.CancelledError
        yield  # an async generator, like the _read_result it replaces

    monkeypatch.setattr(pad, "_read_result", stray_cancel)
    try:
        items = [item async for item in pad.execute_streaming("print('never read')")]
    finally:
        await pad.close()

    assert len(items) == 1
    assert items[0].error.startswith("Cancelled")


async def test_a_cooperative_cancel_restarts_the_pad_and_the_next_cell_runs(monkeypatch):
    # A host that sets _cancel_event and keeps draining makes the turn loop
    # call pad.cancel() when the cell's next item arrives, then stop reading.
    # No task is cancelled, so the pad restarts and the next cell runs.
    _short_timers(monkeypatch=monkeypatch)
    pad = LocalScratchpadRuntime(name="cli-cancel", **_DEFAULTS)
    await pad.start()
    code = (
        "import time\n"
        "for i in range(100):\n"
        "    progress(f'step {i}')\n"
        "    time.sleep(0.2)\n"
    )
    try:
        first_process = pad._proc
        escape_pressed = False
        async for item in pad.execute_streaming(code):
            if escape_pressed:
                await pad.cancel()
                break
            escape_pressed = isinstance(item, str)

        assert first_process.returncode is not None
        assert pad.cells[-1].error == "Cancelled by user."
        cell = await pad.execute("print('after cancel')")
        assert cell.error is None, cell.error
        assert cell.stdout.strip() == "after cancel"
    finally:
        await pad.close()


async def test_a_ctrl_c_during_the_first_run_demo_cell_ends_only_the_demo(monkeypatch):
    # Ctrl+C cancels the CLI's main task. During the first-run demo's cell the
    # demo takes that cancel back and returns the killed cell, so the CLI
    # reports a failed demo, saves first_run_done and starts the chat.
    _short_timers(monkeypatch=monkeypatch)
    pad = LocalScratchpadRuntime(name="first-run-demo", **_DEFAULTS)
    await pad.start()
    console = Console(file=io.StringIO(), theme=build_rich_theme("dark"))
    try:
        task = asyncio.create_task(_run_demo_cell(console=console, pad=pad, code=_LONG_CELL))
        await _wait_for_salvaged_stdout(pad=pad, text="before-cancel")
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=10)
    finally:
        await pad.close()

    assert task in done, "the cancel did not end the demo's cell"
    assert not task.cancelled()
    assert task.cancelling() == 0
    cell = task.result()
    assert cell is pad.cells[-1]
    assert cell.error.startswith("Cancelled")
    assert "before-cancel" in cell.stdout


def _exec_response(*, code: str) -> LLMResponse:
    return LLMResponse(
        content="running",
        tool_calls=[ToolCall(id="tc_cell", name="scratchpad",
                             input={"action": "exec", "name": "main", "code": code})],
        usage=Usage(input_tokens=1, output_tokens=1),
        stop_reason="tool_use",
    )


@pytest.mark.parametrize("cli", [False, True], ids=["host-stop", "cli-ctrl-c"])
async def test_a_cancel_during_a_cell_obeys_the_hosts_turn_policy(
    tmp_path, monkeypatch, cli
):
    # cowork-server's Stop and its idle watchdog call task.cancel() on the task
    # that drains turn_stream. The model asks for one long cell; once the cell
    # runs, the cancel ends a host's turn. The CLI kills only the cell and
    # reaches the next model call, preserving its existing Ctrl+C behavior.
    _short_timers(monkeypatch=monkeypatch)
    workspace = MagicMock(base=tmp_path)
    workspace.artifacts_dir = tmp_path / "artifacts"
    llm = make_mock_llm()
    llm.generate_object_code = AsyncMock(
        return_value=_VerifierVerdict(status="COMPLETE", reason="done")
    )
    model_calls = 0

    def plan_stream(**_kwargs):
        nonlocal model_calls
        model_calls += 1
        response = _exec_response(code=_LONG_CELL) if model_calls == 1 else LLMResponse(
            content="done", tool_calls=[], usage=Usage(input_tokens=1, output_tokens=1),
            stop_reason="end_turn",
        )

        async def gen():
            yield StreamComplete(response=response)

        return gen()

    llm.plan_stream = plan_stream
    session = ChatSession(ChatSessionConfig(llm_client=llm, workspace=workspace))
    factory = get_runtime_factory(
        SimpleNamespace(backend="local"), **({"cancel_ends_turn": False} if cli else {})
    )
    pad = factory(name="main", cells=None, workspace_path=tmp_path, **_DEFAULTS)
    await pad.start()
    session._scratchpads.get_or_create = AsyncMock(return_value=pad)

    async def turn():
        async for _event in session.turn_stream("run the long cell"):
            pass

    try:
        with patch("anton.analytics.send_event"):
            task = asyncio.create_task(turn())
            await _wait_for_salvaged_stdout(pad=pad, text="before-cancel")
            task.cancel()
            done, _ = await asyncio.wait({task}, timeout=10)
    finally:
        await pad.close()
        await session.close()

    assert task in done, "the cancel did not end the turn"
    assert task.cancelled() is not cli
    assert model_calls == (2 if cli else 1)
    assert pad.cells[-1].error.startswith("Cancelled")
