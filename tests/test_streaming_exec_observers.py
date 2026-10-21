"""The scratchpad observers (the cerebellum) see every cell on the streaming path.

`ChatSession._stream_and_handle_tools` runs a scratchpad exec inline, through
`pad.execute_streaming`, and never calls `handle_scratchpad`. Every host that
drives `turn_stream` (the CLI, cowork-server, the cloud_turn pod) runs its cells
there. These tests drive `turn_stream` with a fake LLM and a fake pad, and check
that the observers fire around the cell the way `handle_scratchpad` fires them,
that the cerebellum's end-of-turn flush learns from a failed cell, that copilot
holds the lesson for the person to confirm, and that a cancelled turn's cells
are learned from by no flush.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import anton.core.session as session_mod
from anton.core.backends.base import Cell
from anton.core.llm.provider import (
    LLMResponse,
    StreamComplete,
    StreamToolResult,
    ToolCall,
    Usage,
)
from anton.core.memory.cerebellum import _DiffPassResult, _LessonDraft
from anton.core.memory.cortex import Cortex
from anton.core.memory.hippocampus import Hippocampus
from anton.core.session import ChatSession, ChatSessionConfig, _VerifierVerdict
from tests.conftest import make_mock_llm

_CODE = "import pyodbc"
_DESCRIPTION = "load the ODBC driver"
_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "<cell>", line 1, in <module>\n'
    "ModuleNotFoundError: No module named 'pyodbc'"
)
_LESSON = "Check that pyodbc is installed before importing it."


def _failed_cell() -> Cell:
    return Cell(
        code=_CODE, stdout="", stderr="", error=_TRACEBACK, description=_DESCRIPTION,
    )


class _FakePad:
    """A scratchpad runtime with no subprocess: each `execute_streaming` call
    yields one line of output, then the cell it was given.

    `on_exec` runs at the start of each call with the call's 1-based number, so
    a test can set the cancel event, or raise, on a chosen cell.
    """

    def __init__(
        self,
        *,
        cell: Cell,
        events: list[str],
        on_exec: Callable[[int], None] | None = None,
    ) -> None:
        self._cell = cell
        self._events = events
        self._on_exec = on_exec
        self.calls = 0
        self.cancelled = False

    async def execute_streaming(self, code, **kwargs):
        self.calls += 1
        self._events.append("exec")
        if self._on_exec is not None:
            self._on_exec(self.calls)
        yield "connecting"
        yield self._cell

    async def cancel(self) -> None:
        self.cancelled = True


class _RecordingObserver:
    """Records the cells each hook sees; raises `raises` from both hooks when set."""

    def __init__(self, *, events: list[str], raises: Exception | None = None) -> None:
        self.pre_calls: list[Cell] = []
        self.post_calls: list[Cell] = []
        self._events = events
        self._raises = raises

    async def on_pre_execute(self, cell: Cell) -> None:
        self.pre_calls.append(cell)
        self._events.append("pre")
        if self._raises is not None:
            raise self._raises

    async def on_post_execute(self, cell: Cell) -> None:
        self.post_calls.append(cell)
        self._events.append("post")
        if self._raises is not None:
            raise self._raises


def _raise_on(call: int, exc: BaseException) -> Callable[[int], None]:
    """An `on_exec` that raises `exc` from the given exec call."""

    def on_exec(number: int) -> None:
        if number == call:
            raise exc

    return on_exec


def _llm(*, execs: int = 1) -> AsyncMock:
    """A model that runs `execs` scratchpad execs in its first round, then
    answers every later round.

    `generate_object_code` answers by schema: the cerebellum's diff pass gets
    one lesson, every other structured call (the verifier) gets COMPLETE.
    """
    llm = make_mock_llm()

    async def generate_object_code(schema_class, **kwargs):
        if schema_class is _DiffPassResult:
            return _DiffPassResult(lessons=[_LessonDraft(text=_LESSON, topic="scratchpad")])
        return _VerifierVerdict(status="COMPLETE", reason="done")

    llm.generate_object_code = AsyncMock(side_effect=generate_object_code)
    rounds = 0

    def plan_stream(**kwargs):
        nonlocal rounds
        rounds += 1
        if rounds == 1:
            response = LLMResponse(
                content="Loading the driver.",
                tool_calls=[
                    ToolCall(
                        id=f"tc_exec_{number}", name="scratchpad",
                        input={
                            "action": "exec",
                            "name": "main",
                            "code": _CODE,
                            "one_line_description": _DESCRIPTION,
                            "estimated_execution_time_seconds": 5,
                        },
                    )
                    for number in range(1, execs + 1)
                ],
                usage=Usage(input_tokens=10, output_tokens=20),
                stop_reason="tool_use",
            )
        else:
            response = LLMResponse(
                content="Done.", tool_calls=[],
                usage=Usage(input_tokens=10, output_tokens=20),
                stop_reason="end_turn",
            )

        async def gen():
            yield StreamComplete(response=response)

        return gen()

    llm.plan_stream = plan_stream
    return llm


def _session(
    tmp_path: Path,
    *,
    llm: AsyncMock,
    memory_mode: str = "autopilot",
    background_memory: bool = True,
) -> ChatSession:
    """A session built the way the desktop host builds one: a real Cortex over
    real directories, so the cerebellum's lessons land on disk."""
    workspace = MagicMock(base=tmp_path)
    workspace.artifacts_dir = tmp_path / "artifacts"
    global_dir = tmp_path / "memory" / "global"
    project_dir = tmp_path / "memory" / "project"
    global_dir.mkdir(parents=True)
    project_dir.mkdir(parents=True)
    cortex = Cortex(
        global_hc=Hippocampus(global_dir),
        project_hc=Hippocampus(project_dir),
        mode=memory_mode,
        llm_client=llm,
    )
    return ChatSession(ChatSessionConfig(
        llm_client=llm,
        workspace=workspace,
        cortex=cortex,
        background_memory=background_memory,
    ))


def _track_background(session: ChatSession) -> list[asyncio.Task]:
    """Record every background task the session starts from now on."""
    background: list[asyncio.Task] = []
    spawn = session._spawn_background

    def spawn_and_keep(coro, trace):
        task = spawn(coro, trace)
        background.append(task)
        return task

    session._spawn_background = spawn_and_keep
    return background


async def _run_turn(session: ChatSession, pad: _FakePad) -> None:
    """Drive one `turn_stream` turn, then wait for the background work it started."""
    session._scratchpads.get_or_create = AsyncMock(return_value=pad)
    background = _track_background(session)
    try:
        async for _ in session.turn_stream("load the driver"):
            pass
        await asyncio.gather(*background)
    finally:
        await session.close()


def _diff_calls(llm: AsyncMock) -> list:
    return [
        call for call in llm.generate_object_code.await_args_list
        if call.args[0] is _DiffPassResult
    ]


async def test_a_failed_streaming_cell_reaches_the_observers_in_order(tmp_path, monkeypatch):
    events: list[str] = []
    observe_scratchpad_cell = session_mod.observe_scratchpad_cell

    def observe_and_record(session, name, cell):
        events.append("observe")
        observe_scratchpad_cell(session, name, cell)

    monkeypatch.setattr(session_mod, "observe_scratchpad_cell", observe_and_record)
    session = _session(tmp_path, llm=_llm())
    observer = _RecordingObserver(events=events)
    session._scratchpad_observers.append(observer)
    cell = _failed_cell()

    await _run_turn(session, _FakePad(cell=cell, events=events))

    assert events == ["pre", "exec", "post", "observe"]
    assert observer.pre_calls == [Cell(
        code=_CODE, stdout="", stderr="", error=None,
        description=_DESCRIPTION, estimated_time="5s",
    )]
    assert len(observer.post_calls) == 1
    assert observer.post_calls[0] is cell


async def test_the_end_of_turn_flush_learns_from_a_failed_streaming_cell(tmp_path):
    llm = _llm()
    session = _session(tmp_path, llm=llm)

    await _run_turn(session, _FakePad(cell=_failed_cell(), events=[]))

    [diff_call] = _diff_calls(llm)
    prompt = diff_call.kwargs["messages"][0]["content"]
    assert "No module named 'pyodbc'" in prompt
    assert _DESCRIPTION in prompt
    lessons = (tmp_path / "memory" / "project" / "lessons.md").read_text()
    assert _LESSON in lessons
    assert session._cerebellum.buffered_count == 0


async def test_a_cancelled_streaming_exec_fires_pre_but_not_post(tmp_path):
    """A cancel during the cell leaves no cell, and `handle_scratchpad` fires
    post-execute only for a cell that exists."""
    events: list[str] = []
    llm = _llm()
    session = _session(tmp_path, llm=llm)
    observer = _RecordingObserver(events=events)
    session._scratchpad_observers.append(observer)
    pad = _FakePad(
        cell=_failed_cell(), events=events,
        on_exec=lambda _call: session._cancel_event.set(),
    )

    await _run_turn(session, pad)

    assert pad.cancelled
    assert len(observer.pre_calls) == 1
    assert observer.post_calls == []
    assert _diff_calls(llm) == []


async def test_an_observer_that_raises_never_breaks_the_turn(tmp_path):
    llm = _llm()
    session = _session(tmp_path, llm=llm)
    broken = _RecordingObserver(events=[], raises=RuntimeError("observer bug"))
    healthy = _RecordingObserver(events=[])
    session._scratchpad_observers.insert(0, broken)
    session._scratchpad_observers.append(healthy)

    await _run_turn(session, _FakePad(cell=_failed_cell(), events=[]))

    assert (len(broken.pre_calls), len(broken.post_calls)) == (1, 1)
    # The observers after the broken one still see the cell.
    assert (len(healthy.pre_calls), len(healthy.post_calls)) == (1, 1)
    assert len(_diff_calls(llm)) == 1
    # The turn went on: the cell's result reached the model, and it answered.
    tool_results = [
        m for m in session.history
        if m["role"] == "user" and isinstance(m["content"], list)
    ]
    assert "No module named 'pyodbc'" in tool_results[0]["content"][0]["content"]
    assert session.history[-1] == {"role": "assistant", "content": "Done."}


@pytest.mark.parametrize(
    ("memory_mode", "background_memory"),
    [("off", True), ("autopilot", False)],
    ids=["memory-mode-off", "background-memory-off"],
)
async def test_no_diff_call_when_automatic_memory_passes_are_off(
    tmp_path, memory_mode, background_memory
):
    llm = _llm()
    session = _session(
        tmp_path, llm=llm, memory_mode=memory_mode, background_memory=background_memory,
    )
    observer = _RecordingObserver(events=[])
    session._scratchpad_observers.append(observer)

    await _run_turn(session, _FakePad(cell=_failed_cell(), events=[]))

    assert len(observer.post_calls) == 1
    assert _diff_calls(llm) == []
    # Dropped, not carried into the next turn's flush.
    assert session._cerebellum.buffered_count == 0


def _tool_result_texts(session: ChatSession) -> list[str]:
    return [
        block["content"]
        for m in session.history
        if m["role"] == "user" and isinstance(m["content"], list)
        for block in m["content"]
        if block.get("type") == "tool_result"
    ]


async def test_an_exec_that_raises_fires_pre_but_not_post(tmp_path):
    """No cell exists when the runtime raises, so post-execute never fires,
    the same as `handle_scratchpad`. The raise becomes the tool's result."""
    events: list[str] = []
    llm = _llm()
    session = _session(tmp_path, llm=llm)
    observer = _RecordingObserver(events=events)
    session._scratchpad_observers.append(observer)
    pad = _FakePad(
        cell=_failed_cell(), events=events,
        on_exec=_raise_on(1, RuntimeError("runtime gone")),
    )

    await _run_turn(session, pad)

    assert events == ["pre", "exec"]
    assert len(observer.pre_calls) == 1
    assert observer.post_calls == []
    [result] = _tool_result_texts(session)
    assert result.startswith("Tool 'scratchpad' failed:")
    assert _diff_calls(llm) == []


async def test_copilot_holds_the_lesson_for_the_person_to_confirm(tmp_path):
    """Copilot confirms a medium-confidence lesson before writing it. The
    flush queues it where the CLI asks about it, and writes nothing."""
    llm = _llm()
    session = _session(tmp_path, llm=llm, memory_mode="copilot")

    await _run_turn(session, _FakePad(cell=_failed_cell(), events=[]))

    assert len(_diff_calls(llm)) == 1
    assert not (tmp_path / "memory" / "project" / "lessons.md").exists()
    [held] = session._pending_memory_confirmations
    assert (held.text, held.kind, held.confidence) == (_LESSON, "lesson", "medium")


@pytest.mark.parametrize(
    "stop",
    ["generator-closed", "task-cancelled", "cancel-event"],
)
async def test_a_cancelled_turn_teaches_no_flush_its_failed_cells(tmp_path, stop):
    """The first cell fails and is buffered, then the turn is stopped:
    - generator-closed: the CLI's Escape abandons the stream, which closes it
    - task-cancelled: a host's Stop cancels the task while the second cell runs
    - cancel-event: the CLI's flag stops the second cell and the turn ends cleanly
    Neither the stopped turn's flush nor the next turn's learns from the cell."""
    llm = _llm(execs=2)
    session = _session(tmp_path, llm=llm)
    background = _track_background(session)

    def on_exec(number: int) -> None:
        if number == 2 and stop == "task-cancelled":
            raise asyncio.CancelledError
        if number == 2 and stop == "cancel-event":
            session._cancel_event.set()

    pad = _FakePad(cell=_failed_cell(), events=[], on_exec=on_exec)
    session._scratchpads.get_or_create = AsyncMock(return_value=pad)

    stream = session.turn_stream("load the driver")
    if stop == "generator-closed":
        async for event in stream:
            if isinstance(event, StreamToolResult):
                break
        assert session._cerebellum.buffered_count == 1
        await stream.aclose()
    elif stop == "task-cancelled":
        with pytest.raises(asyncio.CancelledError):
            async for _ in stream:
                pass
        assert pad.calls == 2
    else:
        async for _ in stream:
            pass
        assert pad.calls == 2
    session._cancel_event.clear()  # the CLI clears it before each turn
    assert session._cerebellum.buffered_count == 0

    await _run_turn(session, pad)
    await asyncio.gather(*background)

    assert _diff_calls(llm) == []
    assert not (tmp_path / "memory" / "project" / "lessons.md").exists()
