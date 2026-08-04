"""Session-level test for turn_stream()'s generic tool_progress phase.

Reuses the harness proven in
tests/test_chat_scratchpad.py::TestScratchpadStreaming::test_scratchpad_in_streaming_path
(a real ChatSession driven by a scripted plan_stream, draining
session.turn_stream(...)) — adapted for a generic, non-scratchpad streaming
tool. The tool is registered directly via
session.tool_registry.register_tool(...) rather than depending on
anton/core/tools/test_tool.py, so this test survives that file's eventual
deletion.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from tests.conftest import make_mock_llm

from anton.core.llm.provider import (
    LLMResponse,
    StreamComplete,
    StreamTaskProgress,
    ToolCall,
    Usage,
)
from anton.core.session import ChatSession, ChatSessionConfig
from anton.core.tools.progress import ToolProgress
from anton.core.tools.tool_defs import ToolDef


class _FakeAsyncIter:
    """Wraps items into an async iterator for mocking plan_stream."""

    def __init__(self, items):
        self._items = items

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="end_turn",
    )


def _tool_call_response(tool_name: str, tool_id: str = "tc_1") -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id=tool_id, name=tool_name, input={})],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="tool_use",
    )


async def _two_step_handler(_session, _input):
    yield ToolProgress("step 1")
    yield ToolProgress("step 2")
    yield "final result"


async def _raising_after_progress_handler(_session, _input):
    yield ToolProgress("about to fail")
    raise ValueError("boom")


async def _no_result_handler(_session, _input):
    yield ToolProgress("only progress")


def _make_session_with_tool(handler):
    """A real ChatSession with one custom streaming tool registered.

    Registering a tool BEFORE the first turn_stream() call makes
    ChatSession._build_tools() skip _build_core_tools() (it only runs
    `if not self.tool_registry`, i.e. when empty) — fine here since the
    scripted plan_stream below never asks for any built-in tool by name.
    """
    base = Path(__file__).resolve().parents[1] / ".pytest-workspace"
    base.mkdir(parents=True, exist_ok=True)
    workspace = MagicMock(base=base)
    mock_llm = make_mock_llm()
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))
    session.tool_registry.register_tool(
        ToolDef(
            name="streaming_probe",
            description="test-only streaming tool",
            input_schema={"type": "object", "properties": {}},
            handler=handler,
        )
    )
    return session, mock_llm


def _script_one_tool_call_then_text(mock_llm, tool_name: str, final_text: str):
    """First plan_stream() call returns one tool_use round; second, final text."""
    call_count = 0

    def fake_plan_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _FakeAsyncIter([StreamComplete(response=_tool_call_response(tool_name))])
        return _FakeAsyncIter([StreamComplete(response=_text_response(final_text))])

    mock_llm.plan_stream = fake_plan_stream


def _tool_result_texts(session) -> list[str]:
    return [
        str(item.get("content"))
        for msg in session.history
        if isinstance(msg.get("content"), list)
        for item in msg["content"]
        if item.get("type") == "tool_result"
    ]


class TestTurnStreamToolProgress:
    async def test_progress_events_arrive_in_order_with_the_tool_call_id(self):
        session, mock_llm = _make_session_with_tool(_two_step_handler)
        _script_one_tool_call_then_text(mock_llm, "streaming_probe", "Done.")
        try:
            events = [e async for e in session.turn_stream("run the probe")]
            progress = [
                e for e in events
                if isinstance(e, StreamTaskProgress) and e.phase == "tool_progress"
            ]
            assert [e.message for e in progress] == ["step 1", "step 2"]
            assert all(e.id == "tc_1" for e in progress)
        finally:
            await session.close()

    async def test_progress_markers_never_reach_tool_result_history(self):
        session, mock_llm = _make_session_with_tool(_two_step_handler)
        _script_one_tool_call_then_text(mock_llm, "streaming_probe", "Done.")
        try:
            async for _ in session.turn_stream("run the probe"):
                pass
            texts = _tool_result_texts(session)
            assert any("final result" in t for t in texts)
            assert not any("step 1" in t or "step 2" in t for t in texts)
        finally:
            await session.close()

    async def test_exception_mid_stream_produces_clean_failed_result_not_a_crash(self):
        session, mock_llm = _make_session_with_tool(_raising_after_progress_handler)
        _script_one_tool_call_then_text(mock_llm, "streaming_probe", "Done.")
        try:
            events = [e async for e in session.turn_stream("run the probe")]
            assert any(isinstance(e, StreamComplete) for e in events)
            texts = _tool_result_texts(session)
            assert any("failed" in t and "boom" in t for t in texts)
        finally:
            await session.close()

    async def test_generator_with_no_result_produces_clean_failed_result_not_a_crash(self):
        session, mock_llm = _make_session_with_tool(_no_result_handler)
        _script_one_tool_call_then_text(mock_llm, "streaming_probe", "Done.")
        try:
            events = [e async for e in session.turn_stream("run the probe")]
            assert any(isinstance(e, StreamComplete) for e in events)
            texts = _tool_result_texts(session)
            assert any("failed" in t and "produced no result" in t for t in texts)
        finally:
            await session.close()
