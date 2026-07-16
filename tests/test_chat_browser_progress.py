"""Unit tests for the ``browser_control`` progress special-case in the
``ChatSession`` streaming tool path (WS3-T3).

The generic tool path yields ``StreamTaskProgress(phase="tool_start"/"tool_done",
message=tc.name)`` for most tools. For ``browser_control`` (a host-injected
tool) it instead yields ``phase="browser_action"`` with the agent-supplied
``progress_message`` so the UI shows a human-readable per-action line.

These tests drive ``turn_stream`` with a fully-mocked LLM (no HTTP), keeping
them in the fast unit suite.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from tests.conftest import make_mock_llm

from anton.core.session import ChatSession, ChatSessionConfig
from anton.core.tools.tool_defs import ToolDef
from anton.core.llm.provider import (
    LLMResponse,
    StreamComplete,
    StreamTaskProgress,
    ToolCall,
    Usage,
)


class _FakeAsyncIter:
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


def _tool_response(name: str, tc_input: dict, *, tool_id: str = "tc_1") -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id=tool_id, name=name, input=tc_input)],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="tool_use",
    )


def _make_browser_tool(recorded: list[dict]) -> ToolDef:
    async def handle(session, tc_input: dict) -> str:  # noqa: ANN001
        del session
        recorded.append(dict(tc_input))
        return '{"status": "ok", "observed": {"text": "x"}, "citations": []}'

    return ToolDef(
        name="browser_control",
        description="fake browser control",
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "reason": {"type": "string"},
                "progress_message": {"type": "string"},
            },
            "required": ["action", "reason", "progress_message"],
        },
        handler=handle,
    )


def _make_generic_tool(name: str, recorded: list[dict]) -> ToolDef:
    async def handle(session, tc_input: dict) -> str:  # noqa: ANN001
        del session
        recorded.append(dict(tc_input))
        return "generic-ok"

    return ToolDef(
        name=name,
        description="generic tool",
        input_schema={
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": [],
        },
        handler=handle,
    )


def _session_with_tool(tool: ToolDef, tool_call: LLMResponse) -> ChatSession:
    mock_llm = make_mock_llm()
    # Completion verifier uses plan(); keep it COMPLETE so the loop exits.
    mock_llm.plan = AsyncMock(
        return_value=_text_response("STATUS: COMPLETE — task done")
    )

    call_count = 0

    def fake_plan_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _FakeAsyncIter([StreamComplete(response=tool_call)])
        return _FakeAsyncIter([StreamComplete(response=_text_response("Done."))])

    mock_llm.plan_stream = fake_plan_stream

    return ChatSession(
        ChatSessionConfig(
            llm_client=mock_llm,
            tools=[tool],
            web_search_enabled=False,
            web_fetch_enabled=False,
        )
    )


async def _progress(session: ChatSession, user_input: str) -> list[StreamTaskProgress]:
    out: list[StreamTaskProgress] = []
    async for event in session.turn_stream(user_input):
        if isinstance(event, StreamTaskProgress):
            out.append(event)
    return out


class TestBrowserControlProgress:
    async def test_browser_action_phase_carries_human_message(self):
        recorded: list[dict] = []
        tool = _make_browser_tool(recorded)
        call = _tool_response(
            "browser_control",
            {
                "action": "inspect",
                "reason": "no connector",
                "progress_message": "Reading account list",
            },
        )
        session = _session_with_tool(tool, call)
        progress = await _progress(session, "read my accounts")

        browser = [p for p in progress if p.phase == "browser_action"]
        # Both the pre- and post-dispatch progress events fire.
        assert len(browser) == 2
        messages = [p.message for p in browser]
        assert messages == ["Reading account list", "Reading account list"]
        # The pre-dispatch event has no eta (action running); the
        # post-dispatch event carries the elapsed — this is the completion
        # signal CLI/host displays use to mark the activity done.
        assert browser[0].eta_seconds is None
        assert browser[1].eta_seconds is not None
        assert browser[1].eta_seconds >= 0
        # The raw tool name is never surfaced as the message.
        assert "browser_control" not in messages
        # The generic tool_start/tool_done phases are NOT used for this tool.
        assert not [p for p in progress if p.phase in ("tool_start", "tool_done")]
        # Progress correlates to the originating tool_use id.
        assert all(p.id == "tc_1" for p in browser)
        assert recorded and recorded[0]["action"] == "inspect"

    async def test_falls_back_to_tool_name_without_progress_message(self):
        """When progress_message is absent, the tool takes the generic path
        (tool_start/tool_done with the raw name) — the special-case keys on the
        presence of a progress_message input field."""
        recorded: list[dict] = []
        tool = _make_browser_tool(recorded)
        call = _tool_response(
            "browser_control",
            {"action": "inspect", "reason": "no connector"},
        )
        session = _session_with_tool(tool, call)
        progress = await _progress(session, "read my accounts")

        assert not [p for p in progress if p.phase == "browser_action"]
        generic = [p for p in progress if p.phase in ("tool_start", "tool_done")]
        assert generic, "Expected generic tool_start/tool_done fallback"
        assert all(p.message == "browser_control" for p in generic)

    async def test_other_tools_unaffected(self):
        """A non-browser tool still uses the generic tool_start/tool_done path
        even if it happens to carry a progress_message input."""
        recorded: list[dict] = []
        tool = _make_generic_tool("something_else", recorded)
        call = _tool_response(
            "something_else", {"progress_message": "should be ignored"}
        )
        session = _session_with_tool(tool, call)
        progress = await _progress(session, "do a thing")

        assert not [p for p in progress if p.phase == "browser_action"]
        generic = [p for p in progress if p.phase in ("tool_start", "tool_done")]
        assert generic
        assert all(p.message == "something_else" for p in generic)
