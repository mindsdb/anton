"""A scratchpad call that leaves out `action`.

The tools go to the model without strict schemas, so nothing stops a call
like `scratchpad(name="main", code="...")`. It used to come back as the
plain string "Unknown scratchpad action: " on both exec paths.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from anton.core.backends.base import Cell
from anton.core.llm.provider import LLMResponse, StreamComplete, StreamToolResult, ToolCall, Usage
from anton.core.session import ChatSession, ChatSessionConfig, _VerifierVerdict
from anton.core.tools import tool_handlers
from anton.core.tools.registry import ToolOutcome
from tests.conftest import make_mock_llm


class _FakeAsyncIter:
    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


class _RecordingPad:
    def __init__(self) -> None:
        self.ran: list[str] = []

    async def install_packages(self, packages):  # pragma: no cover - unused
        return "nothing to install"

    async def execute_streaming(self, code, **kwargs):
        self.ran.append(code)
        yield Cell(code=code, stdout="ran it", stderr="", error=None, description="run")


def _response(text: str, tool_input: dict | None = None) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[ToolCall(id="tc_1", name="scratchpad", input=tool_input)] if tool_input else [],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="tool_use" if tool_input else "end_turn",
    )


async def test_a_call_with_code_and_no_action_runs_the_code(tmp_path: Path):
    """The streaming turn is the path the product takes."""
    workspace = MagicMock(base=tmp_path)
    workspace.artifacts_dir = tmp_path / "artifacts"
    mock_llm = make_mock_llm()
    mock_llm.generate_object_code = AsyncMock(return_value=_VerifierVerdict(status="COMPLETE", reason="done"))
    replies = [_response("Running.", {"name": "main", "code": "print(1)"}), _response("Done.")]
    mock_llm.plan_stream = lambda **kw: _FakeAsyncIter([StreamComplete(response=replies.pop(0))])
    pad = _RecordingPad()
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))
    session._scratchpads.get_or_create = AsyncMock(return_value=pad)
    streamed: list[StreamToolResult] = []
    try:
        async for event in session.turn_stream("run it"):
            if isinstance(event, StreamToolResult):
                streamed.append(event)
    finally:
        await session.close()

    assert pad.ran == ["print(1)"]
    # cowork-server's cell replay takes this action for a call that named none.
    assert [(e.name, e.action, e.id) for e in streamed] == [("scratchpad", "exec", "tc_1")]
    results = [
        block["content"]
        for message in session.history if message["role"] == "user" and isinstance(message["content"], list)
        for block in message["content"] if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert results == ["[output]\nran it"]
    # The call as history stores it says what ran: cowork-server rebuilds an
    # earlier turn's cells from these rows and counts only `exec` steps.
    calls = [
        block["input"]
        for message in session.history if message["role"] == "assistant" and isinstance(message["content"], list)
        for block in message["content"] if isinstance(block, dict) and block.get("type") == "tool_use"
    ]
    assert calls == [{"name": "main", "code": "print(1)", "action": "exec"}]


async def test_handle_scratchpad_runs_a_call_with_code_and_no_action_as_exec(monkeypatch):
    """The generator's own loops call `handle_scratchpad` directly."""
    seen: list[dict] = []

    async def fake_prepare(session, tc_input):
        seen.append(tc_input)
        return ToolOutcome(content="prepared", ok=True, reason="")

    monkeypatch.setattr(tool_handlers, "prepare_scratchpad_exec", fake_prepare)
    outcome = await tool_handlers.handle_scratchpad(SimpleNamespace(), {"name": "main", "code": "print(1)"})
    assert seen == [{"name": "main", "code": "print(1)"}]
    assert outcome.content == "prepared"


async def test_a_call_with_neither_action_nor_code_names_the_actions():
    outcome = await tool_handlers.handle_scratchpad(SimpleNamespace(), {"name": "main"})
    assert outcome.ok is False
    assert outcome.reason == "scratchpad_unknown_action"
    assert outcome.content == (
        "No `action` given and no `code` to run. Valid actions: exec, view, reset, remove, dump, install."
    )
    unknown = await tool_handlers.handle_scratchpad(SimpleNamespace(), {"name": "main", "action": "run"})
    assert unknown.ok is False
    assert unknown.content.startswith("`run` is not a scratchpad action.")


def test_the_schema_describes_the_action():
    from anton.core.tools.tool_defs import SCRATCHPAD_TOOL

    action = SCRATCHPAD_TOOL.input_schema["properties"]["action"]
    assert action["enum"] == ["exec", "view", "reset", "remove", "dump", "install"]
    assert "`exec` runs `code`" in action["description"]
