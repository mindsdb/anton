"""A token a cell prints never leaves the scratchpad readable.

A skill that fetches an OAuth token and prints the token endpoint's response
puts the token in the cell's output. Every copy of that output is scrubbed:
the streamed tool result a host saves, the progress strings, the tool result
the model reads next round, the episodic log, and the notebook dump. A cell
killed after printing the token keeps its salvaged output, scrubbed too.

The cells build the token at runtime so it never appears in the cell's code,
which is the model's own text and is not scrubbed.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from anton.chat_session import get_runtime_factory
from anton.core.backends.base import Cell
from anton.core.backends.local import LocalScratchpadRuntime
from anton.core.llm.provider import (
    LLMResponse,
    StreamComplete,
    StreamTaskProgress,
    StreamToolResult,
    ToolCall,
    Usage,
)
from anton.core.memory.episodes import EpisodicMemory
from anton.core.session import ChatSession, ChatSessionConfig, _VerifierVerdict
from tests.conftest import make_mock_llm

_DEFAULTS = dict(
    coding_provider="anthropic",
    coding_model="",
    coding_api_key="",
    coding_base_url="",
)

TOKEN = "opq" + "Z9" * 15
_MAKE_TOKEN = "tok = 'opq' + 'Z9' * 15\n"
_PRINT_TOKEN_RESPONSE = (
    "import json\n"
    + _MAKE_TOKEN
    + "progress('Bearer ' + tok)\n"
    + "print(json.dumps({'access_token': tok, 'expires_in': 3600}))\n"
)


def _exec_response(code: str) -> LLMResponse:
    """A model round that asks the scratchpad to run ``code``."""
    return LLMResponse(
        content="running",
        tool_calls=[ToolCall(id="tc_cell", name="scratchpad",
                             input={"action": "exec", "name": "main", "code": code})],
        usage=Usage(input_tokens=1, output_tokens=1),
        stop_reason="tool_use",
    )


async def test_a_printed_token_is_scrubbed_from_every_copy_of_the_output(tmp_path):
    """Stream, progress, next model input, episodic log and dump never hold the token."""
    workspace = MagicMock(base=tmp_path)
    workspace.artifacts_dir = tmp_path / "artifacts"
    llm = make_mock_llm()
    llm.generate_object_code = AsyncMock(
        return_value=_VerifierVerdict(status="COMPLETE", reason="done")
    )
    model_inputs: list[str] = []

    def plan_stream(**kwargs):
        """Record what the model is sent; run the cell first, then finish."""
        model_inputs.append(repr(kwargs.get("messages")))
        response = _exec_response(_PRINT_TOKEN_RESPONSE) if len(model_inputs) == 1 else LLMResponse(
            content="done", tool_calls=[], usage=Usage(input_tokens=1, output_tokens=1),
            stop_reason="end_turn",
        )

        async def gen():
            yield StreamComplete(response=response)

        return gen()

    llm.plan_stream = plan_stream
    episodic = EpisodicMemory(tmp_path / "episodes")
    episodic.start_session()
    session = ChatSession(
        ChatSessionConfig(llm_client=llm, workspace=workspace, episodic=episodic)
    )
    pad = get_runtime_factory(SimpleNamespace(backend="local"))(
        name="main", cells=None, workspace_path=tmp_path, **_DEFAULTS
    )
    await pad.start()
    session._scratchpads.get_or_create = AsyncMock(return_value=pad)
    events = []
    try:
        with patch("anton.analytics.send_event"):
            async for event in session.turn_stream("connect to the API"):
                events.append(event)
        notebook = pad.render_notebook()
    finally:
        await pad.close()
        await session.close()

    results = [e.content for e in events if isinstance(e, StreamToolResult)]
    progress = [e.message or "" for e in events if isinstance(e, StreamTaskProgress)]
    episodes = "".join(p.read_text() for p in (tmp_path / "episodes").glob("*.jsonl"))
    assert len(model_inputs) == 2, "the model never got the tool result"
    assert results, "no tool result was streamed"
    for copy in [*results, *progress, model_inputs[1], episodes, notebook]:
        assert TOKEN not in copy
    assert "access_token" in results[0]
    assert "[REDACTED_TOKEN]" in results[0]
    assert "[REDACTED_TOKEN]" in model_inputs[1]
    assert any("Bearer [REDACTED_TOKEN]" in m for m in progress)


def test_cells_replayed_from_stored_history_are_scrubbed():
    """Cells a host rebuilds from stored events come back scrubbed in dump and view."""
    stored = Cell(code="fetch()", stdout=f'{{"access_token": "{TOKEN}"}}', stderr="", error=None)
    pad = LocalScratchpadRuntime(name="replayed", cells=[stored], **_DEFAULTS)

    assert TOKEN not in pad.cells[0].stdout
    assert TOKEN not in pad.render_notebook()
    assert TOKEN not in pad.view()
    assert '"access_token": "[REDACTED_TOKEN]"' in pad.cells[0].stdout


async def test_a_cell_killed_after_printing_a_token_keeps_scrubbed_output(monkeypatch):
    """A cell killed by its timeout keeps its salvaged output, with the token redacted."""
    monkeypatch.setenv("ANTON_CELL_INACTIVITY_TIMEOUT", "1")
    monkeypatch.setenv("ANTON_CELL_INACTIVITY_MAX", "1")
    monkeypatch.setenv("ANTON_CELL_INACTIVITY_AFTER_PROGRESS", "1")
    monkeypatch.setenv("ANTON_CELL_TIMEOUT_DEFAULT", "2")
    monkeypatch.setenv("ANTON_SCRATCHPAD_HEARTBEAT_INTERVAL", "0.2")
    pad = LocalScratchpadRuntime(name="killed-token-cell", **_DEFAULTS)
    await pad.start()
    code = _MAKE_TOKEN + "print('access_token=' + tok, flush=True)\nimport time; time.sleep(30)\n"
    try:
        items = [item async for item in pad.execute_streaming(code)]
    finally:
        await pad.close()

    cells = [item for item in items if isinstance(item, Cell)]
    assert len(cells) == 1
    assert "timed out" in cells[0].error.lower()
    assert TOKEN not in cells[0].stdout
    assert "access_token=[REDACTED_TOKEN]" in cells[0].stdout
    assert TOKEN not in pad.cells[-1].stdout
