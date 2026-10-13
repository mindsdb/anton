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

import pytest

from anton.chat_session import get_runtime_factory
from anton.core.backends.base import Cell
from anton.core.backends.local import LocalScratchpadRuntime, _add_salvage_chunk, _salvage_tail
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
from anton.core.tools.tool_defs import ToolDef
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


@pytest.mark.parametrize("key, filler", [
    ("' access_token: ', end=''", "'line\\n' * 1600"),
    ("' access_token: ', end=''", "'x' * 8000"),
    ("' Authorization: Bearer'", "'line\\n' * 1600"),
], ids=["lines", "one-line", "bearer-on-its-own-line"])
async def test_a_killed_cell_over_the_salvage_budget_does_not_strip_a_token_key(monkeypatch, key, filler):
    """Salvage arrives in heartbeat chunks; trimming the oldest chunk must not
    drop a token's key and leave the token itself in the kept output. The first
    chunk ends with the key and the second starts with the token, and the
    salvage is cut to its budget after the scrub."""
    monkeypatch.setenv("ANTON_CELL_INACTIVITY_TIMEOUT", "2")
    monkeypatch.setenv("ANTON_CELL_INACTIVITY_MAX", "2")
    monkeypatch.setenv("ANTON_CELL_INACTIVITY_AFTER_PROGRESS", "2")
    monkeypatch.setenv("ANTON_CELL_TIMEOUT_DEFAULT", "3")
    monkeypatch.setenv("ANTON_SCRATCHPAD_HEARTBEAT_INTERVAL", "0.2")
    pad = LocalScratchpadRuntime(name="salvage-split-cell", **_DEFAULTS)
    await pad.start()
    code = (
        _MAKE_TOKEN
        + "import time\n"
        + f"print('a' * 2000 + {key}, flush=True)\n"
        + "time.sleep(0.6)\n"
        + f"print(tok + ' ' + {filler}, flush=True)\n"
        + "time.sleep(30)\n"
    )
    try:
        items = [item async for item in pad.execute_streaming(code)]
    finally:
        await pad.close()

    cells = [item for item in items if isinstance(item, Cell)]
    assert len(cells) == 1
    assert "timed out" in cells[0].error.lower()
    assert cells[0].stdout
    assert TOKEN not in cells[0].stdout


def _salvage_from(stream: str) -> tuple[str, bool]:
    """Feed `stream` through the salvage path in worker-sized chunks."""
    kept: list[str] = []
    dropped = False
    for start in range(0, len(stream), 8192):
        dropped = _add_salvage_chunk(kept, stream[start : start + 8192]) or dropped
    return _salvage_tail(kept, dropped)


_LONG_TOKEN = "opq" + "Z9" * 500


@pytest.mark.parametrize("key", ["access_token: ", "Authorization: Bearer\n", "Authorization: Bearer"])
def test_dropped_salvage_never_leaves_a_token_without_its_key(key):
    """Token-dense output scrubs to far less than the budget, so the start of
    the kept chunks reaches the output; a token whose key was in a dropped
    chunk must not be there. Sized so only the first chunk is dropped."""
    first = "x" * (8192 - len(key) - 1) + "\n" + key
    rest = ("\n" if key.endswith("Bearer") else "") + TOKEN + "\n"
    rest += "".join(f"access_token: {_LONG_TOKEN}\n" for _ in range(35))
    salvaged, truncated = _salvage_from(first + rest)

    assert TOKEN not in salvaged
    assert "access_token: [REDACTED_TOKEN]" in salvaged
    assert truncated


def test_dropped_salvage_on_one_long_line_reports_nothing_rather_than_a_bare_token():
    """With no line start to cut at, the kept text cannot be told safe."""
    line = "x" * 8178 + "access_token: " + TOKEN + " "
    line += "".join(f"access_token: {_LONG_TOKEN} " for _ in range(32))
    salvaged, truncated = _salvage_from(line)

    assert TOKEN not in salvaged
    assert truncated


def test_salvage_under_the_budget_is_kept_whole():
    salvaged, truncated = _salvage_from("sent 1/3\nsent 2/3\n")

    assert salvaged == "sent 1/3\nsent 2/3\n"
    assert not truncated


async def test_another_tool_result_keeps_quoted_placeholders(tmp_path):
    """Only scratchpad output gets the token filter; source another tool returns reaches the model as written."""
    source = 'export const config = { accessToken: "some-placeholder-value" };'

    async def read_source(_session, _input):
        """Return artifact source the way open_artifact does."""
        return source

    workspace = MagicMock(base=tmp_path)
    workspace.artifacts_dir = tmp_path / "artifacts"
    llm = make_mock_llm()
    llm.generate_object_code = AsyncMock(
        return_value=_VerifierVerdict(status="COMPLETE", reason="done")
    )
    model_inputs: list[str] = []

    def plan_stream(**kwargs):
        """Ask for the tool once, then finish."""
        model_inputs.append(repr(kwargs.get("messages")))
        response = LLMResponse(
            content="reading",
            tool_calls=[ToolCall(id="tc_read", name="read_source", input={})],
            usage=Usage(input_tokens=1, output_tokens=1),
            stop_reason="tool_use",
        ) if len(model_inputs) == 1 else LLMResponse(
            content="done", tool_calls=[], usage=Usage(input_tokens=1, output_tokens=1),
            stop_reason="end_turn",
        )

        async def gen():
            yield StreamComplete(response=response)

        return gen()

    llm.plan_stream = plan_stream
    session = ChatSession(ChatSessionConfig(llm_client=llm, workspace=workspace))
    session.tool_registry.register_tool(ToolDef(
        name="read_source", description="read source", input_schema={"type": "object", "properties": {}},
        handler=read_source,
    ))
    try:
        with patch("anton.analytics.send_event"):
            async for _event in session.turn_stream("read the source"):
                pass
    finally:
        await session.close()

    assert len(model_inputs) == 2
    assert 'some-placeholder-value' in model_inputs[1]


def test_a_shared_replay_list_stays_correct_across_pads():
    """Pads built from one replayed list share it; scrubbing it again for each pad changes nothing more."""
    stored = Cell(code="fetch()", stdout=f'{{"access_token": "{TOKEN}"}}', stderr="", error=None)
    shared = [stored]

    first = LocalScratchpadRuntime(name="first", cells=shared, **_DEFAULTS)
    after_first = list(shared)
    second = LocalScratchpadRuntime(name="second", cells=shared, **_DEFAULTS)

    assert first.cells is shared and second.cells is shared
    assert shared == after_first
    assert shared[0].stdout == '{"access_token": "[REDACTED_TOKEN]"}'
    assert shared[0].code == "fetch()"
