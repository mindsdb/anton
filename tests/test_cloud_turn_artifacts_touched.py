"""The artifacts a real cloud turn names on `turn_completed`.

`_stream_turn` reads the record through `getattr`, so a session double that
defines it (test_cloud_turn_entrypoint.py) would pass with the real
`ChatSession` record renamed or gone, and every turn would silently go out
with no list. These drive a real `ChatSession`, and for whole turns a real
`LocalScratchpadRuntime`, so the cells really write into the artifact folder.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from tests.conftest import make_mock_llm

import anton.core.tools.tool_handlers as tool_handlers
from anton.cloud_turn.__main__ import stream_turn
from anton.core.artifacts import ArtifactStore
from anton.core.backends.base import Cell
from anton.core.backends.local import LocalScratchpadRuntime
from anton.core.llm.provider import LLMResponse, StreamComplete, ToolCall, Usage
from anton.core.session import ChatSession, ChatSessionConfig, _VerifierVerdict

_PAD_DEFAULTS = dict(
    coding_provider="anthropic",
    coding_model="",
    coding_api_key="",
    coding_base_url="",
)
_REQUEST = json.dumps({"protocol_version": 1, "conversation_id": "c", "input": "edit the deck"})


def _session(tmp_path: Path, *, cells: list[str] = ()) -> tuple[ChatSession, list[str]]:
    """A real ChatSession whose model runs each of `cells` as a scratchpad
    exec, then answers; and the prompts the model saw, in order."""
    workspace = MagicMock(base=tmp_path)
    workspace.artifacts_dir = tmp_path / "artifacts"
    llm = make_mock_llm()
    llm.generate_object_code = AsyncMock(return_value=_VerifierVerdict(status="COMPLETE", reason="done"))
    responses = iter([
        LLMResponse(
            content="running",
            tool_calls=[ToolCall(id=f"tc_{i}", name="scratchpad",
                                 input={"action": "exec", "name": "main", "code": code})],
            usage=Usage(input_tokens=1, output_tokens=1),
            stop_reason="tool_use",
        )
        for i, code in enumerate(cells)
    ])
    seen_by_model: list[str] = []

    def plan_stream(**kwargs):
        seen_by_model.append(str(kwargs.get("messages")))

        async def gen():
            yield StreamComplete(response=next(responses, LLMResponse(
                content="done", tool_calls=[], usage=Usage(input_tokens=1, output_tokens=1),
                stop_reason="end_turn",
            )))
        return gen()

    llm.plan_stream = plan_stream
    return ChatSession(ChatSessionConfig(llm_client=llm, workspace=workspace)), seen_by_model


def _deck(tmp_path: Path) -> tuple[ArtifactStore, str, Path]:
    store = ArtifactStore(tmp_path / "artifacts")
    slug = store.create(name="Deck", description="d", type="html-app", primary="index.html").slug
    index = store.folder_for(slug) / "index.html"
    index.write_text("<html>v1</html>", encoding="utf-8")
    return store, slug, index


def _edit_cell(index: Path) -> str:
    return f"open({str(index)!r}, 'w').write('<html>version two</html>')\nprint('the deck is edited')\n"


async def _run_turn(session: ChatSession) -> list[dict]:
    pad = LocalScratchpadRuntime(name="main", **_PAD_DEFAULTS)
    await pad.start()
    session._scratchpads.get_or_create = AsyncMock(return_value=pad)
    events: list[dict] = []
    try:
        with patch("anton.analytics.send_event"):
            await stream_turn(_REQUEST, events.append, session_builder=lambda request: session)
    finally:
        await pad.close()
    return events


def test_a_real_session_exposes_what_its_tools_touched(tmp_path):
    session, _ = _session(tmp_path)
    store, slug, _ = _deck(tmp_path)

    tool_handlers._track_artifact(session, store, slug)

    assert session.artifacts_touched == {slug}
    session.artifacts_touched.add("someone-else")
    assert session.artifacts_touched == {slug}, "a host gets a copy, never the session's own record"
    assert session.artifacts_touched_complete is True


async def test_a_turn_whose_cell_edits_an_artifact_names_it(tmp_path):
    _, slug, index = _deck(tmp_path)
    session, _ = _session(tmp_path, cells=[_edit_cell(index)])

    events = await _run_turn(session)

    assert events[-1] == {"kind": "turn_completed", "artifacts_touched": [slug]}


async def test_a_turn_whose_after_cell_snapshot_fails_sends_no_list(tmp_path, monkeypatch):
    """The cell edited the deck, but the snapshot after it could not read the
    folder, so the record cannot say so. Sent as it stands, the list would
    leave the deck's edit out and the host would never open its folder."""
    _, slug, index = _deck(tmp_path)
    session, seen_by_model = _session(tmp_path, cells=[_edit_cell(index)])
    real = tool_handlers.read_artifacts_snapshot

    def after_the_edit_fails(*, store):
        if "version two" in index.read_text(encoding="utf-8"):
            raise OSError(5, "Input/output error", str(store.root))
        return real(store=store)

    monkeypatch.setattr(tool_handlers, "read_artifacts_snapshot", after_the_edit_fails)

    events = await _run_turn(session)

    assert events[-1] == {"kind": "turn_completed"}
    assert "the deck is edited" in seen_by_model[1], "the cell's result still reaches the model"
    assert "Tool 'scratchpad' failed" not in seen_by_model[1]


async def test_a_slow_snapshot_around_a_cell_leaves_the_event_loop_free(tmp_path, monkeypatch):
    """On a slow mount a snapshot can take long enough that a turn whose
    heartbeat shares the loop looks stalled to the controller."""
    session, _ = _session(tmp_path)
    _deck(tmp_path)
    pad = MagicMock()
    pad.execute = AsyncMock(return_value=Cell(code="print(1)", stdout="1\n", stderr="", error=None))
    session._scratchpads.get_or_create = AsyncMock(return_value=pad)
    real = tool_handlers.read_artifacts_snapshot

    def slow(*, store):
        time.sleep(0.5)
        return real(store=store)

    monkeypatch.setattr(tool_handlers, "read_artifacts_snapshot", slow)
    ticks: list[float] = []

    async def ticker():
        while True:
            ticks.append(time.monotonic())
            await asyncio.sleep(0.02)

    ticking = asyncio.create_task(ticker())
    await asyncio.sleep(0.05)
    try:
        await tool_handlers.handle_scratchpad(session, {"action": "exec", "name": "main", "code": "print(1)"})
    finally:
        ticking.cancel()

    gaps = [later - earlier for earlier, later in zip(ticks, ticks[1:])]
    assert pad.execute.await_count == 1
    assert max(gaps) < 0.4, f"the loop stopped for {max(gaps):.2f} s"
