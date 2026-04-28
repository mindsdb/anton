"""Comprehensive export → import roundtrip tests for /share command.

Flow under test:
  1. Build a session with conversation history, memory entries, scratchpad cells.
  2. Export via handle_share_export → .anton file.
  3. Verify .anton file structure and content.
  4. Import via handle_share_import → new session.
  5. Compare state before and after, noting expected differences.

Expected differences after roundtrip:
  - session._session_id  : new ID (import always creates a fresh session)
  - system_prompt_context: new session has provenance suffix; original had none
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console


class RecordingConsole:
    """Minimal console stub that captures print() calls for test assertions."""

    def __init__(self):
        self._lines: list[str] = []

    def print(self, *args, **kwargs):
        self._lines.append(" ".join(str(a) for a in args))

    def getvalue(self) -> str:
        return "\n".join(self._lines)

from anton.commands.share import handle_share_export, handle_share_import, handle_share_status, handle_share_history
from anton.core.backends.base import Cell
from anton.core.backends.manager import ScratchpadManager
from anton.core.memory.episodes import EpisodicMemory
from anton.core.session import ChatSession, ChatSessionConfig
from tests.conftest import make_mock_llm


# ── fixtures ──────────────────────────────────────────────────────────────────



@pytest.fixture()
def workspace(tmp_path: Path):
    return MagicMock(base=tmp_path)


# Canonical conversation used throughout the tests
HISTORY = [
    {"role": "user",      "content": "What is the revenue breakdown by region?"},
    {"role": "assistant", "content": "Let me query that for you."},
    {"role": "user",      "content": "Can you also show the YoY change?"},
    {"role": "assistant", "content": "Sure, here is the YoY comparison."},
]

SESSION_BORN_MEMORY  = {"content": "Always use CTEs for readability", "kind": "lesson", "topic": "sql"}
PROJECT_MEMORY       = {"content": "Never use SELECT * in production",  "kind": "never",  "topic": ""}
PROFILE_MEMORY       = {"content": "User prefers camel-case",              "kind": "profile", "topic": ""}
SCRATCHPAD_CELL      = Cell(code="df.head()", stdout="   col1\n0  1\n", stderr="", error=None, description="Preview data")


class _SimpleHistoryStore:
    """Minimal in-memory history store for tests."""
    def __init__(self):
        self._data: dict = {}

    def save(self, sid: str, history: list) -> None:
        self._data[sid] = list(history)

    def load(self, sid: str):
        return self._data.get(sid)

    def list_sessions(self, limit: int = 20):
        return []


def _build_exporter_session(
    tmp_path: Path,
    workspace,
    *,
    include_profile_memory: bool = False,
) -> tuple[ChatSession, EpisodicMemory, str]:
    """Return (session, episodic, session_id) ready for export."""
    episodes_dir = tmp_path / "episodes"
    episodic = EpisodicMemory(episodes_dir)
    sid = episodic.start_session()

    # log conversation turns to episodic (mirrors what session.py does during turn())
    turn = 0
    for msg in HISTORY:
        role = msg["role"]
        if role == "user":
            turn += 1
        episodic.log_turn(turn, role, msg["content"])

    # log memories that the export should pick up
    episodic.log_turn(0, "memory_write", **SESSION_BORN_MEMORY)
    episodic.log_turn(0, "memory_read",  **PROJECT_MEMORY)
    if include_profile_memory:
        episodic.log_turn(0, "memory_write", **PROFILE_MEMORY)

    mock_llm = make_mock_llm()
    session = ChatSession(ChatSessionConfig(
        llm_client=mock_llm,
        session_id=sid,
        episodic=episodic,
        workspace=workspace,
    ))
    session._history = list(HISTORY)
    session._turn_count = sum(1 for m in HISTORY if m.get("role") == "user")

    # wire a fake scratchpad runtime with one cell
    mock_runtime = MagicMock()
    mock_runtime.cells = [SCRATCHPAD_CELL]
    session._scratchpads._pads = {"main": mock_runtime}

    return session, episodic, sid



async def _do_import(
    tmp_path: Path,
    console,
    workspace,
    anton_file: Path,
    *,
    current_history: list[dict] | None = None,
    cortex=None,
) -> tuple[ChatSession, EpisodicMemory]:
    """Run handle_share_import and return (new_session, new_episodic).

    get_or_create is mocked so no real venv subprocess is started.
    Pad runtimes are injected into session._scratchpads._pads so callers
    can inspect restored cells via result._scratchpads._pads[name].cells.
    """
    mock_llm = make_mock_llm()
    history_store = _SimpleHistoryStore()

    # episodic for the new session (recipient side)
    new_episodic = EpisodicMemory(tmp_path / "new_episodes")

    # empty current session (no active history unless specified)
    current_session = ChatSession(ChatSessionConfig(
        llm_client=mock_llm,
        workspace=workspace,
    ))
    if current_history:
        current_session._history = list(current_history)

    # pre-build the session that rebuild_session will return.
    # Uses the session_id already set by handle_share_import's start_session() call.
    def _fake_rebuild(**kwargs):
        return ChatSession(ChatSessionConfig(
            llm_client=mock_llm,
            session_id=kwargs.get("session_id"),
            episodic=new_episodic,
            workspace=workspace,
        ))

    # Mock get_or_create so no real venv is started.
    # The mock sets _pads[name] so callers can inspect cells afterward.
    async def _mock_get_or_create(mgr_self, name):
        if name not in mgr_self._pads:
            rt = MagicMock()
            rt.cells = []
            mgr_self._pads[name] = rt
        return mgr_self._pads[name]

    with patch.object(ScratchpadManager, "get_or_create", _mock_get_or_create):
        with patch("anton.commands.session.rebuild_session", side_effect=_fake_rebuild):
            result = await handle_share_import(
                console,
                current_session,
                workspace,
                MagicMock(),              # settings
                {"llm_client": mock_llm}, # state
                None,                     # self_awareness
                cortex,
                new_episodic,
                history_store,
                filepath=str(anton_file),
            )

    return result, new_episodic


class TestShareRoundtrip:
    async def test_full_roundtrip(self, tmp_path, workspace):
        """Export a live session, import it, compare state point by point."""
        console = RecordingConsole()

        # ── 1. build exporter session ──────────────────────────────────────
        session, episodic, original_sid = _build_exporter_session(tmp_path, workspace)
        mock_llm = make_mock_llm()

        with patch("anton.commands.share._generate_meta",
                   AsyncMock(return_value=("revenue-region-yoy", "Analyzed revenue. APAC leads YoY."))):
            await handle_share_export(console, session, workspace, mock_llm, episodic)

        output_dir = tmp_path / ".anton" / "output"
        anton_file = next(output_dir.glob("*.anton"))
        payload = json.loads(anton_file.read_text())

        # ── 2. import into fresh session ───────────────────────────────────
        mock_hc = MagicMock()
        mock_cortex = MagicMock()
        mock_cortex.project_hc = mock_hc

        new_session, new_episodic = await _do_import(
            tmp_path / "recipient", console, workspace, anton_file,
            cortex=mock_cortex,
        )

        # ── 3. compare ─────────────────────────────────────────────────────

        # conversation history: EQUAL
        assert new_session._history == HISTORY, "Conversation history must be fully preserved"
        assert new_session._turn_count == 2

        # session_id: comes from a fresh start_session(), not copied from the .anton file
        # (the .anton file's session.id is the original exporter's ID)
        # We verify the new session got its ID from the new episodic, not the payload
        assert new_session._session_id == new_episodic._session_id

        # memory in new episodic: session_born → memory_write
        mem_eps = new_episodic.get_memory_usage()
        writes = [e for e in mem_eps if e.role == "memory_write"]
        reads  = [e for e in mem_eps if e.role == "memory_read"]

        assert len(writes) == 1
        assert writes[0].content == SESSION_BORN_MEMORY["content"]

        assert len(reads) == 1
        assert reads[0].content == PROJECT_MEMORY["content"]

        # scratchpad: one cell in .anton file, and it is restored into the runtime
        assert len(payload["scratchpad"]["cells"]) == 1
        assert "main" in new_session._scratchpads._pads
        assert len(new_session._scratchpads._pads["main"].cells) == 1

        # hippocampus: session_born (kind=lesson) written to project_hc
        mock_hc.encode_lesson.assert_called_once_with(
            SESSION_BORN_MEMORY["content"],
            topic=SESSION_BORN_MEMORY["topic"],
            source="import",
        )
        mock_hc.encode_rule.assert_not_called()

        # ── 4. status ─────────────────────────────────────────────────────

        status_console = RecordingConsole()
        handle_share_status(status_console, new_session, workspace)
        status_out = status_console.getvalue()

        assert "revenue-region-yoy" in status_out, "Title from export must appear in status"
        assert "Analyzed revenue. APAC leads YoY." in status_out, "Summary must appear in status"
        assert "No data sources" in status_out, "No datasources in HISTORY, so should say none referenced"

        # ── 5. history ─────────────────────────────────────────────────────

        history_console = RecordingConsole()
        handle_share_history(history_console, workspace)
        history_out = history_console.getvalue()

        assert "Exported sessions" in history_out, "Header must appear"
        assert "revenue-region-yoy" in history_out, "Title must appear in history listing"
        assert "Analyzed revenue. APAC leads YoY." in history_out, "Summary must appear in history listing"
        assert "imported by" in history_out, "After roundtrip the file has imported metadata"
