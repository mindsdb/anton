"""Tests for anton.memory.episodes — episodic memory system."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from anton.core.memory.episodes import Episode, EpisodicMemory


@pytest.fixture()
def episodes_dir(tmp_path: Path) -> Path:
    return tmp_path / "episodes"


@pytest.fixture()
def em(episodes_dir: Path) -> EpisodicMemory:
    return EpisodicMemory(episodes_dir)


class TestStartSession:
    def test_creates_file(self, em: EpisodicMemory, episodes_dir: Path):
        sid = em.start_session()
        assert (episodes_dir / f"{sid}.jsonl").exists()

    def test_filename_format(self, em: EpisodicMemory):
        sid = em.start_session()
        # Format: YYYYMMDD_HHMMSS
        assert len(sid) == 15
        assert sid[8] == "_"

    def test_creates_dir(self, episodes_dir: Path):
        assert not episodes_dir.exists()
        em = EpisodicMemory(episodes_dir)
        em.start_session()
        assert episodes_dir.is_dir()


class TestLog:
    def test_appends_jsonl(self, em: EpisodicMemory, episodes_dir: Path):
        sid = em.start_session()
        ep = Episode(
            ts="2026-02-27T14:30:00",
            session=sid,
            turn=1,
            role="user",
            content="hello",
        )
        em.log(ep)
        path = episodes_dir / f"{sid}.jsonl"
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["content"] == "hello"
        assert data["role"] == "user"

    def test_multiple_episodes(self, em: EpisodicMemory, episodes_dir: Path):
        sid = em.start_session()
        for i in range(5):
            em.log(Episode(
                ts=f"2026-02-27T14:30:0{i}",
                session=sid,
                turn=i,
                role="user",
                content=f"msg {i}",
            ))
        path = episodes_dir / f"{sid}.jsonl"
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 5

    def test_timestamp_preserved(self, em: EpisodicMemory, episodes_dir: Path):
        sid = em.start_session()
        ts = "2026-02-27T14:30:52"
        em.log(Episode(ts=ts, session=sid, turn=1, role="user", content="test"))
        path = episodes_dir / f"{sid}.jsonl"
        data = json.loads(path.read_text().strip())
        assert data["ts"] == ts

    def test_metadata(self, em: EpisodicMemory, episodes_dir: Path):
        sid = em.start_session()
        em.log(Episode(
            ts="2026-02-27T14:30:00",
            session=sid,
            turn=1,
            role="scratchpad",
            content="code",
            meta={"description": "Fetch data", "stdout": "ok"},
        ))
        path = episodes_dir / f"{sid}.jsonl"
        data = json.loads(path.read_text().strip())
        assert data["meta"]["description"] == "Fetch data"
        assert data["meta"]["stdout"] == "ok"

    def test_noop_on_error(self, em: EpisodicMemory, episodes_dir: Path):
        """log() should never raise."""
        sid = em.start_session()
        # Make the file path invalid
        em._file = Path("/nonexistent/dir/bad.jsonl")
        # Should not raise
        em.log(Episode(
            ts="2026-02-27T14:30:00",
            session=sid,
            turn=1,
            role="user",
            content="test",
        ))

    def test_noop_when_disabled(self, episodes_dir: Path):
        em = EpisodicMemory(episodes_dir, enabled=False)
        sid = em.start_session()
        em.log(Episode(
            ts="2026-02-27T14:30:00",
            session="test",
            turn=1,
            role="user",
            content="should not appear",
        ))
        # File should exist but be empty (only created by start_session touch)
        path = episodes_dir / f"{sid}.jsonl"
        assert path.read_text() == ""

class TestLogTurn:
    def test_convenience_method(self, em: EpisodicMemory, episodes_dir: Path):
        sid = em.start_session()
        em.log_turn(1, "user", "What's up?")
        path = episodes_dir / f"{sid}.jsonl"
        data = json.loads(path.read_text().strip())
        assert data["role"] == "user"
        assert data["content"] == "What's up?"
        assert data["turn"] == 1

    def test_kwargs_as_meta(self, em: EpisodicMemory, episodes_dir: Path):
        sid = em.start_session()
        em.log_turn(2, "tool_call", "scratchpad input", tool="scratchpad")
        path = episodes_dir / f"{sid}.jsonl"
        data = json.loads(path.read_text().strip())
        assert data["meta"]["tool"] == "scratchpad"

    def test_tool_call_truncation(self, em: EpisodicMemory, episodes_dir: Path):
        sid = em.start_session()
        long_content = "x" * 5000
        em.log_turn(1, "tool_call", long_content)
        path = episodes_dir / f"{sid}.jsonl"
        data = json.loads(path.read_text().strip())
        assert len(data["content"]) == 2000

    def test_tool_result_truncation(self, em: EpisodicMemory, episodes_dir: Path):
        sid = em.start_session()
        long_content = "x" * 5000
        em.log_turn(1, "tool_result", long_content)
        path = episodes_dir / f"{sid}.jsonl"
        data = json.loads(path.read_text().strip())
        assert len(data["content"]) == 2000

class TestRecall:
    def _populate(self, em: EpisodicMemory, messages: list[str]) -> str:
        sid = em.start_session()
        for i, msg in enumerate(messages):
            em.log(Episode(
                ts=f"2026-02-27T14:30:{i:02d}",
                session=sid,
                turn=i,
                role="user",
                content=msg,
            ))
        return sid

    def test_finds_matches(self, em: EpisodicMemory):
        self._populate(em, ["hello world", "goodbye world", "hello again"])
        results = em.recall("hello")
        assert len(results) == 2
        assert all("hello" in r.content.lower() for r in results)

    def test_case_insensitive(self, em: EpisodicMemory):
        self._populate(em, ["Hello World", "HELLO AGAIN"])
        results = em.recall("hello")
        assert len(results) == 2

    def test_max_results(self, em: EpisodicMemory):
        self._populate(em, [f"hello {i}" for i in range(30)])
        results = em.recall("hello", max_results=5)
        assert len(results) == 5

    def test_days_back(self, em: EpisodicMemory, episodes_dir: Path):
        # Create a "recent" session
        self._populate(em, ["recent bitcoin talk"])
        # Create an "old" session by writing a file with old timestamp name
        old_file = episodes_dir / "20200101_120000.jsonl"
        old_ep = Episode(
            ts="2020-01-01T12:00:00",
            session="20200101_120000",
            turn=1,
            role="user",
            content="old bitcoin talk",
        )
        old_file.write_text(json.dumps({"ts": old_ep.ts, "session": old_ep.session,
                                         "turn": old_ep.turn, "role": old_ep.role,
                                         "content": old_ep.content, "meta": {}}) + "\n")
        # Search with days_back=30 — should only find recent
        results = em.recall("bitcoin", days_back=30)
        assert len(results) == 1
        assert "recent" in results[0].content

    def test_no_matches(self, em: EpisodicMemory):
        self._populate(em, ["hello world"])
        results = em.recall("nonexistent")
        assert results == []

    def test_newest_first(self, em: EpisodicMemory):
        self._populate(em, ["first message", "second message", "third message"])
        results = em.recall("message")
        # newest first means last logged comes first
        assert "third" in results[0].content
        assert "first" in results[-1].content

    def test_empty_dir(self, episodes_dir: Path):
        em = EpisodicMemory(episodes_dir)
        # Don't start a session — dir doesn't exist
        results = em.recall("anything")
        assert results == []


class TestRecallFormatted:
    def test_formatted_string(self, em: EpisodicMemory):
        sid = em.start_session()
        em.log(Episode(
            ts="2026-02-27T14:30:00",
            session=sid,
            turn=1,
            role="user",
            content="bitcoin price check",
        ))
        result = em.recall_formatted("bitcoin")
        assert "bitcoin" in result
        assert "2026-02-27" in result

    def test_no_matches(self, em: EpisodicMemory):
        em.start_session()
        result = em.recall_formatted("nonexistent")
        assert "No episodes found" in result

    def test_includes_timestamps(self, em: EpisodicMemory):
        sid = em.start_session()
        em.log(Episode(
            ts="2026-02-27T14:30:52",
            session=sid,
            turn=1,
            role="user",
            content="test message",
        ))
        result = em.recall_formatted("test")
        assert "2026-02-27T14:30:52" in result


class TestSessionCount:
    def test_counts_files(self, em: EpisodicMemory):
        em.start_session()
        # Create additional fake session files
        em._dir.mkdir(parents=True, exist_ok=True)
        (em._dir / "20260101_120000.jsonl").touch()
        (em._dir / "20260102_120000.jsonl").touch()
        assert em.session_count() == 3  # 1 real + 2 fake

    def test_empty_zero(self, episodes_dir: Path):
        em = EpisodicMemory(episodes_dir)
        assert em.session_count() == 0


class TestDisabled:
    def test_log_noop(self, episodes_dir: Path):
        em = EpisodicMemory(episodes_dir, enabled=False)
        em.start_session()
        em.log_turn(1, "user", "should not log")
        # File should be empty
        path = em._file
        assert path.read_text() == ""

    def test_recall_empty(self, episodes_dir: Path):
        em = EpisodicMemory(episodes_dir, enabled=False)
        # No dir created
        results = em.recall("anything")
        assert results == []

    def test_runtime_toggle(self, em: EpisodicMemory, episodes_dir: Path):
        sid = em.start_session()
        em.log_turn(1, "user", "message one")

        # Disable at runtime
        em.enabled = False
        em.log_turn(2, "user", "message two")

        # Re-enable
        em.enabled = True
        em.log_turn(3, "user", "message three")

        path = episodes_dir / f"{sid}.jsonl"
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2  # only messages 1 and 3


class TestWorkspaceIsolation:
    def test_recall_isolated_across_workspaces(self, tmp_path: Path):
        """Episodes logged in workspace A must not appear in workspace B's recall."""
        em_a = EpisodicMemory(tmp_path / "project_a" / ".anton" / "episodes")
        em_b = EpisodicMemory(tmp_path / "project_b" / ".anton" / "episodes")

        em_a.start_session()
        em_a.log_turn(1, "user", "secret project A data")

        assert em_b.recall("secret") == []
