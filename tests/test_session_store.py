from __future__ import annotations

import json

import pytest

from anton.memory.store import SessionStore


@pytest.fixture()
def store(tmp_path):
    return SessionStore(tmp_path)


class TestStartSession:
    async def test_creates_directory_and_files(self, store, tmp_path):
        session_id = await store.start_session("test task")

        session_dir = tmp_path / "sessions" / session_id
        assert session_dir.exists()
        assert (session_dir / "meta.json").exists()
        assert (session_dir / "transcript.jsonl").exists()

    async def test_meta_json_content(self, store, tmp_path):
        session_id = await store.start_session("test task")

        meta = json.loads((tmp_path / "sessions" / session_id / "meta.json").read_text())
        assert meta["id"] == session_id
        assert meta["task"] == "test task"
        assert meta["status"] == "running"
        assert meta["started_at"] > 0
        assert meta["completed_at"] is None

    async def test_updates_index(self, store, tmp_path):
        session_id = await store.start_session("test task")

        index = json.loads((tmp_path / "sessions" / "index.json").read_text())
        assert len(index) == 1
        assert index[0]["id"] == session_id
        assert index[0]["task"] == "test task"
        assert index[0]["status"] == "running"


class TestAppend:
    async def test_writes_to_transcript(self, store, tmp_path):
        session_id = await store.start_session("test task")

        await store.append(session_id, {"type": "plan", "reasoning": "do stuff"})

        transcript_path = tmp_path / "sessions" / session_id / "transcript.jsonl"
        lines = transcript_path.read_text().strip().splitlines()
        # First line is the task entry, second is our appended entry
        assert len(lines) == 2
        entry = json.loads(lines[1])
        assert entry["type"] == "plan"
        assert entry["reasoning"] == "do stuff"
        assert "ts" in entry

    async def test_preserves_existing_entries(self, store, tmp_path):
        session_id = await store.start_session("test task")
        await store.append(session_id, {"type": "step", "index": 0})
        await store.append(session_id, {"type": "step", "index": 1})

        transcript_path = tmp_path / "sessions" / session_id / "transcript.jsonl"
        lines = transcript_path.read_text().strip().splitlines()
        # task + 2 steps
        assert len(lines) == 3


class TestCompleteSession:
    async def test_updates_meta_and_writes_summary(self, store, tmp_path):
        session_id = await store.start_session("test task")
        await store.complete_session(session_id, "Task completed successfully")

        session_dir = tmp_path / "sessions" / session_id
        meta = json.loads((session_dir / "meta.json").read_text())
        assert meta["status"] == "completed"
        assert meta["completed_at"] is not None

        summary = (session_dir / "summary.md").read_text()
        assert summary == "Task completed successfully"

    async def test_updates_index_with_summary_preview(self, store, tmp_path):
        session_id = await store.start_session("test task")
        await store.complete_session(session_id, "Task completed successfully")

        index = json.loads((tmp_path / "sessions" / "index.json").read_text())
        assert index[0]["status"] == "completed"
        assert index[0]["summary_preview"] == "Task completed successfully"


class TestFailSession:
    async def test_updates_status_to_failed(self, store, tmp_path):
        session_id = await store.start_session("test task")
        await store.fail_session(session_id, "Something broke")

        session_dir = tmp_path / "sessions" / session_id
        meta = json.loads((session_dir / "meta.json").read_text())
        assert meta["status"] == "failed"
        assert meta["completed_at"] is not None

    async def test_updates_index(self, store, tmp_path):
        session_id = await store.start_session("test task")
        await store.fail_session(session_id, "Something broke")

        index = json.loads((tmp_path / "sessions" / "index.json").read_text())
        assert index[0]["status"] == "failed"


class TestListSessions:
    async def test_returns_recent_sessions(self, store):
        await store.start_session("task 1")
        await store.start_session("task 2")
        await store.start_session("task 3")

        sessions = store.list_sessions(limit=2)
        assert len(sessions) == 2
        # Most recent first
        assert sessions[0]["task"] == "task 3"

    async def test_empty_when_no_sessions(self, store):
        assert store.list_sessions() == []


class TestGetSession:
    async def test_returns_session_with_summary(self, store):
        session_id = await store.start_session("test task")
        await store.complete_session(session_id, "Done!")

        session = store.get_session(session_id)
        assert session is not None
        assert session["id"] == session_id
        assert session["summary"] == "Done!"

    async def test_returns_none_for_missing_session(self, store):
        assert store.get_session("nonexistent") is None


class TestGetTranscript:
    async def test_parses_jsonl_correctly(self, store):
        session_id = await store.start_session("test task")
        await store.append(session_id, {"type": "plan", "reasoning": "think"})
        await store.append(session_id, {"type": "step", "index": 0, "skill": "read_file"})

        transcript = store.get_transcript(session_id)
        assert len(transcript) == 3  # task + plan + step
        assert transcript[0]["type"] == "task"
        assert transcript[1]["type"] == "plan"
        assert transcript[2]["type"] == "step"

    async def test_returns_empty_for_missing_session(self, store):
        assert store.get_transcript("nonexistent") == []


class TestGetRecentSummaries:
    async def test_returns_summaries_from_completed_sessions(self, store):
        s1 = await store.start_session("task 1")
        await store.complete_session(s1, "Summary 1")
        s2 = await store.start_session("task 2")
        await store.complete_session(s2, "Summary 2")

        summaries = store.get_recent_summaries(limit=3)
        assert len(summaries) == 2
        # Most recent first
        assert summaries[0] == "Summary 2"
        assert summaries[1] == "Summary 1"

    async def test_excludes_failed_sessions(self, store):
        s1 = await store.start_session("task 1")
        await store.complete_session(s1, "Summary 1")
        s2 = await store.start_session("task 2")
        await store.fail_session(s2, "Error")

        summaries = store.get_recent_summaries()
        assert len(summaries) == 1
        assert summaries[0] == "Summary 1"

    async def test_returns_empty_when_no_completed(self, store):
        s1 = await store.start_session("task 1")
        await store.fail_session(s1, "Error")
        assert store.get_recent_summaries() == []
