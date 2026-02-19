from __future__ import annotations

import json
import time
import uuid
from pathlib import Path


class SessionStore:
    """File-system session manager for ~/.anton/sessions/."""

    def __init__(self, base_dir: Path) -> None:
        self._sessions_dir = base_dir / "sessions"
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._sessions_dir / "index.json"

    def _read_index(self) -> list[dict]:
        if self._index_path.exists():
            return json.loads(self._index_path.read_text())
        return []

    def _write_index(self, index: list[dict]) -> None:
        self._index_path.write_text(json.dumps(index, indent=2))

    async def start_session(self, task: str) -> str:
        session_id = uuid.uuid4().hex[:12]
        session_dir = self._sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "id": session_id,
            "task": task,
            "status": "running",
            "started_at": time.time(),
            "completed_at": None,
        }
        (session_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        # Append task entry to transcript
        await self.append(session_id, {"type": "task", "content": task})

        # Update index
        index = self._read_index()
        index.append({
            "id": session_id,
            "task": task,
            "status": "running",
            "started_at": meta["started_at"],
            "completed_at": None,
            "summary_preview": None,
        })
        self._write_index(index)

        return session_id

    async def append(self, session_id: str, entry: dict) -> None:
        session_dir = self._sessions_dir / session_id
        transcript_path = session_dir / "transcript.jsonl"

        if "ts" not in entry:
            entry["ts"] = time.time()

        line = json.dumps(entry) + "\n"
        with open(transcript_path, "a") as f:
            f.write(line)

    async def complete_session(self, session_id: str, summary: str) -> None:
        session_dir = self._sessions_dir / session_id
        now = time.time()

        # Update meta.json
        meta_path = session_dir / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["status"] = "completed"
        meta["completed_at"] = now
        meta_path.write_text(json.dumps(meta, indent=2))

        # Write summary.md
        (session_dir / "summary.md").write_text(summary)

        # Append completion entry to transcript
        await self.append(session_id, {"type": "complete", "summary": summary})

        # Update index
        index = self._read_index()
        for entry in index:
            if entry["id"] == session_id:
                entry["status"] = "completed"
                entry["completed_at"] = now
                entry["summary_preview"] = summary[:200]
                break
        self._write_index(index)

    async def fail_session(self, session_id: str, error: str) -> None:
        session_dir = self._sessions_dir / session_id
        now = time.time()

        # Update meta.json
        meta_path = session_dir / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["status"] = "failed"
        meta["completed_at"] = now
        meta_path.write_text(json.dumps(meta, indent=2))

        # Append failure entry to transcript
        await self.append(session_id, {"type": "failed", "error": error})

        # Update index
        index = self._read_index()
        for entry in index:
            if entry["id"] == session_id:
                entry["status"] = "failed"
                entry["completed_at"] = now
                break
        self._write_index(index)

    def list_sessions(self, limit: int = 20) -> list[dict]:
        index = self._read_index()
        # Return most recent first
        return sorted(index, key=lambda e: e.get("started_at", 0), reverse=True)[:limit]

    def get_session(self, session_id: str) -> dict | None:
        session_dir = self._sessions_dir / session_id
        meta_path = session_dir / "meta.json"
        if not meta_path.exists():
            return None

        meta = json.loads(meta_path.read_text())
        summary_path = session_dir / "summary.md"
        if summary_path.exists():
            meta["summary"] = summary_path.read_text()
        return meta

    def get_transcript(self, session_id: str) -> list[dict]:
        session_dir = self._sessions_dir / session_id
        transcript_path = session_dir / "transcript.jsonl"
        if not transcript_path.exists():
            return []

        entries = []
        for line in transcript_path.read_text().splitlines():
            line = line.strip()
            if line:
                entries.append(json.loads(line))
        return entries

    def get_recent_summaries(self, limit: int = 3) -> list[str]:
        index = self._read_index()
        completed = [
            e for e in index if e.get("status") == "completed"
        ]
        completed.sort(key=lambda e: e.get("completed_at", 0), reverse=True)

        summaries = []
        for entry in completed[:limit]:
            session_dir = self._sessions_dir / entry["id"]
            summary_path = session_dir / "summary.md"
            if summary_path.exists():
                summaries.append(summary_path.read_text())
        return summaries
