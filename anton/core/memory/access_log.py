"""Access log — append-only record of which project memories were delivered each session.

The access log is written by the system at the moment of memory delivery — not by
Anton, not by an LLM judgment call. It's the ground truth for which project memories
were active in a session, used by /share export to populate memory.project_accessed.

File location: <project>/.anton/memory/access_log.jsonl
Format: one JSON object per line (append-only, never overwritten).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class AccessLog:
    """Append-only log of memories delivered to Anton's context per session.

    Writes one entry per memory per delivery. Multiple deliveries of the
    same memory in the same session produce multiple entries — deduplicate
    at export time.
    """

    def __init__(self, base_dir: Path) -> None:
        self._path = base_dir / "access_log.jsonl"

    def log_delivered(
        self,
        records: list[dict],
        scope: str,
        session_id: str,
    ) -> None:
        """Append one log entry per delivered memory record.

        Args:
            records: Raw JSONL records that were delivered (from hippocampus).
            scope: 'global' or 'project' — where these records live.
            session_id: The session in which delivery occurred.
        """
        if not records or not session_id:
            return

        now = _now_iso()
        lines = []
        for r in records:
            entry = {
                "session_id": session_id,
                "memory_id": r.get("id", ""),
                "memory_text": r.get("text", ""),
                "memory_scope": scope,
                "memory_kind": r.get("kind", ""),
                "memory_topic": r.get("topic", ""),
                "delivered_at": now,
            }
            lines.append(json.dumps(entry))

        if not lines:
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(lines) + "\n"
        with open(self._path, "a", encoding="utf-8") as f:
            if sys.platform != "win32":
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(content)
                f.flush()
            finally:
                if sys.platform != "win32":
                    import fcntl
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def get_session_entries(self, session_id: str) -> list[dict]:
        """Return all log entries for the given session.

        Used by /share export to find which project memories were active.
        Returns entries for all scopes — caller filters out global/profile
        as needed per the Memory Export Filter spec.
        """
        if not self._path.is_file():
            return []
        entries = []
        try:
            for line in self._path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("session_id") == session_id:
                        entries.append(entry)
                except json.JSONDecodeError:
                    pass
        except (OSError, UnicodeDecodeError):
            pass
        return entries
