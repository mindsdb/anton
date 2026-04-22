"""Access log — append-only record of which project memories were delivered each session.

The access log is written by the system at the moment of memory delivery — not by
Anton, not by an LLM judgment call. It's the ground truth for which project memories
were active in a session, used by /share export to populate memory.project_accessed.

File location: <project>/.anton/memory/access_log.jsonl
Format: one JSON object per line (append-only except for session pruning on init).

Size management:
  - Dedup: each (session_id, memory_id) pair is written at most once per process
    lifetime (in-memory _seen set). Same memory delivered on every turn only
    produces one log entry per session.
  - Pruning: on init, entries from sessions beyond the most recent _MAX_SESSIONS
    are dropped. Rewrite is atomic and file-locked.

Global scope entries are retained in the file for auditability but excluded
from get_session_entries() — only project-scoped memories are exportable.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_MAX_SESSIONS = 30


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_entries(path: Path) -> list[dict]:
    entries = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    except (OSError, UnicodeDecodeError):
        pass
    return entries


class AccessLog:
    """Append-only log of memories delivered to Anton's context per session.

    Writes at most one entry per (session_id, memory_id) pair — deduplicated
    in memory so the same memory delivered on every turn only appears once.
    On init, entries from sessions beyond the most recent _MAX_SESSIONS are pruned.
    """

    def __init__(self, base_dir: Path) -> None:
        self._path = base_dir / "access_log.jsonl"
        self._seen: set[tuple[str, str]] = set()
        self._prune()

    # ── Write ─────────────────────────────────────────────────────────────────

    def log_delivered(
        self,
        records: list[dict],
        scope: str,
        session_id: str,
    ) -> None:
        """Append one log entry per delivered memory record (deduped per session).

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
            memory_id = r.get("id", "")
            key = (session_id, memory_id)
            if key in self._seen:
                continue
            self._seen.add(key)
            entry = {
                "session_id": session_id,
                "memory_id": memory_id,
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

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_session_entries(self, session_id: str) -> list[dict]:
        """Return project-scoped log entries for the given session.

        Filters out global and profile entries — only project memories are
        exported in the .anton bundle. Global entries are retained in the file
        for auditability but never returned here.

        Used by /share export (not yet implemented) to populate
        memory.project_accessed in the session bundle.
        """
        entries = []
        for entry in _read_entries(self._path):
            if entry.get("session_id") != session_id:
                continue
            if entry.get("memory_scope") == "global":
                continue
            if entry.get("memory_kind") == "profile":
                continue
            entries.append(entry)
        return entries

    # ── Pruning ───────────────────────────────────────────────────────────────

    def _prune(self) -> None:
        """Drop entries from sessions beyond the most recent _MAX_SESSIONS.

        No-op if the file doesn't exist or has <= _MAX_SESSIONS unique sessions.
        Rewrite is atomic (tmp file + rename) and protected by a file lock.
        """
        if not self._path.is_file():
            return

        entries = _read_entries(self._path)
        if not entries:
            return

        # Session IDs are YYYYMMDD_HHMMSS — lexicographic sort is chronological.
        all_sessions = sorted({e.get("session_id", "") for e in entries if e.get("session_id")})
        if len(all_sessions) <= _MAX_SESSIONS:
            return

        keep_sessions = set(all_sessions[-_MAX_SESSIONS:])
        kept = [e for e in entries if e.get("session_id") in keep_sessions]

        tmp_path = self._path.with_suffix(".tmp")
        content = "\n".join(json.dumps(e) for e in kept) + "\n"

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
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
            os.replace(tmp_path, self._path)
        except OSError:
            # Non-fatal — log remains unpruned until next init.
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
