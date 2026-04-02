"""Episodic memory — timestamped, searchable archive of conversations.

Brain analog: Medial Temporal Lobe episodic memory system.  Logs every
turn (user input, assistant response, tool calls, scratchpad cells) as
JSONL.  Fire-and-forget: never blocks anything.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path


@dataclass
class Episode:
    ts: str  # ISO 8601
    session: str  # Session ID (matches filename stem)
    turn: int
    role: str  # "user" | "assistant" | "tool_call" | "tool_result" | "scratchpad"
    content: str
    meta: dict = field(default_factory=dict)


_MAX_TOOL_INPUT = 2000
_MAX_TOOL_RESULT = 2000


class EpisodicMemory:
    """Append-only conversation archive stored as per-session JSONL files."""

    def __init__(self, episodes_dir: Path, *, enabled: bool = True) -> None:
        self._dir = episodes_dir
        self._enabled = enabled
        self._session_id: str | None = None
        self._file: Path | None = None


    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def start_session(self) -> str:
        """Create a new JSONL file for this session and return the session ID."""
        now = datetime.now(timezone.utc)
        self._session_id = now.strftime("%Y%m%d_%H%M%S")
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = self._dir / f"{self._session_id}.jsonl"
        self._file.touch()
        return self._session_id

    def resume_session(self, session_id: str) -> str:
        """Resume an existing session by reusing its session ID and file."""
        self._session_id = session_id
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = self._dir / f"{self._session_id}.jsonl"
        if not self._file.exists():
            self._file.touch()
        return self._session_id

    def log(self, episode: Episode) -> None:
        """Append an episode to the current session file.  Never raises."""
        if not self._enabled or self._file is None:
            return
        try:
            import sys

            line = json.dumps(asdict(episode), ensure_ascii=False) + "\n"
            with self._file.open("a", encoding="utf-8") as f:
                if sys.platform != "win32":
                    import fcntl
                    fcntl.flock(f, fcntl.LOCK_EX)
                    f.write(line)
                    fcntl.flock(f, fcntl.LOCK_UN)
                else:
                    f.write(line)
        except Exception:
            pass  # Fire-and-forget

    def log_turn(
        self,
        turn: int,
        role: str,
        content: str,
        **meta: object,
    ) -> None:
        """Convenience wrapper around log()."""
        if not self._enabled or self._session_id is None:
            return
        # Truncate tool content
        if role == "tool_call":
            content = content[:_MAX_TOOL_INPUT]
        elif role == "tool_result":
            content = content[:_MAX_TOOL_RESULT]

        self.log(Episode(
            ts=datetime.now(timezone.utc).isoformat(),
            session=self._session_id,
            turn=turn,
            role=role,
            content=content,
            meta=dict(meta),
        ))

    def recall(
        self,
        query: str,
        *,
        max_results: int = 20,
        days_back: int | None = None,
    ) -> list[Episode]:
        """Search episodes for *query* (case-insensitive substring match).

        When a user turn matches, returns the full episode context: the
        matching turn plus the assistant response, tool calls, and scratchpad
        results from the same turn. This mirrors real episodic recall — you
        remember the whole episode, not just the cue.

        Returns newest-first, capped at *max_results* episodes (each episode
        may include multiple turns).
        """
        if not self._dir.is_dir():
            return []

        cutoff: datetime | None = None
        if days_back is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)

        pattern = re.compile(re.escape(query), re.IGNORECASE)
        matches: list[Episode] = []
        seen_turns: set[tuple[str, int]] = set()  # (session, turn) dedup

        # Iterate files newest-first (filenames sort chronologically)
        for path in sorted(self._dir.glob("*.jsonl"), reverse=True):
            if cutoff is not None:
                stem = path.stem
                try:
                    file_dt = datetime.strptime(stem, "%Y%m%d_%H%M%S").replace(
                        tzinfo=timezone.utc,
                    )
                    if file_dt < cutoff:
                        continue
                except ValueError:
                    pass

            try:
                lines = path.read_text(encoding="utf-8").strip().splitlines()
            except Exception:
                continue

            # Parse all episodes in this file for context lookups
            all_episodes: list[Episode] = []
            for line in lines:
                if not line.strip():
                    continue
                try:
                    all_episodes.append(Episode(**json.loads(line)))
                except Exception:
                    continue

            # Build turn index: (session, turn) -> list of episodes
            turn_index: dict[tuple[str, int], list[Episode]] = {}
            for ep in all_episodes:
                key = (ep.session, ep.turn)
                turn_index.setdefault(key, []).append(ep)

            # Search newest-first
            for ep in reversed(all_episodes):
                if not pattern.search(ep.content):
                    continue

                key = (ep.session, ep.turn)
                if key in seen_turns:
                    continue
                seen_turns.add(key)

                # Include the matching turn's full context
                turn_episodes = turn_index.get(key, [ep])
                matches.extend(turn_episodes)

                # Also grab the next turn if it has an assistant response
                if ep.role == "user":
                    next_key = (ep.session, ep.turn + 1)
                    if next_key not in seen_turns:
                        next_eps = turn_index.get(next_key, [])
                        has_response = any(
                            e.role in ("assistant", "tool_result", "scratchpad")
                            for e in next_eps
                        )
                        if next_eps and has_response:
                            seen_turns.add(next_key)
                            matches.extend(next_eps)

                if len(seen_turns) >= max_results:
                    return matches

        return matches

    def recall_formatted(
        self,
        query: str,
        **kwargs: object,
    ) -> str:
        """Return a human-readable string of matching episodes."""
        episodes = self.recall(query, **kwargs)  # type: ignore[arg-type]
        if not episodes:
            return f"No episodes found matching '{query}'."
        lines: list[str] = []
        for ep in episodes:
            # Show more content for assistant/scratchpad responses
            max_len = 2000 if ep.role in ("assistant", "scratchpad", "tool_result") else 500
            lines.append(f"[{ep.ts}] ({ep.role}) {ep.content[:max_len]}")
        return "\n".join(lines)

    def session_count(self) -> int:
        """Count the number of session files."""
        if not self._dir.is_dir():
            return 0
        return sum(1 for _ in self._dir.glob("*.jsonl"))
