"""Episodic memory — timestamped, searchable archive of conversations.

Ported from anton/memory/episodes.py (Linux-only, uses fcntl).
"""

from __future__ import annotations

import fcntl
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .constants import _MAX_TOOL_INPUT, _MAX_TOOL_RESULT


@dataclass
class Episode:
    ts: str  # ISO 8601
    session: str
    turn: int
    role: str  # "user" | "assistant" | "tool_call" | "tool_result" | "scratchpad"
    content: str
    meta: dict = field(default_factory=dict)


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
        now = datetime.now(timezone.utc)
        self._session_id = now.strftime("%Y%m%d_%H%M%S")
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = self._dir / f"{self._session_id}.jsonl"
        self._file.touch()
        return self._session_id

    def log(self, episode: Episode) -> None:
        if not self._enabled or self._file is None:
            return
        try:
            line = json.dumps(asdict(episode), ensure_ascii=False) + "\n"
            with self._file.open("a", encoding="utf-8") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(line)
                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception:
            pass

    def log_turn(
        self,
        turn: int,
        role: str,
        content: str,
        **meta: object,
    ) -> None:
        if not self._enabled or self._session_id is None:
            return
        if role == "tool_call":
            content = content[:_MAX_TOOL_INPUT]
        elif role == "tool_result":
            content = content[:_MAX_TOOL_RESULT]

        self.log(
            Episode(
                ts=datetime.now(timezone.utc).isoformat(),
                session=self._session_id,
                turn=turn,
                role=role,
                content=content,
                meta=dict(meta),
            )
        )

    def recall(
        self,
        query: str,
        *,
        max_results: int = 20,
        days_back: int | None = None,
    ) -> list[Episode]:
        if not self._dir.is_dir():
            return []

        cutoff: datetime | None = None
        if days_back is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)

        pattern = re.compile(re.escape(query), re.IGNORECASE)
        matches: list[Episode] = []

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

            for line in reversed(lines):
                if not line.strip():
                    continue
                if not pattern.search(line):
                    continue
                try:
                    data = json.loads(line)
                    matches.append(Episode(**data))
                except Exception:
                    continue
                if len(matches) >= max_results:
                    return matches

        return matches

    def recall_formatted(
        self,
        query: str,
        **kwargs: object,
    ) -> str:
        episodes = self.recall(query, **kwargs)  # type: ignore[arg-type]
        if not episodes:
            return f"No episodes found matching '{query}'."
        lines: list[str] = []
        for ep in episodes:
            lines.append(f"[{ep.ts}] ({ep.role}) {ep.content[:200]}")
        return "\n".join(lines)

    def session_count(self) -> int:
        if not self._dir.is_dir():
            return 0
        return sum(1 for _ in self._dir.glob("*.jsonl"))
