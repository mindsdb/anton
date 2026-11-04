"""The sidekick's memory between delegations, persisted by the host per conversation.

Cowork builds a new session every turn, so the host loads this into
`ChatSessionConfig.sidekick_state` and saves `ChatSession.sidekick_state`
after the turn. Within one session the state simply carries over.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

# Reuse the sidekick's messages only while its prompt cache is still warm: the
# gateway's default lifetime is 5 minutes, and past it a long message list is
# paid for in full again. Older work restarts from `summary`.
REUSE_WINDOW = timedelta(minutes=5)


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class SidekickState:
    """``status`` is "open" when the last report was `needs_main` or
    `budget_exhausted` (the main agent still owes the work a decision), else
    "idle". ``signals`` counts the times Jev found the sidekick stuck or
    wanting to hand back; it drives the upgrade ladder. ``last_call_at`` is an
    ISO-8601 UTC time, or None before the first delegation."""

    sidekick_id: str
    model: str
    messages: list[dict] = field(default_factory=list)
    summary: str = ""
    status: Literal["idle", "open"] = "idle"
    last_call_at: str | None = None
    signals: int = 0

    @classmethod
    def new(cls, model: str) -> "SidekickState":
        return cls(sidekick_id=uuid.uuid4().hex, model=model)

    @classmethod
    def from_dict(cls, data: object) -> "SidekickState | None":
        """The state a host saved, or None when it is not one. Tolerant of
        missing or malformed fields, since a bad row must not fail the turn."""
        if not isinstance(data, dict):
            return None
        sidekick_id = data.get("sidekick_id")
        if not isinstance(sidekick_id, str) or not sidekick_id:
            return None
        messages = data.get("messages")
        signals = data.get("signals")
        last = data.get("last_call_at")
        return cls(
            sidekick_id=sidekick_id,
            model=str(data.get("model") or ""),
            messages=[m for m in messages if isinstance(m, dict)] if isinstance(messages, list) else [],
            summary=str(data.get("summary") or ""),
            status="open" if data.get("status") == "open" else "idle",
            last_call_at=last if isinstance(last, str) else None,
            signals=signals if isinstance(signals, int) and not isinstance(signals, bool) and signals > 0 else 0,
        )

    def to_dict(self) -> dict:
        """A JSON-serialisable copy."""
        return asdict(self)

    def last_call(self) -> datetime | None:
        if not self.last_call_at:
            return None
        try:
            when = datetime.fromisoformat(self.last_call_at)
        except ValueError:
            return None
        return when if when.tzinfo is not None else when.replace(tzinfo=timezone.utc)

    def reusable(self, now: datetime | None = None) -> bool:
        """Whether the next delegation continues the saved messages: there
        are some, and the last call was under `REUSE_WINDOW` ago."""
        last = self.last_call()
        if not self.messages or last is None:
            return False
        return (now or _now()) - last < REUSE_WINDOW

    def resume(self, now: datetime | None = None) -> list[dict]:
        """The messages the next delegation starts from: the saved ones while
        reusable, else none (the kickoff then carries `summary`)."""
        if self.reusable(now):
            return list(self.messages)
        self.messages = []
        return []

    def record(self, *, messages: list[dict], summary: str, report_status: str, model: str,
               now: datetime | None = None) -> None:
        """Keep what a finished delegation leaves for the next one."""
        self.messages = messages
        self.summary = summary
        self.model = model
        self.status = "open" if report_status in ("needs_main", "budget_exhausted") else "idle"
        self.last_call_at = (now or _now()).isoformat()
