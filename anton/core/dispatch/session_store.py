"""Per-session message store — the single IO surface between host and agent.

Following nanoclaw's "the DB is the only IO mechanism" principle, every
dispatch session has its own SQLite database with two tables:

  - ``messages_in`` — written by the host (router), read by the agent runtime.
  - ``messages_out`` — written by the agent runtime, read by the host (delivery).

Everything is a message: chat, webhooks, system actions, scheduled triggers,
action-card responses, agent-to-agent. This collapses what would otherwise be
several IPC channels (stdin pipes, control sockets, status files) into one
inspectable, restartable, debuggable surface.

The store is intentionally narrow — append, read-since, mark-delivered. No
joins, no transactions across rows, no schema migrations beyond the bootstrap.
If the agent runtime crashes, the next process opens the same SQLite file
and resumes from the last unread row.

This module implements :class:`SessionStoreProtocol` so cloud deployments
(Postgres, Redis Streams, etc.) can supply alternate backends without
inheriting from the file-based one — same pattern as :class:`HippocampusProtocol`.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Protocol, runtime_checkable

Direction = Literal["in", "out"]


@dataclass
class StoredMessage:
    """A row from ``messages_in`` or ``messages_out``."""

    rowid: int
    direction: Direction
    kind: str
    """Free-form tag: ``"chat"``, ``"webhook"``, ``"system"``,
    ``"action_response"``, ``"scheduled"``, ``"reply"``, etc."""
    content: Any
    """JSON-decoded payload."""
    timestamp: datetime
    delivered: bool
    """``True`` once the host has delivered an outbound row to the platform,
    or once the agent runtime has consumed an inbound row. Acts as the
    'has been processed' flag for both directions."""


@runtime_checkable
class SessionStoreProtocol(Protocol):
    """Backend-agnostic message store interface."""

    def append(self, direction: Direction, kind: str, content: Any) -> int:
        """Append a row; return its rowid."""
        ...

    def read_undelivered(self, direction: Direction) -> list[StoredMessage]:
        """Return undelivered rows for the given direction, oldest first."""
        ...

    def mark_delivered(self, rowid: int, direction: Direction) -> None:
        """Flip the ``delivered`` flag for one row in the given direction's table."""
        ...

    def close(self) -> None:
        """Release backend resources."""
        ...


# ---------------------------------------------------------------------------
# SQLite implementation
# ---------------------------------------------------------------------------


_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages_in (
    rowid     INTEGER PRIMARY KEY AUTOINCREMENT,
    kind      TEXT NOT NULL,
    content   TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    delivered INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages_out (
    rowid     INTEGER PRIMARY KEY AUTOINCREMENT,
    kind      TEXT NOT NULL,
    content   TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    delivered INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_in_undelivered  ON messages_in  (delivered, rowid);
CREATE INDEX IF NOT EXISTS idx_out_undelivered ON messages_out (delivered, rowid);
"""


class SQLiteSessionStore(SessionStoreProtocol):
    """File-based message store mounted into the agent runtime.

    The SQLite file lives at ``<session_dir>/session.db``. The host writes
    to ``messages_in`` and reads from ``messages_out``; the agent runtime
    does the inverse. SQLite's file locking handles the concurrent access
    safely when both sides use ``WAL`` journaling.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(db_path),
            isolation_level=None,  # autocommit
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        # This DB has concurrent writers from separate connections: the
        # orchestrator appends replies to messages_out while the router's
        # delivery loop marks those rows delivered, and on_inbound writes
        # messages_in. WAL serializes writers; without a busy timeout the
        # loser of a collision gets an immediate `database is locked`
        # OperationalError. Wait up to 5s for the lock instead.
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)

    def append(self, direction: Direction, kind: str, content: Any) -> int:
        """Append a row to ``messages_in`` or ``messages_out``."""
        table = self._table(direction)
        ts = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            f"INSERT INTO {table} (kind, content, timestamp) VALUES (?, ?, ?)",
            (kind, json.dumps(content), ts),
        )
        return cur.lastrowid

    def read_undelivered(self, direction: Direction) -> list[StoredMessage]:
        """Return undelivered rows for the direction, oldest first."""
        table = self._table(direction)
        rows = self._conn.execute(
            f"SELECT rowid, kind, content, timestamp, delivered "
            f"FROM {table} WHERE delivered = 0 ORDER BY rowid ASC"
        ).fetchall()
        return [
            StoredMessage(
                rowid=r[0],
                direction=direction,
                kind=r[1],
                content=json.loads(r[2]),
                timestamp=datetime.fromisoformat(r[3]),
                delivered=bool(r[4]),
            )
            for r in rows
        ]

    def mark_delivered(self, rowid: int, direction: Direction) -> None:
        """Mark a single row delivered in the given direction's table.

        ``direction`` is required because ``messages_in`` and ``messages_out``
        have independent rowid sequences — a rowid alone is ambiguous, and
        guessing the wrong table silently flips the wrong row.
        """
        table = self._table(direction)
        self._conn.execute(
            f"UPDATE {table} SET delivered = 1 WHERE rowid = ?",
            (rowid,),
        )

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    @staticmethod
    def _table(direction: Direction) -> str:
        if direction == "in":
            return "messages_in"
        if direction == "out":
            return "messages_out"
        raise ValueError(f"Invalid direction: {direction!r}")


# ---------------------------------------------------------------------------
# Convenience helpers used by the router
# ---------------------------------------------------------------------------


def open_store(session_dir: Path) -> SQLiteSessionStore:
    """Open (or create) the standard ``session.db`` for a session directory."""
    return SQLiteSessionStore(session_dir / "session.db")
