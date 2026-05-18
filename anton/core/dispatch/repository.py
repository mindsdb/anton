"""Concrete :class:`DispatchRepository` backed by a central SQLite database.

The central DB holds the entity tables — agent groups, messaging groups,
the ``messaging_group_agents`` wiring table, and a sessions index. It does
**not** hold messages; those live in per-session stores
(:class:`anton.core.dispatch.session_store.SQLiteSessionStore`).

Schema:

    agent_groups        (id, name, workspace, policy_json, created_at)
    messaging_groups    (id, channel_type, platform_id, display_name,
                         is_group, created_at,
                         UNIQUE(channel_type, platform_id))
    messaging_group_agents
                        (messaging_group_id, agent_group_id, session_mode,
                         trigger_rule, trigger_pattern, priority,
                         PRIMARY KEY(messaging_group_id, agent_group_id))
    sessions            (id, agent_group_id, session_key, store_path,
                         created_at, last_active_at,
                         UNIQUE(agent_group_id, session_key))

Sessions are keyed by ``(agent_group_id, session_key)`` where
``session_key`` is derived from the wiring's :class:`SessionMode`.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from anton.core.dispatch.adapter import PlatformAddress
from anton.core.dispatch.entities import (
    AgentGroup,
    MessagingGroup,
    MessagingGroupAgent,
    Session,
    SessionMode,
    TriggerRule,
)
from anton.core.dispatch.policy import FileScope, PermissionPolicy
from anton.core.dispatch.router import DispatchRepository
from anton.core.dispatch.session_store import (
    SessionStoreProtocol,
    SQLiteSessionStore,
)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_groups (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    workspace   TEXT NOT NULL,
    policy_json TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messaging_groups (
    id            TEXT PRIMARY KEY,
    channel_type  TEXT NOT NULL,
    platform_id   TEXT NOT NULL,
    display_name  TEXT,
    is_group      INTEGER,
    created_at    TEXT NOT NULL,
    UNIQUE(channel_type, platform_id)
);

CREATE TABLE IF NOT EXISTS messaging_group_agents (
    messaging_group_id TEXT NOT NULL,
    agent_group_id     TEXT NOT NULL,
    session_mode       TEXT NOT NULL,
    trigger_rule       TEXT NOT NULL,
    trigger_pattern    TEXT,
    priority           INTEGER NOT NULL DEFAULT 100,
    PRIMARY KEY (messaging_group_id, agent_group_id),
    FOREIGN KEY (messaging_group_id) REFERENCES messaging_groups(id),
    FOREIGN KEY (agent_group_id)     REFERENCES agent_groups(id)
);

CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,
    agent_group_id  TEXT NOT NULL,
    session_key     TEXT NOT NULL,
    store_path      TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    last_active_at  TEXT NOT NULL,
    UNIQUE(agent_group_id, session_key),
    FOREIGN KEY (agent_group_id) REFERENCES agent_groups(id)
);

CREATE INDEX IF NOT EXISTS idx_mga_mg ON messaging_group_agents (messaging_group_id, priority);
"""


class SqliteDispatchRepository(DispatchRepository):
    """File-based dispatch repository.

    All methods are ``async`` to satisfy the :class:`DispatchRepository`
    protocol, but SQLite work runs synchronously — fine for local
    deployments. Cloud backends (Postgres, etc.) implement the same
    protocol with truly-async drivers.
    """

    def __init__(self, db_path: Path, sessions_root: Path) -> None:
        self.db_path = db_path
        self.sessions_root = sessions_root
        db_path.parent.mkdir(parents=True, exist_ok=True)
        sessions_root.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(db_path),
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)

    # -----------------------------------------------------------------
    # Agent groups
    # -----------------------------------------------------------------

    async def create_agent_group(self, group: AgentGroup) -> AgentGroup:
        """Persist a new agent group; idempotent on the ``id`` column."""
        ts = (group.created_at or datetime.now(timezone.utc)).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO agent_groups "
            "(id, name, workspace, policy_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                group.id,
                group.name,
                str(group.workspace),
                _serialize_policy(group.policy),
                ts,
            ),
        )
        return dataclasses.replace(
            group, created_at=datetime.fromisoformat(ts)
        )

    async def get_agent_group(self, agent_group_id: str) -> AgentGroup:
        """Load an agent group by id; raises :class:`KeyError` if missing."""
        row = self._conn.execute(
            "SELECT id, name, workspace, policy_json, created_at "
            "FROM agent_groups WHERE id = ?",
            (agent_group_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"agent_group not found: {agent_group_id}")
        return AgentGroup(
            id=row[0],
            name=row[1],
            workspace=Path(row[2]),
            policy=_deserialize_policy(row[3]),
            created_at=datetime.fromisoformat(row[4]),
        )

    # -----------------------------------------------------------------
    # Messaging groups
    # -----------------------------------------------------------------

    async def get_or_create_messaging_group(
        self,
        channel_type: str,
        platform_id: str,
    ) -> MessagingGroup:
        """Find an existing messaging group or insert a new one."""
        row = self._conn.execute(
            "SELECT id, channel_type, platform_id, display_name, is_group, created_at "
            "FROM messaging_groups WHERE channel_type = ? AND platform_id = ?",
            (channel_type, platform_id),
        ).fetchone()
        if row is not None:
            return _row_to_messaging_group(row)

        new_id = str(uuid.uuid4())
        ts = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO messaging_groups (id, channel_type, platform_id, created_at) "
            "VALUES (?, ?, ?, ?)",
            (new_id, channel_type, platform_id, ts),
        )
        return MessagingGroup(
            id=new_id,
            channel_type=channel_type,
            platform_id=platform_id,
            created_at=datetime.fromisoformat(ts),
        )

    async def update_messaging_group_metadata(
        self,
        messaging_group_id: str,
        display_name: str | None = None,
        is_group: bool | None = None,
    ) -> None:
        """Patch metadata fields learned via the adapter's ``on_metadata`` callback."""
        sets: list[str] = []
        params: list = []
        if display_name is not None:
            sets.append("display_name = ?")
            params.append(display_name)
        if is_group is not None:
            sets.append("is_group = ?")
            params.append(1 if is_group else 0)
        if not sets:
            return
        params.append(messaging_group_id)
        self._conn.execute(
            f"UPDATE messaging_groups SET {', '.join(sets)} WHERE id = ?",
            params,
        )

    # -----------------------------------------------------------------
    # Wirings
    # -----------------------------------------------------------------

    async def add_wiring(self, wiring: MessagingGroupAgent) -> None:
        """Insert or replace a messaging-group → agent-group wiring."""
        self._conn.execute(
            "INSERT OR REPLACE INTO messaging_group_agents "
            "(messaging_group_id, agent_group_id, session_mode, "
            " trigger_rule, trigger_pattern, priority) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                wiring.messaging_group_id,
                wiring.agent_group_id,
                wiring.session_mode.value,
                wiring.trigger_rule.value,
                wiring.trigger_pattern,
                wiring.priority,
            ),
        )

    async def get_wirings(self, messaging_group_id: str) -> list[MessagingGroupAgent]:
        """Return every wiring for a messaging group, lowest priority first."""
        rows = self._conn.execute(
            "SELECT messaging_group_id, agent_group_id, session_mode, "
            "trigger_rule, trigger_pattern, priority "
            "FROM messaging_group_agents "
            "WHERE messaging_group_id = ? ORDER BY priority ASC",
            (messaging_group_id,),
        ).fetchall()
        return [
            MessagingGroupAgent(
                messaging_group_id=r[0],
                agent_group_id=r[1],
                session_mode=SessionMode(r[2]),
                trigger_rule=TriggerRule(r[3]),
                trigger_pattern=r[4],
                priority=r[5],
            )
            for r in rows
        ]

    # -----------------------------------------------------------------
    # Sessions
    # -----------------------------------------------------------------

    async def resolve_session(
        self,
        agent_group: AgentGroup,
        wiring: MessagingGroupAgent,
        address: PlatformAddress,
    ) -> Session:
        """Find or create the session matching ``wiring.session_mode``."""
        session_key = _session_key_for(wiring, address)
        row = self._conn.execute(
            "SELECT id, agent_group_id, session_key, store_path, "
            "created_at, last_active_at "
            "FROM sessions WHERE agent_group_id = ? AND session_key = ?",
            (agent_group.id, session_key),
        ).fetchone()

        now = datetime.now(timezone.utc).isoformat()
        if row is not None:
            self._conn.execute(
                "UPDATE sessions SET last_active_at = ? WHERE id = ?",
                (now, row[0]),
            )
            return Session(
                id=row[0],
                agent_group_id=row[1],
                session_key=row[2],
                store_path=Path(row[3]),
                created_at=datetime.fromisoformat(row[4]),
                last_active_at=datetime.fromisoformat(now),
            )

        # Fresh session.
        new_id = str(uuid.uuid4())
        store_path = self.sessions_root / agent_group.id / new_id
        store_path.mkdir(parents=True, exist_ok=True)
        self._conn.execute(
            "INSERT INTO sessions "
            "(id, agent_group_id, session_key, store_path, created_at, last_active_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (new_id, agent_group.id, session_key, str(store_path), now, now),
        )
        return Session(
            id=new_id,
            agent_group_id=agent_group.id,
            session_key=session_key,
            store_path=store_path,
            created_at=datetime.fromisoformat(now),
            last_active_at=datetime.fromisoformat(now),
        )

    async def open_session_store(self, session: Session) -> SessionStoreProtocol:
        """Open the per-session SQLite message store."""
        return SQLiteSessionStore(session.store_path / "session.db")

    # -----------------------------------------------------------------
    # Listing / deletion (control-plane API for the dispatch UI)
    # -----------------------------------------------------------------

    async def list_agent_groups(self) -> list[AgentGroup]:
        """Every persisted agent group, ordered by name."""
        rows = self._conn.execute(
            "SELECT id, name, workspace, policy_json, created_at "
            "FROM agent_groups ORDER BY name COLLATE NOCASE"
        ).fetchall()
        return [
            AgentGroup(
                id=r[0],
                name=r[1],
                workspace=Path(r[2]),
                policy=_deserialize_policy(r[3]),
                created_at=datetime.fromisoformat(r[4]),
            )
            for r in rows
        ]

    async def delete_agent_group(self, agent_group_id: str) -> bool:
        """Remove an agent group. Fails if any wirings or sessions still reference it."""
        used = self._conn.execute(
            "SELECT 1 FROM messaging_group_agents WHERE agent_group_id = ? "
            "UNION SELECT 1 FROM sessions WHERE agent_group_id = ? LIMIT 1",
            (agent_group_id, agent_group_id),
        ).fetchone()
        if used is not None:
            raise ValueError(
                f"agent_group {agent_group_id} still has wirings or sessions; "
                "remove those first."
            )
        cursor = self._conn.execute(
            "DELETE FROM agent_groups WHERE id = ?", (agent_group_id,)
        )
        return cursor.rowcount > 0

    async def collapse_to_single_agent_group(self, keep_id: str) -> int:
        """Re-point every wiring and session onto ``keep_id``, drop the rest.

        One-time migration to the single-agent ("Anton") model: any agent
        group other than ``keep_id`` has its wirings and sessions reassigned,
        then its row is deleted. ``UPDATE OR REPLACE`` collapses the rare
        case where two groups were wired to the same messaging group (or
        held the same session_key) — the duplicate is dropped, not errored.

        Returns the number of agent groups removed. Idempotent: a second
        call finds nothing to collapse and returns ``0``.
        """
        others = [
            r[0]
            for r in self._conn.execute(
                "SELECT id FROM agent_groups WHERE id != ?", (keep_id,)
            ).fetchall()
        ]
        if not others:
            return 0
        self._conn.execute(
            "UPDATE OR REPLACE messaging_group_agents SET agent_group_id = ? "
            "WHERE agent_group_id != ?",
            (keep_id, keep_id),
        )
        self._conn.execute(
            "UPDATE OR REPLACE sessions SET agent_group_id = ? "
            "WHERE agent_group_id != ?",
            (keep_id, keep_id),
        )
        self._conn.executemany(
            "DELETE FROM agent_groups WHERE id = ?",
            [(group_id,) for group_id in others],
        )
        return len(others)

    async def list_messaging_groups(self) -> list[MessagingGroup]:
        """Every messaging group seen so far. Created lazily on first inbound event."""
        rows = self._conn.execute(
            "SELECT id, channel_type, platform_id, display_name, is_group, created_at "
            "FROM messaging_groups ORDER BY created_at DESC"
        ).fetchall()
        return [_row_to_messaging_group(r) for r in rows]

    async def list_wirings(self) -> list[MessagingGroupAgent]:
        """Every wiring across all messaging groups."""
        rows = self._conn.execute(
            "SELECT messaging_group_id, agent_group_id, session_mode, "
            "trigger_rule, trigger_pattern, priority "
            "FROM messaging_group_agents ORDER BY priority ASC"
        ).fetchall()
        return [
            MessagingGroupAgent(
                messaging_group_id=r[0],
                agent_group_id=r[1],
                session_mode=SessionMode(r[2]),
                trigger_rule=TriggerRule(r[3]),
                trigger_pattern=r[4],
                priority=r[5],
            )
            for r in rows
        ]

    async def delete_wiring(
        self,
        messaging_group_id: str,
        agent_group_id: str,
    ) -> bool:
        """Remove a single wiring; returns True if a row was deleted."""
        cursor = self._conn.execute(
            "DELETE FROM messaging_group_agents "
            "WHERE messaging_group_id = ? AND agent_group_id = ?",
            (messaging_group_id, agent_group_id),
        )
        return cursor.rowcount > 0

    async def list_sessions(
        self,
        *,
        agent_group_id: str | None = None,
    ) -> list[Session]:
        """Every session, optionally scoped to one agent group, newest activity first."""
        if agent_group_id is None:
            rows = self._conn.execute(
                "SELECT id, agent_group_id, session_key, store_path, "
                "created_at, last_active_at "
                "FROM sessions ORDER BY last_active_at DESC"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, agent_group_id, session_key, store_path, "
                "created_at, last_active_at "
                "FROM sessions WHERE agent_group_id = ? ORDER BY last_active_at DESC",
                (agent_group_id,),
            ).fetchall()
        return [
            Session(
                id=r[0],
                agent_group_id=r[1],
                session_key=r[2],
                store_path=Path(r[3]),
                created_at=datetime.fromisoformat(r[4]),
                last_active_at=datetime.fromisoformat(r[5]),
            )
            for r in rows
        ]

    async def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session_key_for(
    wiring: MessagingGroupAgent,
    address: PlatformAddress,
) -> str:
    """Compute the deterministic session_key used to resolve sessions.

    Mirrors the three :class:`SessionMode` values:

      - ``AGENT_SHARED``        → ``"agent"`` (one session per agent group).
      - ``PER_MESSAGING_GROUP`` → ``"mg:<messaging_group_id>"``.
      - ``PER_THREAD``          → ``"mg:<messaging_group_id>:thread:<thread_id>"``,
        falling back to per-MG when the platform has no thread.
    """
    if wiring.session_mode is SessionMode.AGENT_SHARED:
        return "agent"
    if wiring.session_mode is SessionMode.PER_THREAD and address.thread_id:
        return f"mg:{wiring.messaging_group_id}:thread:{address.thread_id}"
    return f"mg:{wiring.messaging_group_id}"


def _row_to_messaging_group(row) -> MessagingGroup:
    return MessagingGroup(
        id=row[0],
        channel_type=row[1],
        platform_id=row[2],
        display_name=row[3],
        is_group=bool(row[4]) if row[4] is not None else None,
        created_at=datetime.fromisoformat(row[5]),
    )


def _serialize_policy(policy: PermissionPolicy) -> str:
    return json.dumps(
        {
            "file_scopes": [
                {"path": s.path, "mode": s.mode} for s in policy.file_scopes
            ],
            "network_allowlist": policy.network_allowlist,
            "mcp_allowlist": policy.mcp_allowlist,
            "act_without_asking": policy.act_without_asking,
            "require_approval_for_destructive": policy.require_approval_for_destructive,
            "scheduled_dispatch_allowed": policy.scheduled_dispatch_allowed,
            "scheduled_destructive_blocked": policy.scheduled_destructive_blocked,
        }
    )


def _deserialize_policy(blob: str) -> PermissionPolicy:
    data = json.loads(blob)
    return PermissionPolicy(
        file_scopes=[
            FileScope(path=s["path"], mode=s["mode"])
            for s in data.get("file_scopes", [])
        ],
        network_allowlist=list(data.get("network_allowlist", [])),
        mcp_allowlist=list(data.get("mcp_allowlist", [])),
        act_without_asking=bool(data.get("act_without_asking", False)),
        require_approval_for_destructive=bool(
            data.get("require_approval_for_destructive", True)
        ),
        scheduled_dispatch_allowed=bool(data.get("scheduled_dispatch_allowed", False)),
        scheduled_destructive_blocked=bool(
            data.get("scheduled_destructive_blocked", True)
        ),
    )
