"""Entity model — agent groups, messaging groups, and the wiring between them.

The dispatch system separates **what an agent is** (an agent group with a
filesystem, memory, CLAUDE.md, and policy) from **where it's reachable**
(a messaging group: a specific Telegram chat, Slack channel, or CLI
session). The wiring between them — :class:`MessagingGroupAgent` — encodes
the *isolation mode*, lifted from nanoclaw's three-level model.

Three isolation modes:

  1. **Shared session** — multiple messaging groups feed one conversation.
     Webhook + chat use case (GitHub events arriving alongside Slack
     discussion). Session lookup ignores the messaging group.
  2. **Same agent, separate sessions** — one agent identity, independent
     threads. Personal multi-platform use case. Workspace and memory are
     shared; conversation context is not.
  3. **Separate agent groups** — full isolation. Different memory, CLAUDE.md,
     workspace, and container. The cross-channel privacy boundary.

The first two are configurations of the same agent group; the third is
simply two agent groups that don't know about each other. So the entity
model only needs the first two as enum values — separate-agent-groups is
implicit in having distinct ``agent_group_id`` values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from anton.core.dispatch.policy import PermissionPolicy


class SessionMode(str, Enum):
    """How sessions are resolved within an agent group."""

    AGENT_SHARED = "agent-shared"
    """All messaging groups wired to this agent group share one session.
    Used for shared-session isolation (level 1)."""

    PER_MESSAGING_GROUP = "per-messaging-group"
    """Each messaging group gets its own session within the agent group.
    Used for same-agent-separate-sessions isolation (level 2). Default."""

    PER_THREAD = "per-thread"
    """Each ``(messaging_group, thread_id)`` pair gets its own session.
    Useful for platforms with first-class threads (Slack, Discord forum
    channels) when each thread should have independent context."""


class TriggerRule(str, Enum):
    """When the router considers a message worth waking the agent for."""

    ALWAYS = "always"
    """Every inbound message triggers the agent. Right for DMs and
    dedicated bot channels."""

    MENTION_ONLY = "mention-only"
    """Only messages where ``is_mention`` is ``True`` (or the agent's
    name appears, as a fallback) trigger the agent. Right for shared
    channels where the bot is a participant, not the focus."""

    REGEX = "regex"
    """A custom regex must match the message content. ``trigger_pattern``
    on the wiring carries the pattern."""


@dataclass
class AgentGroup:
    """An agent identity — the unit of memory, workspace, and policy.

    Attributes:
        id: Stable identifier, used as the directory name under
            ``groups/`` and as the routing key for sessions.
        name: Human display name. Used in mention-matching fallback when
            the platform doesn't supply ``is_mention``.
        workspace: Filesystem root for this agent's CLAUDE.md, skills,
            and any persistent files.
        policy: Permission policy governing what this agent may do.
        created_at: Creation timestamp.
    """

    id: str
    name: str
    workspace: Path
    policy: PermissionPolicy = field(default_factory=PermissionPolicy)
    created_at: datetime | None = None


@dataclass
class MessagingGroup:
    """A specific conversation on a platform — a chat ID, channel ID, etc.

    The ``(channel_type, platform_id)`` pair is the natural key. ``thread_id``
    is *not* part of the key here — threads are resolved at session time
    based on the wiring's :class:`SessionMode`.

    Attributes:
        id: Internal stable ID (router uses this for joins).
        channel_type: Adapter name (``"telegram"``, ``"slack"``).
        platform_id: Platform's own conversation identifier.
        display_name: Human-readable name learned via ``on_metadata``.
            May be ``None`` until the adapter reports it.
        is_group: ``True`` for group/channel, ``False`` for DM.
        created_at: Creation timestamp.
    """

    id: str
    channel_type: str
    platform_id: str
    display_name: str | None = None
    is_group: bool | None = None
    created_at: datetime | None = None


@dataclass
class MessagingGroupAgent:
    """Wiring: a messaging group routed to an agent group.

    A given messaging group may be wired to multiple agent groups (one
    Slack channel hosting two different bots), and an agent group may be
    wired to multiple messaging groups (one bot present in many places).
    The cross-product is the *messaging_group_agents* table.

    Attributes:
        messaging_group_id: FK to :class:`MessagingGroup`.
        agent_group_id: FK to :class:`AgentGroup`.
        session_mode: How sessions are resolved for this wiring.
        trigger_rule: When inbound messages wake this agent.
        trigger_pattern: Regex source when ``trigger_rule == REGEX``.
        priority: When multiple wirings match, lower numbers win. Lets
            you have a "default" agent (priority=100) plus a specialist
            (priority=10) that handles only its regex.
    """

    messaging_group_id: str
    agent_group_id: str
    session_mode: SessionMode = SessionMode.PER_MESSAGING_GROUP
    trigger_rule: TriggerRule = TriggerRule.ALWAYS
    trigger_pattern: str | None = None
    priority: int = 100


@dataclass
class Session:
    """A live conversation with persistent message history.

    Sessions are derived from wirings — you don't create them directly,
    the router resolves or creates them when an inbound event arrives.

    Attributes:
        id: Stable session identifier.
        agent_group_id: Which agent owns this session.
        session_key: The deterministic key used to resolve this session
            (``"agent:<id>"`` for shared, ``"mg:<id>"`` for per-MG,
            ``"mg:<id>:thread:<tid>"`` for per-thread). Lets the router
            recover the same session on restart.
        store_path: Filesystem path to the per-session message store.
        created_at: Creation timestamp.
        last_active_at: Updated whenever a message is appended.
    """

    id: str
    agent_group_id: str
    session_key: str
    store_path: Path
    created_at: datetime | None = None
    last_active_at: datetime | None = None
