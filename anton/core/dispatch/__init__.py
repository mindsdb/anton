"""Anton Dispatch — bring agents to messaging platforms safely.

Dispatch is Anton's bridge between external messaging platforms (Telegram,
Slack, Discord, Gmail, CLI, …) and agent runtimes. It combines:

  - **Channel adapters + isolation modes** — modeled on nanoclaw's design:
    pluggable adapters, three isolation levels (shared session /
    same-agent-separate-sessions / separate agent groups), per-session
    SQLite as the single IO surface.
  - **Cowork-style safety** — modeled on Claude Cowork's permission model:
    declarative :class:`PermissionPolicy` per agent group, action-card
    prompts for destructive actions, conservative defaults for scheduled
    dispatch.

Public surface:

  - :class:`ChannelAdapter`, :class:`InboundEvent`, :class:`OutboundMessage`,
    :class:`ActionCard`, :class:`ActionResponse` — the adapter contract.
  - :class:`AgentGroup`, :class:`MessagingGroup`, :class:`MessagingGroupAgent`,
    :class:`Session`, :class:`SessionMode`, :class:`TriggerRule` — entities.
  - :class:`PermissionPolicy`, :class:`GateDecision`, :func:`evaluate` —
    safety gates.
  - :class:`SQLiteSessionStore`, :class:`SessionStoreProtocol`,
    :func:`open_store` — message persistence.
  - :class:`DispatchRouter`, :class:`DispatchRepository`,
    :class:`RuntimeOrchestrator`, :func:`matches_trigger` — orchestration.
  - :class:`SqliteDispatchRepository` — concrete repository.
  - :class:`InProcessRuntimeOrchestrator` — concrete runtime for in-process
    agents (tests, CLI demos).
  - :func:`register_channel_adapter`, :func:`init_channel_adapters` —
    adapter discovery.
"""

from anton.core.dispatch.adapter import (
    ActionCard,
    ActionOption,
    ActionResponse,
    Attachment,
    ChannelAdapter,
    ChannelSetup,
    InboundEvent,
    InboundMessage,
    MessageKind,
    OutboundMessage,
    PlatformAddress,
)
from anton.core.dispatch.entities import (
    AgentGroup,
    MessagingGroup,
    MessagingGroupAgent,
    Session,
    SessionMode,
    TriggerRule,
)
from anton.core.dispatch.local_runtime import (
    AgentCallable,
    InProcessRuntimeOrchestrator,
    LocalScratchpadOrchestrator,
)
from anton.core.dispatch.policy import (
    FileScope,
    GateDecision,
    GateResult,
    PermissionPolicy,
    ProposedAction,
    evaluate,
)
from anton.core.dispatch.registry import (
    AdapterFactory,
    ChannelRegistration,
    get_active_adapter,
    get_active_adapters,
    get_registered_channel_types,
    init_channel_adapters,
    register_channel_adapter,
    shutdown_channel_adapters,
)
from anton.core.dispatch.repository import SqliteDispatchRepository
from anton.core.dispatch.router import (
    DispatchRepository,
    DispatchRouter,
    PendingAction,
    RuntimeOrchestrator,
    matches_trigger,
)
from anton.core.dispatch.session_store import (
    Direction,
    SQLiteSessionStore,
    SessionStoreProtocol,
    StoredMessage,
    open_store,
)

__all__ = [
    # adapter
    "ActionCard",
    "ActionOption",
    "ActionResponse",
    "Attachment",
    "ChannelAdapter",
    "ChannelSetup",
    "InboundEvent",
    "InboundMessage",
    "MessageKind",
    "OutboundMessage",
    "PlatformAddress",
    # entities
    "AgentGroup",
    "MessagingGroup",
    "MessagingGroupAgent",
    "Session",
    "SessionMode",
    "TriggerRule",
    # policy
    "FileScope",
    "GateDecision",
    "GateResult",
    "PermissionPolicy",
    "ProposedAction",
    "evaluate",
    # registry
    "AdapterFactory",
    "ChannelRegistration",
    "get_active_adapter",
    "get_active_adapters",
    "get_registered_channel_types",
    "init_channel_adapters",
    "register_channel_adapter",
    "shutdown_channel_adapters",
    # repository
    "SqliteDispatchRepository",
    # router
    "DispatchRepository",
    "DispatchRouter",
    "PendingAction",
    "RuntimeOrchestrator",
    "matches_trigger",
    # local_runtime
    "AgentCallable",
    "InProcessRuntimeOrchestrator",
    "LocalScratchpadOrchestrator",
    # session_store
    "Direction",
    "SQLiteSessionStore",
    "SessionStoreProtocol",
    "StoredMessage",
    "open_store",
]
