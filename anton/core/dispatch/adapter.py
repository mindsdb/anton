"""Channel adapter protocol — the boundary between platforms and Anton.

Adapters bridge messaging platforms (Telegram, Slack, Discord, Gmail, CLI, …)
with the dispatch router. Each adapter is responsible for three things and
three things only:

  1. **Ingress** — receive platform events (webhook / poll / websocket) and
     normalize them into :class:`InboundEvent` objects.
  2. **Filtering** — decide which events warrant agent attention. This may
     be stateless (regex, mention-only) or stateful (e.g. "bot was once
     mentioned in this thread → forward all subsequent messages"). How it
     decides is the adapter's business; the router does not care.
  3. **Egress** — deliver outbound :class:`OutboundMessage` payloads back
     to the platform, including action cards (permission prompts) and
     attachments.

Adapters do **not** know about agent groups, sessions, or vault credentials.
They speak in :class:`PlatformAddress` tuples ``(channel_type, platform_id,
thread_id)``; the router maps those to the entity model.

Two patterns are supported:
  - **Native adapters** implement :class:`ChannelAdapter` directly.
  - **Bridge adapters** wrap an existing SDK (e.g. python-telegram-bot)
    and translate to/from the protocol.

Inspired by nanoclaw's channel adapter design (qwibitai/nanoclaw); the
safety primitives (:class:`PermissionPolicy`, action gates) are modeled on
Claude Cowork's permission semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Literal, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Address & event primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlatformAddress:
    """A platform-level conversation address.

    The router maps ``(channel_type, platform_id, thread_id)`` triples to
    ``(agent_group, session)`` pairs via the entity model. Adapters never
    construct or read agent-group / session IDs — they only deal in
    platform-level identifiers.

    Attributes:
        channel_type: Adapter name (``"telegram"``, ``"slack"``, ``"cli"``).
        platform_id: Conversation identifier on the platform (chat ID, channel ID).
        thread_id: Optional sub-context (Slack thread, GitHub PR comment thread).
            ``None`` means "the conversation as a whole".
    """

    channel_type: str
    platform_id: str
    thread_id: str | None = None


@dataclass
class Attachment:
    """A file/media attachment carried with a message."""

    filename: str
    mime_type: str
    data: bytes | None = None  # inline payload
    url: str | None = None     # or remote reference


MessageKind = Literal["chat", "system", "webhook", "scheduled", "action_response"]


@dataclass
class InboundMessage:
    """Normalized inbound message from a platform.

    Adapters construct these from platform-specific event payloads. The
    ``content`` field is a free-form JSON-serializable object — the router
    forwards it to the agent runtime as-is.
    """

    id: str
    content: Any  # dict / str — host JSON-encodes before persisting
    timestamp: datetime
    kind: MessageKind = "chat"
    sender_id: str | None = None
    sender_name: str | None = None
    is_mention: bool | None = None
    """Platform-confirmed mention signal. ``None`` means the adapter doesn't
    know — the router falls back to text-matching the agent name."""
    is_group: bool | None = None
    """``True`` when the source is a group/channel; ``False`` for DMs."""
    attachments: list[Attachment] = field(default_factory=list)


@dataclass
class InboundEvent:
    """A fully-addressed inbound message handed to the router."""

    address: PlatformAddress
    message: InboundMessage
    reply_to: PlatformAddress | None = None
    """Override delivery destination. Used by admin/CLI transports that want
    to inject a message into one channel but route replies elsewhere. Agents
    cannot set this; only adapters may."""


# ---------------------------------------------------------------------------
# Outbound primitives
# ---------------------------------------------------------------------------


@dataclass
class OutboundMessage:
    """A reply produced by an agent, ready for delivery."""

    address: PlatformAddress
    text: str
    attachments: list[Attachment] = field(default_factory=list)
    reply_to_message_id: str | None = None


@dataclass
class ActionOption:
    """One choice in an action card."""

    id: str
    label: str
    style: Literal["default", "primary", "destructive"] = "default"


@dataclass
class ActionCard:
    """An interactive prompt requiring a user decision.

    Used for Cowork-style permission gates: when an agent attempts a gated
    operation (file deletion, network egress to a non-allowlisted host,
    spending money, etc.), the router holds the call and emits an action
    card via the originating adapter. The agent resumes only after the
    user clicks an option.

    Attributes:
        question_id: Stable identifier; the corresponding
            :class:`ActionResponse` references it.
        prompt: Human-readable question.
        options: Choices presented to the user.
        timeout_seconds: If the user doesn't respond within this window,
            the gated call fails closed (denied). ``None`` = wait forever.
    """

    question_id: str
    prompt: str
    options: list[ActionOption]
    timeout_seconds: int | None = 300


@dataclass
class ActionResponse:
    """User's response to an :class:`ActionCard`."""

    question_id: str
    selected_option_id: str
    user_id: str | None = None
    timestamp: datetime | None = None


# ---------------------------------------------------------------------------
# Adapter protocol
# ---------------------------------------------------------------------------


@dataclass
class ChannelSetup:
    """Callbacks handed to an adapter at startup.

    The adapter retains references to these callbacks and invokes them as
    platform events arrive.
    """

    on_inbound: Callable[[InboundEvent], Awaitable[None]]
    on_metadata: Callable[[PlatformAddress, dict[str, Any]], Awaitable[None]]
    """Called when the adapter learns metadata about a conversation
    (display name, member count, group/DM flag). The router persists this
    for routing decisions and observability."""
    on_action_response: Callable[[ActionResponse], Awaitable[None]]


@runtime_checkable
class ChannelAdapter(Protocol):
    """Structural interface every channel adapter must satisfy.

    Adapters are instantiated by :mod:`anton.core.dispatch.registry` at
    startup and given a :class:`ChannelSetup` to wire up callbacks. They
    run for the lifetime of the dispatch process.
    """

    @property
    def channel_type(self) -> str:
        """Stable adapter name. Used in :class:`PlatformAddress`."""
        ...

    async def setup(self, setup: ChannelSetup) -> None:
        """Begin listening for platform events and remember the callbacks."""
        ...

    async def shutdown(self) -> None:
        """Cleanly stop listening; release platform resources."""
        ...

    async def deliver(self, message: OutboundMessage) -> None:
        """Send a reply back to the platform."""
        ...

    async def show_action_card(
        self,
        address: PlatformAddress,
        card: ActionCard,
    ) -> None:
        """Render an interactive prompt in the originating channel.

        The adapter chooses the platform-native rendering (Telegram inline
        keyboard, Slack block kit, Discord button row, CLI numbered prompt).
        """
        ...
