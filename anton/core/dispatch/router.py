"""Dispatch router — the brain that maps platform events to agent sessions.

The router sits between channel adapters and the agent runtime. For every
inbound :class:`InboundEvent` it:

  1. **Resolves the wiring**: looks up which agent groups are wired to the
     event's ``(channel_type, platform_id)`` and applies trigger rules
     (mention-only, regex, always) and priority.
  2. **Resolves the session**: based on the wiring's :class:`SessionMode`
     it picks (or creates) the session whose store will receive the message.
  3. **Persists**: appends the message to ``messages_in`` in the session
     store. The agent runtime, polling its store, picks it up.
  4. **Wakes the runtime**: ensures the runtime for that session is up.
     If it's already running, the new row is enough.

The router is also the **safety gate**. When the agent runtime emits a
*proposed action* (a tool call awaiting approval), the router consults
the agent group's :class:`PermissionPolicy` and:

  - :attr:`GateDecision.ALLOW` → forwards the call.
  - :attr:`GateDecision.DENY`  → returns a synthetic failure to the runtime.
  - :attr:`GateDecision.PROMPT` → emits an :class:`ActionCard` via the
    originating adapter and pauses the runtime until the user clicks.

Outbound delivery uses a separate :func:`run_delivery_loop` task that
polls every active session's store for ``messages_out`` rows and dispatches
them through the originating adapter — keeping the runtime decoupled from
network I/O.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from anton.core.dispatch.adapter import (
    ActionCard,
    ActionResponse,
    InboundEvent,
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
from anton.core.dispatch.policy import (
    GateDecision,
    GateResult,
    PermissionPolicy,
    ProposedAction,
    evaluate as evaluate_policy,
)
from anton.core.dispatch.registry import get_active_adapter
from anton.core.dispatch.session_store import SessionStoreProtocol


_log = logging.getLogger("anton.dispatch.router")


# ---------------------------------------------------------------------------
# Repository protocol — abstracts the central DB
# ---------------------------------------------------------------------------


class DispatchRepository:
    """Persistence interface for the entity model.

    Wraps the central database holding agent groups, messaging groups,
    wirings, and sessions. Kept as a separate Protocol so the router has
    no SQL knowledge — and so cloud deployments can supply a managed
    backend without changes here.

    The concrete file-backed implementation is
    :class:`anton.core.dispatch.repository.SqliteDispatchRepository`.
    """

    async def get_or_create_messaging_group(
        self,
        channel_type: str,
        platform_id: str,
    ) -> MessagingGroup:
        raise NotImplementedError

    async def update_messaging_group_metadata(
        self,
        messaging_group_id: str,
        display_name: str | None = None,
        is_group: bool | None = None,
    ) -> None:
        raise NotImplementedError

    async def get_wirings(self, messaging_group_id: str) -> list[MessagingGroupAgent]:
        raise NotImplementedError

    async def get_agent_group(self, agent_group_id: str) -> AgentGroup:
        raise NotImplementedError

    async def resolve_session(
        self,
        agent_group: AgentGroup,
        wiring: MessagingGroupAgent,
        address: PlatformAddress,
    ) -> Session:
        raise NotImplementedError

    async def open_session_store(self, session: Session) -> SessionStoreProtocol:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Runtime hook — abstracts how sessions are spun up
# ---------------------------------------------------------------------------


class RuntimeOrchestrator:
    """Starts/wakes per-session agent runtimes.

    In-process implementations (tests, CLI) live in
    :mod:`anton.core.dispatch.local_runtime`. Container-backed
    implementations bridge to ``LocalScratchpadRuntime`` or
    scratchpad_service.
    """

    async def wake(self, session: Session, agent_group: AgentGroup) -> None:
        """Ensure a runtime is running for this session.

        Idempotent: if the runtime is already up, returns immediately.
        """
        raise NotImplementedError

    async def stop(self, session: Session) -> None:
        """Tear down a session's runtime (used for pause/cleanup)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Pending action — bookkeeping for ActionCard prompts
# ---------------------------------------------------------------------------


@dataclass
class PendingAction:
    """A gated action awaiting user approval via :class:`ActionCard`.

    The router stashes one of these per emitted card; when an
    :class:`ActionResponse` arrives it's matched by ``question_id`` and
    the awaiting coroutine is unblocked.
    """

    question_id: str
    address: PlatformAddress
    action: ProposedAction
    future: asyncio.Future[bool]


# ---------------------------------------------------------------------------
# The router
# ---------------------------------------------------------------------------


@dataclass
class DispatchRouter:
    """Central dispatch coordinator.

    Holds references to the persistence and runtime layers and the
    in-process pending-action table. Every adapter callback flows through
    here.

    The :attr:`adapter_lookup` field defaults to
    :func:`anton.core.dispatch.registry.get_active_adapter` so the router
    plays nicely with the registry; tests can pass a custom callable to
    bypass the global registry.
    """

    repo: DispatchRepository
    runtime: RuntimeOrchestrator
    adapter_lookup: Callable[[str], object | None] = field(default=get_active_adapter)
    _pending: dict[str, PendingAction] = field(default_factory=dict)
    _delivery_task: asyncio.Task[None] | None = None
    _active_sessions: dict[str, tuple[Session, AgentGroup, PlatformAddress]] = field(
        default_factory=dict
    )
    _stop_delivery: asyncio.Event = field(default_factory=asyncio.Event)

    # -----------------------------------------------------------------
    # Inbound
    # -----------------------------------------------------------------

    async def on_inbound(self, event: InboundEvent) -> None:
        """Adapter → router entry point for inbound platform events."""
        addr = event.address
        mg = await self.repo.get_or_create_messaging_group(
            addr.channel_type, addr.platform_id
        )
        # Opportunistically record DM-vs-group hints from the message.
        if event.message.is_group is not None and mg.is_group is None:
            await self.repo.update_messaging_group_metadata(
                mg.id, is_group=event.message.is_group
            )

        wirings = await self.repo.get_wirings(mg.id)
        if not wirings:
            _log.debug("no wirings for %s/%s", addr.channel_type, addr.platform_id)
            return

        for wiring in wirings:
            agent_group = await self.repo.get_agent_group(wiring.agent_group_id)
            if not matches_trigger(wiring, event, agent_group.name):
                continue

            session = await self.repo.resolve_session(agent_group, wiring, addr)

            # Remember which platform address last produced traffic for
            # this session so outbound delivery knows where to send replies.
            reply_addr = event.reply_to or addr
            self._active_sessions[session.id] = (session, agent_group, reply_addr)

            store = await self.repo.open_session_store(session)
            try:
                store.append(
                    "in",
                    event.message.kind,
                    {
                        "id": event.message.id,
                        "content": event.message.content,
                        "sender_id": event.message.sender_id,
                        "sender_name": event.message.sender_name,
                        "thread_id": addr.thread_id,
                    },
                )
            finally:
                store.close()

            await self.runtime.wake(session, agent_group)

    async def on_metadata(
        self,
        address: PlatformAddress,
        metadata: dict,
    ) -> None:
        """Adapter → router metadata channel.

        Persists display name and group/DM flag when the adapter learns
        them. Tolerant of partial updates.
        """
        mg = await self.repo.get_or_create_messaging_group(
            address.channel_type, address.platform_id
        )
        await self.repo.update_messaging_group_metadata(
            mg.id,
            display_name=metadata.get("display_name"),
            is_group=metadata.get("is_group"),
        )

    # -----------------------------------------------------------------
    # Action cards / gates
    # -----------------------------------------------------------------

    async def evaluate_gate(
        self,
        agent_group: AgentGroup,
        action: ProposedAction,
    ) -> GateResult:
        """Apply :class:`PermissionPolicy` to a proposed runtime action.

        Pure delegation to :func:`anton.core.dispatch.policy.evaluate`,
        kept as a method so future enhancements (per-session overrides,
        rate limiting, cumulative-spend caps) have a hook point.
        """
        return evaluate_policy(agent_group.policy, action)

    async def request_approval(
        self,
        address: PlatformAddress,
        action: ProposedAction,
        prompt: str,
        *,
        timeout_seconds: int | None = 300,
    ) -> bool:
        """Emit an :class:`ActionCard` and wait for the user's choice.

        Returns ``True`` on approve, ``False`` on deny or timeout. Used
        by runtime adapters when :meth:`evaluate_gate` returns
        :attr:`GateDecision.PROMPT`.
        """
        from anton.core.dispatch.adapter import ActionOption

        question_id = str(uuid.uuid4())
        card = ActionCard(
            question_id=question_id,
            prompt=prompt,
            options=[
                ActionOption(id="approve", label="Approve", style="primary"),
                ActionOption(id="deny", label="Deny", style="destructive"),
            ],
            timeout_seconds=timeout_seconds,
        )
        loop = asyncio.get_event_loop()
        future: asyncio.Future[bool] = loop.create_future()
        self._pending[question_id] = PendingAction(
            question_id=question_id,
            address=address,
            action=action,
            future=future,
        )

        await self.emit_action_card(address, card)

        try:
            if timeout_seconds is None:
                return await future
            return await asyncio.wait_for(future, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            self._pending.pop(question_id, None)
            return False

    async def on_action_response(self, response: ActionResponse) -> None:
        """Adapter → router entry point for ActionCard responses."""
        pending = self._pending.pop(response.question_id, None)
        if pending is None:
            _log.debug("no pending action for question_id=%s", response.question_id)
            return
        approved = response.selected_option_id == "approve"
        if not pending.future.done():
            pending.future.set_result(approved)

    async def emit_action_card(
        self,
        address: PlatformAddress,
        card: ActionCard,
    ) -> None:
        """Render an action card via the appropriate adapter."""
        adapter = self.adapter_lookup(address.channel_type)
        if adapter is None:
            _log.warning(
                "no adapter for channel_type=%s; auto-denying card %s",
                address.channel_type,
                card.question_id,
            )
            pending = self._pending.pop(card.question_id, None)
            if pending and not pending.future.done():
                pending.future.set_result(False)
            return
        await adapter.show_action_card(address, card)  # type: ignore[attr-defined]

    # -----------------------------------------------------------------
    # Outbound
    # -----------------------------------------------------------------

    async def deliver(self, message: OutboundMessage) -> None:
        """Forward one outbound message to its adapter.

        Failures (adapter offline, platform error) are logged. The caller
        decides whether to mark the source store row delivered — typically
        :meth:`run_delivery_loop` only marks on success.
        """
        adapter = self.adapter_lookup(message.address.channel_type)
        if adapter is None:
            raise RuntimeError(
                f"no active adapter for channel_type={message.address.channel_type}"
            )
        await adapter.deliver(message)  # type: ignore[attr-defined]

    async def run_delivery_loop(self, *, poll_interval_s: float = 0.5) -> None:
        """Background task: drain ``messages_out`` from active sessions.

        Polls each active session's store and ships any undelivered
        ``out`` rows through :meth:`deliver`. Successful deliveries are
        marked on the store; failures are logged and retried on the next
        pass (the row stays undelivered).
        """
        self._stop_delivery.clear()
        while not self._stop_delivery.is_set():
            for sid, (session, _ag, addr) in list(self._active_sessions.items()):
                try:
                    store = await self.repo.open_session_store(session)
                except Exception as e:
                    _log.warning("delivery: open_session_store failed for %s: %r", sid, e)
                    continue
                try:
                    rows = store.read_undelivered("out")
                    for row in rows:
                        text = ""
                        if isinstance(row.content, dict):
                            text = str(row.content.get("text", ""))
                        elif isinstance(row.content, str):
                            text = row.content
                        if not text:
                            store.mark_delivered(row.rowid, "out")
                            continue
                        out = OutboundMessage(address=addr, text=text)
                        try:
                            await self.deliver(out)
                            store.mark_delivered(row.rowid, "out")
                        except Exception as e:
                            _log.warning(
                                "delivery failed for session=%s row=%d: %r",
                                sid, row.rowid, e,
                            )
                finally:
                    store.close()

            try:
                await asyncio.wait_for(
                    self._stop_delivery.wait(),
                    timeout=poll_interval_s,
                )
            except asyncio.TimeoutError:
                pass

    def start_delivery_loop(self) -> None:
        """Spawn :meth:`run_delivery_loop` as a background task."""
        if self._delivery_task is not None and not self._delivery_task.done():
            return
        self._delivery_task = asyncio.create_task(
            self.run_delivery_loop(),
            name="dispatch-delivery-loop",
        )

    async def stop_delivery_loop(self) -> None:
        """Stop the background delivery loop."""
        self._stop_delivery.set()
        if self._delivery_task is not None:
            try:
                await asyncio.wait_for(self._delivery_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._delivery_task.cancel()
        self._delivery_task = None


# ---------------------------------------------------------------------------
# Trigger filtering — pure helpers, no IO
# ---------------------------------------------------------------------------


def matches_trigger(
    wiring: MessagingGroupAgent,
    event: InboundEvent,
    agent_group_name: str,
) -> bool:
    """Return ``True`` if this wiring should fire for this event.

    Implements the three :class:`TriggerRule` modes:

      - ``ALWAYS`` — every message fires.
      - ``MENTION_ONLY`` — fires when the platform-confirmed
        ``is_mention`` is set, falling back to a case-insensitive
        substring match of the agent's display name in the message
        text. The fallback handles platforms (or adapters) that don't
        report mentions natively.
      - ``REGEX`` — fires when ``wiring.trigger_pattern`` matches
        ``event.message.content`` interpreted as text.
    """
    rule = wiring.trigger_rule
    if rule is TriggerRule.ALWAYS:
        return True

    if rule is TriggerRule.MENTION_ONLY:
        if event.message.is_mention is True:
            return True
        if event.message.is_mention is False:
            return False
        return _text_contains(event.message.content, agent_group_name)

    if rule is TriggerRule.REGEX:
        if not wiring.trigger_pattern:
            return False
        import re

        try:
            return bool(
                re.search(wiring.trigger_pattern, _as_text(event.message.content))
            )
        except re.error:
            return False

    return False


def _as_text(content: object) -> str:
    """Best-effort string view of message content for regex matching."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text = content.get("text") or content.get("content") or ""
        return text if isinstance(text, str) else ""
    return ""


def _text_contains(content: object, needle: str) -> bool:
    """Case-insensitive substring check used by the mention fallback."""
    return needle.lower() in _as_text(content).lower()
