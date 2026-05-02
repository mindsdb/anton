"""CLI channel adapter — terminal-friendly transport for the dispatcher.

Two operating modes:

  - **Programmatic** (default): Other code calls :meth:`feed_message` to
    inject inbound events and reads :attr:`delivered` to inspect
    outbound replies. This is the workhorse for tests and for embedded
    use (notebook cells, scripted demos).
  - **Interactive**: :func:`run_interactive` pipes stdin/stdout to the
    same adapter, giving you a REPL where each line is a chat message
    and replies print back. Action cards render as numbered prompts.

The CLI adapter is the simplest possible :class:`ChannelAdapter` impl
and serves as the reference for new adapters. It deliberately avoids
async I/O on stdin to keep the example readable.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

from anton.core.dispatch.adapter import (
    ActionCard,
    ActionResponse,
    ChannelAdapter,
    ChannelSetup,
    InboundEvent,
    InboundMessage,
    OutboundMessage,
    PlatformAddress,
)


CLI_CHANNEL_TYPE = "cli"


class CliChannelAdapter(ChannelAdapter):
    """In-process CLI adapter — feeds inbound events and captures replies.

    Attributes:
        delivered: Every :class:`OutboundMessage` the dispatcher sent to
            this adapter. Tests assert against this list.
        cards: Every :class:`ActionCard` the dispatcher emitted. The
            interactive helper picks them up to render numbered prompts;
            tests can answer them via :meth:`respond_to_card`.
        setup_obj: The :class:`ChannelSetup` handed in at startup. Stored
            so :meth:`feed_message` and :meth:`respond_to_card` can call
            back into the router.
    """

    channel_type = CLI_CHANNEL_TYPE

    def __init__(self, default_platform_id: str = "local") -> None:
        self.default_platform_id = default_platform_id
        self.delivered: list[OutboundMessage] = []
        self.cards: list[tuple[PlatformAddress, ActionCard]] = []
        self.setup_obj: ChannelSetup | None = None

    async def setup(self, setup: ChannelSetup) -> None:
        """Remember callbacks; the CLI has no platform connection to open."""
        self.setup_obj = setup

    async def shutdown(self) -> None:
        """No-op — no sockets to close."""
        self.setup_obj = None

    async def deliver(self, message: OutboundMessage) -> None:
        """Capture an outbound message; print it if no listener is registered."""
        self.delivered.append(message)

    async def show_action_card(
        self,
        address: PlatformAddress,
        card: ActionCard,
    ) -> None:
        """Capture a card; tests resolve it via :meth:`respond_to_card`."""
        self.cards.append((address, card))

    # -----------------------------------------------------------------
    # Programmatic API
    # -----------------------------------------------------------------

    async def feed_message(
        self,
        text: str,
        *,
        platform_id: str | None = None,
        thread_id: str | None = None,
        sender_id: str | None = "local-user",
        sender_name: str | None = "local",
        is_mention: bool | None = None,
        is_group: bool | None = False,
    ) -> None:
        """Inject one chat message into the dispatcher.

        Mirrors what a real adapter does on a webhook event: build an
        :class:`InboundEvent` and pass it to ``setup_obj.on_inbound``.
        """
        if self.setup_obj is None:
            raise RuntimeError("CliChannelAdapter not set up; call setup() first")
        event = InboundEvent(
            address=PlatformAddress(
                channel_type=CLI_CHANNEL_TYPE,
                platform_id=platform_id or self.default_platform_id,
                thread_id=thread_id,
            ),
            message=InboundMessage(
                id=str(uuid.uuid4()),
                content={"text": text},
                timestamp=datetime.now(timezone.utc),
                kind="chat",
                sender_id=sender_id,
                sender_name=sender_name,
                is_mention=is_mention,
                is_group=is_group,
            ),
        )
        await self.setup_obj.on_inbound(event)

    async def respond_to_card(
        self,
        question_id: str,
        option_id: str,
        *,
        user_id: str = "local-user",
    ) -> None:
        """Answer a previously-emitted :class:`ActionCard`.

        Looks up the card, calls back into the router via
        ``setup_obj.on_action_response``, and removes it from
        :attr:`cards`.
        """
        if self.setup_obj is None:
            raise RuntimeError("CliChannelAdapter not set up; call setup() first")
        # Drop the card from our queue if present (best-effort).
        self.cards = [(a, c) for (a, c) in self.cards if c.question_id != question_id]
        await self.setup_obj.on_action_response(
            ActionResponse(
                question_id=question_id,
                selected_option_id=option_id,
                user_id=user_id,
                timestamp=datetime.now(timezone.utc),
            )
        )

    def drain_delivered(self) -> list[OutboundMessage]:
        """Return all captured outbound messages and clear the buffer."""
        out, self.delivered = self.delivered, []
        return out


# ---------------------------------------------------------------------------
# Optional interactive REPL helper
# ---------------------------------------------------------------------------


async def run_interactive(
    adapter: CliChannelAdapter,
    *,
    prompt: str = "you> ",
    print_replies: bool = True,
) -> None:
    """Run a tiny stdin/stdout REPL on top of ``adapter``.

    Each line typed becomes a chat message. Replies (already captured by
    :meth:`CliChannelAdapter.deliver`) are flushed after each turn.
    Action cards prompt for ``approve`` / ``deny`` (or any option id).

    This is a convenience for demos — production CLIs will want richer
    rendering (color, multi-line input, history) but the protocol is
    the same.
    """
    loop = asyncio.get_event_loop()
    while True:
        # Drain pending action cards first.
        while adapter.cards:
            address, card = adapter.cards.pop(0)
            print(f"\n[gate] {card.prompt}")
            for opt in card.options:
                print(f"  - {opt.id}: {opt.label}")
            choice = await loop.run_in_executor(None, lambda: input("choice> ").strip())
            await adapter.respond_to_card(card.question_id, choice or "deny")

        try:
            line = await loop.run_in_executor(None, lambda: input(prompt))
        except (EOFError, KeyboardInterrupt):
            print()
            return
        line = line.strip()
        if not line:
            continue
        if line in (":quit", ":exit"):
            return
        await adapter.feed_message(line)

        # Give the runtime a moment to produce a reply, then flush.
        await asyncio.sleep(0.6)
        if print_replies:
            for msg in adapter.drain_delivered():
                print(f"agent> {msg.text}")


def make_cli_adapter_factory(platform_id: str = "local"):
    """Return an adapter factory suitable for :func:`register_channel_adapter`.

    Usage::

        from anton.core.dispatch import register_channel_adapter
        from anton.core.dispatch.channels.cli import make_cli_adapter_factory

        register_channel_adapter("cli", make_cli_adapter_factory())
    """

    async def _factory() -> ChannelAdapter:
        return CliChannelAdapter(default_platform_id=platform_id)

    return _factory
