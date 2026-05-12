"""Channel adapter registry — discovery and lifecycle.

Adapters self-register at import time via :func:`register_channel_adapter`.
The dispatch service calls :func:`init_channel_adapters` at startup, which
instantiates every registered adapter, hands it a :class:`ChannelSetup`,
and retains the live instance for outbound delivery.

Registration is by ``channel_type`` (the adapter's own stable name). At
most one adapter per ``channel_type`` may be active at a time — the second
``register_channel_adapter("telegram", …)`` call overwrites the first,
matching nanoclaw's behavior and supporting hot-reload during development.

Adapter factories may return ``None`` to indicate missing or invalid
credentials; the registry skips those without crashing the dispatcher.
This lets you ship a build with telegram + slack + discord adapters all
linked in but only run the ones the user has configured.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from anton.core.dispatch.adapter import ChannelAdapter, ChannelSetup


AdapterFactory = Callable[[], Awaitable[ChannelAdapter | None]]


@dataclass
class ChannelRegistration:
    """Everything the registry needs to instantiate an adapter."""

    channel_type: str
    factory: AdapterFactory
    """Async callable returning a ready (but not yet ``setup``-ed) adapter,
    or ``None`` if credentials are missing."""


_registry: dict[str, ChannelRegistration] = {}
_active: dict[str, ChannelAdapter] = {}


def register_channel_adapter(channel_type: str, factory: AdapterFactory) -> None:
    """Register an adapter factory by channel type.

    Called at module import time by adapter modules. Idempotent: re-registering
    the same channel type replaces the prior factory, useful for tests and
    hot-reload.
    """
    _registry[channel_type] = ChannelRegistration(channel_type, factory)


def get_active_adapter(channel_type: str) -> ChannelAdapter | None:
    """Return the live adapter for a channel type, or ``None`` if not running."""
    return _active.get(channel_type)


def get_active_adapters() -> list[ChannelAdapter]:
    """Return all currently-running adapters."""
    return list(_active.values())


def get_registered_channel_types() -> list[str]:
    """Return the names of every registered adapter (running or not)."""
    return list(_registry.keys())


async def init_channel_adapters(
    setup_factory: Callable[[ChannelAdapter], ChannelSetup],
    *,
    network_retry_delays_s: tuple[float, ...] = (2.0, 5.0, 10.0),
) -> None:
    """Instantiate and start every registered adapter.

    Args:
        setup_factory: Called once per adapter to build its
            :class:`ChannelSetup`. Lets the router inject per-adapter
            callbacks (the same router handles all adapters, so we
            don't share one ``ChannelSetup``).
        network_retry_delays_s: Backoff delays for transient network
            failures during ``setup``. Misconfigurations (bad tokens)
            still fail fast — only ``OSError``-like network errors
            trigger retries.
    """
    for channel_type, registration in _registry.items():
        try:
            adapter = await registration.factory()
        except Exception as e:
            # Factory failed entirely — log and continue. Don't take down
            # the dispatcher because one adapter has a bug.
            _log_warn(f"adapter factory failed: {channel_type}: {e!r}")
            continue

        if adapter is None:
            _log_warn(f"adapter credentials missing, skipping: {channel_type}")
            continue

        setup = setup_factory(adapter)
        for attempt, delay in enumerate((0.0, *network_retry_delays_s)):
            if delay:
                await asyncio.sleep(delay)
            try:
                await adapter.setup(setup)
                _active[channel_type] = adapter
                break
            except OSError as e:
                if attempt == len(network_retry_delays_s):
                    _log_warn(
                        f"adapter setup gave up after retries: {channel_type}: {e!r}"
                    )
                    break
            except Exception as e:
                _log_warn(f"adapter setup failed (no retry): {channel_type}: {e!r}")
                break


async def shutdown_channel_adapters() -> None:
    """Cleanly stop every active adapter."""
    for adapter in list(_active.values()):
        try:
            await adapter.shutdown()
        except Exception as e:
            _log_warn(f"adapter shutdown error: {adapter.channel_type}: {e!r}")
    _active.clear()


def _reset_for_tests() -> None:
    """Clear the registry. Test-only helper."""
    _registry.clear()
    _active.clear()


def _log_warn(msg: str) -> None:
    """Lightweight logging hook — replaced by the host logger when wired."""
    # Defer to the standard library so this module has no required deps.
    import logging

    logging.getLogger("anton.dispatch.registry").warning(msg)
