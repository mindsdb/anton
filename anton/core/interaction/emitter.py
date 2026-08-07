"""Out-of-band event path for one turn.

Anything holding a ``ChatSession`` — a tool handler, a sub-agent — can push
an event here, and ``turn_stream`` forwards it to the host. This exists
because a tool runs inside ``await dispatch_tool(...)``, where there is no
access to the generator's yield point.

A concrete class, not a Protocol: ``turn_stream`` constructs it and hosts
never implement it. The queue is deliberately unbounded — with a maxsize,
``emit`` from a tool whose dispatch is not being drained would block
forever with no symptom.
"""

from __future__ import annotations

import asyncio

__all__ = ["TurnEmitter"]


class TurnEmitter:
    def __init__(self) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()

    # ── producer side (tools, sub-agents) ────────────────────────────
    async def emit(self, event) -> None:
        await self._queue.put(event)

    # ── consumer side (turn_stream's drain loop only) ────────────────
    async def get(self):
        return await self._queue.get()

    def empty(self) -> bool:
        return self._queue.empty()

    def get_nowait(self):
        return self._queue.get_nowait()
