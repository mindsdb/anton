from __future__ import annotations

import asyncio

from anton.events.types import AntonEvent


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[AntonEvent]] = []

    async def publish(self, event: AntonEvent) -> None:
        for queue in self._subscribers:
            await queue.put(event)

    def subscribe(self) -> asyncio.Queue[AntonEvent]:
        queue: asyncio.Queue[AntonEvent] = asyncio.Queue()
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[AntonEvent]) -> None:
        self._subscribers.remove(queue)
