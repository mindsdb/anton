from __future__ import annotations

from abc import ABC, abstractmethod

from anton.events.types import AntonEvent


class Channel(ABC):
    @abstractmethod
    async def emit(self, event: AntonEvent) -> None: ...

    @abstractmethod
    async def prompt(self, question: str) -> str: ...

    @abstractmethod
    async def close(self) -> None: ...
