from __future__ import annotations

from anton.channel.types import ChannelInfo


class ChannelRegistry:
    def __init__(self) -> None:
        self._channels: dict[str, ChannelInfo] = {}
        self._aliases: dict[str, str] = {}

    def register(self, info: ChannelInfo) -> None:
        self._channels[info.meta.id] = info
        for alias in info.meta.aliases:
            self._aliases[alias] = info.meta.id

    def get(self, channel_id: str) -> ChannelInfo | None:
        if channel_id in self._channels:
            return self._channels[channel_id]
        resolved_id = self._aliases.get(channel_id)
        if resolved_id is not None:
            return self._channels.get(resolved_id)
        return None

    def list_all(self) -> list[ChannelInfo]:
        return list(self._channels.values())

    def resolve(self, channel_id: str) -> ChannelInfo:
        info = self.get(channel_id)
        if info is None:
            raise ValueError(f"Unknown channel: {channel_id!r}")
        return info
