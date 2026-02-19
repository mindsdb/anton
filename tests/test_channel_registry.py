from __future__ import annotations

import pytest

from anton.channel.base import Channel
from anton.channel.registry import ChannelRegistry
from anton.channel.types import ChannelCapability, ChannelInfo, ChannelMeta


class _DummyChannel(Channel):
    async def emit(self, event):
        pass

    async def prompt(self, question):
        return ""

    async def close(self):
        pass


def _make_info(
    id: str = "test",
    label: str = "Test",
    aliases: list[str] | None = None,
) -> ChannelInfo:
    return ChannelInfo(
        meta=ChannelMeta(
            id=id,
            label=label,
            description="A test channel",
            capabilities=[ChannelCapability.TEXT_OUTPUT],
            aliases=aliases or [],
        ),
        factory=_DummyChannel,
    )


class TestChannelRegistry:
    def test_register_and_get_by_id(self):
        reg = ChannelRegistry()
        info = _make_info(id="cli")
        reg.register(info)
        assert reg.get("cli") is info

    def test_get_by_alias(self):
        reg = ChannelRegistry()
        info = _make_info(id="cli", aliases=["terminal", "term"])
        reg.register(info)
        assert reg.get("terminal") is info
        assert reg.get("term") is info

    def test_get_unknown_returns_none(self):
        reg = ChannelRegistry()
        assert reg.get("nonexistent") is None

    def test_list_all_empty(self):
        reg = ChannelRegistry()
        assert reg.list_all() == []

    def test_list_all_populated(self):
        reg = ChannelRegistry()
        a = _make_info(id="a")
        b = _make_info(id="b")
        reg.register(a)
        reg.register(b)
        result = reg.list_all()
        assert len(result) == 2
        assert a in result
        assert b in result

    def test_resolve_success(self):
        reg = ChannelRegistry()
        info = _make_info(id="cli", aliases=["term"])
        reg.register(info)
        assert reg.resolve("cli") is info
        assert reg.resolve("term") is info

    def test_resolve_raises_for_unknown(self):
        reg = ChannelRegistry()
        with pytest.raises(ValueError, match="Unknown channel"):
            reg.resolve("missing")
