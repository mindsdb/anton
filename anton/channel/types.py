from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anton.channel.base import Channel


class ChannelCapability(str, Enum):
    TEXT_OUTPUT = "text_output"
    TEXT_INPUT = "text_input"
    INTERACTIVE = "interactive"
    RICH_FORMATTING = "rich"


@dataclass
class ChannelMeta:
    id: str
    label: str
    description: str
    capabilities: list[ChannelCapability]
    icon: str = ""
    aliases: list[str] = field(default_factory=list)


@dataclass
class ChannelInfo:
    meta: ChannelMeta
    factory: type[Channel]
