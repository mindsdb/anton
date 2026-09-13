"""Channel adapter data contract — the boundary between platforms and Anton.

The orchestration layer that used to live alongside this contract (adapter
registry, router, repository, permission policy, session store, local
runtime) was never wired into any production entrypoint in this repo or in
cowork-server and has been removed. cowork-server built its own parallel
`PluginRegistry`/`LiveAdapterRegistry` (`cowork/channels/registry.py`,
`runtime.py`) against these same data types instead of consuming the
removed orchestration.

Public surface:

  - :class:`ChannelAdapter`, :class:`InboundEvent`, :class:`OutboundMessage`,
    :class:`ActionCard`, :class:`ActionResponse` — the adapter contract.
"""

from anton.core.dispatch.adapter import (
    ActionCard,
    ActionOption,
    ActionResponse,
    Attachment,
    ChannelAdapter,
    ChannelSetup,
    InboundEvent,
    InboundMessage,
    MessageKind,
    OutboundMessage,
    PlatformAddress,
)

__all__ = [
    "ActionCard",
    "ActionOption",
    "ActionResponse",
    "Attachment",
    "ChannelAdapter",
    "ChannelSetup",
    "InboundEvent",
    "InboundMessage",
    "MessageKind",
    "OutboundMessage",
    "PlatformAddress",
]
