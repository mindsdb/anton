"""Channel-adapter infrastructure shared across Slack, WhatsApp, and future channels.

The submodules are framework-agnostic helpers (operate on bytes / dict[str, str])
so the bridge stays testable without FastAPI request fixtures.

Modules
-------
signatures   Webhook HMAC verifiers (Slack v0, WhatsApp Cloud, generic SHA-256).
text         Outbound text splitting honoring per-platform character limits.
secrets      DataVault token loading via the canonical DS_<CHANNEL>_<ACCOUNT>__* layout.
bridge       :class:`ChatBridgeBase` — base class channel adapters extend.
"""
from .bridge import ChatBridgeBase, PartialSendError, WebhookHandshake
from .secrets import (
    MissingChannelSecret,
    load_channel_secrets,
    require_secret,
    secret_var_name,
)
from .signatures import (
    SignatureMismatch,
    verify_hmac_sha256,
    verify_slack,
    verify_whatsapp,
)
from .text import split_for_limit

__all__ = [
    "ChatBridgeBase",
    "MissingChannelSecret",
    "PartialSendError",
    "SignatureMismatch",
    "WebhookHandshake",
    "load_channel_secrets",
    "require_secret",
    "secret_var_name",
    "split_for_limit",
    "verify_hmac_sha256",
    "verify_slack",
    "verify_whatsapp",
]
