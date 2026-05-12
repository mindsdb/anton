"""DataVault helpers for channel adapter credentials.

Channel tokens follow the same ``DS_<CHANNEL>_<ACCOUNT>__<FIELD>`` env-var
layout used by the rest of Anton's data vault, so a Slack workspace's bot
token sits at, e.g., ``DS_SLACK_PROD__BOT_TOKEN``.

Adapters call :func:`load_channel_secrets` at setup time to materialize the
field map. The function reads ``os.environ`` after :class:`LocalDataVault`
has injected credentials — so it works regardless of which path put the
values there (Anton CLI, cowork's data-vault wiring, anton_services bundle).

Security contract
-----------------
* No filesystem fallback. No non-prefixed env-var fallback. A miss returns
  an empty dict so the factory pattern (return ``None`` adapter) keeps
  working; a request that actually NEEDS a secret should call
  :func:`require_secret` or the bridge's ``require_secret()`` method
  instead of ``.get()`` so the absence raises loudly with a clear error.
* Account isolation: the prefix slug applies uppercase + non-alphanumeric
  collapse, so ``DS_SLACK_ACMEA__*`` and ``DS_SLACK_ACMEB__*`` are
  disjoint name-spaces — secrets configured for one account cannot bleed
  into another.
"""
from __future__ import annotations

import os
import re

_SAFE_RE = re.compile(r"[^A-Z0-9]+")


class MissingChannelSecret(KeyError):
    """Raised when a required ``DS_<CHANNEL>_<ACCOUNT>__<FIELD>`` is unset.

    Carries the canonical env-var name so the operator can fix the config
    without having to grep for the slug rules.
    """

    def __init__(self, channel_type: str, account: str, field: str) -> None:
        self.channel_type = channel_type
        self.account = account
        self.field = field
        self.var_name = secret_var_name(channel_type, account, field)
        super().__init__(
            f"channel secret {self.var_name!s} is not set "
            f"(channel={channel_type}, account={account}, field={field})"
        )


def _slug(value: str) -> str:
    """Uppercase + collapse non-alphanumerics into single underscores."""
    return _SAFE_RE.sub("_", value.upper()).strip("_")


def secret_var_name(channel_type: str, account: str, field: str) -> str:
    """Compose the canonical env var name for a channel credential field.

    >>> secret_var_name("slack", "prod-workspace", "bot_token")
    'DS_SLACK_PROD_WORKSPACE__BOT_TOKEN'
    """
    prefix = f"DS_{_slug(channel_type)}_{_slug(account)}"
    return f"{prefix}__{_slug(field)}"


def load_channel_secrets(channel_type: str, account: str) -> dict[str, str]:
    """Return all ``DS_<CHANNEL>_<ACCOUNT>__*`` env vars as a flat field→value map.

    Field names come back lowercased; values are unmodified. Returns an empty
    dict if no matching vars are set — channels treat that as "credentials
    missing, skip startup" via the registry's None-factory pattern.

    Never falls back to non-prefixed env vars or filesystem reads — a miss
    here means "this account isn't configured", not "borrow from somewhere
    else." Use :func:`require_secret` at the point of use when a specific
    field is mandatory.
    """
    prefix = f"DS_{_slug(channel_type)}_{_slug(account)}__"
    out: dict[str, str] = {}
    for key, value in os.environ.items():
        if key.startswith(prefix) and value:
            field = key[len(prefix):].lower()
            if field:
                out[field] = value
    return out


def require_secret(
    secrets: dict[str, str],
    field: str,
    *,
    channel_type: str,
    account: str,
) -> str:
    """Return ``secrets[field]`` stripped, or raise :class:`MissingChannelSecret`.

    Never returns an empty string — an empty / whitespace-only value is
    treated as missing so a half-configured operator setup fails fast
    instead of silently calling a platform API with a blank token.
    """
    raw = secrets.get(field)
    if raw is None:
        raise MissingChannelSecret(channel_type, account, field)
    stripped = raw.strip()
    if not stripped:
        raise MissingChannelSecret(channel_type, account, field)
    return stripped
