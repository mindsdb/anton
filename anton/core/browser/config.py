from __future__ import annotations

from dataclasses import dataclass

#: One browser profile for the user unless the host says otherwise, so a
#: login made in one conversation is there in the next.
DEFAULT_PROFILE = "main"


@dataclass(frozen=True)
class BrowserConfig:
    """Where the user's browser instance is, and which profile to drive.

    ``base_url`` is the instance origin, e.g. ``https://br-ab12cd34.4nton.ai``.
    The credential is not here: the session resolves it per call from its
    MindsHub connection (the user's key or token on desktop, the turn key in
    a cloud pod), because a turn key can be rotated mid-session.
    """

    base_url: str
    profile: str = DEFAULT_PROFILE

    @staticmethod
    def from_dict(raw: object) -> "BrowserConfig | None":
        """Parse a host's ``{"base_url": ..., "profile": ...}`` block; None if unusable."""
        if not isinstance(raw, dict):
            return None
        base_url = raw.get("base_url")
        if not isinstance(base_url, str) or not base_url.startswith("https://"):
            return None
        profile = raw.get("profile")
        if not isinstance(profile, str) or not profile:
            profile = DEFAULT_PROFILE
        return BrowserConfig(base_url=base_url.rstrip("/"), profile=profile)
