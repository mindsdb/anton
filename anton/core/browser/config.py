from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlparse

#: One browser profile for the user unless the host says otherwise, so a
#: login made in one conversation is there in the next.
DEFAULT_PROFILE = "main"

#: Hosts a plain-http browser service may run on: a standalone service on the
#: user's own machine. Anywhere else the API token would cross a network in
#: the clear.
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


@dataclass(frozen=True)
class BrowserConfig:
    """Where the user's browser instance is, and which profile to drive.

    Two ways to reach one:

    - MindsHub (``token`` empty): ``base_url`` is the user's instance, e.g.
      ``https://br-ab12cd34.4nton.ai``. The credential is not here: the
      session resolves it per call from its MindsHub connection (the user's
      key or token on desktop, the turn key in a cloud pod), because a turn
      key can be rotated mid-session.
    - Standalone (``token`` set): ``base_url`` is a browser service someone
      runs themselves, and ``token`` is its ``BROWSER_SERVICE_TOKEN``. No
      MindsHub connection is needed.
    """

    base_url: str
    profile: str = DEFAULT_PROFILE
    # Kept out of repr so a logged config never carries the service token.
    token: str = field(default="", repr=False)

    @staticmethod
    def from_dict(raw: object) -> "BrowserConfig | None":
        """Parse a host's ``{"base_url", "profile", "token"}`` block; None if unusable."""
        if not isinstance(raw, dict):
            return None
        base_url = raw.get("base_url")
        if not isinstance(base_url, str) or not _usable_url(base_url):
            return None
        profile = raw.get("profile")
        if not isinstance(profile, str) or not profile:
            profile = DEFAULT_PROFILE
        token = raw.get("token")
        if not isinstance(token, str):
            token = ""
        return BrowserConfig(base_url=base_url.rstrip("/"), profile=profile, token=token.strip())


def _usable_url(url: str) -> bool:
    """https anywhere; plain http only for a service on this machine."""
    parsed = urlparse(url)
    if parsed.scheme == "https":
        return bool(parsed.hostname)
    return parsed.scheme == "http" and parsed.hostname in _LOCAL_HOSTS
