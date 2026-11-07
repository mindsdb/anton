from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

#: One browser profile for the user unless the host says otherwise, so a
#: login made in one conversation is there in the next.
DEFAULT_PROFILE = "main"

#: Hosts a plain-http browser service may run on: a standalone service on the
#: user's own machine. Anywhere else the API token would cross a network in
#: the clear.
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

#: A project key the service accepts for saved tabs: opaque, 1-64 characters.
_PROJECT_KEY = re.compile(r"[A-Za-z0-9_-]{1,64}")


@dataclass(frozen=True)
class BrowserConfig:
    """Where the user's browser instance is, and which profile to drive.

    Two ways to reach one:

    - MindsHub (``token`` empty): ``base_url`` is the MindsHub browser router,
      e.g. ``https://cowork.mindshub.ai/browser``: one address for everyone,
      and the router finds the caller's own browser. The credential is not
      here: the session resolves it per call from its MindsHub connection
      (the user's key or token on desktop, the turn key in a cloud pod),
      because a turn key can be rotated mid-session.
    - Standalone (``token`` set): ``base_url`` is a browser service someone
      runs themselves, and ``token`` is its ``BROWSER_SERVICE_TOKEN``. No
      MindsHub connection is needed.

    ``project`` keys the tabs saved for the conversation's project; empty
    when there is none, and saved tabs are then unavailable.
    """

    base_url: str
    profile: str = DEFAULT_PROFILE
    # Kept out of repr so a logged config never carries the service token.
    token: str = field(default="", repr=False)
    project: str = ""

    @staticmethod
    def from_dict(raw: object) -> "BrowserConfig | None":
        """Parse a host's ``{"base_url", "profile", "token", "project"}`` block; None if unusable."""
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
        project = raw.get("project")
        if not isinstance(project, str) or not _PROJECT_KEY.fullmatch(project):
            project = ""
        return BrowserConfig(base_url=base_url.rstrip("/"), profile=profile, token=token.strip(), project=project)


def _usable_url(url: str) -> bool:
    """https anywhere; plain http only for a service on this machine."""
    parsed = urlparse(url)
    if parsed.scheme == "https":
        return bool(parsed.hostname)
    return parsed.scheme == "http" and parsed.hostname in _LOCAL_HOSTS
