"""Fire-and-forget anonymous analytics events.

Every call spawns a daemon thread that issues a single GET request to the
configured analytics URL.  The request carries only the action name and a
timestamp — no PII, no payload beyond what the query string contains.

Guarantees:
  • Never blocks the caller.
  • Never raises — all exceptions are silently swallowed.
  • Daemon threads die automatically when the process exits.
"""

from __future__ import annotations

import threading
import time
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anton.config.settings import AntonSettings

_TIMEOUT = 3  # seconds


def send_event(settings: AntonSettings, action: str, **extra: str) -> None:
    """Send an analytics event in a background thread.

    Args:
        settings: Resolved AntonSettings (checked for analytics_enabled / analytics_url).
        action: Event name, e.g. ``"anton_started"``.
        **extra: Additional key=value pairs appended as query parameters.
    """
    try:
        if not settings.analytics_enabled:
            return
        url = settings.analytics_url
        if not url:
            return

        params: dict[str, str] = {
            "action": action,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "_": str(int(time.time() * 1000)),
        }
        params.update(extra)

        full_url = f"{url}?{urllib.parse.urlencode(params)}"

        t = threading.Thread(target=_fire, args=(full_url,), daemon=True)
        t.start()
    except Exception:
        pass


def _fire(url: str) -> None:
    """Perform the actual HTTP GET.  Runs inside a daemon thread."""
    try:
        urllib.request.urlopen(url, timeout=_TIMEOUT)
    except Exception:
        pass
