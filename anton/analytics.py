"""Fire-and-forget anonymous analytics events.

Every call spawns a daemon thread that issues a single GET request to the
configured analytics URL. The request carries only the action name, a
timestamp, and anonymous installation IDs in the query string.
"""

from __future__ import annotations

import hashlib
import threading
import time
import urllib.parse
import urllib.request
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anton.config.settings import AntonSettings

_TIMEOUT = 3  # seconds
_cached_aid: str | None = None
_RESERVED_PARAMS = {"action", "aid", "distinct_id", "timestamp", "_"}


def get_installation_id() -> str:
    """Return a deterministic, anonymous installation ID."""
    global _cached_aid
    if _cached_aid is not None:
        return _cached_aid

    try:
        node = uuid.getnode()
        is_random_fallback = bool(node & (1 << 40))

        if is_random_fallback:
            from pathlib import Path

            path = Path("~/.anton/.installation_id").expanduser()
            if path.is_file():
                _cached_aid = path.read_text(encoding="utf-8").strip()[:16]
            else:
                _cached_aid = uuid.uuid4().hex[:16]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(_cached_aid + "\n", encoding="utf-8")
        else:
            _cached_aid = hashlib.sha256(str(node).encode()).hexdigest()[:16]
    except Exception:
        # Avoid collapsing all failures into a single ID; keep per-process entropy.
        _cached_aid = uuid.uuid4().hex[:16]

    return _cached_aid


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

        aid = get_installation_id()
        params: dict[str, str] = {
            "action": action,
            "aid": aid,
            "distinct_id": aid,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "_": str(int(time.time() * 1000)),
        }
        # Keep analytics identity fields immutable from call sites.
        params.update({k: v for k, v in extra.items() if k not in _RESERVED_PARAMS})

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
