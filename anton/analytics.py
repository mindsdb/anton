"""Fire-and-forget anonymous analytics events.

Every call spawns a daemon thread that issues a single GET request to the
configured analytics URL.  The request carries only the action name, a
timestamp, and an anonymous machine fingerprint — no PII, no payload
beyond what the query string contains.

Guarantees:
  • Never blocks the caller.
  • Never raises — all exceptions are silently swallowed.
  • Daemon threads die automatically when the process exits.

Machine fingerprint
===================

Each event includes an ``aid`` (Anton Installation ID) — a deterministic
SHA-256 hash of the machine's MAC address (``uuid.getnode()``).  This is:

  • **Anonymous**: the hash is one-way; the raw MAC never leaves the
    device.  No hostname, no platform, no PII.
  • **Stateless**: no file on normal machines — computed from the MAC.
  • **Stable**: changes only if the primary network adapter changes.

Fallback for Docker / containers: if Python can't find a real MAC
(detected via the multicast bit), a random UUID is persisted to
``~/.anton/.installation_id`` so it stays stable across restarts.
File I/O only happens in this edge case — normal desktops never
touch disk.

The ``aid`` is truncated to 16 hex characters (~64 bits of entropy) —
enough to be collision-free across millions of installations, short
enough to be a readable query parameter.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
import urllib.parse
import urllib.request
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anton.config.settings import AntonSettings

_TIMEOUT = 3  # seconds

# Cached after first computation — the fingerprint never changes within
# a process, so computing it once is sufficient.
_cached_aid: str | None = None

# Cached CI detection. Env-derived (no PII), so it stays consistent with this
# module's anonymous design while letting the funnel filter out CI/automation
# traffic via the ``is_ci`` flag (ENG-385). There is no email/staff signal on
# this path — events are keyed on the anonymous ``aid``, not a user account.
_cached_is_ci: bool | None = None

# CI markers across common providers. ``CI`` is set by virtually all of them;
# the rest catch providers that don't, or set it inconsistently.
_CI_ENV_VARS = (
    "CI",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "BUILDKITE",
    "CIRCLECI",
    "JENKINS_URL",
    "TF_BUILD",
)


def _is_ci() -> bool:
    """Return True when running under CI/automation (cached, env-only)."""
    global _cached_is_ci
    if _cached_is_ci is None:
        _cached_is_ci = any(os.environ.get(v) for v in _CI_ENV_VARS)
    return _cached_is_ci


def get_installation_id() -> str:
    """Return a deterministic, anonymous machine fingerprint.

    The fingerprint is a truncated SHA-256 of the MAC address on normal
    machines. If no real MAC is available (Docker containers with stripped
    networking), a random UUID is persisted to ``~/.anton/.installation_id``
    as a one-time fallback. Computed once per process and cached.

    Returns:
        A 16-character hex string (64 bits of entropy).
    """
    global _cached_aid
    if _cached_aid is not None:
        return _cached_aid

    try:
        node = uuid.getnode()
        is_random_fallback = bool(node & (1 << 40))  # multicast bit = Python faked it

        if is_random_fallback:
            # No real MAC (e.g. Docker with stripped networking).
            # Persist a UUID to disk so it's stable across restarts.
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
        _cached_aid = "unknown"
    return _cached_aid


def send_event(settings: "AntonSettings", action: str, **extra: str) -> None:
    """Send an analytics event in a background thread.

    Args:
        settings: Resolved AntonSettings (checked for analytics_enabled / analytics_url).
        action: Event name, e.g. ``"anton_started"``.
        **extra: Additional key=value pairs appended as query parameters.

    Every event also carries ``is_ci`` (env-derived, no PII) so CI/automation
    traffic can be excluded from the product funnel.
    """
    try:
        if not settings.analytics_enabled:
            return
        url = settings.analytics_url
        if not url:
            return

        params: dict[str, str] = {
            "action": action,
            "aid": get_installation_id(),
            "is_ci": "true" if _is_ci() else "false",
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
