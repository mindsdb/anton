"""Fire-and-forget analytics events.

Every call spawns a daemon thread that issues a single GET request to the
configured analytics URL.  The request carries the action name, a timestamp,
an anonymous machine fingerprint, and whatever the caller passes as
``extra`` query parameters.  No conversation content, ever.

Where these events actually land
================================

The collector relays into PostHog project 355390 ("Anton"), where these events
appear tagged ``source = mindshub-analytics-collector``.  Rows written before
2026-08-11 carry ``source = mindsdb-zoominfo-lambda`` instead: that was a
different collector, in a different AWS account, and installs old enough to
still hold the previous ``analytics_url`` keep reporting to it.  A query
covering both needs to accept either value.

Consequences worth knowing before adding a caller:

* **Every ``extra`` kwarg becomes a queryable PostHog event property**, apart
  from a small denylist of credential- and address-shaped names, and any value
  that looks like an email address.  ``ds_connect_success`` carries its
  ``engine=postgres`` through and ``turn_completed`` carries its full token
  breakdown, with no collector change needed for either.  So the parameter
  names here ARE the analytics schema; renaming one silently breaks whatever
  queries or dashboards read it.
* **This was not true until 2026-08-11.**  The previous collector copied a
  fixed list of five property names and relayed only actions starting with
  a lowercase ``anton_`` or ``ds_connect_``, discarding everything else and
  answering HTTP 200 regardless.  ``turn_completed`` and its 27 properties
  therefore went nowhere at all from the day they shipped.  If a property
  seems to be missing, check what the collector accepts before assuming the
  caller is at fault.
* **``distinct_id`` is the ``aid`` fingerprint, so these events are
  per-INSTALL, not per-user.**  They do not join to the Keycloak ``sub`` that
  the console, the desktop renderer's PostHog client, and the billing mirror
  all key on.  Per-user questions need either that identified client or the
  ``conversation_id`` -> Langfuse -> user hop.
* PostHog additionally enriches server-side with IP and GeoIP.

Identifier policy
=================

Events were originally fingerprint-only.  ``turn_completed`` (the per-turn
cost event) also sends an opaque **join key** — ``conversation_id``, the same
value emitted as ``Langfuse-Session-Id`` — so a turn's cost can be tied to
its trace when investigating a runaway.  The id carries no personal data on
its own; resolving it to a person requires Langfuse access, and Langfuse
already holds the full conversation content for the same session, so this
adds no disclosure that path doesn't already have.

Still never sent, by any caller: message text, prompts, tool output, file
paths, credentials, hostnames, or email addresses.

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

# Sent instead of urllib's default ``Python-urllib/3.x``, which Cloudflare's bot
# protection answers with 403 on the mindshub.ai zone.  The collector host is
# deliberately not proxied, so this is belt and braces rather than the only
# guard, but ``_fire`` discards its response either way: anything that blocks
# this request disappears without a trace, so it is worth not looking like a
# script.
_USER_AGENT = "anton-analytics/1.0"

# Cached after first computation — the fingerprint never changes within
# a process, so computing it once is sufficient.
_cached_aid: str | None = None

# Cached CI detection. Env-derived (no PII), consistent with this module's
# anonymous design. CI/automation traffic is dropped entirely (see send_event)
# rather than tagged, so it can't pollute the product funnel. Driven by an
# explicit Anton-owned signal (ANTON_IS_CI) with known provider markers as a
# convenience fallback; the bare ``CI`` var is intentionally not consulted —
# it's frequently set to "false" or leaks into local dev shells (ENG-385).
_cached_is_ci: bool | None = None


def _env_true(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _is_ci() -> bool:
    """Return True for Anton automation/CI traffic (cached, env-only)."""
    global _cached_is_ci
    if _cached_is_ci is None:
        _cached_is_ci = (
            _env_true("ANTON_IS_CI")
            or _env_true("GITHUB_ACTIONS")
            or _env_true("GITLAB_CI")
            or _env_true("BUILDKITE")
            or _env_true("CIRCLECI")
            or _env_true("TF_BUILD")
            or bool(os.environ.get("JENKINS_URL"))
        )
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

    CI/automation traffic (ANTON_IS_CI, or a known CI provider) is dropped
    rather than sent, so it can't pollute the product funnel (no PII either way).
    """
    try:
        if not settings.analytics_enabled:
            return
        # Drop CI/automation traffic entirely — no value in product analytics
        # from CI runs, and dropping avoids a per-query exclusion filter.
        if _is_ci():
            return
        url = settings.analytics_url
        if not url:
            return

        params: dict[str, str] = {
            "action": action,
            "aid": get_installation_id(),
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
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        urllib.request.urlopen(req, timeout=_TIMEOUT)
    except Exception:
        pass
