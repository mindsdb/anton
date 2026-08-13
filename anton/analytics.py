"""Fire-and-forget analytics events.

Every call spawns a daemon thread that issues a single GET request to the
configured analytics URL.  The request carries the action name, a timestamp,
an anonymous machine fingerprint, and whatever the caller passes as
``extra`` query parameters.  No conversation content, ever.

Where these events actually land
================================

Two sinks, chosen per event name — see ``_POSTHOG_EVENTS``.

**Direct to PostHog, project 424726 ("MindsHub main").**  Events named in
``_POSTHOG_EVENTS`` POST straight to the Capture API.  Every property
survives, because nothing sits in between to drop them.

**Everything else goes to the collector lambda,** which relays into PostHog
project 355390 ("Anton") — verified 2026-08-06, where those events appear
tagged ``source = mindsdb-zoominfo-lambda``.  That path filters **twice**:
it relays only actions beginning ``anton_`` or ``ds_connect`` (a prefix
rule, case-sensitive), and of the properties only ``action``, ``aid``,
``engine``, ``llm_provider`` and ``has_mdb_key`` survive.  It returns
**HTTP 200 while dropping**, and ``_fire`` never reads the response, so a
caller cannot tell (ENG-1355).

This docstring asserted the opposite until ENG-1495 — it claimed every
``extra`` kwarg became a queryable property, "not an allowlist".  That was
false, and it is the likeliest reason ``turn_completed`` shipped with a sink
that produced nothing at all: it told the author no collector work was
needed.  Do not restore that claim for either path.

Consequences worth knowing before adding a caller:

* **Choose the sink deliberately.**  A new event left on the collector path
  arrives with its properties stripped unless they are the five above, and
  does not arrive at all unless its name carries one of the two prefixes.
* **``distinct_id`` is the ``aid`` fingerprint on both paths, so these events
  are per-INSTALL, not per-user.**  They do not join to the Keycloak ``sub``
  that the console, the desktop renderer's PostHog client, and the billing
  mirror all key on.  Per-user questions need either that identified client
  or the ``conversation_id`` -> Langfuse -> user hop.
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
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anton.config.settings import AntonSettings

logger = logging.getLogger(__name__)

_TIMEOUT = 3  # seconds

# ── The PostHog sink (ENG-1495) ─────────────────────────────────────
# The register of events that bypass the collector and post straight to
# PostHog, each with its own properties documented — the same shape as
# cowork's renderer register (``src/renderer/cowork/lib/analytics.js``), so
# both clients read the same way.
#
#   turn_completed   ended_by, tokens_total, input_tokens, output_tokens,
#                    cache_read_tokens, cache_creation_tokens, llm_calls,
#                    rounds, continuations, peak_context_tokens, duration_ms,
#                    {planning,coding,router}_{model,tokens,calls},
#                    unknown_{tokens,calls}, llm_provider, harness,
#                    anton_version, conversation_id, turn_index
#
#   rule_retrieval   outcome, when_rules, kept_rules, rules_chars,
#                    stop_reason, input_tokens, output_tokens, duration_ms
#                    (anton/core/memory/cortex.py::_emit_rule_retrieval)
#
# An event NOT listed here keeps the collector path, so moving one is an
# explicit decision rather than something that happens by default.
_POSTHOG_EVENTS = frozenset({"turn_completed", "rule_retrieval"})

# `$lib` names the sender, matching the convention the other emitters follow
# (`cowork-desktop`, `mindshub-site-beacon`), so a per-emitter breakdown in
# project 424726 stays honest.
_LIB = "anton-library"

# urllib's default ``Python-urllib/3.x`` is answered with 403 by some edges.
# Named for its one consumer (``_fire_posthog``) rather than the module: ``_fire``
# still sends urllib's default, and a bare ``_USER_AGENT`` here collides with the
# identically-valued constant anton#333 adds for the collector path, dragging
# ``_POSTHOG_EVENTS`` into a conflict region it has nothing to do with.
_POSTHOG_USER_AGENT = "anton-posthog/1.0"

# Transport noise, not analytics: appended as a cache buster for the GET path.
# Carries no meaning as a property, so the PostHog payload drops it.
_CACHE_BUSTER_KEY = "_"

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


def _posthog_body(key: str, action: str, params: dict[str, str]) -> bytes:
    """Build the PostHog Capture API payload for one event.

    ``aid`` becomes ``distinct_id`` and is kept as a property too, so a query
    can group on the install without touching person data.  ``timestamp`` is
    promoted to PostHog's own field: it is the moment the turn ended, not the
    moment a queued daemon thread got around to sending, and for a cost event
    that difference is the one you would go on to plot.

    Deliberately no ``$insert_id``.  The natural key would be
    ``(conversation_id, turn_index)``, but an abandoned turn's books and a
    later retry can legitimately share both, and dropping that row would lose
    exactly the runaway a cancel was investigating (anton#309 review).
    ``TurnCost.emitted`` already stops the same books emitting twice, so
    dedupe here could only add a way to lose real events.
    """
    properties = {
        k: v for k, v in params.items()
        if k not in (_CACHE_BUSTER_KEY, "action", "timestamp")
    }
    properties["$lib"] = _LIB
    # Store the event without creating a Person for the install fingerprint.
    # Without this every `aid` becomes a "person" in project 424726, which is
    # keyed on the Keycloak `sub` — so machine fingerprints would be counted
    # alongside real identified users in every person metric and cohort there,
    # and no later query can separate them again.
    properties["$process_person_profile"] = False

    return json.dumps(
        {
            "api_key": key,
            "event": action,
            "distinct_id": params.get("aid") or "unknown",
            "timestamp": params.get("timestamp"),
            "properties": properties,
        }
    ).encode()


def _fire_posthog(url: str, body: bytes) -> None:
    """POST one Capture payload.  Runs inside a daemon thread.

    Reads the status and logs a non-2xx at debug.  Still fire-and-forget — no
    retry, no raise, no effect on the caller — but the failure leaves a trace on
    the machine where it happened instead of only in an aggregate nobody has
    built yet.

    **What this does and does not catch.**  Measured against the live endpoint,
    2026-08-13:

        bogus api_key      -> HTTP 200  {"status":"Ok"}     <- NOT detectable
        real api_key       -> HTTP 200  {"status":"Ok"}
        malformed payload  -> HTTP 400  "failed to hydrate events..."
        absent  api_key    -> HTTP 401  "event submitted without an api_key"

    So this catches a malformed payload and any transport failure, and it
    **cannot** catch a wrong, rotated or revoked project token: PostHog accepts
    an invalid key with 200 and drops the event server-side.  The 401 is
    unreachable from here — ``send_event`` returns early when the key is empty,
    so a request without one is never built.  A token problem is therefore
    invisible from this side by construction, which is exactly why the
    zero-volume alert on ``turn_completed`` is not optional.

    Note the rejection is caught as ``HTTPError``, not read off a response.
    ``urlopen`` **raises** on 4xx/5xx rather than returning something with a
    ``.status`` to inspect — an earlier version of this function checked the
    status inside a ``with`` block that a rejection never reaches, and its test
    passed only because the stub returned where urllib raises.
    """
    try:
        # Inside the try on purpose: `Request()` raises ValueError on a malformed
        # URL, and a bad `ANTON_POSTHOG_HOST` is a user-supplied value. This runs
        # in a daemon thread, outside `send_event`'s guard, so anything escaping
        # here reaches the user as a traceback mid-session — which would break
        # this module's stated guarantee that it never raises.
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "User-Agent": _POSTHOG_USER_AGENT,
            },
        )
        urllib.request.urlopen(request, timeout=_TIMEOUT)
    except urllib.error.HTTPError as exc:
        # The status is worth more than a traceback to whoever is reading logs
        # on a user's machine.
        logger.debug("posthog capture rejected: HTTP %s", exc.code)
    except Exception:
        logger.debug("posthog capture failed", exc_info=True)


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

        params: dict[str, str] = {
            "action": action,
            "aid": get_installation_id(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            _CACHE_BUSTER_KEY: str(int(time.time() * 1000)),
        }
        params.update(extra)

        if action in _POSTHOG_EVENTS:
            # POST, not GET: a query string is copied into gateway, CDN and WAF
            # access logs, which is how the sibling endpoint ended up with real
            # email addresses in them (ENG-1355 §2). A body is not.
            key = getattr(settings, "posthog_key", "") or ""
            host = (getattr(settings, "posthog_host", "") or "").rstrip("/")
            if not key or not host:
                return
            t = threading.Thread(
                target=_fire_posthog,
                args=(f"{host}/capture/", _posthog_body(key, action, params)),
                daemon=True,
            )
        else:
            url = settings.analytics_url
            if not url:
                return
            t = threading.Thread(
                target=_fire,
                args=(f"{url}?{urllib.parse.urlencode(params)}",),
                daemon=True,
            )
        t.start()
    except Exception:
        pass


def _fire(url: str) -> None:
    """Perform the actual HTTP GET.  Runs inside a daemon thread."""
    try:
        urllib.request.urlopen(url, timeout=_TIMEOUT)
    except Exception:
        pass
