"""Webhook HMAC verifiers for inbound channel events.

Each verifier is a pure function that takes raw bytes + headers and either
returns the verified payload bytes, or raises :class:`SignatureMismatch`.
Verifiers never log or read the request — that's the caller's job.

Constant-time comparison is mandatory; never use ``==`` on signatures.
"""
from __future__ import annotations

import hashlib
import hmac
import time

DEFAULT_REPLAY_WINDOW_S = 5 * 60  # Slack's recommended window


class SignatureMismatch(Exception):
    """Raised when a webhook payload's HMAC doesn't match the expected secret."""


def _hex_eq(a: str, b: str) -> bool:
    """Constant-time hex-string equality."""
    return hmac.compare_digest(a.encode("ascii"), b.encode("ascii"))


def verify_hmac_sha256(
    body: bytes,
    *,
    secret: str,
    expected: str,
    prefix: str = "",
) -> None:
    """Generic HMAC-SHA256 verifier.

    Computes ``HMAC(secret, body).hex()`` and compares against ``expected``,
    which may carry an optional ``prefix`` (e.g. ``"sha256="``).
    """
    if not secret:
        raise SignatureMismatch("missing signing secret")
    if not expected:
        raise SignatureMismatch("missing signature header")

    bare = expected[len(prefix):] if prefix and expected.startswith(prefix) else expected
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    if not _hex_eq(digest, bare):
        raise SignatureMismatch("HMAC mismatch")


def verify_slack(
    body: bytes,
    *,
    signing_secret: str,
    timestamp: str,
    signature: str,
    replay_window_s: int = DEFAULT_REPLAY_WINDOW_S,
    now: float | None = None,
) -> None:
    """Verify an inbound Slack request.

    Signature scheme is ``v0=hex(HMAC-SHA256(secret, "v0:{ts}:{body}"))``.
    Reject requests older than ``replay_window_s`` seconds to bound replay.

    See https://api.slack.com/authentication/verifying-requests-from-slack
    """
    if not signing_secret:
        raise SignatureMismatch("missing slack signing secret")
    if not timestamp:
        raise SignatureMismatch("missing X-Slack-Request-Timestamp")
    if not signature.startswith("v0="):
        raise SignatureMismatch("invalid signature version")

    try:
        ts = int(timestamp)
    except ValueError as exc:
        raise SignatureMismatch("non-integer timestamp") from exc

    current = int(now if now is not None else time.time())
    if abs(current - ts) > replay_window_s:
        raise SignatureMismatch("timestamp outside replay window")

    base = f"v0:{timestamp}:".encode("utf-8") + body
    digest = hmac.new(signing_secret.encode("utf-8"), base, hashlib.sha256).hexdigest()
    if not _hex_eq(f"v0={digest}", signature):
        raise SignatureMismatch("HMAC mismatch")


def verify_whatsapp(
    body: bytes,
    *,
    app_secret: str,
    signature_header: str,
) -> None:
    """Verify an inbound WhatsApp Cloud webhook.

    Signature header is ``X-Hub-Signature-256: sha256=<hex>``.
    See https://developers.facebook.com/docs/graph-api/webhooks/getting-started#event-notifications
    """
    verify_hmac_sha256(
        body,
        secret=app_secret,
        expected=signature_header,
        prefix="sha256=",
    )
