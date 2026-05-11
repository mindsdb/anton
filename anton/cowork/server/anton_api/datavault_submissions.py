"""In-memory staging area for `data-vault-form` submissions.

The chat conversation never carries credential VALUES — only form
ids and submission ids. When the user fills a form in the side
panel, the cowork frontend posts the values here; we hand back a
submission id, the conversation continues with that id, and Anton's
tool fetches the values just-in-time when it actually needs them
to test a connection.

Entries TTL after `_TTL_SECONDS` so abandoned conversations don't
keep credential material in process memory indefinitely. The store
is process-local — restarting the server drops every staged
submission, which is intentional: persisted credentials live in the
real datasource vault (saved via the existing /v1/datasources POST)
once a connection succeeds.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Optional


logger = logging.getLogger(__name__)

# 24h — long enough for the user to walk away and resume after a
# meeting; short enough that abandoned conversations don't leak
# credentials forever. Cleanup runs lazily on every read/write.
_TTL_SECONDS = 24 * 60 * 60

_store: dict[str, dict[str, Any]] = {}


def _now() -> float:
    return time.time()


def _purge_expired(now: float | None = None) -> int:
    """Drop any entry older than `_TTL_SECONDS`. Called lazily."""
    threshold = (now if now is not None else _now()) - _TTL_SECONDS
    stale = [sid for sid, e in _store.items() if e.get("created_at", 0) < threshold]
    for sid in stale:
        _store.pop(sid, None)
    return len(stale)


def stage_submission(
    *,
    form_id: str,
    conversation_id: Optional[str],
    values: dict[str, Any],
    skipped: list[str] | None = None,
) -> str:
    """Stage a form submission. Returns a submission id the chat
    continuation can reference; Anton's tool calls
    `get_submission(sid)` to retrieve the values when it needs them.
    """
    _purge_expired()
    submission_id = "sub_" + uuid.uuid4().hex[:12]
    _store[submission_id] = {
        "submission_id": submission_id,
        "form_id": form_id,
        "conversation_id": conversation_id,
        "values": dict(values or {}),
        "skipped": list(skipped or []),
        "created_at": _now(),
        "status": "received",
    }
    return submission_id


def get_submission(submission_id: str) -> dict[str, Any] | None:
    _purge_expired()
    entry = _store.get(submission_id)
    if entry is None:
        return None
    # Return a shallow copy so callers can't mutate the live entry.
    return {**entry, "values": dict(entry.get("values", {}))}


def consume_submission(submission_id: str) -> dict[str, Any] | None:
    """Like get_submission but also removes the entry from the store.
    Use after the values have been applied so they don't linger.
    """
    _purge_expired()
    entry = _store.pop(submission_id, None)
    return entry
