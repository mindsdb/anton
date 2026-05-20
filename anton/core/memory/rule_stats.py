"""Per-rule retrieval and outcome counters — Layer 3 of the ACC path.

Tracks two things per rule:

  - **retrievals**: how often a rule has been selected into a turn's
    system prompt by the ranker (Phase B).
  - **ignored**: how often a rule was loaded into the prompt AND the
    pattern it warns against fired anyway (Phase C — outcome bridge).

Stored as a sidecar JSON file (``rules.stats.json`` next to
``rules.md``) rather than inline in the markdown metadata. Two reasons
for the sidecar:

  1. Writing per-rule counters to ``rules.md`` would require parsing,
     mutating, and re-emitting the whole markdown file on every turn.
     A JSON sidecar is one ``json.dump`` away from atomic.
  2. The stats are operationally cheap and high-churn; the rule text
     is canon. Keeping them separate means a typo in stats can't
     corrupt the canonical store.

Rule identity is ``sha256(rule.text.strip().lower())[:16]``. Stable
for the rule's lifetime; if the consolidator rewrites a rule, the
hash changes and its counters reset. Acceptable v1 trade-off — v2
should attach a stable UUID in the rule's HTML-comment metadata so
edits preserve identity. Without that, large-scale rephrasing by the
consolidator would zero out the retrieval data we want to use to
*decide* which rules to keep.

Concurrency: ``flush()`` writes via ``.tmp`` + ``os.replace`` under
``fcntl.flock(LOCK_EX)``. Same shape every other anton storage layer
already uses; the inherited POSIX-only constraint is intentional.

Usage pattern: callers should mutate via ``record_retrieval`` /
``record_ignored`` and call ``flush()`` once at end of the operation
(typically once per ``build_memory_context()``). Don't flush after
every record — that's one disk write per rule retrieved per turn,
which is silly when the natural batch boundary is the turn itself.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any


_STATS_VERSION = 1


def rule_id(text: str) -> str:
    """Stable 64-bit hash of the rule text.

    Public so test fixtures, debug surfaces, and the eventual
    ``/memory --rankings`` view can compute the same ID without
    duplicating the hashing logic. Changes if the text changes — see
    module docstring for the v1 trade-off.
    """
    norm = (text or "").strip().lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _blank_record() -> dict[str, Any]:
    return {
        "retrievals": 0,
        "ignored": 0,
        "last_retrieved": None,
    }


class RuleStats:
    """Lightweight stats sidecar with a buffer-and-flush write pattern.

    Reads on construction (best-effort: a corrupt file becomes a fresh
    state and gets overwritten on next flush, rather than crashing the
    turn). Mutations buffer in memory; ``flush()`` writes atomically
    when the buffer is dirty. Callers should flush once per turn
    rather than after every record.
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        self._data: dict[str, Any] = {"version": _STATS_VERSION, "rules": {}}
        self._dirty = False
        self._load()

    # ── persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # Corrupted or unreadable — keep fresh in-memory state and
            # overwrite cleanly on next flush. Logging is intentionally
            # absent here; an unreadable stats file shouldn't be a
            # turn-stopping incident.
            return
        if isinstance(raw, dict) and isinstance(raw.get("rules"), dict):
            self._data = raw
            self._data.setdefault("version", _STATS_VERSION)

    def flush(self) -> None:
        """Persist the in-memory state if dirty. Idempotent."""
        if not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            json.dump(self._data, f, indent=2, sort_keys=True)
        os.replace(tmp, self._path)
        self._dirty = False

    # ── mutators ─────────────────────────────────────────────────────

    def record_retrieval(self, rule_text: str) -> None:
        """Bump the retrieval counter for ``rule_text`` and update its
        ``last_retrieved`` timestamp. No-op on empty input."""
        if not (rule_text or "").strip():
            return
        rid = rule_id(rule_text)
        rec = self._data["rules"].setdefault(rid, _blank_record())
        rec["retrievals"] = int(rec.get("retrievals", 0)) + 1
        rec["last_retrieved"] = _now_iso()
        self._dirty = True

    def record_ignored(self, rule_text: str) -> None:
        """Bump the ignored counter — used by Phase C (outcome bridge)
        when an ACC detector fires for a pattern whose corresponding
        rule WAS loaded into this turn's prompt. No-op on empty input."""
        if not (rule_text or "").strip():
            return
        rid = rule_id(rule_text)
        rec = self._data["rules"].setdefault(rid, _blank_record())
        rec["ignored"] = int(rec.get("ignored", 0)) + 1
        self._dirty = True

    # ── readers ──────────────────────────────────────────────────────

    def get(self, rule_text: str) -> dict[str, Any]:
        """Stats for one rule. Returns a blank record if absent —
        callers don't need to distinguish 'never seen' from 'zero
        counters' for ranking-tiebreak purposes."""
        rid = rule_id(rule_text)
        return dict(self._data["rules"].get(rid, _blank_record()))

    def all(self) -> dict[str, dict[str, Any]]:
        """Snapshot of every recorded rule keyed by id. Used by tests
        and the eventual ``/memory --rankings`` debug surface."""
        return {rid: dict(rec) for rid, rec in self._data["rules"].items()}
