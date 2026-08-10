"""Structured result contract for side-effecting tools (ENG-696).

Side-effecting tools (publish, create/update artifact, launch backend) used to
return a human-readable string or an ad-hoc JSON dict. Neither made the
*committed state* machine-readable: a consumer (eval, monitoring, the model
itself) could not reliably tell what was committed, where it lives, or whether a
retry would duplicate it.

`SideEffectResult` is the single envelope every such tool returns. It carries:

- `success` — did the side effect commit.
- `resource_id` — stable identity of the committed resource (slug / report id).
- `external_url` — where the resource can be reached, if it has a URL.
- `idempotency_key` — stable key for the operation; a re-run with the same key
  targets the same resource (dedup handle), so retries are recognisable.
- `committed_at` — ISO-8601 UTC instant the side effect committed; `None` when
  nothing was committed (validation failure, pre-commit error).
- `content_hash` — hash of the committed content, when the tool has content.
- `details` — tool-specific machine-readable fields that don't fit the common
  ones above (e.g. launch_backend's `port` / `pid` / `log_path`); `None` when
  the tool has no extras.
- `message` — the human-readable line kept for the model / desktop UI.

It serialises to a JSON string inside a `ToolOutcome` and sets `ToolOutcome.ok`
from `success`, so the ENG-1276 error streak keys on the explicit verdict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

from anton.core.tools.registry import ToolOutcome


def now_iso() -> str:
    """Current instant as an ISO-8601 UTC string (the `committed_at` basis)."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SideEffectResult:
    """Machine-readable outcome of a side-effecting tool call (see module doc)."""

    success: bool
    message: str
    resource_id: str | None = None
    external_url: str | None = None
    idempotency_key: str | None = None
    committed_at: str | None = None
    content_hash: str | None = None
    details: dict | None = None

    def to_outcome(self, reason: str = "") -> ToolOutcome:
        """Render to a `ToolOutcome`: JSON payload as content, verdict as `ok`."""
        payload = {
            "success": self.success,
            "message": self.message,
            "resource_id": self.resource_id,
            "external_url": self.external_url,
            "idempotency_key": self.idempotency_key,
            "committed_at": self.committed_at,
            "content_hash": self.content_hash,
            "details": self.details,
        }
        return ToolOutcome(
            content=json.dumps(payload, indent=2),
            ok=self.success,
            reason=reason,
        )

    @classmethod
    def failed(cls, message: str, reason: str = "") -> ToolOutcome:
        """Shorthand for a non-committing failure (`committed_at` stays None)."""
        return cls(success=False, message=message).to_outcome(reason=reason)
