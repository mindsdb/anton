"""The artifact editor's own record: one JSON line per event worth reading later.

Yolocoder's lesson (``recordFailure`` in ``internal/agent/agent.go``): the one
thing worth keeping from a turn that went wrong is the patch the model sent and
what the rejection said, and full traces are never on when you need them. So a
rejected patch is ALWAYS written here, whatever tracing is configured, and so
is the summary of every run (the ticket's EDIT_RUN / EDIT_ATTEMPT rows).

Where: ``ANTON_ARTIFACT_EDITOR_LOG`` when set (a file path), else
``<artifact folder's parent>/.artifact-editor.jsonl``, i.e. beside the
artifacts, never inside one (an artifact folder is the user's content).
Best-effort: a record that cannot be written is logged and dropped, never
raised into the edit.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["EditRecord", "LOG_ENV_VAR", "record_path"]

_log = logging.getLogger(__name__)

LOG_ENV_VAR = "ANTON_ARTIFACT_EDITOR_LOG"
FILE_NAME = ".artifact-editor.jsonl"
# A patch longer than this is cut in the record: the record is for reading why
# a patch failed, and the head of it says that.
MAX_PATCH_CHARS = 20_000
MAX_DETAIL_CHARS = 8_000


def record_path(folder: Path) -> Path:
    configured = os.environ.get(LOG_ENV_VAR, "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(folder).parent / FILE_NAME


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n... ({len(text) - limit} more characters)"


@dataclass
class EditRecord:
    """Writes one artifact editor run's lines to the JSONL record."""

    folder: Path
    task: str
    slug: str = ""
    conversation_id: str = ""
    run_id: str = field(default_factory=lambda: os.urandom(6).hex())
    _path: Path | None = None

    def __post_init__(self) -> None:
        self._path = record_path(self.folder)

    def _write(self, kind: str, payload: dict[str, Any]) -> None:
        line = {
            "kind": kind,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "run_id": self.run_id,
            "slug": self.slug or Path(self.folder).name,
            "conversation_id": self.conversation_id,
            **payload,
        }
        try:
            assert self._path is not None
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
        except Exception:  # noqa: BLE001 — a record must never fail an edit
            _log.warning("could not write the artifact editor record to %s", self._path, exc_info=True)

    def patch_rejected(
        self, *, attempt: int, patch: str, paths: list[str], reasons: list[dict[str, Any]], detail: str,
        placed: list[str],
    ) -> None:
        """A patch that did not (fully) apply. Always written."""
        self._write("patch_rejected", {
            "task": _clip(self.task, 2000),
            "attempt": attempt,
            "paths": paths,
            "placed_files": placed,
            "reasons": reasons,
            "patch": _clip(patch, MAX_PATCH_CHARS),
            "detail": _clip(detail, MAX_DETAIL_CHARS),
        })

    def check(self, *, channel: str, status: str, ms: int, findings: list[dict[str, Any]], note: str) -> None:
        self._write("check", {"channel": channel, "status": status, "ms": ms, "findings": findings[:20],
                              "note": _clip(note, 2000)})

    def run(self, **summary: Any) -> None:
        """The run's outcome: status, model calls, attempts, files, usage."""
        self._write("edit_run", {"task": _clip(self.task, 2000), **summary})
