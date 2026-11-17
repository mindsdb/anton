"""Quick edit: a change typed in the artifact viewer's bar, made without a chat turn (ENG-4227).

The server routes a bar message with :func:`router.route_request`; an ``edit``
comes here. :func:`run_quick_edit` runs the same editor loop as
``edit_artifact`` (:class:`ArtifactEditor`, PR #790) on the artifact folder,
with three differences that suit a bar with no agent around it:

* the model may hand the request back (``hand_to_agent``) when it needs data,
  a query or an analysis the files do not have: status ``escalate``, every
  file restored, and the web sends the message to the agent instead;
* up to ``max_fix_rounds`` failed checks (any channel) go back to the model
  with the checker's findings before it gives up;
* earlier bar messages (``history``) come with the request, so "make it
  bigger" knows what "it" is.

``failed`` and ``escalate`` leave the folder exactly as it was. The result's
``summary`` is one short sentence for the user. Nothing here raises except a
cancellation (after the folder has been restored).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from anton.core.artifact_editor.checkers import CheckResult, Checker

__all__ = ["QuickEditResult", "quick_edit_llm_from_settings", "run_quick_edit"]

_log = logging.getLogger(__name__)

QuickStatus = Literal["applied", "escalate", "failed", "noop"]

MAX_HISTORY = 6
MAX_HISTORY_CHARS = 500
MAX_SUMMARY_CHARS = 240


@dataclass
class QuickEditResult:
    status: QuickStatus
    changed: list[str]
    summary: str  # one short user-facing sentence of what changed (or why not)
    escalate_reason: str | None
    check: CheckResult | None  # the final check
    rounds: int  # fix rounds used
    llm_calls: int
    ms: int
    # Beyond the contract: the editor's own words on a failure, for logs (never user-facing).
    error: str = ""


def _sentence(text: str, limit: int = MAX_SUMMARY_CHARS) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[: end + 1] if end > limit // 3 else cut.rstrip() + "..."


def _task(message: str, history: list[dict] | None) -> str:
    """The request, with the earlier bar messages it may refer to."""
    lines = []
    for entry in (history or [])[-MAX_HISTORY:]:
        if not isinstance(entry, dict):
            continue
        text = " ".join(str(entry.get("text") or "").split())
        if not text:
            continue
        who = "user" if entry.get("role") == "user" else "you (the editor)"
        if len(text) > MAX_HISTORY_CHARS:
            text = text[: MAX_HISTORY_CHARS - 3] + "..."
        lines.append(f"- {who}: {text}")
    message = message.strip()
    if not lines:
        return message
    return (
        "Earlier messages in this artifact's quick-edit bar, oldest first (already done; context only, so a "
        "follow-up like \"bigger\" or \"undo that\" can be read against them):\n" + "\n".join(lines)
        + f"\n\nThe request now:\n{message}"
    )


class _StageProgress:
    """The editor's Progress, forwarding stages to ``on_progress`` as ``{"stage": ..., ...}``."""

    def __init__(self, on_progress: Callable[[dict], None] | None) -> None:
        self._on_progress = on_progress
        self.trail: list[str] = []

    def _emit(self, event: dict) -> None:
        if self._on_progress is None:
            return
        try:
            self._on_progress(event)
        except Exception:  # noqa: BLE001 — a host callback failure never fails the edit
            _log.debug("a quick-edit progress callback failed", exc_info=True)

    def stage(self, name: str, **info: Any) -> None:
        self._emit({"stage": name, **info})

    def status(self, text: str) -> None:
        pass

    def log(self, text: str) -> None:
        if len(self.trail) < 200:
            self.trail.append(text.strip())


def _metadata(folder: Path) -> dict:
    try:
        meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        return meta if isinstance(meta, dict) else {}
    except (OSError, ValueError):
        return {}


def _failure_summary(error: str, check: CheckResult | None) -> str:
    if check is not None and check.status == "failed":
        first = check.findings[0].message if check.findings else ""
        first = re.sub(r"\s+", " ", first)[:120]
        return _sentence("I couldn't make that change without breaking the page"
                         + (f" ({first})" if first else "") + ", so the artifact is unchanged.")
    return "I couldn't make that change safely, so the artifact is unchanged."


async def run_quick_edit(
    folder: Path,
    message: str,
    *,
    kind: str,
    llm,
    checker: Checker | None,
    history: list[dict] | None = None,
    on_progress: Callable[[dict], None] | None = None,
    max_fix_rounds: int = 2,
    primary: str | None = None,
    name: str = "",
    role: Literal["planning", "coding"] = "planning",
    images: bool = True,
) -> QuickEditResult:
    """Make the change MESSAGE asks for in the artifact at FOLDER, check it, fix it, or hand it back.

    ``checker`` None picks the environment's checks for ``kind``
    (:func:`select_checker`: static, then a browser when one is configured).
    ``on_progress`` gets ``{"stage": "reading" | "writing" | "checking" |
    "fixing", ...}`` as the edit moves on ("fixing" carries ``round`` and
    ``findings``). ``primary`` / ``name`` default to the folder's metadata.
    """
    from anton.core.artifact_editor.checkers import select_checker
    from anton.core.artifact_editor.editor import ArtifactEditor
    from anton.core.artifact_editor.workspace import ArtifactWorkspace

    started = time.monotonic()
    folder = Path(folder)

    def result(status: QuickStatus, summary: str, **kw: Any) -> QuickEditResult:
        values: dict[str, Any] = {"changed": [], "escalate_reason": None, "check": None, "rounds": 0,
                                  "llm_calls": 0}
        values.update(kw)
        return QuickEditResult(status=status, summary=_sentence(summary),
                               ms=int((time.monotonic() - started) * 1000), **values)

    if not str(message or "").strip():
        return result("noop", "There was nothing to change: the message is empty.")
    meta = _metadata(folder)
    if primary is None:
        candidate = meta.get("primary")
        primary = candidate if isinstance(candidate, str) and candidate and (folder / candidate).is_file() else None
    try:
        ws = ArtifactWorkspace(folder, kind=kind, primary=primary)
    except (ValueError, OSError) as exc:
        return result("failed", "This artifact can't be edited from the bar.", error=str(exc))
    if checker is None:
        checker = select_checker(kind, slug=folder.name)
    progress = _StageProgress(on_progress)
    editor = ArtifactEditor(
        ws, llm, checker, progress, role=role, slug=folder.name, name=name or str(meta.get("name") or ""),
        images=images, escalation=True, max_fix_rounds=max(0, int(max_fix_rounds)),
    )
    try:
        outcome = await editor.edit(_task(message, history))
    except asyncio.CancelledError:
        raise  # the editor has restored every file it wrote
    except Exception as exc:  # noqa: BLE001 — edit() does not raise, but a quick edit must never fail the request
        _log.warning("the quick edit crashed", exc_info=True)
        return result("failed", "I couldn't make that change safely, so the artifact is unchanged.",
                      error=f"{type(exc).__name__}: {exc}")

    common = {"rounds": outcome.fix_rounds, "llm_calls": outcome.model_calls, "check": outcome.check}
    if outcome.status == "escalated":
        reason = _sentence(outcome.reply or "This needs more than editing the artifact's files.")
        return result("escalate", reason, escalate_reason=reason, **common)
    if outcome.status == "restored":
        return result("failed", _failure_summary(outcome.error, outcome.check), error=outcome.error, **common)
    if outcome.status == "answered" or not outcome.files:
        return result("noop", outcome.reply or "Nothing needed to change.", **common)
    summary = outcome.reply or f"Updated {', '.join(outcome.files)}."
    return result("applied", summary, changed=list(outcome.files), **common)


def quick_edit_llm_from_settings(settings) -> object:
    """The editor's LLM client, built from AntonSettings the way ChatSession gets its own.

    ``LLMClient.from_settings`` plus the per-call idle deadline that
    ChatSession applies to its client. Run the editor with
    ``role=editor_role(settings)`` (``ANTON_ARTIFACT_EDITOR_ROLE``).
    """
    from anton.core.llm.client import LLMClient

    llm = LLMClient.from_settings(settings)
    timeout = getattr(settings, "model_call_idle_timeout_s", None)
    if timeout is not None:
        llm.model_call_idle_timeout_s = timeout
    return llm
