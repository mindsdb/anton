"""Headless-browser lint for `.html` artifacts (ENG-1204 Fix 3).

Catches a page that throws on load or references a missing local script —
things nothing else in the pipeline ever checks. Reuses cowork's own
bundled Electron/Chromium via `ANTON_HTML_LINT_BROWSER` (set by cowork's
Electron main process to `process.execPath`) instead of a new dependency.
Desktop only; silently no-ops wherever the env var isn't set (cloud has no
browser to find).

Page load + network policy live in `_html_lint_runner.js` (checked in, not
agent-authored) — see that file for details.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

_RUNNER_SCRIPT = Path(__file__).with_name("_html_lint_runner.js")
_TIMEOUT_SECONDS = 8
_RESULT_START = "RESULT_JSON_START"
_RESULT_END = "RESULT_JSON_END"


@dataclass(frozen=True)
class HtmlFinding:
    """A console error, failed local-asset request, or crash from a headless load."""

    kind: str  # "console_error" | "failed_request" | "crashed"
    detail: str

    def message(self) -> str:
        return f"{self.kind}: {self.detail}"


def lint_html(path: Path) -> list[HtmlFinding]:
    """Load the page headless and flag console errors + failed local assets.

    Best-effort: no browser configured, a crash, a timeout, or malformed
    output all yield `[]` rather than raising — a checker must never fail
    the artifact read/cell that triggered it.
    """
    try:
        return _lint_html(path)
    except Exception:
        return []


def _lint_html(path: Path) -> list[HtmlFinding]:
    browser = _discover_browser()
    if browser is None:
        return []

    env = {**os.environ, "ANTON_HTML_LINT_TARGET": str(path)}
    try:
        proc = subprocess.run(
            [browser, str(_RUNNER_SCRIPT), "--headless=new", "--disable-gpu"],
            env=env,
            capture_output=True,
            timeout=_TIMEOUT_SECONDS,
            text=True,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []

    result = _extract_result(proc.stdout)
    if result is None:
        return []
    return _findings_from_result(result)


def _discover_browser() -> str | None:
    browser = os.environ.get("ANTON_HTML_LINT_BROWSER")
    if not browser:
        return None
    if os.path.isfile(browser) and os.access(browser, os.X_OK):
        return browser
    return shutil.which(browser)


def _extract_result(stdout: str) -> dict | None:
    start = stdout.find(_RESULT_START)
    end = stdout.find(_RESULT_END)
    if start == -1 or end == -1 or end <= start:
        return None
    blob = stdout[start + len(_RESULT_START) : end].strip()
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _findings_from_result(result: dict) -> list[HtmlFinding]:
    findings: list[HtmlFinding] = []
    for err in result.get("consoleErrors") or []:
        message = err.get("message", "")
        line = err.get("line")
        detail = f"{message} (line {line})" if line else message
        findings.append(HtmlFinding(kind="console_error", detail=detail))
    for req in result.get("failedRequests") or []:
        detail = f"{req.get('url', '?')} — {req.get('error', 'unknown error')}"
        findings.append(HtmlFinding(kind="failed_request", detail=detail))
    if result.get("crashed"):
        findings.append(
            HtmlFinding(kind="crashed", detail="renderer process crashed while loading the page")
        )
    return findings
