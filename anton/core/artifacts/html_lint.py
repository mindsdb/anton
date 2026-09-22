"""Headless-browser lint for `.html` artifacts.

Catches a page that throws on load or references a missing local script —
things nothing else in the pipeline ever checks. Two engines: desktop reuses
cowork's own bundled Electron/Chromium via `ANTON_HTML_LINT_BROWSER` (set by
cowork's Electron main process to `process.execPath`); the cloud sandbox pod
has no Electron/Node, so it falls back to Playwright's Python binding when
that env var isn't set. `_resolve_command` picks between them.

Page load + network policy live in `_html_lint_runner.js` (Electron,
checked in, not agent-authored) and `_html_lint_runner.py` (Playwright) —
see those files for details. Both emit the same JSON contract on stdout.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

_log = logging.getLogger(__name__)

_RUNNER_JS = Path(__file__).with_name("_html_lint_runner.js")
_RUNNER_PY = Path(__file__).with_name("_html_lint_runner.py")
_TIMEOUT_SECONDS_ELECTRON = 8
# Placeholder — a cold Chromium start under gVisor is slower than desktop;
# replace with the real cold-start measurement from plan step 8 before release.
_TIMEOUT_SECONDS_PLAYWRIGHT = 20
_RESULT_START = "RESULT_JSON_START"
_RESULT_END = "RESULT_JSON_END"


@dataclass(frozen=True)
class HtmlFinding:
    """A console error, failed local-asset request, crash, or empty-body render."""

    kind: str  # "console_error" | "failed_request" | "crashed" | "empty_page"
    detail: str

    def message(self) -> str:
        return f"{self.kind}: {self.detail}"


@dataclass(frozen=True)
class HtmlLintRun:
    """Diagnostic detail behind a `lint_html` call — for logging/telemetry, never
    exposed to the agent (the tool-facing contract is `lint_html`'s bare findings).
    """

    engine: str  # "electron" | "playwright" | "none"
    outcome: str  # "clean" | "findings" | "no_result" | "timeout" | "launch_failed"
    findings: list[HtmlFinding] | None
    duration_ms: int


def lint_html(path: Path) -> list[HtmlFinding] | None:
    """Load the page headless and flag console errors + failed local assets.

    Returns `None` when the page could not actually be checked — no browser
    configured, a crash, a timeout, or malformed output — as distinct from
    `[]`, which means it loaded cleanly. Best-effort either way: none of
    this ever raises, so a checker failure can't fail the artifact
    read/cell that triggered it.
    """
    return lint_html_run(path).findings


def lint_html_run(path: Path) -> HtmlLintRun:
    """Same contract as `lint_html`, plus which engine ran and why it didn't
    if it didn't. Never raises, for the same reason `lint_html` never does.
    """
    try:
        run = _lint_html_run(path)
    except Exception:
        run = HtmlLintRun(engine="none", outcome="no_result", findings=None, duration_ms=0)
    _log.info("html lint run: engine=%s outcome=%s duration_ms=%d", run.engine, run.outcome, run.duration_ms)
    return run


def _lint_html_run(path: Path) -> HtmlLintRun:
    resolved = _resolve_command(path)
    if resolved is None:
        return HtmlLintRun(engine="none", outcome="no_result", findings=None, duration_ms=0)
    engine, argv, extra_env, timeout_seconds = resolved

    env = {**os.environ, **extra_env}
    start = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            env=env,
            capture_output=True,
            timeout=timeout_seconds,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return HtmlLintRun(engine=engine, outcome="timeout", findings=None, duration_ms=_elapsed_ms(start))
    except OSError:
        return HtmlLintRun(engine=engine, outcome="launch_failed", findings=None, duration_ms=_elapsed_ms(start))
    duration_ms = _elapsed_ms(start)

    result = _extract_result(proc.stdout)
    if result is None:
        return HtmlLintRun(engine=engine, outcome="no_result", findings=None, duration_ms=duration_ms)

    findings = _findings_from_result(result)
    outcome = "findings" if findings else "clean"
    return HtmlLintRun(engine=engine, outcome=outcome, findings=findings, duration_ms=duration_ms)


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def _resolve_command(path: Path) -> tuple[str, list[str], dict[str, str], int] | None:
    """(engine, argv, extra_env, timeout_seconds), or None when no engine is usable.

    Electron is probed first, so a developer machine that happens to have
    Playwright installed keeps using the already-verified desktop path.
    """
    browser = _discover_browser()
    if browser is not None:
        # Runner goes through both argv and env on purpose: a bare `electron`
        # binary loads argv[1] as the app to run, but a packaged app ignores
        # argv[1] (always loads its own app.asar) and picks the runner up
        # from the env instead, in its lint-mode entry point.
        return (
            "electron",
            [browser, str(_RUNNER_JS), "--headless=new", "--disable-gpu"],
            {"ANTON_HTML_LINT_TARGET": str(path), "ANTON_HTML_LINT_RUNNER": str(_RUNNER_JS)},
            _TIMEOUT_SECONDS_ELECTRON,
        )
    if find_spec("playwright") is not None:
        return (
            "playwright",
            [sys.executable, str(_RUNNER_PY)],
            {"ANTON_HTML_LINT_TARGET": str(path)},
            _TIMEOUT_SECONDS_PLAYWRIGHT,
        )
    return None


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
    if result.get("emptyPage"):
        findings.append(
            HtmlFinding(
                kind="empty_page",
                detail=(
                    "page rendered with no visible text or elements — could be "
                    "intentional (content added later) or a sign the markup/script "
                    "is broken; worth opening to confirm"
                ),
            )
        )
    return findings
