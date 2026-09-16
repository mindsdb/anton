"""Checked-in runner for html_lint.py's cloud (Playwright) path — not agent-authored.

Python peer of `_html_lint_runner.js`: same JSON contract on stdout, same
network policy and console/crash semantics, so `_findings_from_result` in
`html_lint.py` cannot tell which engine produced its input. Target file path
comes via ANTON_HTML_LINT_TARGET (env, matching the JS runner).

Run as a subprocess, never imported: `sync_playwright()` raises if called
from a thread that already has a running asyncio loop, which is exactly the
thread `lint_changed_artifact_files` runs on.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import ConsoleMessage, Error as PlaywrightError, Page, Request, Route, sync_playwright

_SETTLE_MS = 500
_NAV_TIMEOUT_MS = 10_000
_HARD_TIMEOUT_MS = 15_000  # safety net; the real enforcement is the caller's subprocess timeout
_RESULT_START = "RESULT_JSON_START"
_RESULT_END = "RESULT_JSON_END"

_EMPTY_PAGE_CHECK = Path(__file__).with_name("_html_lint_empty_page.js").read_text()
_LAUNCH_ARGS = ["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]


def _emit(result: dict) -> None:
    print(_RESULT_START)
    print(json.dumps(result))
    print(_RESULT_END)


def _run(target_path: Path, result: dict) -> None:
    file_url = target_path.as_uri()
    dir_url = target_path.parent.as_uri() + "/"

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=_LAUNCH_ARGS)
        try:
            page = browser.new_page()

            def on_console(msg: ConsoleMessage) -> None:
                if msg.type != "error":
                    return
                location = msg.location
                if not location.get("url", "").startswith(dir_url):
                    return
                result["consoleErrors"].append({"message": msg.text, "line": location.get("line")})

            def on_pageerror(error: Exception) -> None:
                result["consoleErrors"].append({"message": str(error), "line": None})

            def on_requestfailed(request: Request) -> None:
                if not request.url.startswith(dir_url):
                    return
                result["failedRequests"].append({"url": request.url, "error": request.failure or "unknown error"})

            def on_crash(_page: Page) -> None:
                result["crashed"] = True

            def on_route(route: Route) -> None:
                request = route.request
                url = urlsplit(request.url)
                is_local = url.scheme == "file"
                is_loopback = url.hostname in ("localhost", "127.0.0.1", "::1")
                if is_local or is_loopback or request.method == "GET":
                    route.continue_()
                    return
                # External, state-changing request: neutralize as an empty
                # SUCCESS rather than an abort, so a correctly-written
                # artifact's own `.catch()` doesn't fire from our interception.
                route.fulfill(status=200, content_type="text/plain", body="")

            page.on("console", on_console)
            page.on("pageerror", on_pageerror)
            page.on("requestfailed", on_requestfailed)
            page.on("crash", on_crash)
            page.route("**/*", on_route)

            try:
                page.goto(file_url, wait_until="load", timeout=_NAV_TIMEOUT_MS)
            except PlaywrightError:
                pass

            page.wait_for_timeout(_SETTLE_MS)

            if not result["crashed"]:
                try:
                    result["emptyPage"] = bool(page.evaluate(_EMPTY_PAGE_CHECK))
                except PlaywrightError:
                    pass
        finally:
            browser.close()


def main() -> None:
    result = {"consoleErrors": [], "failedRequests": [], "crashed": False, "crashDetails": None, "emptyPage": False}
    target = os.environ.get("ANTON_HTML_LINT_TARGET")

    timer = threading.Timer(_HARD_TIMEOUT_MS / 1000, lambda: (_emit(result), os._exit(0)))
    timer.daemon = True
    timer.start()

    try:
        if target:
            _run(Path(target).resolve(), result)
    except Exception as exc:  # never raise, never hang — a checker must not fail the caller
        result["crashDetails"] = str(exc)
    finally:
        timer.cancel()
        _emit(result)


if __name__ == "__main__":
    main()
