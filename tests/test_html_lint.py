"""Coverage for the html structural lint."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from anton.core.artifacts.html_lint import lint_html, lint_url

requires_browser = pytest.mark.skipif(
    not os.environ.get("ANTON_HTML_LINT_BROWSER"),
    reason="ANTON_HTML_LINT_BROWSER not set to a real browser binary",
)

_BROKEN_HTML = """<!DOCTYPE html>
<html>
<head><script src="missing.js"></script></head>
<body>
<script>
console.error("intentional error for test");
undefinedFunctionCallXYZ();
</script>
</body>
</html>
"""

_CLEAN_HTML = """<!DOCTYPE html>
<html><body><h1>All good</h1><script>console.log("all good");</script></body></html>
"""

_EMPTY_BODY_HTML = """<!DOCTYPE html>
<html><body><script>a = 1 / 0;</script></body></html>
"""


@pytest.fixture
def broken_page(tmp_path: Path) -> Path:
    path = tmp_path / "broken.html"
    path.write_text(_BROKEN_HTML)
    return path


@pytest.fixture
def clean_page(tmp_path: Path) -> Path:
    path = tmp_path / "clean.html"
    path.write_text(_CLEAN_HTML)
    return path


@pytest.fixture
def empty_body_page(tmp_path: Path) -> Path:
    """No console error, no failed request — division by zero doesn't throw
    in JS — but nothing visible ever gets rendered either (a real case found
    manually: an agent-generated page with only a script, no markup)."""
    path = tmp_path / "empty_body.html"
    path.write_text(_EMPTY_BODY_HTML)
    return path


def test_no_browser_configured_yields_none(monkeypatch, broken_page: Path):
    monkeypatch.delenv("ANTON_HTML_LINT_BROWSER", raising=False)
    assert lint_html(broken_page) is None


def test_browser_path_that_does_not_exist_yields_none(monkeypatch, broken_page: Path):
    monkeypatch.setenv("ANTON_HTML_LINT_BROWSER", "/no/such/binary-xyz")
    assert lint_html(broken_page) is None


@requires_browser
def test_flags_console_error_and_missing_local_asset(broken_page: Path):
    findings = lint_html(broken_page)
    kinds = {f.kind for f in findings}
    assert "console_error" in kinds
    assert "failed_request" in kinds
    messages = " ".join(f.message() for f in findings)
    assert "intentional error for test" in messages
    assert "missing.js" in messages


@requires_browser
def test_clean_page_has_no_findings(clean_page: Path):
    assert lint_html(clean_page) == []


@requires_browser
def test_flags_empty_body_with_no_other_signal(empty_body_page: Path):
    """Division by zero is not a JS error, and there's no missing asset —
    console_error/failed_request stay silent. empty_page is the only thing
    that can catch this shape of bug."""
    findings = lint_html(empty_body_page)
    kinds = {f.kind for f in findings}
    assert kinds == {"empty_page"}


@requires_browser
def test_message_format(broken_page: Path):
    findings = lint_html(broken_page)
    failed = next(f for f in findings if f.kind == "failed_request")
    assert failed.message().startswith("failed_request:")


# ── lint_url: the same runner over http (S-02) ──────────────────────────────

def test_lint_url_refuses_anything_that_is_not_an_http_url(tmp_path: Path, monkeypatch):
    """A file path belongs to lint_html; `None` means "not checked", never
    "clean", so a wrong argument cannot pass a page."""
    monkeypatch.setenv("ANTON_HTML_LINT_BROWSER", "/bin/true")
    assert lint_url(str(tmp_path / "index.html")) is None
    assert lint_url("file:///tmp/index.html") is None
    assert lint_url("") is None
    assert lint_url(None) is None  # type: ignore[arg-type]


def test_lint_url_without_a_browser_yields_none(monkeypatch):
    monkeypatch.delenv("ANTON_HTML_LINT_BROWSER", raising=False)
    assert lint_url("http://127.0.0.1:1/") is None


def test_lint_url_hands_the_runner_the_url_as_target(monkeypatch):
    import subprocess

    from anton.core.artifacts import html_lint

    seen: dict = {}

    def fake_run(cmd, env, **kw):
        seen["target"] = env["ANTON_HTML_LINT_TARGET"]
        return subprocess.CompletedProcess(cmd, 0, stdout="RESULT_JSON_START\n{}\nRESULT_JSON_END\n", stderr="")
    monkeypatch.setattr(html_lint, "discover_browser", lambda: "/usr/bin/electron")
    monkeypatch.setattr(html_lint.subprocess, "run", fake_run)
    assert lint_url("http://127.0.0.1:5000") == []
    assert seen["target"] == "http://127.0.0.1:5000"


@pytest.fixture
def served_broken_page(tmp_path: Path):
    """The broken page over http, on a loopback port: the runner treats the
    origin as the page's own scope, and a 404 for `missing.js` is a failed
    request the way ERR_FILE_NOT_FOUND is under file://."""
    import functools
    import http.server
    import threading

    (tmp_path / "index.html").write_text(_BROKEN_HTML)

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args, **kwargs):
            pass

    handler = functools.partial(Quiet, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/index.html"
    finally:
        server.shutdown()


@requires_browser
def test_lint_url_flags_console_error_and_http_404(served_broken_page: str):
    findings = lint_url(served_broken_page)
    assert findings is not None
    kinds = {f.kind for f in findings}
    assert "console_error" in kinds
    assert "failed_request" in kinds
    messages = " ".join(f.message() for f in findings)
    assert "intentional error for test" in messages
    assert "missing.js" in messages and "HTTP 404" in messages
