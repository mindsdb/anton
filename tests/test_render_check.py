"""The render-check client and the three places that use it.

A local HTTP server stands in for the render-check service, so every test
sees the exact request anton sends (path, content type, JSON body) and the
client's handling of each answer the contract allows: a verdict, an error
status, a dropped connection, a slow answer, a body that does not parse.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest

from anton.core.artifacts import ArtifactStore, render_check
from anton.core.artifacts.render_check import (
    RenderCheckRequest,
    RenderCheckResult,
    RenderCheckUnavailable,
    check_file,
    check_html,
)
from anton.core.backends.sandbox import CellSandbox
from anton.core.tools import tool_handlers
from anton.core.tools.generate_artifact import orchestrator, verifiers
from anton.core.tools.generate_artifact.state import GenState, VerifyResult

_PAGE = (
    '<!doctype html><html><head><meta name="viewport" content="width=device-width">'
    '</head><body><div id="a">hi</div></body></html>'
)
_SHOT = base64.b64encode(b"\xff\xd8\xff\xe0 not really a jpeg").decode()


def _answer(status: str = "passed", *, findings=(), unreachable=(), screenshot: bool = True) -> dict:
    return {
        "status": status,
        "findings": list(findings),
        "unreachable": list(unreachable),
        "screenshot": {"media_type": "image/jpeg", "data": _SHOT} if screenshot else None,
        "policy": "default-src 'self'",
    }


class _Service:
    """What the fake service received, and what it answers next."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        # Each answer: (HTTP status, body bytes, delay in seconds). The last
        # one repeats once the list runs out.
        self.answers: list[tuple[int, bytes, float]] = [(200, json.dumps(_answer()).encode(), 0.0)]
        # When set, answers each request from its JSON body instead.
        self.respond = None

    def answer(self, *answers) -> None:
        self.answers = [
            (code, body if isinstance(body, bytes) else json.dumps(body).encode(), delay)
            for code, body, delay in answers
        ]


@pytest.fixture
def service(monkeypatch):
    state = _Service()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - the stdlib's name
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)
            state.requests.append({
                "path": self.path,
                "content_type": self.headers.get("Content-Type"),
                "body": json.loads(raw),
            })
            if state.respond is not None:
                code, body, delay = state.respond(state.requests[-1]["body"])
                body = body if isinstance(body, bytes) else json.dumps(body).encode()
            else:
                code, body, delay = state.answers.pop(0) if len(state.answers) > 1 else state.answers[0]
            if delay:
                time.sleep(delay)
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the client stopped reading an answer over its cap

        def log_message(self, *args):  # keep pytest output clean
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    monkeypatch.setenv("ANTON_RENDER_CHECK_URL", f"http://127.0.0.1:{server.server_address[1]}/")
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """The person's project, where the sandboxed tests keep their pages."""
    path = tmp_path / "project"
    path.mkdir()
    return path


@pytest.fixture
def policy(tmp_path: Path, project: Path) -> CellSandbox:
    for name in ("home", "tmp"):
        (tmp_path / name).mkdir()
    return CellSandbox(
        project_dir=project, home_dir=tmp_path / "home", tmp_dir=tmp_path / "tmp", user_name="bob",
    )


# ── The client ───────────────────────────────────────────────────────────────


def test_the_request_is_the_contract_body_posted_to_v1_check(service):
    result = check_html(_PAGE)

    assert service.requests == [{
        "path": "/v1/check",
        "content_type": "application/json",
        "body": {"html": _PAGE, "screenshot": True},
    }]
    assert isinstance(result, RenderCheckResult)
    assert result.status == "passed"
    assert result.screenshot is not None and result.screenshot.data == _SHOT


def test_a_check_without_a_screenshot_asks_for_none(service):
    check_html(_PAGE, screenshot=False)
    assert service.requests[0]["body"] == {"html": _PAGE, "screenshot": False}


def test_the_request_model_is_the_contract_and_nothing_more():
    assert RenderCheckRequest(html="x").model_dump() == {"html": "x", "screenshot": True}
    with pytest.raises(ValueError):
        RenderCheckRequest(html="x", extra=1)


def test_a_failed_verdict_keeps_every_finding(service):
    findings = [
        {"kind": "csp_violation", "viewport": "desktop", "detail": "Refused to load https://unpkg.com/leaflet.js"},
        {"kind": "uncaught_exception", "viewport": "phone", "detail": "ReferenceError: L is not defined"},
    ]
    service.answer((200, _answer("failed", findings=findings), 0.0))

    result = check_html(_PAGE)

    assert result.status == "failed"
    assert [(f.kind, f.viewport) for f in result.findings] == [
        ("csp_violation", "desktop"), ("uncaught_exception", "phone"),
    ]


@pytest.mark.parametrize("code,needle", [
    (503, "busy or not ready (HTTP 503)"),
    (413, "size limit (HTTP 413)"),
    (400, "rejected the request (HTTP 400)"),
    (500, "answered HTTP 500"),
])
def test_an_error_status_is_unverified_with_its_reason(service, code, needle):
    service.answer((code, {"error": "x"}, 0.0))
    result = check_html(_PAGE)
    assert isinstance(result, RenderCheckUnavailable)
    assert result.status == "unverified"
    assert needle in result.unverified_reason


def test_a_body_that_does_not_parse_is_unverified_never_a_pass(service):
    service.answer((200, b'{"status": "passed", "findings": "nope"}', 0.0))
    result = check_html(_PAGE)
    assert result.status == "unverified"
    assert "does not parse" in result.unverified_reason


def test_a_finding_kind_outside_the_contract_does_not_parse(service):
    service.answer((200, _answer("failed", findings=[{"kind": "new_kind", "viewport": "desktop", "detail": "x"}]), 0.0))
    assert check_html(_PAGE).status == "unverified"


def test_a_screenshot_that_is_not_base64_does_not_parse(service):
    body = _answer()
    body["screenshot"]["data"] = "not base64 at all!"
    service.answer((200, body, 0.0))
    assert check_html(_PAGE).status == "unverified"


def test_an_oversized_screenshot_is_dropped_but_the_verdict_kept(service, monkeypatch):
    monkeypatch.setattr(render_check, "MAX_SCREENSHOT_BASE64_CHARS", 4)
    result = check_html(_PAGE)
    assert result.status == "passed" and result.screenshot is None


def test_a_slow_service_is_unverified_after_the_timeout(service):
    service.answer((200, _answer(), 2.0))
    started = time.monotonic()
    result = check_html(_PAGE, timeout=0.3)
    assert time.monotonic() - started < 1.5
    assert result.status == "unverified"
    assert "no answer within" in result.unverified_reason


def test_a_service_nobody_listens_on_is_unverified(monkeypatch):
    import socket

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()  # nothing listens there now: the connection is refused
    monkeypatch.setenv("ANTON_RENDER_CHECK_URL", f"http://127.0.0.1:{port}")
    result = check_html(_PAGE)
    assert result.status == "unverified"
    assert "could not be reached (ConnectError)" in result.unverified_reason


def test_unset_asks_nobody(monkeypatch):
    monkeypatch.delenv("ANTON_RENDER_CHECK_URL", raising=False)
    result = check_html(_PAGE)
    assert result.status == "unverified"
    assert "ANTON_RENDER_CHECK_URL unset" in result.unverified_reason


def test_a_page_over_the_size_limit_is_not_sent(service, monkeypatch):
    monkeypatch.setattr(render_check, "MAX_PAGE_BYTES", 100)
    result = check_html(_PAGE * 3)
    assert result.status == "unverified"
    assert result.unverified_reason == (
        f"the page is {len(_PAGE * 3) // 1024} KB, over the render check's 0 MB limit"
    )
    assert service.requests == []


def _inline_data_page(size: int) -> str:
    """A dashboard with its records inline as `const D = [...]`, the way the
    skill tells the agent to embed data, at least `size` bytes long."""
    row = json.dumps({"name": "Lisbon", "region": "south", "value": 1234})
    count = size // (len(row) + 2) + 1
    return (
        "<!doctype html><html><body><div id=\"chart\"></div><script>const D = ["
        + ", ".join([row] * count) + "];</script></body></html>"
    )


def test_a_page_with_inline_data_under_the_limit_is_sent_whole(service):
    """The cap counted the JSON-escaped request, which doubles every quote of
    inline records, so a 7 MiB page the skill allows came back unverified as
    "the page is 8495 KB, over the render check's 8 MB limit"."""
    page = _inline_data_page(7 * 1024 * 1024)
    escaped = len(RenderCheckRequest(html=page).model_dump_json().encode("utf-8"))
    assert len(page.encode("utf-8")) < render_check.MAX_PAGE_BYTES < escaped

    result = check_html(page)

    assert result.status == "passed"
    assert len(service.requests) == 1 and service.requests[0]["body"]["html"] == page


def test_a_request_over_its_cap_once_escaped_says_so(service, monkeypatch):
    """A page of control characters grows six times once JSON-escaped. Its
    reason names the request, not a page size the file does not have."""
    monkeypatch.setattr(render_check, "MAX_BODY_BYTES", 4096)
    page = "\x01" * 1024

    result = check_html(page)

    assert result.status == "unverified"
    assert result.unverified_reason == (
        "the page's request is 6 KB once JSON-escaped, over the render check's 0 MB request limit"
    )
    assert service.requests == []


def test_check_file_reads_no_more_than_the_page_limit(service, tmp_path, monkeypatch):
    monkeypatch.setattr(render_check, "MAX_PAGE_BYTES", 100)
    page = tmp_path / "index.html"
    page.write_text(_PAGE * 3)

    result = check_file(page, cell_sandbox=None)

    assert result.status == "unverified"
    assert result.unverified_reason == "the page is over the render check's 0 MB limit"
    assert service.requests == []


# The render-check service's 503 answer when one check runs and two wait.
# The service owns the wording; anton keys its one retry on the "busy" start.
_BUSY = {"detail": "busy: one check runs and two wait"}


@pytest.fixture
def quick_retry(monkeypatch):
    monkeypatch.setattr(render_check, "BUSY_RETRY_SECONDS", 0.05)


def _record_post_timeouts(monkeypatch) -> list[float]:
    timeouts: list[float] = []
    post = render_check._post

    def recording(base, *, body, timeout):
        timeouts.append(timeout)
        return post(base, body=body, timeout=timeout)

    monkeypatch.setattr(render_check, "_post", recording)
    return timeouts


def test_a_busy_service_is_asked_once_more_within_the_time_left(service, quick_retry, monkeypatch):
    """With several conversations finishing pages at once, the fourth check
    got 503 at once and its page read as not verified in a browser."""
    service.answer((503, _BUSY, 0.0), (200, _answer(), 0.0))
    timeouts = _record_post_timeouts(monkeypatch)

    result = check_html(_PAGE, timeout=20.0)

    assert result.status == "passed"
    assert len(service.requests) == 2 and service.requests[0] == service.requests[1]
    assert timeouts[0] == 20.0
    assert timeouts[1] <= 20.0 - render_check.BUSY_RETRY_SECONDS


def test_a_service_busy_twice_is_not_asked_a_third_time(service, quick_retry):
    service.answer((503, _BUSY, 0.0))

    result = check_html(_PAGE)

    assert isinstance(result, RenderCheckUnavailable)
    assert result.unverified_reason == "the render-check service is busy or not ready (HTTP 503)"
    assert len(service.requests) == 2


@pytest.mark.parametrize("body", [
    pytest.param({"detail": "the render check isn't ready: starting"}, id="not-ready"),
    pytest.param({"detail": "the browser couldn't run the check: gone"}, id="browser-failed"),
    pytest.param({"error": "busy"}, id="no-detail"),
    pytest.param(b"busy", id="not-json"),
    pytest.param({"detail": "busy: " + "x" * render_check.MAX_ERROR_BODY_BYTES}, id="over-the-read-cap"),
])
def test_a_503_that_is_not_the_busy_answer_is_not_asked_again(service, quick_retry, body):
    service.answer((503, body, 0.0), (200, _answer(), 0.0))

    result = check_html(_PAGE)

    assert result.status == "unverified"
    assert len(service.requests) == 1


def test_a_busy_service_is_not_asked_again_without_the_time_for_it(service, monkeypatch):
    service.answer((503, _BUSY, 0.0), (200, _answer(), 0.0))
    timeouts = _record_post_timeouts(monkeypatch)
    too_short = render_check.BUSY_RETRY_SECONDS + render_check.BUSY_RETRY_MIN_SECONDS - 0.5

    started = time.monotonic()
    result = check_html(_PAGE, timeout=too_short)

    assert result.status == "unverified"
    assert timeouts == [too_short]
    assert time.monotonic() - started < render_check.BUSY_RETRY_SECONDS, "no wait for an ask that never comes"


def test_the_generators_check_is_asked_again_after_a_busy_answer(service, quick_retry, tmp_path):
    page = tmp_path / "index.html"
    page.write_text(_PAGE)
    service.answer((503, _BUSY, 0.0), (200, _answer("failed", findings=[
        {"kind": "uncaught_exception", "viewport": "desktop", "detail": "ReferenceError: L is not defined"},
    ]), 0.0))

    check = verifiers.verify_frontend_live(page, cell_sandbox=None)

    assert check.status == "failed"
    assert check.verdict.errors == [
        "Loaded in a headless browser, the page threw an uncaught exception: "
        "ReferenceError: L is not defined (desktop)",
    ]
    assert len(service.requests) == 2


@pytest.mark.parametrize("url", ["http://[::1", "http://render-check:abc"])
def test_a_malformed_service_url_is_unverified_not_raised(monkeypatch, tmp_path, url):
    """`httpx.InvalidURL` is not an `httpx.HTTPError`: it escaped, so the
    cell lost every artifact lint line and generate_artifact crashed."""
    monkeypatch.setenv("ANTON_RENDER_CHECK_URL", url)
    page = tmp_path / "index.html"
    page.write_text(_PAGE)

    result = check_html(_PAGE)

    assert isinstance(result, RenderCheckUnavailable)
    assert result.unverified_reason == "ANTON_RENDER_CHECK_URL is not a valid URL"
    check = verifiers.verify_frontend_live(page, cell_sandbox=None)
    assert check.status == "unverified"
    assert check.reason == "unverified: ANTON_RENDER_CHECK_URL is not a valid URL"


# The render-check service's deadline for one check (25 s) plus the most it
# spends closing the page after that deadline (3 s). The service owns both
# numbers; this copy moves with them.
_SERVICE_ANSWERS_WITHIN = 25.0 + 3.0


def test_anton_waits_longer_than_the_service_takes_to_answer():
    """anton gave up at 30 s, about 2 s after the service's latest answer, so
    a slow cleanup on a host short of CPU turned a failed page into an
    unverified one and the agent lost the findings."""
    assert render_check.TIMEOUT_SECONDS >= _SERVICE_ANSWERS_WITHIN + 5.0
    # A check asked again after a busy answer still outlasts the service.
    assert render_check.TIMEOUT_SECONDS - render_check.BUSY_RETRY_SECONDS >= _SERVICE_ANSWERS_WITHIN + 1.0
    # The checks after one cell share a budget that still fits one whole check.
    assert tool_handlers._RENDER_CHECK_CELL_BUDGET_SECONDS >= render_check.TIMEOUT_SECONDS


_TIMEOUT = {"kind": "timeout", "viewport": "desktop", "detail": "the check's 25 s ran out while measuring the page"}
_THROWN = {"kind": "uncaught_exception", "viewport": "desktop", "detail": "ReferenceError: renderChart is not defined"}
_BLANK = {"kind": "blank_view", "viewport": "desktop", "detail": "#world-map has no drawn content"}
_CSP = {"kind": "csp_violation", "viewport": "desktop", "detail": "script-src blocked https://unpkg.com/x.js"}
_MISSING = {"url": "https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js", "reason": "not in the check's mirror"}


@pytest.mark.parametrize("status,findings,unreachable", [
    pytest.param("failed", [_CSP, _TIMEOUT], [_MISSING], id="a-definite-finding-fails-whatever-else"),
    pytest.param("unverified", [_THROWN], [_MISSING], id="a-library-the-check-could-not-load"),
    pytest.param("failed", [_THROWN, _TIMEOUT], [], id="an-exception-fails-despite-the-deadline"),
    pytest.param("unverified", [_BLANK, _TIMEOUT], [], id="a-blank-view-on-a-slow-host"),
    pytest.param("failed", [_BLANK], [], id="a-blank-view"),
    pytest.param("passed", [], [], id="nothing-found"),
])
def test_each_status_the_rule_gives_parses(status, findings, unreachable):
    result = RenderCheckResult.model_validate(_answer(status, findings=findings, unreachable=unreachable))
    assert result.status == status


@pytest.mark.parametrize("answer", [
    pytest.param({"status": "failed"}, id="failed-with-no-findings"),
    pytest.param(_answer("failed", findings=[_TIMEOUT]), id="failed-with-only-a-timeout"),
    pytest.param(_answer("passed", unreachable=[_MISSING]), id="passed-with-unreachable"),
    pytest.param(_answer("passed", findings=[
        {"kind": "console_error", "viewport": "desktop", "detail": "boom"},
    ]), id="passed-with-a-console-error"),
    pytest.param(_answer("unverified", findings=[_THROWN, _TIMEOUT]), id="unverified-beside-an-exception"),
])
def test_a_status_its_findings_contradict_is_unverified_never_a_pass(service, tmp_path, answer):
    """The generator's check rebuilt the status from the findings, so a
    `failed` answer with nothing it could name read as a pass."""
    service.answer((200, answer, 0.0))
    page = tmp_path / "index.html"
    page.write_text(_PAGE)

    result = check_html(_PAGE)

    assert isinstance(result, RenderCheckUnavailable)
    assert result.unverified_reason == "the render-check service answered a status its own findings contradict"
    check = verifiers.verify_frontend_live(page, cell_sandbox=None)
    assert check.status == "unverified" and check.verdict.errors == []


def test_the_service_is_reached_directly_whatever_proxy_the_process_has(service, monkeypatch):
    """The service sits on an internal network; the egress proxy a
    deployment sets for everything else would not reach it."""
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    assert check_html(_PAGE).status == "passed"
    assert len(service.requests) == 1


def test_an_unverified_answer_says_what_could_not_load(service):
    service.answer((200, _answer("unverified", unreachable=[
        {"url": "https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js",
         "reason": "pin an exact version the check carries: echarts@5.5.1"},
    ], screenshot=False), 0.0))
    result = check_html(_PAGE)
    assert result.unverified_reason == (
        "could not load https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js "
        "(pin an exact version the check carries: echarts@5.5.1)"
    )


def test_check_file_reads_the_page_and_sends_its_text(service, tmp_path):
    page = tmp_path / "index.html"
    page.write_text(_PAGE)
    assert check_file(page, cell_sandbox=None).status == "passed"
    assert service.requests[0]["body"]["html"] == _PAGE


def test_check_file_never_follows_a_link_out_of_the_persons_files(service, tmp_path, project, policy):
    """This process runs outside the sandbox: a page linked to another
    person's file must not be read and shown back as a screenshot."""
    outside = tmp_path / "other-person.html"
    outside.write_text("<html><body>someone else's data</body></html>")
    link = project / "index.html"
    os.symlink(outside, link)
    result = check_file(link, cell_sandbox=policy)
    assert result.status == "unverified"
    assert "outside the files this conversation may read" in result.unverified_reason
    assert service.requests == []


def _swap_after_the_check(monkeypatch, swap) -> None:
    """Run `swap` right after the policy approves a path: agent code racing
    the host between the check and the read."""
    approve = CellSandbox.allows_read

    def racing(self, path):
        allowed = approve(self, path)
        swap()
        return allowed

    monkeypatch.setattr(CellSandbox, "allows_read", racing)


def test_a_link_swapped_in_after_the_check_is_not_followed(service, tmp_path, project, policy, monkeypatch):
    """The page was checked by name and then read by name, so a thread in
    the agent's kernel that swaps the page for a link between the two had
    the host read the link's target and send it to the service."""
    secret = tmp_path / "other-person.html"
    secret.write_text("<html><body>someone else's data</body></html>")
    page = project / "index.html"
    page.write_text(_PAGE)

    def swap():
        link = project / "swap.tmp"
        os.symlink(secret, link)
        os.replace(link, page)

    _swap_after_the_check(monkeypatch, swap)
    result = check_file(page, cell_sandbox=policy)

    assert result.status == "unverified"
    assert service.requests == []


def test_a_folder_swapped_for_a_link_after_the_check_is_not_followed(service, tmp_path, project, policy, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "index.html").write_text("<html><body>someone else's data</body></html>")
    site = project / "site"
    site.mkdir()
    (site / "index.html").write_text(_PAGE)

    def swap():
        os.rename(site, project / "site-old")
        os.symlink(elsewhere, site)

    _swap_after_the_check(monkeypatch, swap)
    result = check_file(site / "index.html", cell_sandbox=policy)

    assert result.status == "unverified"
    assert service.requests == []


def test_a_hard_link_to_another_persons_file_is_not_read(service, tmp_path, project, policy):
    """A hard link is another name for the same file: its resolved path
    sits in the project, but its bytes are someone else's."""
    secret = tmp_path / "other-person.html"
    secret.write_text("<html><body>someone else's data</body></html>")
    page = project / "index.html"
    os.link(secret, page)

    result = check_file(page, cell_sandbox=policy)

    assert result.status == "unverified"
    assert service.requests == []


def test_a_fifo_at_the_name_neither_hangs_nor_is_sent(service, project, policy):
    """Reading a FIFO with no writer never returns, which held the cell and
    a thread of the server forever."""
    page = project / "index.html"
    os.mkfifo(page)
    outcome: list = []
    worker = threading.Thread(target=lambda: outcome.append(check_file(page, cell_sandbox=policy)), daemon=True)
    worker.start()
    worker.join(timeout=5)

    assert not worker.is_alive(), "check_file hung on a FIFO"
    assert outcome[0].status == "unverified"
    assert outcome[0].unverified_reason == "the page could not be read (refused a file that is not a regular file)"
    assert service.requests == []


def test_a_name_that_cannot_be_resolved_is_unverified_not_raised(service, project, policy, monkeypatch):
    """Resolving a name agent code is swapping can raise (macOS: EINVAL
    from readlink); `check_file` never raises."""
    def failing(path):
        raise OSError(22, "Invalid argument")

    monkeypatch.setattr(render_check.os.path, "realpath", failing)
    result = check_file(project / "index.html", cell_sandbox=policy)
    assert result.status == "unverified"
    assert result.unverified_reason == "the page could not be read (Invalid argument)"
    assert service.requests == []

def test_an_answer_over_the_response_cap_is_unverified(service):
    """anton used to read the whole answer, whatever its size, into a
    process every person on the server shares."""
    detail = "x" * (64 * 1024)
    service.answer((200, _answer("failed", findings=[
        {"kind": "console_error", "viewport": "desktop", "detail": f"{i} {detail}"} for i in range(80)
    ]), 0.0))

    result = check_html(_PAGE)

    assert isinstance(result, RenderCheckUnavailable)
    assert result.reason == "the render check's answer is over 4 MB, more than anton reads"


def _flood(count: int) -> dict:
    """A page that logs one error per data row."""
    return _answer("failed", findings=[
        {"kind": "console_error", "viewport": "desktop", "detail": f"missing value for row {i}"} for i in range(count)
    ])


def test_a_page_with_thousands_of_findings_reaches_the_agent_bounded(service, tmp_path):
    """Every distinct finding used to become a line of the cell's result and
    a verifier error of the retry prompt: 10,000 rows, 10,000 lines."""
    service.answer((200, _flood(10_000), 0.0))
    store, slug = _artifact_with_pages(tmp_path, "index.html")

    lint = tool_handlers.lint_changed_artifact_files(store, {slug: tool_handlers.ArtifactFiles()})

    shown = render_check.MAX_FINDINGS_SHOWN
    assert len(lint.messages) == shown + 1
    assert lint.messages[0] == f"{slug}/index.html — console_error (desktop): missing value for row 0"
    assert lint.messages[-1] == f"{slug}/index.html — and {10_000 - shown} more distinct findings, not listed"

    check = verifiers.verify_frontend_live(store.root / slug / "index.html", cell_sandbox=None)
    assert check.status == "failed"
    assert len(check.verdict.errors) == shown + 1
    assert check.verdict.errors[-1] == (
        f"Loaded in a headless browser, the page had {10_000 - shown} more distinct problems, "
        "not listed here: fix the ones above first."
    )


def test_the_url_never_reaches_agent_code(policy, monkeypatch):
    """The service holds no secret, but its address is the server's to use:
    a cell has no business calling it."""
    env = policy.child_env({"ANTON_RENDER_CHECK_URL": "http://render-check:8000", "PATH": "/usr/bin"})
    assert "ANTON_RENDER_CHECK_URL" not in env
    assert env["PATH"] == "/usr/bin"


# ── generate_artifact's browser step ─────────────────────────────────────────


def test_the_service_loads_the_page_even_under_a_sandbox_policy(service, project, policy):
    page = project / "index.html"
    page.write_text(_PAGE)

    check = verifiers.verify_frontend_live(page, cell_sandbox=policy)

    assert check.status == "passed"
    assert check.screenshot is not None
    assert service.requests[0]["body"] == {"html": _PAGE, "screenshot": True}


def test_failed_findings_become_verifier_errors_named_once_per_finding(service, tmp_path):
    page = tmp_path / "index.html"
    page.write_text(_PAGE)
    service.answer((200, _answer("failed", findings=[
        {"kind": "csp_violation", "viewport": "desktop", "detail": "script-src blocked https://unpkg.com/x.js"},
        {"kind": "csp_violation", "viewport": "phone", "detail": "script-src blocked https://unpkg.com/x.js"},
        {"kind": "separate_file", "viewport": "desktop", "detail": "data.js"},
        {"kind": "blank_view", "viewport": "phone", "detail": "#map"},
        {"kind": "overflow", "viewport": "phone", "detail": "scrollWidth 512 > 390"},
        {"kind": "uncaught_exception", "viewport": "desktop", "detail": "ReferenceError: L is not defined"},
        {"kind": "console_error", "viewport": "desktop", "detail": "boom"},
        {"kind": "crashed", "viewport": "desktop", "detail": "renderer gone"},
    ]), 0.0))

    check = verifiers.verify_frontend_live(page, cell_sandbox=None)

    assert check.status == "failed"
    assert check.verdict.errors == [
        "Loaded in a headless browser, the page broke the app preview's content security policy "
        "(desktop and phone): script-src blocked https://unpkg.com/x.js",
        "Loaded in a headless browser, the page asked for a separate file, which the app preview "
        "never serves (desktop): data.js",
        "Loaded in a headless browser, a chart, map or canvas stayed blank: #map (phone)",
        "Loaded in a headless browser, the page is wider than the screen: scrollWidth 512 > 390 (phone)",
        "Loaded in a headless browser, the page threw an uncaught exception: "
        "ReferenceError: L is not defined (desktop)",
        "Loaded in a headless browser, the page logged a console error: boom (desktop)",
        "Loaded in a headless browser, the page crashed the renderer process.",
    ]


def test_an_unverified_answer_is_a_skip_with_its_reason_not_an_error(service, tmp_path):
    page = tmp_path / "index.html"
    page.write_text(_PAGE)
    service.answer((503, {"error": "busy"}, 0.0))
    check = verifiers.verify_frontend_live(page, cell_sandbox=None)
    assert check.status == "unverified"
    assert check.reason == "unverified: the render-check service is busy or not ready (HTTP 503)"
    assert check.findings_listed is False
    assert check.verdict.errors == []


@pytest.mark.parametrize("answer,reason", [
    pytest.param(
        _answer("unverified", findings=[_BLANK, {**_BLANK, "viewport": "phone"}, _TIMEOUT]),
        "unverified: the check ran out of time (the check's 25 s ran out while measuring the page)\n"
        "Before it ran out of time, the check found:\n"
        "- blank_view (desktop and phone): #world-map has no drawn content (it may still have been drawing)",
        id="a-blank-map-before-the-deadline",
    ),
    pytest.param(
        _answer("unverified", findings=[_THROWN], unreachable=[_MISSING]),
        "unverified: could not load https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js "
        "(not in the check's mirror)\n"
        "The check still found:\n"
        "- uncaught_exception (desktop): ReferenceError: renderChart is not defined",
        id="an-exception-beside-a-library-the-check-could-not-load",
    ),
])
def test_an_unverified_check_still_lists_what_it_found(service, tmp_path, answer, reason):
    """The reason named only the timeout or the unreachable URL, so the agent
    never saw the blank map or the exception the check had found."""
    page = tmp_path / "index.html"
    page.write_text(_PAGE)
    service.answer((200, answer, 0.0))

    check = verifiers.verify_frontend_live(page, cell_sandbox=None)

    assert check.status == "unverified"
    assert check.reason == reason
    assert check.findings_listed is True
    assert check.verdict.errors == []


def test_an_unverified_check_counts_the_findings_past_the_shown_ones(service, tmp_path):
    page = tmp_path / "index.html"
    page.write_text(_PAGE)
    rows = render_check.MAX_FINDINGS_SHOWN + 3
    service.answer((200, _answer("unverified", findings=[
        {**_BLANK, "detail": f"#chart-{i} has no drawn content"} for i in range(rows)
    ] + [_TIMEOUT]), 0.0))

    check = verifiers.verify_frontend_live(page, cell_sandbox=None)

    lines = check.reason.split("\n")
    assert lines[2] == "- blank_view (desktop): #chart-0 has no drawn content (it may still have been drawing)"
    assert len(lines) == 2 + render_check.MAX_FINDINGS_SHOWN + 1
    assert lines[-1] == "- and 3 more distinct findings, not listed"


def test_an_exception_before_the_deadline_fails_the_generators_check(service, tmp_path):
    """A page that throws at load and then runs the check out of time. The
    service fails it on the exception; the timeout beside it adds no error."""
    page = tmp_path / "index.html"
    page.write_text(_PAGE)
    service.answer((200, _answer("failed", findings=[_THROWN, _TIMEOUT]), 0.0))

    check = verifiers.verify_frontend_live(page, cell_sandbox=None)

    assert check.status == "failed"
    assert check.verdict.errors == [
        "Loaded in a headless browser, the page threw an uncaught exception: "
        "ReferenceError: renderChart is not defined (desktop)",
    ]


def test_the_generators_check_can_ask_for_no_screenshot(service, tmp_path):
    page = tmp_path / "index.html"
    page.write_text(_PAGE)
    verifiers.verify_frontend_live(page, cell_sandbox=None)
    verifiers.verify_frontend_live(page, cell_sandbox=None, screenshot=False)
    assert [r["body"]["screenshot"] for r in service.requests] == [True, False]


def _gen_state(tmp_path: Path, policy=None) -> GenState:
    session = SimpleNamespace(_scratchpads=SimpleNamespace(cell_sandbox=policy), _page_checks={})
    return GenState(
        session=session, artifact_type="html-app", artifact_path=tmp_path, slug="dash",
        brief="b", is_fullstack=False,
    )


def _writes_page(tmp_path: Path, kickoffs: list[str]):
    async def fake_loop(**kw):
        kickoffs.append(kw["kickoff"])
        (tmp_path / "index.html").write_text(_PAGE)
        return {"files_written": ["index.html"], "rounds_used": 1, "summary": "s"}

    return fake_loop


async def test_the_generator_retries_on_what_the_service_found(service, project, policy, monkeypatch):
    """Under the sandbox, the preview's verdict feeds the same retry as the
    static rules, and the second attempt passes."""
    st = _gen_state(project, policy)
    kickoffs: list[str] = []
    monkeypatch.setattr(orchestrator.engine, "_run_loop", _writes_page(project, kickoffs))
    service.answer(
        (200, _answer("failed", findings=[
            {"kind": "csp_violation", "viewport": "desktop", "detail": "blocked https://unpkg.com/leaflet.js"},
        ]), 0.0),
        (200, _answer("passed"), 0.0),
    )

    assert await orchestrator._gen_verify_frontend(st) is None

    assert len(kickoffs) == 2
    assert isinstance(kickoffs[0], str), "no check has run before the first attempt"
    # The retry shows the failed page as the check saw it, beside its findings.
    text, caption, image = kickoffs[1]
    assert "blocked https://unpkg.com/leaflet.js" in text["text"]
    assert caption == {
        "type": "text",
        "text": "[render check screenshot] dash/index.html as the app preview shows it, at desktop width.",
    }
    assert image == {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _SHOT}}
    assert st.browser_status == "passed"
    assert st.browser_screenshot is not None
    assert st.checks_skipped == []
    shell = orchestrator._result_shell(st)
    assert shell["verified_in_browser"] is True
    assert shell["screenshot"] is st.browser_screenshot
    assert shell["screenshot_page"] == "index.html"


async def test_the_retry_shows_no_screenshot_once_the_model_refused_one(service, project, policy, monkeypatch):
    """A model that refused a render screenshot earlier in the session would
    refuse the retry's screenshot too, so the retry gets the findings as text."""
    st = _gen_state(project, policy)
    st.session._render_screenshots_refused = True
    kickoffs: list = []
    monkeypatch.setattr(orchestrator.engine, "_run_loop", _writes_page(project, kickoffs))
    service.answer(
        (200, _answer("failed", findings=[_CSP]), 0.0),
        (200, _answer("passed"), 0.0),
    )

    assert await orchestrator._gen_verify_frontend(st) is None

    assert len(kickoffs) == 2 and isinstance(kickoffs[1], str)
    assert "script-src blocked https://unpkg.com/x.js" in kickoffs[1]
    assert [r["body"]["screenshot"] for r in service.requests] == [False, False]


async def test_a_refused_read_image_leaves_the_retrys_screenshot_on(service, project, policy, monkeypatch):
    """A refused `read_image` image says nothing about render screenshots:
    one corrupt file draws a refusal from a model that takes images too. The
    retry still shows the failed page."""
    st = _gen_state(project, policy)
    st.session._images_refused = True
    kickoffs: list = []
    monkeypatch.setattr(orchestrator.engine, "_run_loop", _writes_page(project, kickoffs))
    service.answer(
        (200, _answer("failed", findings=[_CSP]), 0.0),
        (200, _answer("passed"), 0.0),
    )

    assert await orchestrator._gen_verify_frontend(st) is None

    assert len(kickoffs) == 2
    text, _, image = kickoffs[1]
    assert "script-src blocked https://unpkg.com/x.js" in text["text"]
    assert image["type"] == "image" and image["source"]["data"] == _SHOT
    assert [r["body"]["screenshot"] for r in service.requests] == [True, True]


async def test_a_retry_after_a_static_failure_shows_no_earlier_screenshot(service, project, policy, monkeypatch):
    """The screenshot is of the page the attempt just before wrote. A
    retry after a page the static checks failed, which no browser loaded,
    gets no screenshot of an older page."""
    monkeypatch.setattr(orchestrator, "GEN_VERIFY_MAX_RETRIES", 2)
    st = _gen_state(project, policy)
    pages = [_PAGE, "<html><body>no viewport, no doctype</body></html>", _PAGE]
    kickoffs: list = []

    async def fake_loop(**kw):
        kickoffs.append(kw["kickoff"])
        (project / "index.html").write_text(pages[len(kickoffs) - 1])
        return {"files_written": ["index.html"], "rounds_used": 1, "summary": "s"}

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)
    monkeypatch.setattr(orchestrator.verifiers, "verify_frontend", lambda html, **kw: VerifyResult(
        errors=[] if html == _PAGE else ["the page has no viewport meta"],
    ))
    service.answer((200, _answer("failed", findings=[_CSP]), 0.0), (200, _answer("passed"), 0.0))

    assert await orchestrator._gen_verify_frontend(st) is None

    assert [type(k).__name__ for k in kickoffs] == ["str", "list", "str"]


# The most requests a fake model in these tests serves. Past it, a resend
# bound that broke fails the test at once instead of hanging it.
_REQUEST_CAP = 20


class _RequestCapExceeded(RuntimeError):
    """Not an image refusal, so nothing in anton resends after it."""


class _GeneratorModel:
    """The generator's model behind `plan_stream` and `code_stream`: every
    request it takes writes the page, and finishes unless `finishes(n,
    images)` says otherwise. `refuses(n, images)` picks the requests it
    refuses with an image 400. Both pick by the request's 1-based number and
    the image parts it carries. Records each request's count, and raises
    `_RequestCapExceeded` past `_REQUEST_CAP` requests."""

    def __init__(self, *, refuses, finishes=lambda n, images: True) -> None:
        self.refuses = refuses
        self.finishes = finishes
        self.images: list[int] = []

    def stream(self, *, system, messages, tools=None, max_tokens=None):
        images = sum(
            1
            for m in messages if isinstance(m.get("content"), list)
            for b in m["content"] if isinstance(b, dict) and b.get("type") == "image"
        )
        self.images.append(images)
        return self._reply(len(self.images), images=images)

    async def _reply(self, n: int, *, images: int):
        from anton.core.llm.provider import LLMResponse, StreamComplete, ToolCall, Usage
        from anton.core.tools.generate_artifact.sub_tools import FILE_BEGIN_MARKER, FILE_END_MARKER

        if n > _REQUEST_CAP:
            raise _RequestCapExceeded(f"request {n}: more than {_REQUEST_CAP}")
        if self.refuses(n, images):
            raise _ImagePart400("Invalid content type. image_url is only supported by certain models.")
        finish = [ToolCall(id=f"f{n}", name="finish", input={"summary": "wrote the page"})]
        yield StreamComplete(response=LLMResponse(
            content=f"{FILE_BEGIN_MARKER}\n{_PAGE}\n{FILE_END_MARKER}",
            tool_calls=[
                ToolCall(id=f"w{n}", name="write_file", input={"path": "index.html"}),
                *(finish if self.finishes(n, images) else []),
            ],
            usage=Usage(input_tokens=1, output_tokens=1),
            stop_reason="tool_use",
        ))


def _gen_state_on(project: Path, *, policy, model: _GeneratorModel) -> GenState:
    st = _gen_state(project, policy)
    st.session._render_screenshots_refused = False
    st.session._llm = SimpleNamespace(plan_stream=model.stream, code_stream=model.stream, max_tokens=None)
    return st


async def test_a_refused_retry_screenshot_is_dropped_once_and_later_attempts_send_none(
    service, project, policy, monkeypatch
):
    """A model that takes no image input refuses the retry that shows the
    failed page. The generator sends that request once more without the
    image, the session records the refusal, and the attempt after it carries
    no screenshot: one refused request in the whole run."""
    monkeypatch.setattr(orchestrator, "GEN_VERIFY_MAX_RETRIES", 2)
    model = _GeneratorModel(refuses=lambda n, images: images > 0)
    st = _gen_state_on(project, policy=policy, model=model)
    service.answer(
        (200, _answer("failed", findings=[_CSP]), 0.0),
        (200, _answer("failed", findings=[_CSP]), 0.0),
        (200, _answer("passed"), 0.0),
    )

    assert await orchestrator._gen_verify_frontend(st) is None

    # Attempt 1; attempt 2 with the screenshot, refused, then without it;
    # attempt 3, which knows the model refuses images.
    assert model.images == [0, 1, 0, 0]
    assert st.session._render_screenshots_refused is True
    # The checks after the refusal ask the service for no screenshot.
    assert [r["body"]["screenshot"] for r in service.requests] == [True, False, False]
    assert st.browser_status == "passed"


async def test_the_coding_models_refusal_leaves_the_main_agents_screenshots_on(service, project, policy):
    """Round 0 runs on the planning model, the main agent's own, and later
    rounds on the coding model. Here only the coding model refuses images:
    round 0 takes the retry's screenshot and writes without finishing, round
    1 is refused over it and sent once more without it. That refusal says
    nothing about the main agent's model, so its checks keep screenshots."""
    planner = _GeneratorModel(refuses=lambda n, images: False, finishes=lambda n, images: images == 0)
    coder = _GeneratorModel(refuses=lambda n, images: images > 0)
    st = _gen_state(project, policy)
    st.session._render_screenshots_refused = False
    st.session._llm = SimpleNamespace(plan_stream=planner.stream, code_stream=coder.stream, max_tokens=None)
    service.answer((200, _answer("failed", findings=[_CSP]), 0.0), (200, _answer("passed"), 0.0))

    assert await orchestrator._gen_verify_frontend(st) is None

    # Attempt 1 on the planner; attempt 2's round 0 on the planner with the
    # screenshot, then round 1 on the coder, refused once and sent without it.
    assert planner.images == [0, 1]
    assert coder.images == [1, 0]
    assert st.session._render_screenshots_refused is False
    assert [r["body"]["screenshot"] for r in service.requests] == [True, True]
    assert st.browser_status == "passed"


async def test_a_refusal_after_the_screenshot_drop_ends_the_attempt(service, project, policy, monkeypatch):
    """The drop happens once: a request still refused without the image
    raises, as any provider error in the generator does, after two requests
    and no record of a refusal the screenshot did not cause."""
    model = _GeneratorModel(refuses=lambda n, images: n > 1)
    st = _gen_state_on(project, policy=policy, model=model)
    service.answer((200, _answer("failed", findings=[_CSP]), 0.0))

    with pytest.raises(_ImagePart400):
        await orchestrator._gen_verify_frontend(st)

    assert model.images == [0, 1, 0]
    assert st.session._render_screenshots_refused is False


async def test_a_text_kickoff_refused_over_an_image_is_not_resent(tmp_path):
    """Only the kickoff's screenshot is the loop's to drop. A refusal of a
    request with none in it raises after one request, as before."""
    from anton.core.tools.generate_artifact import engine

    model = _GeneratorModel(refuses=lambda n, images: True)
    session = SimpleNamespace(
        _render_screenshots_refused=False,
        _llm=SimpleNamespace(plan_stream=model.stream, code_stream=model.stream, max_tokens=None),
    )

    with pytest.raises(_ImagePart400):
        await engine._run_loop(
            session=session, system="s", kickoff="write the page", artifact_path=tmp_path,
            node_label="generate_frontend",
        )

    assert model.images == [0]
    assert session._render_screenshots_refused is False


def test_the_generators_trace_writes_no_image_data(tmp_path):
    """The debug trace logs every round's messages, and a retry's kickoff
    carries the screenshot on each of them. It logs its length instead."""
    from anton.core.tools.generate_artifact.debug_trace import GenTrace

    shot = render_check.RenderScreenshot(media_type="image/jpeg", data=_SHOT)
    trace_file = tmp_path / "trace.jsonl"
    kickoff = [{"type": "text", "text": "fix these"}, *render_check.screenshot_blocks(shot, page="dash/index.html")]

    GenTrace(str(trace_file)).llm_call(
        node="generate_frontend", method="plan", system="s", messages=[{"role": "user", "content": kickoff}],
    )

    line = trace_file.read_text()
    assert _SHOT not in line
    image = json.loads(line)["messages"][0]["content"][2]
    assert image["source"]["data"] == f"<{len(_SHOT)} base64 characters>"


async def test_generate_traces_its_result_without_the_screenshot(tmp_path, monkeypatch):
    """`ANTON_DEBUG_ARTIFACT_GENERATE_TOOL` traces every run's result. The
    screenshot in it is up to a megabyte of base64; the handler shows it to
    the agent, the trace leaves it out."""
    from anton.core.tools.generate_artifact import engine

    trace_file = tmp_path / "trace.jsonl"
    monkeypatch.setenv("ANTON_DEBUG_ARTIFACT_GENERATE_TOOL", str(trace_file))
    shot = render_check.RenderScreenshot(media_type="image/jpeg", data=_SHOT)

    async def fake_run(state, *, entry):
        return {"status": "generated", "screenshot": shot, "screenshot_page": "index.html"}

    monkeypatch.setattr(orchestrator, "run", fake_run)
    session = SimpleNamespace(_page_checks={}, _workspace=None, _scratchpads=None)
    result = await engine.generate(
        session=session, slug="dash", artifact_path=tmp_path, artifact_type="html-app",
        user_request="a map", agent_understanding="a map",
    )

    assert result["screenshot"] is shot, "the handler still gets it"
    traced = [json.loads(line) for line in trace_file.read_text().splitlines()]
    run_result = next(t for t in traced if t["event"] == "run_result")
    assert run_result["result"] == {"status": "generated", "screenshot_page": "index.html"}
    assert _SHOT not in trace_file.read_text()


async def test_an_unverified_page_passes_the_step_and_says_why(service, project, policy, monkeypatch):
    st = _gen_state(project, policy)
    monkeypatch.setattr(orchestrator.engine, "_run_loop", _writes_page(project, []))
    service.answer((200, _answer("unverified", unreachable=[
        {"url": "https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js", "reason": "not in the check's mirror"},
    ], screenshot=False), 0.0))

    assert await orchestrator._gen_verify_frontend(st) is None

    assert st.checks_skipped == [
        "browser load of the page: unverified: could not load "
        "https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js (not in the check's mirror)"
    ]
    assert st.browser_status == "unverified"
    assert st.browser_findings_listed is False
    assert orchestrator._result_shell(st)["verified_in_browser"] is False


async def test_an_unverified_page_tells_the_agent_what_the_check_still_found(service, project, policy, monkeypatch):
    st = _gen_state(project, policy)
    monkeypatch.setattr(orchestrator.engine, "_run_loop", _writes_page(project, []))
    service.answer((200, _answer("unverified", findings=[_BLANK, _TIMEOUT], screenshot=False), 0.0))

    assert await orchestrator._gen_verify_frontend(st) is None

    assert orchestrator._result_shell(st)["checks_skipped"] == [
        "browser load of the page: unverified: the check ran out of time "
        "(the check's 25 s ran out while measuring the page)\n"
        "Before it ran out of time, the check found:\n"
        "- blank_view (desktop): #world-map has no drawn content (it may still have been drawing)"
    ]
    assert st.browser_status == "unverified"
    assert st.browser_findings_listed is True


async def test_an_attached_image_is_embedded_so_the_page_passes(service, project, policy, monkeypatch):
    """An html-app with an attached logo: the generator writes
    `<img src="logo.png">` as told, and the preview never serves a file next
    to the page. The pipeline embeds the bytes before the check, so the
    check passes on the first attempt instead of failing every retry."""
    from anton.core.tools.generate_artifact import attachments as att_mod

    upload = project / "upload" / "logo.png"
    upload.parent.mkdir()
    upload.write_bytes(b"\x89PNG fake logo")
    st = _gen_state(project, policy)
    st.attachments = [att_mod.Attachment(path=str(upload), name="logo.png", size=15, kind=att_mod.KIND_ASSET)]
    orchestrator._stage_attachments(st)
    page = _PAGE.replace("<div", '<img src="logo.png" alt="logo"><div')

    kickoffs: list[str] = []

    async def fake_loop(**kw):
        kickoffs.append(kw["kickoff"])
        (project / "index.html").write_text(page)
        return {"files_written": ["index.html"], "rounds_used": 1, "summary": "s"}

    def respond(body):
        if 'src="logo.png"' in body["html"]:
            return 200, _answer("failed", findings=[
                {"kind": "separate_file", "viewport": "desktop", "detail": "logo.png"},
            ]), 0.0
        return 200, _answer("passed"), 0.0

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)
    service.respond = respond

    assert await orchestrator._gen_verify_frontend(st) is None

    assert len(kickoffs) == 1
    assert st.browser_status == "passed"
    sent = service.requests[0]["body"]["html"]
    assert 'src="data:image/png;base64,' + base64.b64encode(b"\x89PNG fake logo").decode() + '"' in sent
    assert (project / "index.html").read_text() == sent


async def test_embedding_and_the_static_checks_run_off_the_event_loop(service, project, policy, monkeypatch):
    """Embedding a large asset, and the static checks over the page it
    grows, take a second or more; on the event loop that would stall every
    conversation the process serves."""
    loop_thread = threading.get_ident()
    threads: dict[str, int] = {}
    real_embed, real_verify = orchestrator._embed_attachments, orchestrator.verifiers.verify_frontend

    def embed(state, entry_file):
        threads["embed"] = threading.get_ident()
        return real_embed(state, entry_file)

    def verify(html, **kw):
        threads["verify"] = threading.get_ident()
        return real_verify(html, **kw)

    monkeypatch.setattr(orchestrator, "_embed_attachments", embed)
    monkeypatch.setattr(orchestrator.verifiers, "verify_frontend", verify)
    monkeypatch.setattr(orchestrator.engine, "_run_loop", _writes_page(project, []))

    assert await orchestrator._gen_verify_frontend(_gen_state(project, policy)) is None

    assert set(threads) == {"embed", "verify"}
    assert loop_thread not in threads.values()


def test_embedding_reads_no_link_out_of_the_persons_files(tmp_path, project, policy):
    """The staged copy sits where agent code writes, and its bytes land in a
    page the agent reads back: a link put there must not carry another
    person's file into the page."""
    from anton.core.tools.generate_artifact import attachments as att_mod

    secret = tmp_path / "other-person.png"
    secret.write_bytes(b"someone else's file")
    os.symlink(secret, project / "logo.png")
    att = att_mod.Attachment(path="x", name="logo.png", size=1, kind=att_mod.KIND_ASSET, staged="logo.png")
    html = '<img src="logo.png">'

    assert att_mod.embed_assets(html, [att], project, cell_sandbox=policy) == html


def test_embedding_covers_the_reference_shapes_and_leaves_text_alone(tmp_path):
    from anton.core.tools.generate_artifact import attachments as att_mod

    (tmp_path / "logo.png").write_bytes(b"png")
    (tmp_path / "font.woff2").write_bytes(b"font")
    atts = [
        att_mod.Attachment(path="x", name="logo.png", size=3, kind=att_mod.KIND_ASSET, staged="logo.png"),
        att_mod.Attachment(path="y", name="font.woff2", size=4, kind=att_mod.KIND_ASSET, staged="font.woff2"),
    ]
    png = "data:image/png;base64," + base64.b64encode(b"png").decode()
    html = (
        "<img src=\"logo.png\"><img src='./logo.png'><img src=logo.png>"
        "<style>@font-face{src:url(font.woff2)} .a{background:url( './logo.png' )}</style>"
        "<script>img.src = 'logo.png'</script><p>see logo.png</p>"
    )

    out = att_mod.embed_assets(html, atts, tmp_path, cell_sandbox=None)

    assert out.count(png) == 5
    assert "src:url(data:font/woff2;base64," + base64.b64encode(b"font").decode() + ")" in out
    assert "<p>see logo.png</p>" in out


async def test_the_browser_line_shows_only_where_a_browser_will_load_the_page(project, policy, monkeypatch):
    """`Loading the page in a browser` claims a browser: under a sandbox with
    no service none runs, so the line must not appear."""
    import asyncio

    monkeypatch.delenv("ANTON_RENDER_CHECK_URL", raising=False)
    monkeypatch.setattr(orchestrator.engine, "_run_loop", _writes_page(project, []))
    st = _gen_state(project, policy)
    st.progress = asyncio.Queue()

    assert await orchestrator._gen_verify_frontend(st) is None
    lines = []
    while not st.progress.empty():
        lines.append(st.progress.get_nowait())
    assert not [line for line in lines if "browser" in line]
    assert st.checks_skipped == [
        "browser load of the page: skipped: agent code runs in a sandbox on this deployment, "
        "which starts no browser"
    ]

    monkeypatch.setenv("ANTON_RENDER_CHECK_URL", "http://127.0.0.1:9")
    st = _gen_state(project, policy)
    st.progress = asyncio.Queue()
    await orchestrator._gen_verify_frontend(st)
    lines = []
    while not st.progress.empty():
        lines.append(st.progress.get_nowait())
    assert "Loading the page in a browser" in lines


async def test_generate_records_how_it_left_the_page(tmp_path, monkeypatch):
    from anton.core.tools.generate_artifact import engine

    session = SimpleNamespace(_page_checks={})
    st = SimpleNamespace(
        slug="dash", browser_status=None, files_written=["index.html"], is_fullstack=False, primary="index.html",
    )
    engine._record_page_check(session, state=st, result="Frontend verification failed after 2 attempt(s): x")
    assert session._page_checks == {("dash", "index.html"): "unverified"}

    st.browser_status = "failed"
    engine._record_page_check(session, state=st, result="Frontend verification failed after 2 attempt(s): x")
    assert session._page_checks == {("dash", "index.html"): "failed"}

    st.is_fullstack, st.browser_status = True, "passed"
    engine._record_page_check(session, state=st, result={"status": "generated"})
    assert session._page_checks[("dash", "static/index.html")] == "passed"

    session._page_checks.clear()
    engine._record_page_check(session, state=st, result={"status": "needs_confirmation"})
    st.files_written, st.browser_status = [], None
    engine._record_page_check(session, state=st, result="Error: refused")
    assert session._page_checks == {}


# ── generate_artifact's result ───────────────────────────────────────────────


async def _handler_result(
    tmp_path: Path, monkeypatch, result: dict, *, screenshots_refused: bool = False, images_refused: bool = False
):
    from anton.core.tools import generate_artifact as gen_pkg

    store = ArtifactStore(tmp_path / "artifacts")
    slug = store.create(name="Map", description="d", type="html-app", primary="index.html").slug

    async def fake_generate(**kw):
        return dict(result)

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    session = SimpleNamespace(
        _workspace=SimpleNamespace(artifacts_dir=tmp_path / "artifacts"),
        _render_screenshots_refused=screenshots_refused,
        _images_refused=images_refused,
    )
    out = None
    async for item in tool_handlers.handle_generate_artifact(
        session, {"slug": slug, "user_request": "a map", "agent_understanding": "a map"}
    ):
        out = item
    return slug, out


async def test_the_screenshot_reaches_the_agent_beside_the_result(tmp_path, monkeypatch):
    shot = render_check.RenderScreenshot(media_type="image/jpeg", data=_SHOT)
    slug, out = await _handler_result(tmp_path, monkeypatch, {
        "status": "generated", "files_written": ["index.html"], "warnings": [], "checks_skipped": [],
        "verified_in_browser": True, "screenshot": shot,
    })

    assert out.ok is True
    text, caption, image = out.content
    payload = json.loads(text["text"])
    assert "screenshot" not in payload
    assert payload["verified_in_browser"] is True
    assert caption == {
        "type": "text",
        "text": f"[render check screenshot] {slug}/index.html as the app preview shows it, at desktop width.",
    }
    assert image == {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _SHOT}}


async def test_the_caption_names_the_page_the_check_loaded(tmp_path, monkeypatch):
    """The generator may write a page under another name than the primary
    the artifact had before the run; the caption names the page the check
    loaded, not a file that may not exist."""
    shot = render_check.RenderScreenshot(media_type="image/jpeg", data=_SHOT)
    slug, out = await _handler_result(tmp_path, monkeypatch, {
        "status": "generated", "files_written": ["report.html"], "warnings": [], "checks_skipped": [],
        "verified_in_browser": True, "screenshot": shot, "screenshot_page": "report.html",
    })

    text, caption, _ = out.content
    assert "screenshot_page" not in json.loads(text["text"])
    assert caption["text"] == (
        f"[render check screenshot] {slug}/report.html as the app preview shows it, at desktop width."
    )


async def test_the_result_names_the_page_the_generator_wrote(service, project, policy, monkeypatch):
    """No primary set, and the generator wrote `report.html`: the result
    names that page beside its screenshot."""
    st = _gen_state(project, policy)

    async def fake_loop(**kw):
        (project / "report.html").write_text(_PAGE)
        return {"files_written": ["report.html"], "rounds_used": 1, "summary": "s"}

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)

    assert await orchestrator._gen_verify_frontend(st) is None

    shell = orchestrator._result_shell(st)
    assert shell["screenshot"] is not None and shell["screenshot_page"] == "report.html"


async def test_an_unverified_page_changes_what_the_agent_is_told(tmp_path, monkeypatch):
    """The old instruction said every file was verified and not to re-verify,
    which a sandboxed agent relayed as a working page."""
    _, out = await _handler_result(tmp_path, monkeypatch, {
        "status": "generated", "files_written": ["index.html"], "warnings": [],
        "checks_skipped": ["browser load of the page: unverified: the render-check service is busy"],
        "verified_in_browser": False,
    })
    instruction = json.loads(out.content)["instruction"]
    assert "unverified in a browser" in instruction
    assert "statically verified by the generation pipeline" not in instruction
    assert "Do NOT re-read, re-parse or re-verify" not in instruction


async def test_an_unverified_page_lets_the_agent_fix_what_the_check_still_found(tmp_path, monkeypatch):
    """`checks_skipped` lists what an unverified check still found, such as
    a blank map measured before the deadline. The instruction must not tell
    the agent to leave it for the user to report, nor to keep editing a page
    that may already work while no check can vouch for it."""
    _, out = await _handler_result(tmp_path, monkeypatch, {
        "status": "generated", "files_written": ["index.html"], "warnings": [],
        "checks_skipped": [
            "browser load of the page: unverified: the check ran out of time\n"
            "Before it ran out of time, the check found:\n"
            "- blank_view (desktop): #world-map has no drawn content (it may still have been drawing)"
        ],
        "verified_in_browser": False,
    })
    instruction = json.loads(out.content)["instruction"]
    assert "When it also lists what the check found" in instruction
    assert (
        "fix those once; if a later check still cannot vouch for the page, report it as "
        "unverified instead of editing it again"
    ) in instruction
    assert "otherwise act only on problems the user actually reports" in instruction


async def test_a_verified_page_keeps_the_do_not_reverify_instruction(tmp_path, monkeypatch):
    _, out = await _handler_result(tmp_path, monkeypatch, {
        "status": "generated", "files_written": ["index.html"], "warnings": [], "checks_skipped": [],
        "verified_in_browser": True,
    })
    instruction = json.loads(out.content)["instruction"]
    assert "Do NOT re-read, re-parse or re-verify" in instruction
    assert "unverified in a browser" not in instruction


# ── The check after a cell ───────────────────────────────────────────────────


def _artifact_with_pages(
    tmp_path: Path, *pages: str, primary: str = "index.html", artifact_type: str = "html-app"
) -> tuple[ArtifactStore, str]:
    store = ArtifactStore(tmp_path / "artifacts")
    slug = store.create(name="Map", description="d", type=artifact_type, primary=primary).slug
    for name in pages:
        (store.root / slug / name).write_text(_PAGE.replace("hi", name))
    return store, slug


def test_only_edited_pages_are_sent_primary_first_with_one_screenshot(service, tmp_path):
    store, slug = _artifact_with_pages(tmp_path, "old.html", primary="z.html")
    before = tool_handlers.snapshot_existing_artifacts(store)  # the artifact before the cell: only old.html
    for name in ("a.html", "z.html"):
        (store.root / slug / name).write_text(_PAGE.replace("hi", name))

    lint = tool_handlers.lint_changed_artifact_files(store, before)

    sent = [r["body"] for r in service.requests]
    assert [b["html"] for b in sent] == [_PAGE.replace("hi", "z.html"), _PAGE.replace("hi", "a.html")]
    # Each page is asked for a screenshot until one fails; none did, so the
    # first page's is kept.
    assert [b["screenshot"] for b in sent] == [True, True]
    assert lint.screenshot is not None and lint.screenshot_page == f"{slug}/z.html"
    assert lint.page_checks == {(slug, "z.html"): "passed", (slug, "a.html"): "passed"}
    assert lint.messages == [
        f"{slug}/z.html — loaded in a browser the way the app preview loads it: no problems found",
        f"{slug}/a.html — loaded in a browser the way the app preview loads it: no problems found",
    ]


def test_a_failed_page_lists_its_findings_and_wins_over_a_passed_one(service, tmp_path):
    store, slug = _artifact_with_pages(tmp_path, "index.html", "b.html")
    service.answer(
        (200, _answer("failed", findings=[
            {"kind": "separate_file", "viewport": "desktop", "detail": "data.js"},
            {"kind": "separate_file", "viewport": "phone", "detail": "data.js"},
        ]), 0.0),
        (200, _answer("passed", screenshot=False), 0.0),
    )
    lint = tool_handlers.lint_changed_artifact_files(store, {slug: tool_handlers.ArtifactFiles()})
    assert lint.messages[0] == f"{slug}/index.html — separate_file (desktop and phone): data.js"
    assert lint.page_checks == {(slug, "index.html"): "failed", (slug, "b.html"): "passed"}


def test_a_check_that_cannot_run_says_so_instead_of_nothing(service, project, policy):
    store, slug = _artifact_with_pages(project, "index.html")
    service.answer((503, {"error": "busy"}, 0.0))
    lint = tool_handlers.lint_changed_artifact_files(store, {slug: tool_handlers.ArtifactFiles()}, cell_sandbox=policy)
    assert lint.messages == [
        f"{slug}/index.html — not checked in a browser: the render-check service is busy or not ready (HTTP 503)"
    ]
    assert lint.page_checks == {(slug, "index.html"): "unverified"}
    assert lint.screenshot is None


def test_pages_past_the_cell_budget_are_named_as_not_checked(service, tmp_path, monkeypatch):
    monkeypatch.setattr(tool_handlers, "_RENDER_CHECK_CELL_BUDGET_SECONDS", 0.5)
    monkeypatch.setattr(tool_handlers, "_RENDER_CHECK_MIN_SECONDS", 0.4)
    store, slug = _artifact_with_pages(tmp_path, "index.html", "b.html")
    service.answer((200, _answer("passed"), 0.2))

    lint = tool_handlers.lint_changed_artifact_files(store, {slug: tool_handlers.ArtifactFiles()})

    assert len(service.requests) == 1
    assert lint.messages[-1] == (
        f"{slug}/b.html — not checked in a browser: the browser checks after this cell ran out of time before this page"
    )
    assert lint.page_checks == {(slug, "index.html"): "passed", (slug, "b.html"): "unverified"}


def _dashboard_with_data_file(tmp_path: Path, *, artifact_type: str = "html-app") -> tuple[ArtifactStore, str]:
    """A dashboard built under the old skill: its page loads a sibling data file."""
    store, slug = _artifact_with_pages(tmp_path, "details.html", artifact_type=artifact_type)
    folder = store.root / slug
    (folder / "index.html").write_text(_PAGE.replace("</body>", '<script src="dashboard_data.js"></script></body>'))
    (folder / "dashboard_data.js").write_text("window.DATA = [1, 2, 3];")
    return store, slug


_SEPARATE_FILE = {"kind": "separate_file", "viewport": "desktop", "detail": "dashboard_data.js"}


@pytest.mark.parametrize("artifact_type", ["html-app", "document", "mixed"])
@pytest.mark.parametrize("cell", ["rewrite", "delete"])
def test_a_cell_that_changes_only_a_file_the_page_loads_checks_the_page(service, tmp_path, cell, artifact_type):
    """The cell rewrites (or deletes) only `dashboard_data.js`. No page
    changed, but the artifact did: its primary is checked, fails, and the
    turn ends with the line that says so. Its other pages are not sent. Any
    artifact but a fullstack app, whose page needs its backend."""
    store, slug = _dashboard_with_data_file(tmp_path, artifact_type=artifact_type)
    data = store.root / slug / "dashboard_data.js"
    service.answer((200, _answer("failed", findings=[_SEPARATE_FILE]), 0.0))
    session = SimpleNamespace(_page_checks={})

    before = tool_handlers.snapshot_existing_artifacts(store)
    if cell == "rewrite":
        data.write_text("window.DATA = [4, 5, 6, 7];")
    else:
        data.unlink()
    lint = tool_handlers.lint_changed_artifact_files(store, before)
    tool_handlers.record_page_checks(session, checks=lint.page_checks)

    assert [r["body"]["html"] for r in service.requests] == [(store.root / slug / "index.html").read_text()]
    assert lint.page_checks == {(slug, "index.html"): "failed"}
    assert lint.messages == [f"{slug}/index.html — separate_file (desktop): dashboard_data.js"]
    assert tool_handlers.page_check_lines(store, checks=session._page_checks) == [
        "“Map” failed the browser check, so it may not display correctly in the preview."
    ]


def test_a_cell_that_renames_an_older_draft_into_place_checks_the_page(service, tmp_path):
    """`os.replace(draft.html, details.html)` keeps the draft's older mtime,
    so the artifact's newest mtime does not move. The page under
    `details.html` is new all the same, and gets checked. It is not the
    primary, so only its being counted as edited sends it; the primary the
    cell left alone is not sent."""
    store, slug = _artifact_with_pages(tmp_path, "index.html", "details.html", "draft.html")
    folder = store.root / slug
    draft = folder / "draft.html"
    draft.write_text(_PAGE.replace("hi", "the new details page"))
    os.utime(draft, (1, 1))
    service.answer((200, _answer("passed"), 0.0))

    before = tool_handlers.snapshot_existing_artifacts(store)
    os.replace(draft, folder / "details.html")
    lint = tool_handlers.lint_changed_artifact_files(store, before)

    assert [r["body"]["html"] for r in service.requests] == [_PAGE.replace("hi", "the new details page")]
    assert lint.page_checks == {(slug, "details.html"): "passed"}


def _fullstack_app(tmp_path: Path) -> tuple[ArtifactStore, str]:
    """A fullstack app: its page under `static/` calls the backend beside it."""
    store = ArtifactStore(tmp_path / "artifacts")
    slug = store.create(
        name="Orders", description="d", type="fullstack-stateless-app", primary="static/index.html"
    ).slug
    folder = store.root / slug
    (folder / "static").mkdir(exist_ok=True)
    (folder / "static" / "index.html").write_text(_PAGE.replace("</body>", "<script>api('/api/items')</script></body>"))
    (folder / "backend.py").write_text("ITEMS = [1]\n")
    return store, slug


_SAME_ORIGIN = {"kind": "separate_file", "viewport": "desktop", "detail": "api/items"}


def test_a_cell_that_changes_only_a_fullstack_backend_sends_no_page(service, tmp_path):
    """The service fails every request to the page's own origin, which a
    fullstack page makes to its backend. A cell that rewrites only
    `backend.py` checks no page: no request, no line, nothing recorded."""
    store, slug = _fullstack_app(tmp_path)
    service.answer((200, _answer("failed", findings=[_SAME_ORIGIN]), 0.0))

    before = tool_handlers.snapshot_existing_artifacts(store)
    (store.root / slug / "backend.py").write_text("ITEMS = [1, 2]\n")
    lint = tool_handlers.lint_changed_artifact_files(store, before)

    assert service.requests == []
    assert lint.page_checks == {}
    assert lint.messages == []


def test_an_edited_fullstack_page_is_named_as_not_checked(service, tmp_path):
    """The page itself is not sent either, so it gets no false
    `separate_file` for its own API calls: it is named as not checked."""
    store, slug = _fullstack_app(tmp_path)
    service.answer((200, _answer("failed", findings=[_SAME_ORIGIN]), 0.0))

    before = tool_handlers.snapshot_existing_artifacts(store)
    (store.root / slug / "static" / "index.html").write_text(_PAGE)
    lint = tool_handlers.lint_changed_artifact_files(store, before)

    assert service.requests == []
    assert lint.page_checks == {(slug, "static/index.html"): "unverified"}
    assert lint.messages == [
        f"{slug}/index.html — not checked in a browser: a fullstack app's page calls its backend, "
        "which the render check does not run"
    ]


async def test_what_an_unverified_check_found_reaches_the_agent_once_per_page(service, project, policy):
    """On a slow host every load of the page runs out of time after the
    check saw a blank map, which may still have been drawing. The first
    cell's result lists the blank map, so the agent can fix it once. The
    second cell's result says only that the page was not checked, so the
    agent reports it as unverified instead of editing it again and again."""
    service.answer((200, _answer("unverified", findings=[_BLANK, _TIMEOUT], screenshot=False), 0.0))
    session, pad, slug = _session_for(project, [
        _llm_response("One.", {"action": "exec", "name": "main", "code": "write_page()"}, "tc_1"),
        _llm_response("Two.", {"action": "exec", "name": "main", "code": "write_page()"}, "tc_2"),
        _llm_response("Done."),
    ])
    session._scratchpads._cell_sandbox = policy

    streamed, _ = await _final_text(session)

    results = [json.dumps(result["content"], ensure_ascii=False) for result in _tool_results(session)]
    assert len(pad.ran) == 2 and len(service.requests) == 2, "one check per cell"
    assert [result.count("blank_view") for result in results] == [1, 0]
    unchecked = (
        f"{slug}/index.html — not checked in a browser: the check ran out of time "
        "(the check's 25 s ran out while measuring the page)"
    )
    blank = (
        f"{slug}/index.html — before it ran out of time, the check found blank_view (desktop): "
        "#world-map has no drawn content (it may still have been drawing)"
    )
    first, second = (result["content"] for result in _tool_results(session))
    assert first.endswith(f"[artifact lint]\n{unchecked}\n{blank}")
    assert second.endswith(f"[artifact lint]\n{unchecked}")
    assert streamed.endswith(_LINE)


async def test_a_page_generate_artifact_left_unverified_gets_no_findings_again(service, tmp_path):
    """generate_artifact listed what its check found in `checks_skipped` and
    recorded the page as unverified. The agent fixes it in a cell, and that
    check is unverified the same way: it gets only the line that says so."""
    from anton.core.tools.generate_artifact import engine

    store, slug = _artifact_with_pages(tmp_path, "index.html")
    session = SimpleNamespace(_page_checks={}, _unverified_findings_listed=set(), _scratchpads=None)
    engine._record_page_check(session, state=SimpleNamespace(
        slug=slug, browser_status="unverified", browser_findings_listed=True, files_written=["index.html"],
        is_fullstack=False, primary="index.html",
    ), result={"status": "generated"})
    service.answer((200, _answer("unverified", findings=[_BLANK, _TIMEOUT], screenshot=False), 0.0))

    before = tool_handlers.snapshot_existing_artifacts(store)
    (store.root / slug / "index.html").write_text(_PAGE.replace("hi", "fixed"))
    lint = await tool_handlers.check_cell_artifacts(session, store=store, before=before)

    assert len(service.requests) == 1
    assert lint.messages == [
        f"{slug}/index.html — not checked in a browser: the check ran out of time "
        "(the check's 25 s ran out while measuring the page)"
    ]
    assert session._page_checks == {(slug, "index.html"): "unverified"}


_NOT_READY = (503, {"detail": "the render check isn't ready: starting"}, 0.0)
_BLANK_BEFORE_THE_DEADLINE = (200, _answer("unverified", findings=[_BLANK, _TIMEOUT], screenshot=False), 0.0)


@pytest.mark.parametrize("browser_status,result", [
    pytest.param(None, "Frontend verification failed after 2 attempt(s): no viewport meta", id="static-failure"),
    pytest.param("unverified", {"status": "generated"}, id="its-check-got-no-answer"),
])
async def test_a_page_generate_artifact_listed_nothing_for_gets_its_findings_in_the_next_cell(
    service, tmp_path, browser_status, result
):
    """generate_artifact left the page unverified without listing anything
    the check found: its static checks failed and no browser loaded the
    page, or its check got no answer. The agent then builds the page by
    hand, and the cell's check sees a blank map before the deadline. That
    blank map is listed: the agent never saw it."""
    from anton.core.tools.generate_artifact import engine

    store, slug = _artifact_with_pages(tmp_path, "index.html")
    session = SimpleNamespace(_page_checks={}, _unverified_findings_listed=set(), _scratchpads=None)
    engine._record_page_check(session, state=SimpleNamespace(
        slug=slug, browser_status=browser_status, browser_findings_listed=False, files_written=["index.html"],
        is_fullstack=False, primary="index.html",
    ), result=result)
    assert session._page_checks == {(slug, "index.html"): "unverified"}
    assert session._unverified_findings_listed == set()
    service.answer(_BLANK_BEFORE_THE_DEADLINE)

    before = tool_handlers.snapshot_existing_artifacts(store)
    (store.root / slug / "index.html").write_text(_PAGE.replace("hi", "hand built"))
    lint = await tool_handlers.check_cell_artifacts(session, store=store, before=before)

    assert lint.messages == [
        f"{slug}/index.html — not checked in a browser: the check ran out of time "
        "(the check's 25 s ran out while measuring the page)",
        f"{slug}/index.html — before it ran out of time, the check found blank_view (desktop): "
        "#world-map has no drawn content (it may still have been drawing)",
    ]
    assert session._unverified_findings_listed == {(slug, "index.html")}


def test_generate_marks_the_page_listed_only_when_its_result_carries_the_listing(tmp_path):
    """The agent reads what generate_artifact's unverified check found in
    the result's `checks_skipped`, which a failed run's text does not carry."""
    from anton.core.tools.generate_artifact import engine

    key = tool_handlers.PageKey("dash", "index.html")
    st = SimpleNamespace(
        slug="dash", browser_status="unverified", browser_findings_listed=True, files_written=["index.html"],
        is_fullstack=False, primary="index.html",
    )
    session = SimpleNamespace(_page_checks={}, _unverified_findings_listed=set())
    engine._record_page_check(session, state=st, result="Frontend verification failed after 2 attempt(s): x")
    assert session._unverified_findings_listed == set()

    st.browser_status = "failed"
    engine._record_page_check(session, state=st, result={"status": "stopped_over_budget"})
    assert session._unverified_findings_listed == set()

    st.browser_status = "unverified"
    engine._record_page_check(session, state=st, result={"status": "generated"})
    assert session._page_checks == {key: "unverified"}
    assert session._unverified_findings_listed == {key}


async def test_a_check_that_listed_nothing_leaves_the_next_one_free_to_list(service, project):
    """Cell 1's check got no answer, so it listed nothing. Cell 2's check
    sees a blank map before the deadline: it lists the blank map, which the
    agent has not seen. Cell 3's check sees the same and lists nothing,
    since the agent already had its one chance to fix it. One request per
    cell, none resent."""
    service.answer(_NOT_READY, _BLANK_BEFORE_THE_DEADLINE)
    session, pad, slug = _session_for(project, [
        _llm_response("One.", {"action": "exec", "name": "main", "code": "write_page()"}, "tc_1"),
        _llm_response("Two.", {"action": "exec", "name": "main", "code": "write_page()"}, "tc_2"),
        _llm_response("Three.", {"action": "exec", "name": "main", "code": "write_page()"}, "tc_3"),
        _llm_response("Done."),
    ])

    await _final_text(session)

    results = [result["content"] for result in _tool_results(session)]
    assert len(pad.ran) == 3 and len(service.requests) == 3
    unchecked = f"{slug}/index.html — not checked in a browser: the check ran out of time"
    assert results[0].endswith(
        f"[artifact lint]\n{slug}/index.html — not checked in a browser: "
        "the render-check service is busy or not ready (HTTP 503)"
    )
    assert unchecked in results[1] and unchecked in results[2]
    assert [result.count("blank_view") for result in results] == [0, 1, 0]
    assert session._unverified_findings_listed == {(slug, "index.html")}


async def test_each_turn_lists_what_an_unverified_check_found_once(service, project):
    """The record of pages already listed is the turn's: the next turn's
    first unverified check of the page lists its findings again."""
    service.answer(_BLANK_BEFORE_THE_DEADLINE)
    session, _, _ = _session_for(project, [
        _llm_response("One.", {"action": "exec", "name": "main", "code": "write_page()"}, "tc_1"),
        _llm_response("Done."),
        _llm_response("Two.", {"action": "exec", "name": "main", "code": "write_page()"}, "tc_2"),
        _llm_response("Done again."),
    ])
    try:
        for prompt in ("draw the map", "draw it once more"):
            async for _ in session.turn_stream(prompt):
                pass
    finally:
        await session.close()

    results = [result["content"] for result in _tool_results(session)]
    assert len(service.requests) == 2
    assert [result.count("blank_view") for result in results] == [1, 1]


def test_the_screenshot_is_the_failed_pages_when_the_primary_passed(service, tmp_path):
    """The primary passes and the next page fails: the agent sees the page
    it has to fix. Once a failed page sent one, the pages after it are asked
    for none."""
    store, slug = _artifact_with_pages(tmp_path, "index.html", "b.html", "c.html")
    failed_shot = base64.b64encode(b"the failed page").decode()
    failed = _answer("failed", findings=[_BLANK])
    failed["screenshot"]["data"] = failed_shot
    service.answer(
        (200, _answer("passed"), 0.0),
        (200, failed, 0.0),
        (200, _answer("failed", findings=[_BLANK], screenshot=False), 0.0),
    )

    lint = tool_handlers.lint_changed_artifact_files(store, {slug: tool_handlers.ArtifactFiles()})

    assert [r["body"]["screenshot"] for r in service.requests] == [True, True, False]
    assert lint.screenshot_page == f"{slug}/b.html"
    assert lint.screenshot.data == failed_shot


@pytest.mark.parametrize("answer,lines", [
    pytest.param(
        _answer("unverified", findings=[_BLANK, _TIMEOUT], screenshot=False),
        [
            "not checked in a browser: the check ran out of time "
            "(the check's 25 s ran out while measuring the page)",
            "before it ran out of time, the check found blank_view (desktop): "
            "#world-map has no drawn content (it may still have been drawing)",
        ],
        id="a-blank-map-before-the-deadline",
    ),
    pytest.param(
        _answer("unverified", findings=[_THROWN], unreachable=[_MISSING], screenshot=False),
        [
            "not checked in a browser: could not load "
            "https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js (not in the check's mirror)",
            "the check still found uncaught_exception (desktop): ReferenceError: renderChart is not defined",
        ],
        id="an-exception-beside-a-library-the-check-could-not-load",
    ),
])
def test_an_unverified_page_lists_what_the_check_still_found(service, tmp_path, answer, lines):
    """A check that could not vouch for the page still saw a problem: the
    agent sees it under the line that says the page was not checked, and
    can fix it. The deadline is named once, in that line, and a blank map
    seen before it may still have been drawing."""
    store, slug = _artifact_with_pages(tmp_path, "index.html")
    service.answer((200, answer, 0.0))

    lint = tool_handlers.lint_changed_artifact_files(store, {slug: tool_handlers.ArtifactFiles()})

    assert lint.messages == [f"{slug}/index.html — {line}" for line in lines]
    assert lint.page_checks == {(slug, "index.html"): "unverified"}


def test_the_screenshot_rides_a_successful_cell_only(tmp_path):
    shot = render_check.RenderScreenshot(media_type="image/jpeg", data=_SHOT)
    lint = tool_handlers.ArtifactLint(messages=["dash/index.html — ok"], screenshot=shot, screenshot_page="dash/index.html")

    content = tool_handlers.cell_content_with_lint("[output]\nhi", lint, cell_ok=True)
    assert content[0] == {"type": "text", "text": "[output]\nhi\n\n[artifact lint]\ndash/index.html — ok"}
    assert content[1]["text"].startswith(render_check.SCREENSHOT_CAPTION)
    assert content[2]["type"] == "image"

    failed = tool_handlers.cell_content_with_lint("[error]\nboom", lint, cell_ok=False)
    assert failed == "[error]\nboom\n\n[artifact lint]\ndash/index.html — ok"


def test_the_record_keeps_each_pages_latest_check(tmp_path):
    key = tool_handlers.PageKey
    session = SimpleNamespace(_page_checks={key("a", "index.html"): "passed"})
    tool_handlers.record_page_checks(
        session, checks={key("a", "index.html"): "unverified", key("b", "index.html"): "passed"}
    )
    assert session._page_checks == {key("a", "index.html"): "unverified", key("b", "index.html"): "passed"}
    # A session without the record (a test double) is left alone.
    tool_handlers.record_page_checks(SimpleNamespace(), checks={key("a", "index.html"): "failed"})


def test_page_lines_name_each_artifact_with_a_page_that_did_not_pass(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    shipped = store.create(name="Shipments map", description="d", type="html-app").slug
    broken = store.create(name="Sales globe", description="d", type="html-app").slug
    key = tool_handlers.PageKey
    lines = tool_handlers.page_check_lines(store, checks={
        key(shipped, "index.html"): "unverified",
        key(broken, "index.html"): "failed",
        key(broken, "details.html"): "passed",
        key("ok", "index.html"): "passed",
    })
    assert lines == [
        "“Shipments map” was not verified in a browser, so it may not display correctly in the preview.",
        "“Sales globe” failed the browser check, so it may not display correctly in the preview.",
    ]


def test_a_later_pass_of_another_page_keeps_the_failed_one(service, tmp_path):
    """Two cells, two pages of one artifact: the first cell's failed
    index.html stays failed when the second cell edits only details.html and
    it passes. One record per artifact let that pass hide the failure the
    preview still shows."""
    store, slug = _artifact_with_pages(tmp_path, "details.html")
    index, details = store.root / slug / "index.html", store.root / slug / "details.html"
    session = SimpleNamespace(_page_checks={})
    service.answer(
        (200, _answer("failed", findings=[
            {"kind": "csp_violation", "viewport": "desktop", "detail": "style-src blocked leaflet.css"},
        ]), 0.0),
        (200, _answer("passed"), 0.0),
    )

    before = tool_handlers.snapshot_existing_artifacts(store)
    index.write_text(_PAGE.replace("hi", "index.html"))
    checks = tool_handlers.lint_changed_artifact_files(store, before).page_checks
    tool_handlers.record_page_checks(session, checks=checks)
    before = tool_handlers.snapshot_existing_artifacts(store)
    details.write_text(_PAGE)
    checks = tool_handlers.lint_changed_artifact_files(store, before).page_checks
    tool_handlers.record_page_checks(session, checks=checks)

    assert [r["body"]["html"] for r in service.requests] == [_PAGE.replace("hi", "index.html"), _PAGE]
    assert tool_handlers.page_check_lines(store, checks=session._page_checks) == [
        "“Map” failed the browser check, so it may not display correctly in the preview."
    ]


# ── Screenshots in history ───────────────────────────────────────────────────


def _result_with_shot(tool_use_id: str, data: str) -> dict:
    shot = render_check.RenderScreenshot(media_type="image/jpeg", data=data)
    return {
        "role": "user",
        "content": [{
            "type": "tool_result", "tool_use_id": tool_use_id,
            "content": [{"type": "text", "text": "[output]\n"},
                        *render_check.screenshot_blocks(shot, page="dash/index.html")],
        }],
    }


def test_only_the_newest_render_screenshot_stays_in_history():
    other = base64.b64encode(b"second").decode()
    read_image = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "img", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
            {"type": "text", "text": "Loaded chart.png (10x10, 3 B)."},
        ]}],
    }
    history = [_result_with_shot("t1", _SHOT), read_image, _result_with_shot("t2", other)]

    assert render_check.drop_older_screenshots(history) == 1

    first = history[0]["content"][0]["content"]
    assert first[-1] == {"type": "text", "text": render_check.OLDER_SCREENSHOT_PLACEHOLDER}
    assert history[1] == read_image, "an image the agent read itself is not a render screenshot"
    assert history[2]["content"][0]["content"][-1]["source"]["data"] == other
    assert render_check.drop_older_screenshots(history) == 0


# ── The streaming turn: the product path ─────────────────────────────────────


class _FakeAsyncIter:
    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


class _PagePad:
    """A scratchpad that, like the real one, writes the page to disk and
    hands back the cell. Records each code it ran."""

    def __init__(self, page: Path) -> None:
        self.page = page
        self.ran: list[str] = []

    async def install_packages(self, packages):  # pragma: no cover - unused
        return "nothing to install"

    async def execute_streaming(self, code, **kwargs):
        from anton.core.backends.base import Cell

        self.ran.append(code)
        self.page.write_text(_PAGE.replace("hi", f"cell {len(self.ran)}"))
        yield Cell(code=code, stdout="wrote the page", stderr="", error=None, description="write page")


def _llm_response(text: str, tool_input: dict | None = None, tool_id: str = "tc_1"):
    from anton.core.llm.provider import LLMResponse, ToolCall, Usage

    return LLMResponse(
        content=text,
        tool_calls=[ToolCall(id=tool_id, name="scratchpad", input=tool_input)] if tool_input else [],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="tool_use" if tool_input else "end_turn",
    )


def _session_for(project: Path, replies: list, *, verdicts=()):
    """A ChatSession whose model plays `replies` in order and whose pad writes
    the artifact's page. Returns (session, pad, slug)."""
    from unittest.mock import AsyncMock, MagicMock

    from anton.core.llm.provider import StreamComplete, StreamTextDelta
    from anton.core.session import ChatSession, ChatSessionConfig, _VerifierVerdict
    from tests.conftest import make_mock_llm

    workspace = MagicMock(base=project)
    workspace.artifacts_dir = project / "artifacts"
    slug = ArtifactStore(workspace.artifacts_dir).create(
        name="Shipments map", description="d", type="html-app", primary="index.html"
    ).slug
    pad = _PagePad(workspace.artifacts_dir / slug / "index.html")

    mock_llm = make_mock_llm()
    pending = list(verdicts)

    async def verdict(*args, **kwargs):
        return pending.pop(0) if pending else _VerifierVerdict(status="COMPLETE", reason="done")

    mock_llm.generate_object_code = AsyncMock(side_effect=verdict)
    queue = list(replies)

    def plan_stream(**kwargs):
        reply = queue.pop(0) if queue else _llm_response("")
        events = [StreamTextDelta(text=reply.content)] if reply.content else []
        return _FakeAsyncIter([*events, StreamComplete(response=reply)])

    mock_llm.plan_stream = plan_stream
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))
    session._scratchpads.get_or_create = AsyncMock(return_value=pad)
    return session, pad, slug


def _tool_results(session) -> list:
    return [
        block
        for message in session.history if message["role"] == "user" and isinstance(message["content"], list)
        for block in message["content"] if isinstance(block, dict) and block.get("type") == "tool_result"
    ]


async def test_the_cell_result_carries_the_screenshot_on_the_streaming_path(service, project, policy):
    session, _, slug = _session_for(project, [
        _llm_response("Writing.", {"action": "exec", "name": "main", "code": "write_page()"}),
        _llm_response("Done."),
    ])
    session._scratchpads._cell_sandbox = policy
    try:
        async for _ in session.turn_stream("draw the map"):
            pass
    finally:
        await session.close()

    assert [r["body"]["html"] for r in service.requests] == [_PAGE.replace("hi", "cell 1")]
    text, caption, image = _tool_results(session)[0]["content"]
    assert text["text"].endswith(
        f"[artifact lint]\n{slug}/index.html — loaded in a browser the way the app preview loads it: no problems found"
    )
    assert caption["text"].startswith(render_check.SCREENSHOT_CAPTION)
    assert image["source"]["data"] == _SHOT
    assert session._page_checks == {(slug, "index.html"): "passed"}


async def test_two_cells_leave_one_screenshot_in_history(service, project):
    second = base64.b64encode(b"second page").decode()
    service.answer((200, _answer(), 0.0), (200, {**_answer(), "screenshot": {"media_type": "image/jpeg", "data": second}}, 0.0))
    session, _, _ = _session_for(project, [
        _llm_response("One.", {"action": "exec", "name": "main", "code": "write_page()"}, "tc_1"),
        _llm_response("Two.", {"action": "exec", "name": "main", "code": "write_page()"}, "tc_2"),
        _llm_response("Done."),
    ])
    try:
        async for _ in session.turn_stream("draw the map"):
            pass
    finally:
        await session.close()

    images = [
        block["source"]["data"]
        for result in _tool_results(session) if isinstance(result["content"], list)
        for block in result["content"] if block.get("type") == "image"
    ]
    assert images == [second]
    first = _tool_results(session)[0]["content"]
    assert {"type": "text", "text": render_check.OLDER_SCREENSHOT_PLACEHOLDER} in first


async def _final_text(session, prompt: str = "draw the map") -> tuple[str, list]:
    from anton.core.llm.provider import StreamTextDelta

    events = []
    try:
        async for event in session.turn_stream(prompt):
            events.append(event)
    finally:
        await session.close()
    return "".join(e.text for e in events if isinstance(e, StreamTextDelta)), events


_LINE = "“Shipments map” was not verified in a browser, so it may not display correctly in the preview."


async def test_a_sandboxed_turn_ends_with_a_line_for_an_unverified_page(service, project, policy):
    """The model can claim the page works whatever the checks said; this line
    is anton's own, appended after the reply as a paragraph of its own, and
    stays in history and in the turn's explainability record."""
    from anton.explainability import ExplainabilityStore

    service.answer((503, {"error": "busy"}, 0.0))
    session, _, _ = _session_for(project, [
        _llm_response("Writing.", {"action": "exec", "name": "main", "code": "write_page()"}),
        _llm_response("The map is ready and works."),
    ])
    session._scratchpads._cell_sandbox = policy

    streamed, _ = await _final_text(session)

    assert streamed == "Writing.The map is ready and works.\n\n" + _LINE
    assert ExplainabilityStore(project).load_latest().answer_text == streamed
    last = session.history[-1]
    assert last["role"] == "assistant"
    texts = last["content"] if isinstance(last["content"], str) else "".join(b["text"] for b in last["content"])
    assert texts.endswith(_LINE)


@pytest.mark.parametrize("reply", ["", "\n"])
async def test_a_turn_with_no_reply_text_ends_with_only_the_line(service, project, policy, reply):
    """With no reply before it, or one of only whitespace, the line opens
    the message: no blank line goes first."""
    service.answer((503, {"error": "busy"}, 0.0))
    session, _, _ = _session_for(project, [
        _llm_response("", {"action": "exec", "name": "main", "code": "write_page()"}),
        _llm_response(reply),
    ])
    session._scratchpads._cell_sandbox = policy

    streamed, _ = await _final_text(session)

    assert streamed == reply + _LINE


async def test_a_page_that_passed_gets_no_line(service, project, policy):
    session, _, _ = _session_for(project, [
        _llm_response("Writing.", {"action": "exec", "name": "main", "code": "write_page()"}),
        _llm_response("The map is ready."),
    ])
    session._scratchpads._cell_sandbox = policy
    streamed, _ = await _final_text(session)
    assert streamed == "Writing.The map is ready."


async def test_without_a_sandbox_the_turn_gets_no_line(service, project):
    service.answer((503, {"error": "busy"}, 0.0))
    session, _, _ = _session_for(project, [
        _llm_response("Writing.", {"action": "exec", "name": "main", "code": "write_page()"}),
        _llm_response("Done."),
    ])
    streamed, _ = await _final_text(session)
    assert _LINE not in streamed


async def test_the_line_after_an_unspent_continuation_is_marked_as_no_replacement(
    service, project, policy, monkeypatch
):
    """A consumer takes the first text after a continuation boundary for the
    answer that replaces the one it showed. When the continuation said
    nothing, the line must not wipe the answer the user read: a hand-back
    marker goes first."""
    from anton.core.llm.provider import StreamTaskProgress, StreamTextDelta
    from anton.core.session import _VerifierVerdict

    # The completion check, which forces the continuation, skips turns of
    # fewer tool rounds than this.
    monkeypatch.setenv("ANTON_VERIFY_MIN_TOOL_ROUNDS", "1")

    service.answer((503, {"error": "busy"}, 0.0))
    session, _, _ = _session_for(project, [
        _llm_response("Writing.", {"action": "exec", "name": "main", "code": "write_page()"}),
        _llm_response("ANSWER"),
        _llm_response(""),
    ], verdicts=[_VerifierVerdict(status="INCOMPLETE", reason="not done")])
    session._scratchpads._cell_sandbox = policy

    _, events = await _final_text(session)

    phases = [(i, e.phase) for i, e in enumerate(events) if isinstance(e, StreamTaskProgress)]
    line = next(i for i, e in enumerate(events) if isinstance(e, StreamTextDelta) and e.text == "\n\n" + _LINE)
    assert any(p == "continuation" for _, p in phases)
    assert events[line - 1] == StreamTaskProgress(phase="handback", message="")


# ── A model that takes no image input ────────────────────────────────────────

# What OpenAI answers on chat.completions when a request carries an image
# part and the model takes no image input. Its `param` points into the
# content array in a form `classify_content_rejection` does not read, so the
# SDK's own BadRequestError reaches the session.
_NO_IMAGES_CHAT = {"error": {
    "message": "Invalid content type. image_url is only supported by certain models.",
    "type": "invalid_request_error", "param": "messages.[3].content.[1].type", "code": None,
}}
# The same refusal with the `param` spelling the Responses API uses
# (`input[N].content[M].type`), which `classify_content_rejection` reads as
# content: a ContentValidationError reaches the session.
_NO_IMAGES_RESPONSES = {"error": {
    "message": "Invalid content type. input_image is only supported by certain models.",
    "type": "invalid_request_error", "param": "input[4].content[1].type", "code": "invalid_value",
}}
# A refusal that names no content path at all, as a gateway in front of a
# text-only model sends it.
_NO_IMAGES_PLAIN = {"error": {
    "message": "This model does not support image input.",
    "type": "invalid_request_error", "param": None, "code": None,
}}
_EXCEPTION = "TypeError: Cannot read properties of undefined (reading 'kind')"
_FAILED_WITH_SHOT = _answer(
    "failed", findings=[{"kind": "uncaught_exception", "viewport": "desktop", "detail": _EXCEPTION}],
)


def _image_parts(body: dict) -> int:
    """Image parts in a chat.completions or Responses request body."""
    items = body.get("messages") or body.get("input") or []
    return sum(
        1
        for item in items if isinstance(item, dict) and isinstance(item.get("content"), list)
        for part in item["content"] if isinstance(part, dict) and part.get("type") in ("image_url", "input_image")
    )


def _chat_sse(reply: "str | dict") -> bytes:
    def chunk(delta: dict, finish: str | None = None, usage: dict | None = None) -> dict:
        choices = [] if usage else [{"index": 0, "delta": delta, "finish_reason": finish}]
        return {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "gpt-test",
                "choices": choices, **({"usage": usage} if usage else {})}

    if isinstance(reply, str):
        chunks = [chunk({"role": "assistant", "content": reply}), chunk({}, "stop")]
    else:
        call = {"index": 0, "id": reply["id"], "type": "function",
                "function": {"name": reply.get("name", "scratchpad"), "arguments": json.dumps(reply["input"])}}
        chunks = [chunk({"role": "assistant", "tool_calls": [call]}), chunk({}, "tool_calls")]
    chunks.append(chunk({}, usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}))
    return ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode()


def _responses_sse(reply: "str | dict") -> bytes:
    usage = {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 5,
             "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 15}

    def response(status: str) -> dict:
        return {"id": "resp_1", "object": "response", "created_at": 0, "model": "gpt-test", "status": status,
                "output": [], "usage": usage, "parallel_tool_calls": True, "tool_choice": "auto", "tools": []}

    events: list[dict] = [{"type": "response.created", "response": {**response("in_progress"), "usage": None}}]
    if isinstance(reply, str):
        events += [
            {"type": "response.output_item.added", "output_index": 0,
             "item": {"type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": []}},
            {"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0, "content_index": 0,
             "delta": reply, "logprobs": []},
        ]
    else:
        arguments = json.dumps(reply["input"])
        events += [
            {"type": "response.output_item.added", "output_index": 0,
             "item": {"type": "function_call", "id": "fc_1", "call_id": reply["id"],
                      "name": reply.get("name", "scratchpad"), "arguments": "", "status": "in_progress"}},
            {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "output_index": 0,
             "delta": arguments},
            {"type": "response.function_call_arguments.done", "item_id": "fc_1", "output_index": 0,
             "arguments": arguments},
        ]
    events.append({"type": "response.completed", "response": response("completed")})
    return "".join(
        f"event: {e['type']}\ndata: {json.dumps({'sequence_number': i, **e})}\n\n" for i, e in enumerate(events)
    ).encode()


# What `_TextOnlyModel` answers past `_REQUEST_CAP`: the session raises a
# provider 401 at once and never resends it.
_OVER_THE_CAP = {"error": {
    "message": f"more than {_REQUEST_CAP} requests in one test",
    "type": "authentication_error", "param": None, "code": "invalid_api_key",
}}


class _TextOnlyModel:
    """An OpenAI-compatible API for a model that takes no image input.

    Real HTTP, so the real SDK and anton's real provider are in the path. A
    request `refuses` picks gets the 400 `refusal`, here any request carrying
    an image part; any other gets the next scripted reply: a tool call (a
    dict, a scratchpad call unless it names another tool) or an answer (a
    str). Every request past `_REQUEST_CAP` gets a 401, which ends the turn
    at once with a `ProviderAuthError`.
    """

    def __init__(self, refusal: dict, replies: list) -> None:
        self.refusal, self.replies, self.requests = refusal, list(replies), []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):  # noqa: N802 - the stdlib's name
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append(body)
                if len(owner.requests) > _REQUEST_CAP:
                    code, kind, data = 401, "application/json", json.dumps(_OVER_THE_CAP).encode()
                elif owner.refuses(body):
                    code, kind, data = 400, "application/json", json.dumps(owner.refusal).encode()
                else:
                    sse = _responses_sse if self.path.endswith("/responses") else _chat_sse
                    code, kind, data = 200, "text/event-stream", sse(owner.replies.pop(0))
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def refuses(self, body: dict) -> bool:
        return _image_parts(body) > 0

    def provider(self, flavor: str):
        from anton.core.llm.openai import OpenAIProvider

        return OpenAIProvider(
            api_key="k", base_url=f"http://127.0.0.1:{self.server.server_port}/v1", flavor=flavor,
        )


# The person's own image, which `_VisionModel` refuses.
_PERSONS_IMAGE = base64.b64encode(b"the person's logo").decode()


class _VisionModel(_TextOnlyModel):
    """A model that takes images: it refuses only a request carrying the
    person's image (`_PERSONS_IMAGE`), and takes render screenshots."""

    def refuses(self, body: dict) -> bool:
        return _PERSONS_IMAGE in json.dumps(body)


def _cell(tool_id: str) -> dict:
    return {"id": tool_id, "input": {"action": "exec", "name": "main", "code": "write_page()",
                                     "one_line_description": "write the page",
                                     "estimated_execution_time_seconds": 1}}


def _on_model(session, model: _TextOnlyModel, flavor: str) -> None:
    """Route the session's planning calls to `model` through anton's real provider."""
    provider = model.provider(flavor)
    # `_session_for`'s workspace is a MagicMock; its project context would
    # reach the wire as a mock the SDK cannot serialize.
    session._workspace.build_anton_md_context.return_value = ""

    def plan_stream(*, system, messages, tools=None, max_tokens=None, native_web_tools=None):
        return provider.stream(model="gpt-test", system=system, messages=messages, tools=tools,
                               max_tokens=max_tokens or 1024)

    session._llm.plan_stream = plan_stream


class _Refusal(NamedTuple):
    """A transport, a refusal it can get, and how a turn whose request carries
    no render screenshot ends on it, as it did before screenshots were
    dropped: the error that ends the turn and how many requests it sent."""

    flavor: str
    body: dict
    terminal: str
    attempts: int


_REFUSALS = {
    "chat.completions": _Refusal("openai-compatible-generic", _NO_IMAGES_CHAT, "BadRequestError", 4),
    "responses-content-param": _Refusal("openai", _NO_IMAGES_RESPONSES, "ContentValidationError", 1),
    "responses-no-param": _Refusal("openai", _NO_IMAGES_PLAIN, "BadRequestError", 4),
}


@pytest.mark.parametrize("case", sorted(_REFUSALS))
async def test_a_model_that_refuses_images_still_gets_the_findings_in_the_same_turn(service, project, case):
    """The render check attaches a screenshot; a model that takes no image
    input refuses the next request with a 400. The turn drops the screenshot,
    keeps the findings beside it, sends the request once more and ends with
    the model's answer. The next cell's check sends no screenshot."""
    refusal = _REFUSALS[case]
    service.answer((200, _FAILED_WITH_SHOT, 0.0))
    model = _TextOnlyModel(refusal.body, [_cell("call_1"), _cell("call_2"), "The map works now."])
    session, pad, slug = _session_for(project, [])
    _on_model(session, model, refusal.flavor)
    try:
        streamed, _ = await _final_text(session)
    finally:
        model.server.shutdown()

    assert streamed.endswith("The map works now.")
    assert len(pad.ran) == 2 and len(service.requests) == 2, "both cells ran and both pages were checked"
    # The first cell's result with its screenshot, refused once; the same
    # request without it; the second cell's result, which carries none.
    assert [_image_parts(body) for body in model.requests] == [0, 1, 0, 0]
    retried, after_second_cell = (json.dumps(body, ensure_ascii=False) for body in model.requests[2:])
    finding = f"{slug}/index.html — uncaught_exception (desktop): {_EXCEPTION}"
    assert finding in retried
    assert render_check.DROPPED_SCREENSHOT_PLACEHOLDER in retried
    assert render_check.REFUSED_SCREENSHOT_PLACEHOLDER not in retried
    # The resend went through, so the note says the model refused it, and
    # the second check asked the service for no screenshot.
    assert after_second_cell.count(finding) == 2
    assert render_check.REFUSED_SCREENSHOT_PLACEHOLDER in after_second_cell
    assert [r["body"]["screenshot"] for r in service.requests] == [True, False]
    assert session._render_screenshots_refused is True
    history = json.dumps(session.history)
    assert "An error interrupted execution" not in history
    assert "The task has failed" not in history


@pytest.mark.parametrize("case", sorted(_REFUSALS))
async def test_a_refusal_with_no_screenshot_in_the_request_ends_the_turn_as_before(service, project, case):
    """A screenshot is anton's own addition; an image the person attached is
    not. With no render screenshot to drop, the refusal takes the path it
    took before: the same error ends the turn after the same requests."""
    refusal = _REFUSALS[case]
    model = _TextOnlyModel(refusal.body, [])
    session, _, _ = _session_for(project, [])
    _on_model(session, model, refusal.flavor)
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}}
    try:
        with pytest.raises(Exception) as refused:
            async for _ in session.turn_stream([{"type": "text", "text": "what is this?"}, image]):
                pass
    finally:
        await session.close()
        model.server.shutdown()

    assert type(refused.value).__name__ == refusal.terminal
    assert len(model.requests) == refusal.attempts
    assert session._render_screenshots_refused is False


def _read_image(tool_id: str, *, path: str) -> dict:
    return {"id": tool_id, "name": "read_image", "input": {"file_path": path}}


async def _turns(session, *prompts: str) -> list[str]:
    """Run one turn per prompt on the same session, as the CLI does; the
    streamed text of each."""
    from anton.core.llm.provider import StreamTextDelta

    texts = []
    try:
        for prompt in prompts:
            events = [event async for event in session.turn_stream(prompt)]
            texts.append("".join(e.text for e in events if isinstance(e, StreamTextDelta)))
    finally:
        await session.close()
    return texts


@pytest.mark.parametrize("case", sorted(_REFUSALS))
async def test_a_model_that_refuses_images_still_answers_after_read_image(project, case):
    """The agent reads back an image it drew; a model that takes no image
    input refuses the request that carries it. The turn replaces the image
    with a note, sends the request once more and ends with the model's
    answer. From then on `read_image` answers in text, so a later call
    costs no refused request. Render screenshots stay on: they did not
    cause this refusal."""
    refusal = _REFUSALS[case]
    (project / "digits.png").write_bytes(b"\x89PNG\r\n\x1a\n five digits")
    model = _TextOnlyModel(refusal.body, [
        _read_image("call_1", path="digits.png"), "I could not view the image.",
        _read_image("call_2", path="digits.png"), "I still cannot view it.",
    ])
    session, _, _ = _session_for(project, [])
    _on_model(session, model, refusal.flavor)
    try:
        first, second = await _turns(session, "what digits are in digits.png?", "try once more")
    finally:
        model.server.shutdown()

    assert first.endswith("I could not view the image.")
    assert second.endswith("I still cannot view it.")
    # The read_image call; its result with the image, refused; the same
    # request without it. The second turn's call sends no image at all.
    assert [_image_parts(body) for body in model.requests] == [0, 1, 0, 0, 0]
    resent = json.dumps(model.requests[2], ensure_ascii=False)
    assert render_check.DROPPED_TOOL_IMAGE_PLACEHOLDER in resent
    assert "Loaded digits.png" in resent, "the text beside the image stays"
    # The resend went through, so the note says the model refused the image.
    assert render_check.REFUSED_TOOL_IMAGE_PLACEHOLDER in json.dumps(model.requests[3], ensure_ascii=False)
    assert "digits.png was not loaded" in json.dumps(model.requests[4], ensure_ascii=False)
    assert session._images_refused is True
    assert session._render_screenshots_refused is False
    history = json.dumps(session.history)
    assert "An error interrupted execution" not in history
    assert "The task has failed" not in history


@pytest.mark.parametrize("case", sorted(_REFUSALS))
async def test_a_cell_after_a_refused_read_image_drops_its_screenshot_once(service, project, case):
    """The model refused the image `read_image` loaded and took the request
    without it. That says nothing about render screenshots, so the next
    cell's check still asks for one. A model with no image input refuses
    that request too; the turn drops the screenshot once, sends the request
    without it and ends with the model's answer. One drop of each kind."""
    refusal = _REFUSALS[case]
    (project / "digits.png").write_bytes(b"\x89PNG\r\n\x1a\n five digits")
    model = _TextOnlyModel(refusal.body, [
        _read_image("call_1", path="digits.png"), _cell("call_2"), "The map is drawn.",
    ])
    session, pad, _ = _session_for(project, [])
    _on_model(session, model, refusal.flavor)
    try:
        streamed, _ = await _final_text(session)
    finally:
        model.server.shutdown()

    assert streamed.endswith("The map is drawn.")
    assert len(pad.ran) == 1
    assert [r["body"]["screenshot"] for r in service.requests] == [True]
    # The read_image call; its result with the image, refused; the same
    # request without it; the cell's result with its screenshot, refused;
    # the same request without it.
    assert [_image_parts(body) for body in model.requests] == [0, 1, 0, 1, 0]
    assert render_check.DROPPED_SCREENSHOT_PLACEHOLDER in json.dumps(model.requests[4], ensure_ascii=False)
    assert session._images_refused is True
    assert session._render_screenshots_refused is True


# An image `_RefusesOneImage` refuses: a PNG signature with no image after
# it, which a model that takes images cannot decode.
_CORRUPT_PNG = b"\x89PNG\r\n\x1a\n a PNG cut off after its signature"
# A refusal of one image the provider could not read, naming no content path.
_UNREADABLE_400 = {"error": {
    "message": "You uploaded an unsupported image. Please make sure your image is valid.",
    "type": "invalid_request_error", "param": None, "code": None,
}}


class _RefusesOneImage(_TextOnlyModel):
    """A model that takes images and refuses only a request carrying
    `_CORRUPT_PNG`."""

    def refuses(self, body: dict) -> bool:
        return base64.b64encode(_CORRUPT_PNG).decode() in json.dumps(body)


async def test_a_vision_model_that_refuses_one_read_image_file_still_gets_screenshots(service, project):
    """A model that takes images refuses one corrupt file `read_image`
    loaded. `read_image` sends no more images, but the next cell's check
    still asks for a screenshot, and the model takes it."""
    (project / "photo.png").write_bytes(_CORRUPT_PNG)
    model = _RefusesOneImage(_UNREADABLE_400, [
        _read_image("call_1", path="photo.png"), _cell("call_2"), "The map is drawn.",
    ])
    session, pad, _ = _session_for(project, [])
    _on_model(session, model, "openai-compatible-generic")
    try:
        streamed, _ = await _final_text(session)
    finally:
        model.server.shutdown()

    assert streamed.endswith("The map is drawn.")
    assert len(pad.ran) == 1
    assert [r["body"]["screenshot"] for r in service.requests] == [True]
    # The read_image call; its result with the file, refused; the same
    # request without it; the cell's result with its screenshot, taken.
    assert [_image_parts(body) for body in model.requests] == [0, 1, 0, 1]
    assert session._images_refused is True
    assert session._render_screenshots_refused is False


async def test_a_vision_model_that_refuses_a_file_beside_a_screenshot_still_gets_screenshots(service, project):
    """History keeps the newest render screenshot, so a request carrying a
    corrupt file `read_image` loaded carries that screenshot too. The turn
    drops the file first and resends; the model takes the screenshot, so
    only `read_image` stops sending images, and the next cell's check still
    asks for a screenshot."""
    (project / "photo.png").write_bytes(_CORRUPT_PNG)
    model = _RefusesOneImage(_UNREADABLE_400, [
        _cell("call_1"), _read_image("call_2", path="photo.png"), _cell("call_3"), "Done.",
    ])
    session, pad, _ = _session_for(project, [])
    _on_model(session, model, "openai-compatible-generic")
    try:
        streamed, _ = await _final_text(session)
    finally:
        model.server.shutdown()

    assert streamed.endswith("Done.")
    assert len(pad.ran) == 2
    # The first cell's result with its screenshot, taken; the file beside
    # it, refused; the same request with only the screenshot, taken; the
    # second cell's result with its own screenshot, taken.
    assert [_image_parts(body) for body in model.requests] == [0, 1, 2, 1, 1]
    assert [r["body"]["screenshot"] for r in service.requests] == [True, True]
    assert session._images_refused is True
    assert session._render_screenshots_refused is False
    assert not render_check.screenshots_refused(session.history)


def _read_image_round(tool_use_id: str) -> list[dict]:
    """A read_image call and its result with the image, as history holds them."""
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}}
    return [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": tool_use_id, "name": "read_image", "input": {"file_path": "d.png"}},
        ]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": [
            image, {"type": "text", "text": "Loaded d.png"},
        ]}]},
    ]


@pytest.mark.parametrize("case", sorted(_REFUSALS))
async def test_a_text_only_model_refusing_both_kinds_drops_each_once_and_confirms_both(project, case):
    """A request carries a render screenshot and an image `read_image`
    loaded, and a model with no image input refuses it. The turn drops the
    tool image and resends; the screenshot is refused too, so the turn drops
    it and resends once more. That request goes through, which confirms
    both drops."""
    refusal = _REFUSALS[case]
    model = _TextOnlyModel(refusal.body, ["Done."])
    session, _, _ = _session_for(project, [])
    _on_model(session, model, refusal.flavor)
    session._history.extend([*_shot_round("t1"), *_read_image_round("t2")])
    try:
        async for _ in session.turn_stream("go on"):
            pass
    finally:
        await session.close()
        model.server.shutdown()

    assert [_image_parts(body) for body in model.requests] == [2, 1, 0]
    assert session._images_refused is True
    assert session._render_screenshots_refused is True
    assert render_check.screenshots_refused(session.history)
    assert render_check.tool_images_refused(session.history)


@pytest.mark.parametrize("case", sorted(_REFUSALS))
async def test_a_refusal_neither_kind_caused_drops_each_once_and_confirms_neither(project, case):
    """The request is refused with the tool image, without it, and without
    the screenshot as well. Each kind is dropped once, so the turn takes the
    old path from there and ends as it did before, and nothing records that
    the model refused images."""
    refusal = _REFUSALS[case]
    model = _RefusesOnceItSawAnImage(refusal.body, [])
    session, _, _ = _session_for(project, [])
    _on_model(session, model, refusal.flavor)
    session._history.extend([*_shot_round("t1"), *_read_image_round("t2")])
    try:
        with pytest.raises(Exception) as refused:
            async for _ in session.turn_stream("go on"):
                pass
    finally:
        await session.close()
        model.server.shutdown()

    assert type(refused.value).__name__ == refusal.terminal
    # The tool image dropped, then the screenshot, then the requests a
    # refusal with no image of anton's to drop always took.
    assert [_image_parts(body) for body in model.requests] == [2, 1] + [0] * refusal.attempts
    assert session._images_refused is False
    assert session._render_screenshots_refused is False
    assert not render_check.screenshots_refused(session.history)
    assert not render_check.tool_images_refused(session.history)


# JPEG bytes saved as a .png.
_JPEG_NAMED_PNG = b"\xff\xd8\xff\xe0 a JPEG named .png"
# What Anthropic's API answers an image whose bytes are not the media type it
# was sent as.
_MEDIA_TYPE_400 = {"error": {
    "message": "messages.1.content.0.image.source.base64: The image was specified using the image/png "
               "media type, but the image appears to be a image/jpeg image",
    "type": "invalid_request_error", "param": None, "code": None,
}}


class _ChecksMediaType(_TextOnlyModel):
    """A model that takes images and refuses one sent as a PNG whose bytes
    are `_JPEG_NAMED_PNG`."""

    def refuses(self, body: dict) -> bool:
        return f"data:image/png;base64,{base64.b64encode(_JPEG_NAMED_PNG).decode()}" in json.dumps(body)


async def test_a_jpeg_named_png_reaches_the_model_and_read_image_stays_on(project):
    """`read_image` sends the media type the bytes show, so a provider that
    checks the bytes takes a JPEG saved as .png, and a later valid file
    still loads."""
    (project / "photo.png").write_bytes(_JPEG_NAMED_PNG)
    (project / "good.png").write_bytes(b"\x89PNG\r\n\x1a\n five digits")
    model = _ChecksMediaType(_MEDIA_TYPE_400, [
        _read_image("call_1", path="photo.png"), _read_image("call_2", path="good.png"), "Done.",
    ])
    session, _, _ = _session_for(project, [])
    _on_model(session, model, "openai-compatible-generic")
    try:
        streamed, _ = await _final_text(session)
    finally:
        model.server.shutdown()

    assert streamed.endswith("Done.")
    assert [_image_parts(body) for body in model.requests] == [0, 1, 2]
    assert "data:image/jpeg;base64," in json.dumps(model.requests[1])
    assert session._images_refused is False


async def test_read_image_sends_the_media_type_the_bytes_show(project):
    from anton.core.tools.tool_handlers import handle_read_image

    (project / "photo.png").write_bytes(_JPEG_NAMED_PNG)
    session = SimpleNamespace(_workspace=SimpleNamespace(base=project), _scratchpads=None, _images_refused=False)

    out = await handle_read_image(session, {"file_path": "photo.png"})

    assert out.ok is True
    assert out.content[0]["source"]["media_type"] == "image/jpeg"
    assert base64.b64decode(out.content[0]["source"]["data"]) == _JPEG_NAMED_PNG


_GIF_1X1 = (
    b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,"
    b"\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
)
_WEBP_1X1 = b"RIFF\x1a\x00\x00\x00WEBPVP8L\x0d\x00\x00\x00/\x00\x00\x00\x10\x07\x10\x11\x11\x88\x88\xfe\x07\x00"


@pytest.mark.parametrize("name, data, media_type", [
    ("a.gif", _GIF_1X1, "image/gif"),
    ("a.webp", _WEBP_1X1, "image/webp"),
])
async def test_read_image_still_sends_a_gif_or_a_webp(project, name, data, media_type):
    from anton.core.tools.tool_handlers import handle_read_image

    (project / name).write_bytes(data)
    session = SimpleNamespace(_workspace=SimpleNamespace(base=project), _scratchpads=None, _images_refused=False)

    out = await handle_read_image(session, {"file_path": name})

    assert out.ok is True
    assert out.content[0]["source"]["media_type"] == media_type
    assert base64.b64decode(out.content[0]["source"]["data"]) == data


async def test_read_image_reads_bmp_bytes_as_a_bmp_whatever_the_name(project):
    """BMP bytes named .png go to the BMP conversion, not to the "not a
    readable image" answer: out as PNG with Pillow, or the conversion's own
    error without it."""
    import struct

    from anton.core.tools.tool_handlers import handle_read_image

    pixels = b"\x00\x00\xff\x00"
    bmp = (
        b"BM" + struct.pack("<IHHI", 54 + len(pixels), 0, 0, 54)
        + struct.pack("<IiiHHIIiiII", 40, 1, 1, 1, 24, 0, len(pixels), 2835, 2835, 0, 0)
        + pixels
    )
    (project / "chart.png").write_bytes(bmp)
    session = SimpleNamespace(_workspace=SimpleNamespace(base=project), _scratchpads=None, _images_refused=False)

    out = await handle_read_image(session, {"file_path": "chart.png"})

    try:
        import PIL  # noqa: F401
    except ImportError:
        assert (out.ok, out.reason) == (False, "bmp_convert_failed")
    else:
        assert out.ok is True
        assert out.content[0]["source"]["media_type"] == "image/png"


async def test_read_image_of_a_file_that_is_no_image_answers_in_text(project):
    """A .png whose bytes are no PNG, JPEG, GIF or WebP gets a text answer,
    and no image part reaches the model."""
    (project / "notes.png").write_bytes(b"these are notes, not a picture")
    model = _VisionModel(_UNREADABLE_400, [_read_image("call_1", path="notes.png"), "Done."])
    session, _, _ = _session_for(project, [])
    _on_model(session, model, "openai-compatible-generic")
    try:
        streamed, _ = await _final_text(session)
    finally:
        model.server.shutdown()

    assert streamed.endswith("Done.")
    assert [_image_parts(body) for body in model.requests] == [0, 0]
    assert "'notes.png' is not a readable PNG, JPEG, GIF or WebP image." in json.dumps(model.requests[1])
    assert session._images_refused is False


class _RefusesOnceItSawAnImage(_TextOnlyModel):
    """A model that, once a request carried an image, refuses every request
    after it, with or without one."""

    saw_image = False

    def refuses(self, body: dict) -> bool:
        self.saw_image = self.saw_image or _image_parts(body) > 0
        return self.saw_image


@pytest.mark.parametrize("case", sorted(_REFUSALS))
async def test_a_refusal_the_tool_image_did_not_cause_drops_it_once(project, case):
    """The image is dropped once per turn. The resend without it is refused
    as well, so the turn takes the old path from there and ends as it did
    before, and nothing records that the model refused images."""
    refusal = _REFUSALS[case]
    (project / "digits.png").write_bytes(b"\x89PNG\r\n\x1a\n five digits")
    model = _RefusesOnceItSawAnImage(refusal.body, [_read_image("call_1", path="digits.png")])
    session, _, _ = _session_for(project, [])
    _on_model(session, model, refusal.flavor)
    try:
        with pytest.raises(Exception) as refused:
            async for _ in session.turn_stream("what digits are in digits.png?"):
                pass
    finally:
        await session.close()
        model.server.shutdown()

    assert type(refused.value).__name__ == refusal.terminal
    # One request with the image; after it, the resend without it, then
    # the requests a refusal with no image of anton's to drop always took.
    assert [_image_parts(body) for body in model.requests] == [0, 1] + [0] * refusal.attempts
    assert session._images_refused is False


async def test_read_image_answers_in_text_once_the_model_refused_an_image(project):
    from anton.core.tools.tool_handlers import handle_read_image

    (project / "digits.png").write_bytes(b"\x89PNG\r\n\x1a\n five digits")
    session = SimpleNamespace(_workspace=SimpleNamespace(base=project), _scratchpads=None, _images_refused=True)

    out = await handle_read_image(session, {"file_path": "digits.png"})

    assert out.ok is False and out.reason == "images_refused"
    assert isinstance(out.content, str) and "Tell the user you could not view this image." in out.content

    session._images_refused = False
    loaded = await handle_read_image(session, {"file_path": "digits.png"})
    assert loaded.ok is True and loaded.content[0]["type"] == "image"


def _shot_round(tool_use_id: str) -> list[dict]:
    """A scratchpad call and its result with a render screenshot, as history holds them."""
    call = {"role": "assistant", "content": [{
        "type": "tool_use", "id": tool_use_id, "name": "scratchpad",
        "input": {"action": "exec", "name": "main", "code": "write_page()"},
    }]}
    return [call, _result_with_shot(tool_use_id, _SHOT)]


async def test_the_screenshot_is_dropped_once_then_the_refusal_takes_its_old_path(project):
    """When the request still fails without the screenshot, as it does for an
    image the person attached, the turn resends once and then ends the way
    it did before: here the content repair and the error the host shows.
    The screenshot was not the cause, so later checks keep theirs, and the
    note says only what happened."""
    from anton.core.llm.provider import ContentValidationError, ProviderAuthError

    session, _, _ = _session_for(project, [])
    session._history.extend(_shot_round("t1"))
    calls: list[str] = []

    async def refuse(user_msg):
        calls.append(user_msg)
        if len(calls) > _REQUEST_CAP:
            # The session ends the turn on this at once, so a broken bound fails here.
            raise ProviderAuthError(f"more than {_REQUEST_CAP} requests in one test")
        raise ContentValidationError("refused")
        yield  # pragma: no cover - makes this an async generator

    session._stream_and_handle_tools = refuse
    try:
        with pytest.raises(ContentValidationError):
            async for _ in session.turn_stream("check the page again"):
                pass
    finally:
        await session.close()

    assert len(calls) == 2
    shot_result = _tool_results(session)[0]["content"]
    assert {"type": "text", "text": render_check.DROPPED_SCREENSHOT_PLACEHOLDER} in shot_result
    assert shot_result[0]["text"] == "[output]\n", "the text beside the screenshot stays"
    assert session._render_screenshots_refused is False
    assert not render_check.screenshots_refused(session.history)


class _ImagePart400(Exception):
    """A 400 that names an image, as the SDK raises it for a refused image part."""

    status_code = 400
    body = {"message": "Invalid image.", "type": "invalid_request_error"}


async def test_a_later_success_after_a_second_image_refusal_confirms_nothing(project):
    """The request is refused again without the screenshots, so they were
    not the cause. The recovery resend that goes through after it says
    nothing about them: later checks keep their screenshots."""
    from anton.core.llm.provider import StreamComplete

    session, _, _ = _session_for(project, [])
    session._history.extend(_shot_round("t1"))
    outcomes = [_ImagePart400("Invalid image."), _ImagePart400("Invalid image."), None]

    async def stream(user_msg):
        outcome = outcomes.pop(0)
        if outcome is not None:
            raise outcome
        yield StreamComplete(response=_llm_response("Done."))

    session._stream_and_handle_tools = stream
    try:
        async for _ in session.turn_stream("check the page again"):
            pass
    finally:
        await session.close()

    assert outcomes == []
    assert session._render_screenshots_refused is False
    assert not render_check.screenshots_refused(session.history)


async def test_the_screenshot_drop_leaves_both_recovery_resends_for_a_later_error(project):
    """The resend without screenshots is outside the count of resends of an
    unchanged request. A turn whose next attempts fail for another reason
    still gets both recovery-note resends before the explain call."""
    from anton.core.llm.provider import ContentValidationError

    session, _, _ = _session_for(project, [])
    session._history.extend(_shot_round("t1"))
    errors = [ContentValidationError("refused"), *(RuntimeError("the tool crashed") for _ in range(3))]
    calls: list[str] = []

    async def stream(user_msg):
        calls.append(user_msg)
        raise errors.pop(0)
        yield  # pragma: no cover - makes this an async generator

    session._stream_and_handle_tools = stream
    try:
        async for _ in session.turn_stream("check the page again"):
            pass
    finally:
        await session.close()

    assert len(calls) == 4
    history = json.dumps(session.history)
    assert history.count("An error interrupted execution") == 2
    assert "The task has failed 3 times" in history


async def test_a_session_seeded_with_a_refused_screenshot_attaches_none(service, project):
    """cowork-server builds a session for every turn and seeds it with the
    conversation's earlier tool results, cut by anton's own row splitter. The
    note the refusal left tells the new session the model takes no image
    input, so its first check costs no refused request."""
    from anton.cloud_turn.history_rows import split_turn_into_rows

    earlier = _shot_round("t0")
    render_check.drop_refused_screenshots(earlier)
    render_check.confirm_refused_screenshots(earlier)
    session, _, slug = _session_for(project, [
        _llm_response("Writing.", {"action": "exec", "name": "main", "code": "write_page()"}),
        _llm_response("Done."),
    ])
    session._history.extend(split_turn_into_rows(earlier))

    await _final_text(session)

    assert len(service.requests) == 1, "the page was still checked"
    assert service.requests[0]["body"]["screenshot"] is False
    result = _tool_results(session)[-1]["content"]
    assert isinstance(result, str) and f"{slug}/index.html — loaded in a browser" in result
    assert session._render_screenshots_refused is True


@pytest.mark.parametrize("flavor", ["openai", "openai-compatible-generic"])
async def test_an_oversized_image_of_the_persons_keeps_screenshots_on(service, project, flavor):
    """The model takes images and refuses only the person's oversized one. A
    size refusal is about that image, so the render screenshots stay in the
    request and on for the rest of the conversation."""
    from tests.test_bad_request_reaches_the_classifier import _PATCHES_400

    model = _VisionModel(_PATCHES_400, [_cell("call_1"), "Done.", _cell("call_3"), "Checked."])
    session, _, _ = _session_for(project, [])
    _on_model(session, model, flavor)
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _PERSONS_IMAGE}}
    try:
        async for _ in session.turn_stream("draw the map"):
            pass
        with pytest.raises(Exception) as refused:
            async for _ in session.turn_stream([{"type": "text", "text": "use this logo"}, image]):
                pass
        async for _ in session.turn_stream("check the map again"):
            pass
    finally:
        await session.close()
        model.server.shutdown()

    assert type(refused.value).__name__ == "ContentTooLargeError"
    # The refused request held the screenshot and the person's image, and
    # was sent once. The last check's screenshot reached the model.
    assert [_image_parts(body) for body in model.requests] == [0, 1, 2, 0, 1]
    assert [r["body"]["screenshot"] for r in service.requests] == [True, True]
    assert session._render_screenshots_refused is False
    assert not render_check.screenshots_refused(session.history)
    assert render_check.DROPPED_SCREENSHOT_PLACEHOLDER not in json.dumps(session.history)


@pytest.mark.parametrize("flavor", ["openai", "openai-compatible-generic"])
async def test_a_refusal_the_screenshots_did_not_cause_leaves_them_on(service, project, flavor):
    """A 400 that names an image may be about one image the person attached.
    The turn drops the screenshot and resends once; the resend is refused
    the same way, so nothing confirms the drop and the next conversation's
    sessions start with screenshots on."""
    model = _VisionModel(_UNREADABLE_400, [_cell("call_1"), "Done."])
    session, _, _ = _session_for(project, [])
    _on_model(session, model, flavor)
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _PERSONS_IMAGE}}
    try:
        async for _ in session.turn_stream("draw the map"):
            pass
        with pytest.raises(Exception):
            async for _ in session.turn_stream([{"type": "text", "text": "use this logo"}, image]):
                pass
    finally:
        await session.close()
        model.server.shutdown()

    # The refusal with the screenshot, then the same request without it.
    assert [_image_parts(body) for body in model.requests][2:4] == [2, 1]
    assert session._render_screenshots_refused is False
    assert not render_check.screenshots_refused(session.history)
    assert render_check.DROPPED_SCREENSHOT_PLACEHOLDER in json.dumps(session.history)


async def test_generate_artifact_attaches_no_screenshot_once_the_model_refused_one(tmp_path, monkeypatch):
    """A model that refused a render screenshot would refuse this one as well."""
    shot = render_check.RenderScreenshot(media_type="image/jpeg", data=_SHOT)
    _, out = await _handler_result(tmp_path, monkeypatch, {
        "status": "generated", "files_written": ["index.html"], "warnings": [], "checks_skipped": [],
        "verified_in_browser": True, "screenshot": shot,
    }, screenshots_refused=True)

    assert out.ok is True and isinstance(out.content, str)
    payload = json.loads(out.content)
    assert payload["verified_in_browser"] is True and "screenshot" not in payload


async def test_generate_artifact_attaches_the_screenshot_after_a_refused_read_image(tmp_path, monkeypatch):
    """A refused `read_image` image leaves render screenshots on: the model
    may take images and have refused only that file."""
    shot = render_check.RenderScreenshot(media_type="image/jpeg", data=_SHOT)
    _, out = await _handler_result(tmp_path, monkeypatch, {
        "status": "generated", "files_written": ["index.html"], "warnings": [], "checks_skipped": [],
        "verified_in_browser": True, "screenshot": shot,
    }, images_refused=True)

    assert out.ok is True
    _, _, image = out.content
    assert image == {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _SHOT}}


def test_a_refused_screenshot_leaves_a_note_and_the_text_beside_it():
    read_image = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "img", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
        ]}],
    }
    history = [_result_with_shot("t1", _SHOT), read_image, _result_with_shot("t2", _SHOT)]
    assert not render_check.screenshots_refused(history)

    assert render_check.drop_refused_screenshots(history) == 2

    for message in (history[0], history[2]):
        text, caption, note = message["content"][0]["content"]
        assert text == {"type": "text", "text": "[output]\n"}
        assert caption["text"].startswith(render_check.SCREENSHOT_CAPTION)
        assert note == {"type": "text", "text": render_check.DROPPED_SCREENSHOT_PLACEHOLDER}
    assert history[1] == read_image, "an image the agent read itself is not anton's to drop"
    assert not render_check.screenshots_refused(history), "no request went through without them yet"
    assert render_check.drop_refused_screenshots(history) == 0

    assert render_check.confirm_refused_screenshots(history) == 2
    for message in (history[0], history[2]):
        assert message["content"][0]["content"][-1] == {
            "type": "text", "text": render_check.REFUSED_SCREENSHOT_PLACEHOLDER,
        }
    assert render_check.screenshots_refused(history)
    assert render_check.confirm_refused_screenshots(history) == 0


def test_a_refused_tool_image_leaves_a_note_and_every_other_image_alone():
    """A model with no image input refused every request carrying the image
    `read_image` returned, and the screenshot drop found nothing to take."""
    read_image = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "img", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
            {"type": "text", "text": "Loaded digits.png (10x10, 3 B)."},
        ]}],
    }
    image_url = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "url", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}],
    }
    attached = {"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "BBBB"}},
        {"type": "text", "text": "what digits are in this?"},
    ]}
    history = [attached, _result_with_shot("t1", _SHOT), read_image, image_url]
    before = json.loads(json.dumps(history))

    assert render_check.drop_refused_tool_images(history) == 2

    note = {"type": "text", "text": render_check.DROPPED_TOOL_IMAGE_PLACEHOLDER}
    assert history[2]["content"][0]["content"] == [note, {"type": "text", "text": "Loaded digits.png (10x10, 3 B)."}]
    assert history[3]["content"][0]["content"] == [note]
    assert history[0] == before[0], "an image the person attached is the content repair's to judge"
    assert history[1] == before[1], "a render screenshot is drop_refused_screenshots' to drop"
    assert not render_check.screenshots_refused(history)
    assert not render_check.tool_images_refused(history), "no request has gone through without them yet"
    assert render_check.drop_refused_tool_images(history) == 0

    # A request went through without them: the notes say the model refused them.
    assert render_check.confirm_refused_tool_images(history) == 2
    assert history[3]["content"][0]["content"] == [{"type": "text", "text": render_check.REFUSED_TOOL_IMAGE_PLACEHOLDER}]
    assert render_check.tool_images_refused(history)
    assert not render_check.screenshots_refused(history)
    assert render_check.confirm_refused_tool_images(history) == 0


async def test_a_session_seeded_with_a_refused_tool_image_reads_none(project):
    """cowork-server builds a session for every turn and seeds it with the
    conversation's earlier tool results, cut by anton's own row splitter.
    The note a confirmed refusal of a `read_image` image left tells the new
    session the model takes no image input, so `read_image` answers in text
    instead of costing a refused request in every turn."""
    from anton.cloud_turn.history_rows import split_turn_into_rows

    (project / "digits.png").write_bytes(b"\x89PNG\r\n\x1a\n five digits")
    earlier = [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t0", "name": "read_image", "input": {"file_path": "digits.png"}},
        ]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t0", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
            {"type": "text", "text": "Loaded digits.png (10x10, 3 B)."},
        ]}]},
    ]
    render_check.drop_refused_tool_images(earlier)
    render_check.confirm_refused_tool_images(earlier)
    model = _TextOnlyModel(_NO_IMAGES_CHAT, [_read_image("call_1", path="digits.png"), "I cannot view it."])
    session, _, _ = _session_for(project, [])
    session._history.extend(split_turn_into_rows(earlier))
    _on_model(session, model, "openai-compatible-generic")
    try:
        streamed, _ = await _final_text(session, "what digits are in digits.png?")
    finally:
        model.server.shutdown()

    assert streamed.endswith("I cannot view it.")
    assert session._images_refused is True
    assert [_image_parts(body) for body in model.requests] == [0, 0], "no request carried an image"
    assert "digits.png was not loaded" in json.dumps(model.requests[1], ensure_ascii=False)


# ── What the generator and the skill tell the model ──────────────────────────


def _skill() -> str:
    path = Path(__file__).resolve().parents[1] / "anton/core/memory/builtin_skills/build-html-dashboard/SKILL.md"
    return path.read_text(encoding="utf-8")


def test_the_skill_names_the_preview_rules_and_the_tested_recipe():
    skill = _skill()
    for rule in (
        "https://cdn.jsdelivr.net/npm/<package>@<exact version>/",
        "no external stylesheet",
        "No external images or map tiles",
        "No `eval`/`new Function` and no web workers",
        "d3@7.9.0 with topojson-client@3.1.0 and world-atlas@2.0.2",
        "echarts.registerMap(name, geoJson)",
        "globe.gl@2.34.2 with `polygonsData`",
        "https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-110m.json",
        "@tailwindcss/browser@4.3.3",
        "echarts@5.5.1",
    ):
        assert rule in skill, rule


def test_the_skill_embeds_everything_and_no_longer_splits_out_big_data():
    """A sibling `dashboard_data.js` or image never loads in the preview."""
    skill = _skill()
    assert "Anything larger than 100KB is the exception" not in skill
    assert "shutil.copy(src, dest_dir / name)" not in skill
    assert "however large" in skill


def test_the_skill_says_what_checks_a_hand_built_page_now():
    skill = _skill()
    assert "a hand-built page gets no automatic check" not in skill
    assert "A hand-built page gets the browser check too" in skill
    assert "not checked in a browser" in skill
