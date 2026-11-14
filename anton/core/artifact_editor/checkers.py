"""Does the artifact still work? One result shape, several ways to find out.

The editor asks once the model says it is done (never after every edit), and a
failure goes back into the same conversation for repair. Every checker returns
a :class:`CheckResult`: ``passed``, ``failed`` or ``unverified``. Unverified
means nothing could load the page here, and it is never treated as passed:
Mochi soft-passes a check nobody answered, anton does not.

* :class:`StaticChecker`: what can be checked without a browser. JSON parses,
  ``backend.py`` compiles, an HTML page still has its document and its
  ``<script>`` tags balanced, and with ``node`` installed every inline script
  parses. Always first: a page whose script does not parse needs no browser to
  say so, and it costs milliseconds.
* :class:`RenderCheckChecker`: the render-check service (``POST /v1/check``),
  the headless Chromium that loads a page the way the app preview does. Static
  pages, in cowork-server and wherever ``ANTON_RENDER_CHECK_URL`` is set.
* :class:`RenderUrlChecker`: the same service's URL mode (``POST
  /v1/check-url``) for a fullstack app whose live address is known: the page is
  loaded from its running backend, so its API calls are real.
* :class:`ElectronChecker`: the desktop app's own browser (``html_lint``).
* :class:`PreviewChecker`: the user's open preview, through the host
  (``ChatSessionConfig.preview_checker``). No preview open → unverified.

:func:`select_checker` picks per environment and artifact kind.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from anton.core.artifact_editor.workspace import ArtifactWorkspace

__all__ = [
    "CheckResult",
    "Checker",
    "ChainChecker",
    "ElectronChecker",
    "Finding",
    "PreviewChecker",
    "RenderCheckChecker",
    "RenderUrlChecker",
    "StaticChecker",
    "select_checker",
]

_log = logging.getLogger(__name__)

CheckStatus = Literal["passed", "failed", "unverified"]

# Findings shown to the model in one repair message; the rest are counted.
MAX_FINDINGS_SHOWN = 12
NODE_TIMEOUT_S = 10.0
FULLSTACK_KINDS = frozenset({"fullstack-stateless-app", "fullstack-stateful-app"})


@dataclass
class Finding:
    kind: str
    message: str
    viewport: str | None = None
    stack: str | None = None
    source: str | None = None

    def line(self) -> str:
        where = f" [{self.viewport}]" if self.viewport else ""
        src = f" ({self.source})" if self.source else ""
        text = f"- {self.kind}{where}{src}: {self.message}"
        if self.stack:
            text += "\n  " + "\n  ".join(self.stack.strip().splitlines()[:6])
        return text


@dataclass
class CheckResult:
    status: CheckStatus
    channel: str  # "static" | "render" | "render-url" | "preview" | "electron"
    findings: list[Finding] = field(default_factory=list)
    screenshot_b64: str | None = None
    ms: int = 0
    note: str = ""
    # The screenshot's media type, when there is one.
    screenshot_media_type: str = "image/png"
    # The render check's comparison with the page as it was before (a
    # `render_check.Comparison`), for the turn's page record. None elsewhere.
    comparison: object | None = None

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def describe(self) -> str:
        """The findings, worded for the model's repair message."""
        lines = [f"{self.channel} check: {self.status}"]
        if self.note:
            lines.append(self.note)
        shown = self.findings[:MAX_FINDINGS_SHOWN]
        lines += [one.line() for one in shown]
        if len(self.findings) > len(shown):
            lines.append(f"- and {len(self.findings) - len(shown)} more findings")
        return "\n".join(lines)


class Checker(Protocol):
    async def check(self, ws: "ArtifactWorkspace", changed: list[str]) -> CheckResult: ...


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _raw(ws: "ArtifactWorkspace", path: str) -> str | None:
    """The file as it is on disk (unmasked): what a browser would load."""
    target = Path(ws.folder) / path
    try:
        return target.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        return None
    except OSError as exc:
        _log.debug("could not read %s for a check: %s", target, exc)
        return None


def _entry_page(ws: "ArtifactWorkspace", changed: Sequence[str]) -> str | None:
    pages = [p for p in changed if p.lower().endswith((".html", ".htm"))]
    if ws.primary and (ws.primary in pages or not pages):
        return ws.primary
    return pages[0] if pages else ws.primary


# ── static ────────────────────────────────────────────────────────────

_SCRIPT_RE = re.compile(r"<script\b([^>]*)>(.*?)</script\s*>", re.S | re.I)
_TYPE_RE = re.compile(r"""\btype\s*=\s*["']?([^"'\s>]+)""", re.I)
_SRC_RE = re.compile(r"\bsrc\s*=", re.I)
_JS_TYPES = frozenset({"", "text/javascript", "application/javascript", "module", "text/babel"})


def _inline_scripts(html: str) -> list[tuple[str, str, int]]:
    """(type, source, line number) of each inline script a browser would run or parse."""
    found = []
    for match in _SCRIPT_RE.finditer(html):
        attrs, body = match.group(1), match.group(2)
        if _SRC_RE.search(attrs) or not body.strip():
            continue
        kind = (_TYPE_RE.search(attrs).group(1).lower() if _TYPE_RE.search(attrs) else "")
        line = html.count("\n", 0, match.start(2)) + 1
        found.append((kind, body, line))
    return found


async def _node_check(source: str, *, module: bool, node: str) -> str | None:
    """`node --check` on SOURCE: the syntax error, or None when it parses (or node gave no answer)."""
    suffix = ".mjs" if module else ".js"
    with tempfile.TemporaryDirectory(prefix="anton-editor-check-") as tmp:
        path = Path(tmp) / f"script{suffix}"
        path.write_text(source, encoding="utf-8")
        try:
            proc = await asyncio.create_subprocess_exec(
                node, "--check", str(path),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
        except OSError:
            return None
        try:
            _, err = await asyncio.wait_for(proc.communicate(), NODE_TIMEOUT_S)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return None
        except asyncio.CancelledError:
            proc.kill()
            raise
        if proc.returncode == 0:
            return None
        text = err.decode("utf-8", "replace")
        for shown in {str(path), os.path.realpath(path)}:
            text = text.replace(shown, "script")
        # node prints the offending line, a caret line, then "SyntaxError: ...".
        lines = [line for line in text.splitlines() if line.strip() and not line.startswith("Node.js v")]
        return "\n".join(lines[:6]) or "the script does not parse"


class StaticChecker:
    """Checks that need no browser. Fails only on something certain."""

    channel = "static"

    def __init__(self, *, node: str | None | Literal["auto"] = "auto", cell_sandbox=None) -> None:
        # Under a sandbox policy no process is started on the agent's files:
        # the JSON, Python and HTML checks still run in this process.
        if cell_sandbox is not None:
            node = None
        self._node = shutil.which("node") if node == "auto" else node

    async def check(self, ws: "ArtifactWorkspace", changed: list[str]) -> CheckResult:
        started = time.monotonic()
        findings: list[Finding] = []
        for path in changed:
            text = _raw(ws, path)
            if text is None:
                continue
            lower = path.lower()
            if lower.endswith(".json"):
                try:
                    json.loads(text)
                except ValueError as exc:
                    findings.append(Finding("invalid_json", f"{path} is not valid JSON: {exc}", source=path))
            elif lower.endswith(".py"):
                try:
                    compile(text, path, "exec", dont_inherit=True)
                except SyntaxError as exc:
                    findings.append(Finding(
                        "syntax_error", f"{path} line {exc.lineno}: {exc.msg}: {(exc.text or '').strip()}",
                        source=path,
                    ))
            elif lower.endswith((".html", ".htm")):
                findings += await self._html(path, text)
            elif lower.endswith((".js", ".mjs")) and self._node:
                error = await _node_check(text, module=lower.endswith(".mjs"), node=self._node)
                if error:
                    findings.append(Finding("syntax_error", f"{path}: {error}", source=path))
        status: CheckStatus = "failed" if findings else "passed"
        return CheckResult(status=status, channel=self.channel, findings=findings, ms=_ms(started))

    async def _html(self, path: str, text: str) -> list[Finding]:
        findings: list[Finding] = []
        low = text.lower()
        if "<html" not in low and "<body" not in low:
            findings.append(Finding("broken_page", f"{path} has no <html> or <body>: the page lost its document",
                                    source=path))
        opened = len(re.findall(r"<script\b", text, re.I))
        closed = len(re.findall(r"</script\s*>", text, re.I))
        if opened != closed:
            findings.append(Finding(
                "broken_page", f"{path} has {opened} <script> tags but {closed} </script>: a script is not closed",
                source=path,
            ))
        for kind, body, line in _inline_scripts(text):
            if kind in ("application/json", "application/ld+json", "importmap"):
                try:
                    json.loads(body)
                except ValueError as exc:
                    findings.append(Finding("invalid_json", f"{path} line {line}: a JSON script does not parse: {exc}",
                                            source=path))
                continue
            if kind not in _JS_TYPES or kind == "text/babel" or not self._node:
                continue
            error = await _node_check(body, module=kind == "module", node=self._node)
            if error:
                findings.append(Finding(
                    "syntax_error", f"{path}: the inline script starting on line {line} does not parse:\n{error}",
                    source=path,
                ))
        return findings


# ── render check (service) ────────────────────────────────────────────


def _from_render_outcome(outcome, *, channel: str, started: float) -> CheckResult:
    """A render_check outcome as a CheckResult, with what the comparison says in `note`."""
    from anton.core.artifacts import render_check

    status: CheckStatus = outcome.status
    findings = [Finding(kind=f.kind, message=f.detail, viewport=f.viewport) for f in outcome.findings]
    notes: list[str] = []
    if status == "unverified":
        notes.append(f"not verified in a browser: {outcome.unverified_reason}")
    else:
        coverage = getattr(outcome, "coverage", None)
        if coverage is not None:
            notes.append(f"checked: {coverage.checked}")
            if coverage.unchecked:
                notes.append("tabs and toggles not pressed: " + "; ".join(coverage.unchecked))
        comparison = render_check.comparison_notes(outcome.comparison)
        if comparison.broke:
            notes.append(f"these views drew before this edit and fail now: {render_check.joined(comparison.broke)}")
        if comparison.blank:
            notes.append(f"these views showed a graphic before this edit and show none now: "
                         f"{render_check.joined(comparison.blank)}")
        if comparison.removed:
            notes.append(f"this edit removed these views: {render_check.joined(comparison.removed)}; put them back "
                         "unless the user asked for that")
    shot = outcome.screenshot
    return CheckResult(
        status=status, channel=channel, findings=findings,
        screenshot_b64=shot.data if shot is not None else None,
        screenshot_media_type=shot.media_type if shot is not None else "image/png",
        ms=_ms(started), note="\n".join(notes), comparison=outcome.comparison,
    )


class RenderCheckChecker:
    """The render-check service loads the entry page as the app preview does.

    ``previous_html`` is the page as it was before this edit (or this turn),
    so the service can say which views the edit broke, blanked or removed.
    """

    channel = "render"

    def __init__(
        self, *, previous_html: str | None = None, screenshot: bool = True, cell_sandbox=None,
        timeout_s: float | None = None,
    ) -> None:
        self.previous_html = previous_html
        self.screenshot = screenshot
        self.cell_sandbox = cell_sandbox
        self.timeout_s = timeout_s

    async def check(self, ws: "ArtifactWorkspace", changed: list[str]) -> CheckResult:
        from anton.core.artifacts import render_check

        started = time.monotonic()
        if render_check.configured_url() is None:
            return CheckResult("unverified", self.channel, note="no render-check service configured")
        page = _entry_page(ws, changed)
        if page is None:
            return CheckResult("unverified", self.channel, note="the artifact has no page to load")
        outcome = await asyncio.to_thread(
            render_check.check_file, Path(ws.folder) / page, cell_sandbox=self.cell_sandbox,
            previous_html=self.previous_html, screenshot=self.screenshot,
            timeout=self.timeout_s or render_check.TIMEOUT_SECONDS,
        )
        return _from_render_outcome(outcome, channel=self.channel, started=started)


class RenderUrlChecker:
    """The render-check service loads a running app from its address (``POST /v1/check-url``).

    ``address`` is awaited at check time: a live app must be relaunched with
    the edited files before it is loaded, and the host knows how. None means
    no running app, so the page is unverified.
    """

    channel = "render-url"

    def __init__(
        self, address: Callable[[], Awaitable[str | None]], *, screenshot: bool = True, timeout_s: float = 35.0,
    ) -> None:
        self.address = address
        self.screenshot = screenshot
        self.timeout_s = timeout_s

    async def check(self, ws: "ArtifactWorkspace", changed: list[str]) -> CheckResult:
        from anton.core.artifacts import render_check

        started = time.monotonic()
        base = render_check.configured_url()
        if base is None:
            return CheckResult("unverified", self.channel, note="no render-check service configured")
        try:
            url = await self.address()
        except Exception as exc:  # noqa: BLE001 — a relaunch failure is the page's verdict, not a crash
            _log.warning("could not get the live app's address for a check", exc_info=True)
            return CheckResult("unverified", self.channel, note=f"the live app could not be reached ({exc})",
                               ms=_ms(started))
        if not url:
            return CheckResult("unverified", self.channel, note="the live app is not running", ms=_ms(started))
        outcome = await asyncio.to_thread(self._post, base, url)
        return _from_render_outcome(outcome, channel=self.channel, started=started)

    def _post(self, base: str, url: str):
        import httpx2 as httpx
        from pydantic import ValidationError

        from anton.core.artifacts.render_check import RenderCheckResult, RenderCheckUnavailable

        body = {"url": url, "screenshot": self.screenshot}
        try:
            with httpx.Client(timeout=self.timeout_s, trust_env=False) as client:
                response = client.post(f"{base}/v1/check-url", json=body)
        except httpx.TimeoutException:
            return RenderCheckUnavailable(reason=f"the render check gave no answer within {self.timeout_s:.0f} s")
        except httpx.HTTPError as exc:
            return RenderCheckUnavailable(reason=f"the render-check service could not be reached ({type(exc).__name__})")
        if response.status_code == 404:
            return RenderCheckUnavailable(reason="the render-check service has no URL mode (an older version)")
        if response.status_code == 403:
            return RenderCheckUnavailable(reason="the render-check service does not allow this app's address")
        if response.status_code != 200:
            return RenderCheckUnavailable(reason=f"the render-check service answered HTTP {response.status_code}")
        try:
            return RenderCheckResult.model_validate_json(response.content)
        except ValidationError:
            return RenderCheckUnavailable(reason="the render-check service answered something that does not parse")


# ── the desktop's browser ─────────────────────────────────────────────


class ElectronChecker:
    """Cowork desktop's Electron loads the page (``html_lint``)."""

    channel = "electron"

    def __init__(self, *, url: Callable[[], Awaitable[str | None]] | None = None) -> None:
        self.url = url

    async def check(self, ws: "ArtifactWorkspace", changed: list[str]) -> CheckResult:
        from anton.core.artifacts import html_lint

        started = time.monotonic()
        if html_lint.discover_browser() is None:
            return CheckResult("unverified", self.channel, note="no browser on this deployment")
        if self.url is not None:
            address = await self.url()
            if not address:
                return CheckResult("unverified", self.channel, note="the app is not running", ms=_ms(started))
            found = await asyncio.to_thread(html_lint.lint_url, address, cell_sandbox=None)
        else:
            page = _entry_page(ws, changed)
            if page is None:
                return CheckResult("unverified", self.channel, note="the artifact has no page to load")
            found = await asyncio.to_thread(html_lint.lint_html, Path(ws.folder) / page, cell_sandbox=None)
        if found is None:
            return CheckResult("unverified", self.channel, note="the local browser check gave no answer",
                               ms=_ms(started))
        findings = [Finding(kind=f.kind, message=f.detail) for f in found]
        return CheckResult("failed" if findings else "passed", self.channel, findings, ms=_ms(started))


# ── the user's preview ────────────────────────────────────────────────

PreviewCheckFn = Callable[[str, float], Awaitable["dict | None"]]


class PreviewChecker:
    """Asks the host to reload the user's open preview and report its errors.

    ``preview_checker(slug, timeout_s)`` returns ``{"status": "passed" |
    "failed" | "unavailable", "errors": [{type, message, file, line, url}],
    "ms": int}`` or None. None or "unavailable" (nobody has the preview open,
    the turn ended, it timed out) is unverified.
    """

    channel = "preview"

    def __init__(self, preview_checker: PreviewCheckFn, *, slug: str, timeout_s: float = 20.0) -> None:
        self._ask = preview_checker
        self.slug = slug
        self.timeout_s = timeout_s

    async def check(self, ws: "ArtifactWorkspace", changed: list[str]) -> CheckResult:
        started = time.monotonic()
        try:
            answer = await asyncio.wait_for(self._ask(self.slug, self.timeout_s), self.timeout_s + 5)
        except TimeoutError:
            answer = None
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — a host bridge failure is "nobody answered"
            _log.warning("the preview check bridge failed", exc_info=True)
            answer = None
        if not isinstance(answer, dict) or answer.get("status") not in ("passed", "failed"):
            return CheckResult("unverified", self.channel, note="no open preview answered the check", ms=_ms(started))
        findings = []
        for error in answer.get("errors") or []:
            if not isinstance(error, dict):
                continue
            where = error.get("file") or error.get("url") or ""
            if where and error.get("line"):
                where = f"{where}:{error['line']}"
            findings.append(Finding(
                kind=str(error.get("type") or "error"), message=str(error.get("message") or "")[:2000],
                source=where or None,
            ))
        status: CheckStatus = "failed" if answer["status"] == "failed" or findings else "passed"
        return CheckResult(status, self.channel, findings, ms=int(answer.get("ms") or _ms(started)))


# ── composition ───────────────────────────────────────────────────────


class ChainChecker:
    """Static first, then each browser channel in turn until one gives a verdict.

    A static failure ends the check (no browser needed to say a script does not
    parse). A browser channel that comes back unverified hands over to the next
    one; when none can vouch for the page the result is unverified, with every
    reason in its note.
    """

    def __init__(self, checkers: Sequence[Checker]) -> None:
        self.checkers = list(checkers)

    @property
    def channel(self) -> str:
        return "+".join(getattr(c, "channel", "?") for c in self.checkers)

    async def check(self, ws: "ArtifactWorkspace", changed: list[str]) -> CheckResult:
        started = time.monotonic()
        notes: list[str] = []
        last: CheckResult | None = None
        static_passed = False
        for checker in self.checkers:
            result = await checker.check(ws, changed)
            last = result
            if result.status == "failed":
                return result
            if result.channel == "static":
                static_passed = result.status == "passed"
                continue
            if result.status == "passed":
                return result
            if result.note:
                notes.append(f"{result.channel}: {result.note}")
        if last is None:
            return CheckResult("unverified", "none", note="no checker configured")
        if last.channel == "static":
            # Only static checks ran: they never vouch for how a page renders.
            return CheckResult("unverified", "static", note="not loaded in a browser: no browser check is "
                               "available here" + ("; the static checks passed" if static_passed else ""),
                               ms=_ms(started))
        return CheckResult("unverified", last.channel, note="\n".join(notes) or last.note, ms=_ms(started),
                           findings=last.findings)


def select_checker(
    kind: str,
    *,
    slug: str = "",
    cell_sandbox=None,
    previous_html: str | None = None,
    screenshot: bool = True,
    preview_checker: PreviewCheckFn | None = None,
    live_app_address: Callable[[], Awaitable[str | None]] | None = None,
) -> ChainChecker:
    """The checks this environment can run for an artifact of ``kind``.

    * static checks always;
    * a static page: the render-check service when configured, else the
      desktop's Electron when present and agent code is not sandboxed;
    * a fullstack app: the service's URL mode when it is configured and a
      live-app address is known, else Electron against the running app;
    * then the user's preview, when the host can ask it.
    """
    from anton.core.artifacts import html_lint, render_check

    chain: list[Checker] = [StaticChecker(cell_sandbox=cell_sandbox)]
    service = render_check.configured_url() is not None
    if kind in FULLSTACK_KINDS:
        if service and live_app_address is not None:
            chain.append(RenderUrlChecker(live_app_address, screenshot=screenshot))
        elif live_app_address is not None and cell_sandbox is None and html_lint.discover_browser() is not None:
            chain.append(ElectronChecker(url=live_app_address))
    elif kind == "html-app":
        if service:
            chain.append(RenderCheckChecker(previous_html=previous_html, screenshot=screenshot,
                                            cell_sandbox=cell_sandbox))
        elif cell_sandbox is None and html_lint.discover_browser() is not None:
            chain.append(ElectronChecker())
    if preview_checker is not None and slug:
        chain.append(PreviewChecker(preview_checker, slug=slug))
    return ChainChecker(chain)
