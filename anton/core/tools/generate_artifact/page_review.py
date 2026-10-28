"""Review of an html-app draft, for the builder that can still edit it.

`review_draft` runs the page's existing checks (the static verifier and the
browser check, unchanged) and adds what the page actually shows: an outline
of headings, tables, visuals and controls. The outline comes from the local
browser's probe when one runs here (after scripts ran, at desktop and phone
width, with controls pressed), else from the HTML text, so a deployment with
no browser still gets the page's structure. `DraftReview.report` words it
all for the builder; only `blocking` decides whether the draft may finish.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING

from anton.core.artifacts import html_lint, render_check
from anton.utils.datasources import scrub_credentials, scrub_output

from . import verifiers
from .state import BrowserCheck, VerifyResult

if TYPE_CHECKING:
    from anton.core.backends.sandbox import CellSandbox

# Reviews one build may run. The first is always shown to the builder; the
# rest follow edits. Past the last, `finish` is accepted as the draft stands.
MAX_DRAFT_REVIEWS = 3

_MAX_ITEMS = 12


# `scheme://user:password@host`: the password of a connection string.
_URI_PASSWORD = re.compile(r"(?i)\b([a-z][a-z0-9+.-]{1,20}://[^\s/:@'\"<>]+:)([^\s@/'\"<>]+)(@)")


def scrub_page(html: str) -> str:
    """The page with every saved secret value, well-known key shape and
    connection-string password replaced by a marker. A published page is
    shared by link, so nothing of the kind may stay in it, whoever wrote it."""
    return _URI_PASSWORD.sub(r"\1[REDACTED]\3", scrub_credentials(html))


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


@dataclass(frozen=True)
class ReviewVerdict:
    """A reviewer's answer to `finish`: accept it, or answer the call with
    `content` (the review, text or text plus screenshot blocks) instead."""

    accept: bool
    content: "str | list[dict]" = "ok"


@dataclass
class DraftReview:
    """One review of the page at `page` (relative to the artifact folder)."""

    page: str
    digest: str
    static: VerifyResult
    live: BrowserCheck
    outline: list[str] = field(default_factory=list)
    outline_source: str = ""
    screenshot: "render_check.RenderScreenshot | None" = None

    @property
    def blocking(self) -> list[str]:
        """The problems the page's checks fail it for, as the verifier words them."""
        return [*self.static.errors, *self.live.verdict.errors]

    def report(self, *, number: int, limit: int) -> str:
        lines = [f"## Review {number} of {limit}: {self.page}"]
        if self.blocking:
            lines.append("Problems the checks fail this page for:")
            lines += [f"- {scrub_output(e)}" for e in self.blocking]
        else:
            lines.append("The checks found no problem that fails the page.")
        if self.live.status == "unverified" and self.live.reason:
            lines.append(f"Browser check: {scrub_output(self.live.reason)}")
        if self.static.warnings or self.live.verdict.warnings:
            lines.append("Warnings:")
            lines += [f"- {scrub_output(w)}" for w in [*self.static.warnings, *self.live.verdict.warnings]]
        if self.outline:
            lines.append(f"What the page shows ({self.outline_source}):")
            lines += [f"- {line}" for line in self.outline]
        lines.append(
            "Compare this with the request and the data. Fix what is broken, missing or "
            "unsupported with `edit_file` (rewrite only if most of the page must change), "
            "then call `finish`. If the page is right as it is, call `finish` again."
        )
        return "\n".join(lines)


def review_draft(
    entry: Path,
    *,
    artifact_path: Path,
    cell_sandbox: "CellSandbox | None",
    screenshot: bool = True,
    previous_html: str | None = None,
) -> DraftReview:
    """Check the page at `entry` and outline what it shows. Blocking; run in a thread."""
    html = entry.read_text(encoding="utf-8", errors="replace")
    static = verifiers.verify_frontend(html, is_fullstack=False)
    probe: html_lint.HtmlProbe | None = None
    if render_check.configured_url() is None and cell_sandbox is None and html_lint.discover_browser() is not None:
        # One local load gives both the browser verdict and the outline.
        probe = html_lint.probe_html(entry, cell_sandbox=None)
        live = verifiers.local_browser_check(probe.findings if probe else None, sandboxed=False)
    else:
        live = verifiers.verify_frontend_live(
            entry, cell_sandbox=cell_sandbox, screenshot=screenshot, previous_html=previous_html
        )
    shot = live.screenshot
    if probe is not None:
        outline = probe_outline(probe.probe) + dangling_ids_line(html)
        source = "loaded in a browser at desktop and phone width"
        if screenshot and shot is None and probe.screenshot_jpeg_base64:
            shot = render_check.RenderScreenshot(media_type="image/jpeg", data=probe.screenshot_jpeg_base64)
    else:
        outline, source = static_outline(html), "read from the HTML before scripts run"
    return DraftReview(
        page=entry.relative_to(artifact_path).as_posix(),
        digest=digest(html),
        static=static,
        live=live,
        outline=[scrub_output(line) for line in outline],
        outline_source=source,
        screenshot=shot if screenshot else None,
    )


# ── Outline from the local browser's probe ──────────────────────────────────


def _q(text: str) -> str:
    return f"“{text}”"


def probe_outline(probe: dict) -> list[str]:
    """The probe's measurements as report lines."""
    lines: list[str] = []
    desktop = probe.get("desktop") or {}
    phone = probe.get("phone") or {}
    if desktop:
        lines += _view_lines(desktop)
    for label, view in (("desktop", desktop), ("phone", phone)):
        if not view:
            continue
        width, scroll = view.get("width"), view.get("scrollWidth")
        if isinstance(width, int) and isinstance(scroll, int) and scroll > width + 1:
            parts = ", ".join(view.get("overflowing") or [])
            lines.append(f"At {label} width ({width} px) the page is {scroll} px wide" + (f": {parts}" if parts else ""))
    small = phone.get("smallText") if phone else None
    if isinstance(small, int) and small:
        lines.append(f"At phone width {small} text element(s) are smaller than 11 px")
    controls = probe.get("controls") or []
    if controls:
        said = []
        for c in controls[:_MAX_ITEMS]:
            what = "raised an error: " + str(c["error"]) if c.get("error") else (
                "changed the page" if c.get("changed") else "changed nothing visible"
            )
            said.append(f"{_q(str(c.get('label') or '?'))} {what}")
        lines.append("Controls pressed: " + "; ".join(said))
    if probe.get("untried"):
        lines.append(f"{probe['untried']} more control(s) were not pressed")
    return lines


def _view_lines(view: dict) -> list[str]:
    lines = []
    if view.get("title"):
        lines.append(f"Title: {_q(view['title'])}")
    heads = view.get("headings") or []
    if heads:
        lines.append("Headings: " + "; ".join(f"H{h['level']} {_q(h['text'])}" for h in heads[:_MAX_ITEMS]))
    lines.append(f"Visible text: {view.get('textChars', 0)} characters")
    for t in (view.get("tables") or [])[:_MAX_ITEMS]:
        cols = ", ".join(t.get("columns") or [])
        cap = f" {_q(t['caption'])}" if t.get("caption") else ""
        lines.append(f"Table {t.get('name')}{cap} [{cols}]: {t.get('rows', 0)} rows shown")
    visuals = view.get("visuals") or []
    if visuals:
        lines.append("Visuals: " + "; ".join(
            f"{v['kind']} {v['name']} {v['width']}x{v['height']} {'drawn' if v.get('drawn') else 'BLANK'}"
            for v in visuals[:_MAX_ITEMS]
        ))
    if view.get("failedViews"):
        lines.append("Views marked failed: " + ", ".join(map(str, view["failedViews"])))
    return lines


# ── Outline from the HTML text ──────────────────────────────────────────────

_NON_TEXT = {"script", "style", "template", "noscript", "head", "title"}
_ID_LOOKUP = re.compile(
    r"""getElementById\(\s*['"]([\w:-]+)['"]\s*\)|querySelector(?:All)?\(\s*['"]#([\w-]+)[^'"]*['"]\s*\)"""
)


class _Outliner(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.title = ""
        self.headings: list[tuple[int, str]] = []
        self.text_chars = 0
        self.ids: set[str] = set()
        self.tables: list[dict] = []
        self.visuals: list[str] = []
        self.controls: list[str] = []
        self.scripts: list[str] = []
        self.external: list[str] = []
        self._heading: list[str] | None = None
        self._control: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if a.get("id"):
            self.ids.add(a["id"])
        if tag not in ("br", "img", "input", "meta", "link", "hr", "source", "col", "wbr"):
            self.stack.append(tag)
        name = f"#{a['id']}" if a.get("id") else tag
        if tag in ("h1", "h2", "h3"):
            self._heading = [tag]
        elif tag == "table":
            self.tables.append({"name": name, "columns": [], "rows": 0, "in_head": False})
        elif tag == "thead" and self.tables:
            self.tables[-1]["in_head"] = True
        elif tag == "tr" and self.tables:
            t = self.tables[-1]
            if not t["in_head"] and t["columns"]:
                t["rows"] += 1
        elif tag in ("th",) and self.tables:
            self._cell = []
        elif tag in ("canvas", "svg", "img"):
            self.visuals.append(f"{tag} {name}")
        elif tag in ("button", "select", "summary") or (tag == "input" and a.get("type") in ("checkbox", "radio", "range")):
            self._control = [name, a.get("aria-label") or ""]
            if tag == "input":
                self.controls.append(f"{tag} {name}")
                self._control = None
        if tag == "script":
            if a.get("src"):
                self.external.append(a["src"])
            self.scripts.append("")

    def handle_endtag(self, tag):
        if tag in ("h1", "h2", "h3") and self._heading:
            self.headings.append((int(tag[1]), " ".join("".join(self._heading[1:]).split())))
            self._heading = None
        elif tag == "thead" and self.tables:
            self.tables[-1]["in_head"] = False
        elif tag == "th" and self._cell is not None and self.tables:
            t = self.tables[-1]
            if t["rows"] == 0:
                t["columns"].append(" ".join("".join(self._cell).split())[:30])
            self._cell = None
        elif tag in ("button", "select", "summary") and self._control is not None:
            name, label = self._control[0], self._control[1] or " ".join(" ".join(self._control[2:]).split())
            self.controls.append(f"{tag} {name}" + (f" {_q(label[:40])}" if label else ""))
            self._control = None
        if tag in self.stack:
            while self.stack and self.stack.pop() != tag:
                pass

    def handle_data(self, data):
        current = self.stack[-1] if self.stack else ""
        if current == "title":
            self.title += data
        if current == "script" and self.scripts:
            self.scripts[-1] += data
            return
        if any(t in _NON_TEXT for t in self.stack):
            return
        self.text_chars += len(data.strip())
        for buf in (self._heading, self._cell):
            if buf is not None:
                buf.append(data)
        if self._control is not None and current != "select":
            self._control.append(data)


def _parse(html: str) -> _Outliner | None:
    p = _Outliner()
    try:
        p.feed(html)
        p.close()
    except Exception:  # noqa: BLE001
        return None
    return p


def dangling_ids_line(html: str) -> list[str]:
    """One line naming the ids scripts look up that no element has, or none.
    A script that creates an element with such an id names it a second time."""
    p = _parse(html)
    if p is None:
        return []
    code = "\n".join(p.scripts)
    dangling: list[str] = []
    for m in _ID_LOOKUP.finditer(code):
        ident = m.group(1) or m.group(2)
        if ident not in p.ids and len(re.findall(rf"['\"#]{re.escape(ident)}\b", code)) <= 1 and ident not in dangling:
            dangling.append(ident)
    if not dangling:
        return []
    return ["Scripts look up ids no element has: " + ", ".join(f"#{d}" for d in dangling[:_MAX_ITEMS])]


def static_outline(html: str) -> list[str]:
    """What the HTML text says the page holds, before any script runs."""
    p = _parse(html)
    if p is None:
        return ["The HTML could not be parsed for an outline."]
    lines = []
    if p.title.strip():
        lines.append(f"Title: {_q(p.title.strip())}")
    if p.headings:
        lines.append("Headings: " + "; ".join(f"H{lvl} {_q(t)}" for lvl, t in p.headings[:_MAX_ITEMS]))
    lines.append(f"Text in the markup: {p.text_chars} characters")
    for t in p.tables[:_MAX_ITEMS]:
        lines.append(f"Table {t['name']} [{', '.join(t['columns'])}]: {t['rows']} rows in the markup (scripts may add more)")
    if p.visuals:
        lines.append("Visual elements: " + ", ".join(p.visuals[:_MAX_ITEMS]))
    if p.controls:
        lines.append("Controls: " + ", ".join(p.controls[:_MAX_ITEMS]))
    if p.external:
        lines.append("External scripts: " + ", ".join(p.external[:_MAX_ITEMS]))
    return lines + dangling_ids_line(html)
