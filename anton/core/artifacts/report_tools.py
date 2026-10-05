"""Building blocks for self-contained HTML reports, used from the scratchpad.

    from anton.core.artifacts import report_tools as rt
    html = rt.page("Stock cover", rt.section("Summary", rt.para("...")),
                   rt.section("Detail", rt.table(["Part", "Units"], rows)))
    rt.save(folder / "report.html", html)
    rt.check(folder / "report.html")   # structure, local links; returns what the page shows
    rt.update(path, {"summary-section": rt.para("...")})   # an existing page, in place

Every function escapes the text it is given, so values from data files are
shown as text and never run as markup. The output is one offline HTML file:
inline CSS and JavaScript, no external requests. These helpers lay out what
the caller computed; they do not calculate or check business values.

They are imported from the installed package rather than copied into artifact
folders, so a report made with one release opens and edits with the next.
"""

from __future__ import annotations

import html as _html
import json
import os
import re
from html.parser import HTMLParser
from pathlib import Path

__all__ = [
    "Html", "page", "section", "para", "bullets", "table", "bar_chart", "filter_table",
    "link", "inline", "rel_link", "data", "save", "check", "update", "insert",
]


class Html(str):
    """Markup produced by these helpers, or wrapped by the caller as trusted.

    Plain strings passed to the helpers are escaped. Adding Html together keeps
    it Html, and a plain string added to Html is escaped, so
    ``rt.para(a) + rt.bullets(b)`` stays markup and text stays text.
    """

    def __add__(self, other):
        return Html(str.__add__(self, _esc(other)))

    def __radd__(self, other):
        return Html(_esc(other) + str(self))


def _esc(value) -> str:
    if isinstance(value, Html):
        return str(value)
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return _html.escape(str(value), quote=True)


def _attr_id(id: str | None) -> str:
    return f' id="{_esc(id)}"' if id else ""


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-") or "section"


_CSS = """
:root{--bg:#f6f7f9;--fg:#18202b;--muted:#566070;--card:#fff;--line:#d9dee5;--accent:#2462b8;--mark:#b5531e}
[data-theme=dark]{--bg:#11161d;--fg:#e8edf3;--muted:#a3adba;--card:#1a212b;--line:#334050;--accent:#7fb0ff;--mark:#f0a070}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px}h1{font-size:1.7rem;margin:.2em 0 .6em}
h2{font-size:1.15rem;margin:0 0 .6em}section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:0 0 16px}
p{margin:.4em 0}.subtitle{color:var(--muted);margin-top:-.4em}a{color:var(--accent)}
.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.95rem}
caption{text-align:left;color:var(--muted);padding:0 0 6px}th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);overflow-wrap:anywhere}
th{font-weight:600;background:color-mix(in srgb,var(--line) 35%,transparent)}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
svg.bars{width:100%;height:auto}svg.bars text{fill:var(--fg);font-size:13px}svg.bars .bar{fill:var(--accent)}svg.bars .threshold{stroke:var(--mark);stroke-dasharray:4 3}
.filter label{font-weight:600;margin-right:8px}.filter select{font:inherit;padding:4px 8px;margin-bottom:10px}.total{font-weight:600}
@media print{body{background:#fff;color:#000}section{border:0;padding:0;break-inside:avoid}.filter select{display:none}}
"""

_FILTER_JS = """
(function(){document.querySelectorAll('[data-rt-filter]').forEach(function(box){
var sel=box.querySelector('select'),rows=Array.prototype.slice.call(box.querySelectorAll('tbody tr')),
total=box.querySelector('[data-rt-total]'),empty=box.querySelector('[data-rt-empty]'),
hideZero=box.hasAttribute('data-rt-hide-zero');
function apply(){var shown=0,sum=0;rows.forEach(function(r){
var ok=sel.selectedIndex===0||r.getAttribute('data-rt-key')===sel.value;
var v=Number(r.getAttribute('data-rt-value'));if(ok&&hideZero&&r.hasAttribute('data-rt-value')&&!(v>0))ok=false;
r.hidden=!ok;if(ok){shown++;if(r.hasAttribute('data-rt-value'))sum+=v;}});
if(total){total.textContent=total.getAttribute('data-rt-total')+': '+String(Math.round(sum*1e6)/1e6);}
if(empty){empty.hidden=shown>0;}}
sel.addEventListener('change',apply);apply();});})();
"""


def page(title: str, *parts, lang: str = "en", theme: str = "light", subtitle: str | None = None) -> Html:
    """A complete document: lang, charset, viewport, inline CSS, light or dark theme."""
    if theme not in ("light", "dark"):
        raise ValueError('theme must be "light" or "dark"')
    body = "".join(_esc(p) for p in parts)
    script = f"<script>{_FILTER_JS}</script>" if "data-rt-filter" in body else ""
    sub = f'<p class="subtitle">{_esc(subtitle)}</p>' if subtitle else ""
    return Html(
        f'<!doctype html><html lang="{_esc(lang)}" data-theme="{theme}"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{_esc(title)}</title>'
        f"<style>{_CSS}</style></head><body><main><h1>{_esc(title)}</h1>{sub}{body}{script}</main></body></html>\n"
    )


def section(heading: str, *parts, id: str | None = None) -> Html:
    """A titled block. ``id`` defaults to the heading's slug plus "-section"."""
    return Html(f'<section id="{_esc(id or _slug(heading) + "-section")}"><h2>{_esc(heading)}</h2>'
                + "".join(_esc(p) for p in parts) + "</section>")


def para(*texts) -> Html:
    """One paragraph per text."""
    return Html("".join(f"<p>{_esc(t)}</p>" for t in texts))


def bullets(items) -> Html:
    return Html("<ul>" + "".join(f"<li>{_esc(i)}</li>" for i in items) + "</ul>")


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _row_cells(columns, row) -> list:
    if isinstance(row, dict):
        return [row.get(c) for c in columns]
    cells = list(row)
    if len(cells) != len(columns):
        raise ValueError(f"row has {len(cells)} cells for {len(columns)} columns: {row!r}")
    return cells


def _thead(columns, numeric) -> str:
    return "<thead><tr>" + "".join(
        f'<th scope="col"{" class=num" if n else ""}>{_esc(c)}</th>' for c, n in zip(columns, numeric)) + "</tr></thead>"


def table(columns, rows, *, caption: str | None = None, id: str | None = None) -> Html:
    """A data table. Rows are lists in column order or dicts keyed by column name."""
    columns = list(columns)
    body = [_row_cells(columns, r) for r in rows]
    numeric = [bool(body) and all(_is_number(r[i]) or r[i] is None for r in body) for i in range(len(columns))]
    cap = f"<caption>{_esc(caption)}</caption>" if caption else ""
    trs = "".join("<tr>" + "".join(f'<td{" class=num" if n else ""}>{_esc(v)}</td>' for v, n in zip(r, numeric))
                  + "</tr>" for r in body)
    return Html(f'<div class="table-wrap"><table{_attr_id(id)}>{cap}{_thead(columns, numeric)}<tbody>{trs}</tbody></table></div>')


def bar_chart(labels, values, *, name: str, value_label: str = "Value", threshold: float | None = None,
              id: str | None = None) -> Html:
    """Horizontal bars as inline SVG with role="img" and the accessible name ``name``.

    Add a ``table`` of the same values when readers need the exact figures.
    """
    labels, values = [str(l) for l in labels], list(values)
    if len(labels) != len(values) or not labels:
        raise ValueError("labels and values must be non-empty and the same length")
    if not all(_is_number(v) and v >= 0 for v in values):
        raise ValueError("bar values must be non-negative numbers")
    top = max(values + ([threshold] if threshold is not None else [])) or 1
    left, width, row_h = 160, 520, 30
    height = row_h * len(labels) + 20
    bars = []
    for i, (label, value) in enumerate(zip(labels, values)):
        y = 10 + i * row_h
        w = width * value / top
        bars.append(f'<text x="{left - 8}" y="{y + 18}" text-anchor="end">{_esc(label)}</text>'
                    f'<rect class="bar" x="{left}" y="{y + 4}" width="{w:.1f}" height="{row_h - 10}">'
                    f"<title>{_esc(label)}: {_esc(value)} {_esc(value_label)}</title></rect>"
                    f'<text x="{left + w + 6:.1f}" y="{y + 18}">{_esc(value)}</text>')
    mark = ""
    if threshold is not None:
        x = left + width * threshold / top
        mark = f'<line class="threshold" x1="{x:.1f}" x2="{x:.1f}" y1="4" y2="{height - 4}"><title>Threshold {_esc(threshold)}</title></line>'
    return Html(f'<svg class="bars"{_attr_id(id)} role="img" aria-label="{_esc(name)}" '
                f'viewBox="0 0 {left + width + 60} {height}"><title>{_esc(name)}</title>{"".join(bars)}{mark}</svg>')


def filter_table(columns, rows, *, key: str, label: str, region_name: str, value: str | None = None,
                 total_label: str | None = None, empty_text: str = "No matching rows.",
                 hide_zero: bool = False, id: str | None = None) -> Html:
    """A select that filters table rows by the ``key`` column, inside a labelled region.

    ``value`` names a numeric column to total over the visible rows (shown as
    "<total_label>: N"); ``hide_zero`` also hides rows whose value is not positive.
    The All view is rendered in the HTML, so the page reads correctly without JavaScript.
    """
    columns = list(columns)
    if key not in columns or (value is not None and value not in columns):
        raise ValueError("key and value must be column names")
    body = [_row_cells(columns, r) for r in rows]
    k = columns.index(key)
    v = columns.index(value) if value is not None else None
    numeric = [bool(body) and all(_is_number(r[i]) or r[i] is None for r in body) for i in range(len(columns))]
    options = sorted({str(r[k]) for r in body})
    fid = id or _slug(region_name) + "-filter"
    trs, shown, total = [], 0, 0
    for r in body:
        visible = not (hide_zero and v is not None and not (_is_number(r[v]) and r[v] > 0))
        shown += visible
        total += r[v] if visible and v is not None and _is_number(r[v]) else 0
        val_attr = f' data-rt-value="{_esc(r[v])}"' if v is not None and _is_number(r[v]) else ""
        trs.append(f'<tr data-rt-key="{_esc(r[k])}"{val_attr}{"" if visible else " hidden"}>'
                   + "".join(f'<td{" class=num" if n else ""}>{_esc(c)}</td>' for c, n in zip(r, numeric)) + "</tr>")
    total_html = ""
    if v is not None:
        tl = total_label or f"Total {value}"
        total_html = f'<p class="total" data-rt-total="{_esc(tl)}" aria-live="polite">{_esc(tl)}: {_esc(round(total, 6))}</p>'
    return Html(
        f'<div class="filter" id="{_esc(fid)}" data-rt-filter{" data-rt-hide-zero" if hide_zero else ""}>'
        f'<label for="{_esc(fid)}-select">{_esc(label)}</label>'
        f'<select id="{_esc(fid)}-select"><option value="">All</option>'
        + "".join(f'<option value="{_esc(o)}">{_esc(o)}</option>' for o in options) + "</select>"
        f'<div role="region" aria-label="{_esc(region_name)}"><h3>{_esc(region_name)}</h3>{total_html}'
        f'<p data-rt-empty{" hidden" if shown else ""}>{_esc(empty_text)}</p>'
        f'<div class="table-wrap"><table>{_thead(columns, numeric)}<tbody>{"".join(trs)}</tbody></table></div></div></div>'
    )


def link(href: str, text: str) -> Html:
    if re.match(r"\s*javascript:", str(href), re.I):
        raise ValueError("javascript: links are not allowed")
    return Html(f'<a href="{_esc(href)}">{_esc(text)}</a>')


def inline(*parts) -> Html:
    """Text and links run together, for use inside para, bullets or table cells."""
    return Html("".join(_esc(p) for p in parts))


def rel_link(from_file, target) -> str:
    """The relative href from the HTML file at ``from_file`` to ``target``."""
    return Path(os.path.relpath(Path(target).resolve(), Path(from_file).resolve().parent)).as_posix()


def _json_text(value) -> str:
    # A "</script>" inside a value must not close the block.
    text = json.dumps(value, ensure_ascii=False, allow_nan=False)
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def data(id: str, value) -> Html:
    """An inert JSON block, for data the author chooses to embed (nothing is embedded otherwise)."""
    return Html(f'<script type="application/json" id="{_esc(id)}">{_json_text(value)}</script>')


def save(path, html: str) -> Path:
    """Write ``html`` to ``path`` atomically (a failed write leaves the old file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(str(html), encoding="utf-8")
    tmp.replace(path)
    return path


# --- reading a saved page back ------------------------------------------------

_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}

# Markup that reaches the reader as text, e.g. "<h2>Summary</h2>" passed as a plain
# string and escaped. Text inside <code> or <pre> is left alone.
_SHOWN_TAG = re.compile(r"</?(?:h[1-6]|p|a|div|section|span|table|tr|td|th|ul|ol|li|strong|em|b|i|br)\b[^<>]{0,200}>",
                        re.I)


class _Reader(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids, self.links, self.scripts, self.text = [], [], [], []
        self.lang = self.viewport = self.title = None
        self.tables, self._table, self._row, self._cell, self._skip, self._in_title = [], None, None, None, 0, False
        self.shown_tags, self._code = [], 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.append(a["id"])
        if tag == "html":
            self.lang = a.get("lang")
        elif tag == "meta" and a.get("name") == "viewport":
            self.viewport = a.get("content")
        elif tag == "a" and a.get("href") is not None:
            self.links.append(a["href"])
        elif tag in ("script", "link", "img", "iframe") and (a.get("src") or (tag == "link" and a.get("href"))):
            self.scripts.append(a.get("src") or a.get("href"))
        elif tag in ("script", "style"):
            self._skip += 1
        elif tag in ("code", "pre"):
            self._code += 1
        elif tag == "title" and self.title is None:
            self._in_title = True  # the document title; svg <title>s come later
        elif tag == "table":
            self._table = {"id": a.get("id"), "columns": [], "rows": []}
        elif tag == "tr" and self._table is not None:
            self._row = {"cells": [], "head": False, "hidden": "hidden" in a}
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
            self._row["head"] = self._row["head"] or tag == "th"

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
        elif tag in ("code", "pre") and self._code:
            self._code -= 1
        elif tag == "title" and self._in_title:
            self._in_title = False
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row["cells"].append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None and self._table is not None:
            if self._row["head"] and not self._table["columns"]:
                self._table["columns"] = self._row["cells"]
            else:
                self._table["rows"].append(self._row["cells"])
            self._row = None
        elif tag == "table" and self._table is not None:
            self.tables.append(self._table)
            self._table = None

    def handle_data(self, text):
        if self._skip:
            return
        if self._in_title:
            self.title = text
            return
        if self._cell is not None:
            self._cell.append(text)
        if not self._code:
            self.shown_tags.extend(_SHOWN_TAG.findall(text))
        self.text.append(text)


def check(path) -> dict:
    """Read a saved page back. Raises ValueError on structural problems.

    Checks: lang and viewport present, no external scripts, styles or images,
    unique element ids, every local link resolving to an existing file, and no
    HTML tags shown to the reader as text (markup passed as a plain string is
    escaped; wrap trusted markup from these helpers, or pass it to ``section``).
    Returns the title, the visible text and each table's columns and rows (as
    shown text), for comparing with the values the page was built from. This
    is not a browser check and says nothing about business correctness.
    """
    path = Path(path)
    reader = _Reader()
    reader.feed(path.read_text(encoding="utf-8"))
    problems = []
    if not reader.lang:
        problems.append("html element has no lang")
    if not reader.viewport:
        problems.append("no viewport meta tag")
    external = [s for s in reader.scripts if re.match(r"\s*(https?:)?//", s or "")]
    if external:
        problems.append(f"external resources: {external}")
    dupes = sorted({i for i in reader.ids if reader.ids.count(i) > 1})
    if dupes:
        problems.append(f"duplicate ids: {dupes}")
    broken = []
    for href in reader.links:
        target = href.split("#", 1)[0].split("?", 1)[0]
        if not target or re.match(r"\s*([a-z][a-z0-9+.-]*:|//)", target, re.I):
            continue
        if not (path.parent / target).exists():
            broken.append(href)
    if broken:
        problems.append(f"local links that do not resolve: {broken}")
    if reader.shown_tags:
        problems.append(f"HTML tags shown as text: {reader.shown_tags[:3]}")
    if problems:
        raise ValueError("; ".join(problems))
    return {
        "title": (reader.title or "").strip(),
        "text": " ".join(" ".join(reader.text).split()),
        "tables": [{"id": t["id"], "columns": t["columns"], "rows": t["rows"]} for t in reader.tables],
        "links": reader.links,
    }


# --- editing a saved page in place ---------------------------------------------

class _Locator(HTMLParser):
    """Character offsets of each element with an id: its content and the whole element."""

    def __init__(self, text):
        super().__init__(convert_charrefs=False)
        self._text = text
        self._lines = [0]
        for line in text.splitlines(keepends=True):
            self._lines.append(self._lines[-1] + len(line))
        self._stack, self.spans, self.outer = [], {}, {}

    def _offset(self):
        line, col = self.getpos()
        return self._lines[line - 1] + col

    def handle_starttag(self, tag, attrs):
        if tag in _VOID:
            return
        start_tag = self.get_starttag_text() or ""
        here = self._offset()
        self._stack.append((tag, dict(attrs).get("id"), here + len(start_tag), here))

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == tag:
                _, id_, start, outer_start = self._stack[i]
                del self._stack[i:]
                if id_ is not None:
                    end = self._offset()
                    self.spans.setdefault(id_, []).append((start, end))
                    self.outer.setdefault(id_, []).append((outer_start, self._text.index(">", end) + 1))
                return


_LEADING_HEADING = re.compile(r"\s*<h([1-6])\b[^>]*>.*?</h\1\s*>", re.I | re.S)


def _locate(text: str) -> _Locator:
    locator = _Locator(text)
    locator.feed(text)
    locator.close()
    return locator


_HTML_TAG = re.compile(r"<html\b[^>]*>", re.I)
_HEAD_TAG = re.compile(r"<head\b[^>]*>", re.I)
_VIEWPORT = '<meta name="viewport" content="width=device-width,initial-scale=1">'


def _with_basics(text: str) -> tuple[str, list]:
    """``text`` with a lang attribute and a viewport meta tag added if either is missing.

    Pages made elsewhere often lack them; an edited page should still read well
    on a phone and pass ``check``. lang defaults to "en", as in ``page``.
    """
    reader = _Reader()
    reader.feed(text)
    added = []
    html_tag = _HTML_TAG.search(text)
    if not reader.viewport:
        anchor = _HEAD_TAG.search(text) or html_tag
        if anchor:
            text = text[:anchor.end()] + _VIEWPORT + text[anchor.end():]
            added.append("viewport")
    if not reader.lang and html_tag:
        text = text[:html_tag.start() + 5] + ' lang="en"' + text[html_tag.start() + 5:]
        added.append("lang")
    return text, added


def _write_edit(path: Path, before: str, out: str, receipt: dict) -> dict:
    out, added = _with_basics(out)
    save(path, out)
    receipt.update(bytes_before=len(before.encode()), bytes_after=len(out.encode()))
    if added:
        receipt["added"] = added
    return receipt


def _one(found: dict, id_: str) -> tuple[int, int]:
    spans = found.get(id_, [])
    if len(spans) != 1:
        raise ValueError(f"id {id_!r} matches {len(spans)} elements; it must match exactly one")
    return spans[0]


def update(path, changes: dict) -> dict:
    """Replace the content of elements by id, leaving every other byte unchanged.

    Works on any saved page, including ones not made with these helpers; a
    page missing a lang attribute or viewport tag gets them (see ``added``).
    ``changes`` maps an element id to new content: markup from these helpers,
    plain text (escaped), or, for a ``script type="application/json"`` block,
    any JSON value. A heading at the start of the element (a section's title)
    is kept unless the new content starts with a heading of its own. Each id
    must match exactly one element. Returns a receipt of the ids changed and
    the old and new file sizes.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    locator = _locate(text)
    edits = []
    for id_, content in changes.items():
        start, end = _one(locator.spans, id_)
        if isinstance(content, str):
            new = _esc(content)
            heading = _LEADING_HEADING.match(text, start, end)
            if heading and not re.match(r"\s*<h[1-6]\b", new, re.I):
                new = heading.group(0) + new
        else:
            new = _json_text(content)
        edits.append((start, end, new))
    edits.sort()
    for (s1, e1, _), (s2, _e2, _) in zip(edits, edits[1:]):
        if s2 < e1:
            raise ValueError("edited elements overlap")
    out = text
    for start, end, new in reversed(edits):
        out = out[:start] + new + out[end:]
    return _write_edit(path, text, out, {"changed": sorted(changes)})


def insert(path, content, *, before: str | None = None, after: str | None = None) -> dict:
    """Add ``content`` just before or just after the element with that id.

    For new parts of an existing page, e.g. a decision summary above the
    detail: ``rt.insert(path, rt.section("Decision summary", rt.para(...)),
    before="detail-section")``. Every other byte is unchanged. Plain text is
    escaped, as elsewhere. Adds a missing lang or viewport, and returns a
    receipt, like ``update``.
    """
    if (before is None) == (after is None):
        raise ValueError("give exactly one of before= or after=")
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    start, end = _one(_locate(text).outer, before or after)
    at = start if before is not None else end
    out = text[:at] + _esc(content) + text[at:]
    return _write_edit(path, text, out, {"inserted": "before " + before if before is not None else "after " + after})
