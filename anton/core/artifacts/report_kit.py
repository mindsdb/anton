"""report_kit — small, tested building blocks for self-contained local reports.

Available in scratchpad cells as the builtin ``report_kit``. It renders
escaped HTML fragments and a complete offline page; it never computes business
values. The caller supplies every number, row and sentence and remains
responsible for correctness. ``check()`` re-reads a saved report and verifies
structure and exact audit values; it does not run a browser.
"""
from __future__ import annotations

import html as _html
import json as _json
import math as _math
import re as _re
from pathlib import Path as _Path

__all__ = ["esc", "p", "ul", "table", "audit", "section", "details", "data", "bar_chart",
           "filter_table", "filter_preview", "page", "save", "check"]

_CSS = """
:root{--bg:#f6f8fb;--panel:#fff;--ink:#17202c;--muted:#4f5d6e;--line:#d6dde6;--accent:#2a62b8;--hot:#b4371f}
html[data-theme=dark]{--bg:#101820;--panel:#182433;--ink:#eef3fa;--muted:#b3c2d3;--line:#3b4c5f;--accent:#7eaefc;--hot:#ff9b80}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1040px;margin:0 auto;padding:28px 20px 56px}h1{font-size:30px;line-height:1.2;margin:8px 0 18px}h2{font-size:20px;margin:0 0 12px}
section,details{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin:14px 0}
summary{cursor:pointer;font-weight:600}.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;margin:8px 0}
caption{text-align:left;color:var(--muted);padding:4px 0}th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}code{font-family:ui-monospace,Menlo,monospace;overflow-wrap:anywhere}
label{font-weight:600;margin-right:8px}select{font:inherit;padding:6px 12px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink)}
svg{width:100%;height:auto}svg text{fill:currentColor;font:14px system-ui,sans-serif}.muted{color:var(--muted)}
@media(max-width:600px){main{padding:16px 12px}section,details{padding:14px}h1{font-size:24px}}
"""


def esc(value) -> str:
    """Escape a value for HTML text/attributes; dict/list/bool/None render as JSON."""
    if isinstance(value, (dict, list, bool)) or value is None:
        value = _json.dumps(value, ensure_ascii=False)
    return _html.escape(str(value), quote=True)


def p(*texts, cls=None) -> str:
    c = f' class="{esc(cls)}"' if cls else ''
    return ''.join(f'<p{c}>{esc(t)}</p>' for t in texts)


def ul(items) -> str:
    return '<ul>' + ''.join(f'<li>{esc(i)}</li>' for i in items) + '</ul>'


def _is_num(v):
    return type(v) in (int, float) and _math.isfinite(v)


def _rows(columns, rows, where='rows'):
    """Accept rows as sequences (one cell per column) or dicts keyed by column name."""
    cols = list(columns)
    out = []
    for i, r in enumerate(rows):
        if isinstance(r, dict):
            missing = [c for c in cols if c not in r]
            if missing:
                raise ValueError(f'{where}[{i}] is a dict without column(s) {missing}; keys are {list(r)}')
            out.append([r[c] for c in cols])
        else:
            r = list(r)
            if len(r) != len(cols):
                raise ValueError(f'{where}[{i}] has {len(r)} cells but columns has {len(cols)}: {cols}')
            out.append(r)
    return out


def table(columns, rows, *, caption=None, id=None) -> str:
    """Accessible table; rows are lists or dicts keyed by column name; numbers right-aligned; cells escaped."""
    cols = list(columns)
    rows = _rows(cols, rows)
    numeric = [bool(rows) and all(_is_num(r[i]) for r in rows) for i in range(len(cols))]
    head = ''.join(f'<th scope="col"{" class=n" if numeric[i] else ""}>{esc(c)}</th>' for i, c in enumerate(cols))
    body = ''.join('<tr>' + ''.join(f'<td{" class=n" if numeric[i] else ""}>{esc(v)}</td>' for i, v in enumerate(r)) + '</tr>' for r in rows)
    ident = f' id="{esc(id)}"' if id else ''
    cap = f'<caption>{esc(caption)}</caption>' if caption else ''
    return f'<div class="wrap"><table{ident}>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def audit(metrics: dict, *, id='metric-audit', caption=None) -> str:
    """Metric/Value table whose values are the exact JSON of each metric."""
    body = ''.join(f'<tr><th scope="row">{esc(k)}</th><td><code>{_html.escape(_json.dumps(v, ensure_ascii=False))}</code></td></tr>'
                   for k, v in metrics.items())
    cap = f'<caption>{esc(caption)}</caption>' if caption else ''
    return (f'<div class="wrap"><table id="{esc(id)}">{cap}<thead><tr><th scope="col">Metric</th><th scope="col">Value</th>'
            f'</tr></thead><tbody>{body}</tbody></table></div>')


def section(heading, *parts, id=None) -> str:
    ident = f' id="{esc(id)}"' if id else ''
    return f'<section{ident}><h2>{esc(heading)}</h2>' + ''.join(parts) + '</section>'


def details(summary, *parts, open=True, id=None) -> str:
    ident = f' id="{esc(id)}"' if id else ''
    return f'<details{ident}{" open" if open else ""}><summary>{esc(summary)}</summary>' + ''.join(parts) + '</details>'


def _json_text(value) -> str:
    # Inert JSON inside <script>: neutralise </script>, comments and entities.
    return _json.dumps(value, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')


def data(id, value) -> str:
    """Embed a JSON snapshot as an inert script block."""
    return f'<script type="application/json" id="{esc(id)}">{_json_text(value)}</script>'


def bar_chart(labels, values=None, *, name, threshold=None, value_label='Value', id='chart') -> str:
    """Offline accessible horizontal bar chart (role=img, accessible name = name).

    Bars at or above ``threshold`` use the highlight colour; a dashed line marks
    the threshold. Add a data table separately for exact values.
    """
    if values is None and isinstance(labels, dict):
        labels, values = list(labels), list(labels.values())
    labels, values = [str(x) for x in labels], list(values if values is not None else [])
    if len(labels) != len(values) or not labels:
        raise ValueError(f'bar_chart needs matching labels and values (got {len(labels)} labels, {len(values)} values)')
    bad = [(l, v) for l, v in zip(labels, values) if not (_is_num(v) and v >= 0)]
    if bad:
        raise ValueError(f'bar_chart values must be non-negative finite numbers; bad: {bad[:3]}')
    top = max([*values, threshold or 0, 1])
    left, width, row = 130, 520, 44
    h = 30 + row * len(labels) + (30 if threshold is not None else 10)
    parts = []
    for i, (lab, v) in enumerate(zip(labels, values)):
        y = 20 + i * row
        w = width * v / top
        hot = threshold is not None and v >= threshold
        parts.append(f'<text x="8" y="{y + 20}">{esc(lab)}</text><rect x="{left}" y="{y + 4}" width="{w:.2f}" height="24" '
                     f'fill="var({"--hot" if hot else "--accent"})"><title>{esc(lab)}: {esc(v)}</title></rect>'
                     f'<text x="{left + w + 8:.2f}" y="{y + 21}">{esc(v)}</text>')
    if threshold is not None:
        x = left + width * threshold / top
        parts.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="14" y2="{h - 24}" stroke="currentColor" stroke-dasharray="5 4"/>'
                     f'<text x="{x:.2f}" y="{h - 6}" text-anchor="middle">{esc(value_label)} threshold: {esc(threshold)}</text>')
    return (f'<svg id="{esc(id)}" role="img" aria-label="{esc(name)}" viewBox="0 0 740 {h}"><title>{esc(name)}</title>'
            + ''.join(parts) + '</svg>')


def _col(cols, name, role):
    if name not in cols:
        raise ValueError(f'{role}={name!r} is not one of columns {cols}')
    return cols.index(name)


def filter_preview(rows, *, columns, key, value, positive_only=True, all_label='All') -> dict:
    """What filter_table shows for each option: {option: {'rows': [...], 'total': n}}."""
    cols = list(columns)
    k, v = _col(cols, key, 'key'), _col(cols, value, 'value')
    original = list(rows)
    rows = _rows(cols, original)
    bad = [i for i, r in enumerate(rows) if not _is_num(r[v])]
    if bad:
        raise ValueError(f'value column {value!r} must be numbers; rows {bad[:3]} have {[rows[i][v] for i in bad[:3]]}')
    options = [all_label] + sorted({str(r[k]) for r in rows})
    out = {}
    for opt in options:
        idx = [i for i, r in enumerate(rows) if (opt == all_label or str(r[k]) == opt) and (not positive_only or r[v] > 0)]
        total = sum(rows[i][v] for i in idx)
        # Rows come back in the caller's shape (dicts stay dicts, lists stay lists).
        sel = [dict(original[i]) if isinstance(original[i], dict) else list(rows[i]) for i in idx]
        out[opt] = {'rows': sel, 'total': round(total, 10) if isinstance(total, float) else total}
    return out


_FILTER_JS = """(function(){var root=document.getElementById(%(id)s);if(!root)return;
var cfg=JSON.parse(document.getElementById(%(data)s).textContent),sel=document.getElementById(%(sel)s),
body=root.querySelector('tbody'),total=document.getElementById(%(total)s),empty=document.getElementById(%(empty)s);
function cell(v){var td=document.createElement('td');td.textContent=(v!==null&&typeof v==='object')?JSON.stringify(v):String(v);
if(typeof v==='number')td.className='n';return td;}
function render(){var o=sel.value,rows=cfg.rows.filter(function(r){return(o===cfg.all||String(r[cfg.key])===o)&&(!cfg.positive||r[cfg.value]>0);});
body.replaceChildren();rows.forEach(function(r){var tr=document.createElement('tr');r.forEach(function(v){tr.appendChild(cell(v));});body.appendChild(tr);});
var t=rows.reduce(function(s,r){return s+r[cfg.value];},0);total.textContent=cfg.label+': '+String(Math.round(t*1e10)/1e10);
empty.hidden=rows.length>0;}
sel.addEventListener('change',render);render();})();"""


def filter_table(*, label, rows, columns, key, value, results_name, total_label, positive_only=True,
                 all_label='All', empty_text='No matching results for this selection.', id='filter') -> str:
    """Labelled select plus an accessible results region with a live total.

    ``key`` and ``value`` are column names in ``columns``. Options are
    ``all_label`` and every distinct key value (sorted). The default
    (``all_label``) view is pre-rendered so the page is usable without
    JavaScript; JavaScript re-renders rows, the total and the empty state on
    change. Use filter_preview() with the same arguments to verify each option.
    """
    cols = list(columns)
    k, v = _col(cols, key, 'key'), _col(cols, value, 'value')
    rows = _rows(cols, rows)
    bad = [i for i, r in enumerate(rows) if not _is_num(r[v])]
    if bad:
        raise ValueError(f'value column {value!r} must be numbers; rows {bad[:3]} have {[rows[i][v] for i in bad[:3]]}')
    prev = filter_preview(rows, columns=cols, key=key, value=value, positive_only=positive_only, all_label=all_label)
    first = prev[all_label]
    ids = {k2: f'{id}-{k2}' for k2 in ('select', 'total', 'empty', 'data', 'region')}
    opts = ''.join(f'<option value="{esc(o)}">{esc(o)}</option>' for o in prev)
    cfg = {'rows': rows, 'key': k, 'value': v, 'positive': bool(positive_only), 'all': all_label, 'label': total_label}
    js = _FILTER_JS % {n: _json.dumps(ids[n2]) for n, n2 in (('id', 'region'), ('data', 'data'), ('sel', 'select'), ('total', 'total'), ('empty', 'empty'))}
    return (f'<div class="filter"><label for="{ids["select"]}">{esc(label)}</label><select id="{ids["select"]}">{opts}</select></div>'
            f'<section id="{ids["region"]}" role="region" aria-label="{esc(results_name)}"><h2>{esc(results_name)}</h2>'
            f'<p id="{ids["total"]}" aria-live="polite"><strong>{esc(total_label)}: {esc(first["total"])}</strong></p>'
            + table(cols, first['rows']) +
            f'<p id="{ids["empty"]}"{" hidden" if first["rows"] else ""}>{esc(empty_text)}</p></section>'
            + data(ids['data'], cfg) + f'<script>{js}</script>')


def page(title, *parts, theme='light', lang='en') -> str:
    """Complete self-contained HTML document (no external assets)."""
    if theme not in ('light', 'dark'):
        raise ValueError('theme must be light or dark')
    return (f'<!doctype html><html lang="{esc(lang)}" data-theme="{theme}"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{esc(title)}</title>'
            f'<style>{_CSS}</style></head><body><main><h1>{esc(title)}</h1>' + ''.join(parts) + '</main></body></html>')


def save(path, text) -> dict:
    path = _Path(path)
    if path.is_symlink():
        raise ValueError('refusing to write through a symlink')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)
    return {'path': str(path), 'bytes': path.stat().st_size}


def check(path, metrics: dict, *, ids=(), text=(), audit_id='metric-audit') -> dict:
    """Re-read a saved HTML report and verify structure and exact audit values.

    Raises AssertionError listing every problem; returns a receipt otherwise.
    Checks: lang + viewport, no external http(s) assets, every metric shown in
    the audit table with its exact JSON value, every required element id, every
    required text fragment in visible text, and parseable JSON script blocks.
    Not a browser check.
    """
    from html.parser import HTMLParser

    raw = _Path(path).read_text(encoding='utf-8')
    problems = []

    class P(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.ids, self.ext, self.json_ok, self.texts = set(), [], True, []
            self.in_script = None
            self.rows, self.cur, self.cell, self.in_audit = [], None, None, False
            self.depth = 0
            self.lang = self.viewport = False

        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            if a.get('id'):
                self.ids.add(a['id'])
            if tag == 'html' and a.get('lang'):
                self.lang = True
            if tag == 'meta' and a.get('name') == 'viewport':
                self.viewport = True
            for attr in ('src', 'href'):
                if _re.match(r'^(https?:)?//', a.get(attr) or ''):
                    self.ext.append(a[attr])
            if tag == 'script':
                self.in_script = a.get('type', '')
                self.buf = ''
            if tag == 'table' and a.get('id') == audit_id:
                self.in_audit = True
            if self.in_audit and tag == 'tr':
                self.cur = []
            if self.in_audit and tag in ('td', 'th') and self.cur is not None:
                self.cell = ''

        def handle_endtag(self, tag):
            if tag == 'script':
                if self.in_script == 'application/json':
                    try:
                        _json.loads(self.buf)
                    except ValueError:
                        self.json_ok = False
                self.in_script = None
            if self.in_audit and tag in ('td', 'th') and self.cell is not None:
                self.cur.append(self.cell.strip())
                self.cell = None
            if self.in_audit and tag == 'tr' and self.cur is not None:
                self.rows.append(self.cur)
                self.cur = None
            if tag == 'table' and self.in_audit:
                self.in_audit = False

        def handle_data(self, d):
            if self.in_script is not None:
                self.buf += d
                return
            if self.cell is not None:
                self.cell += d
            self.texts.append(d)

    parser = P()
    parser.feed(raw)
    if not parser.lang:
        problems.append('html lang missing')
    if not parser.viewport:
        problems.append('viewport meta missing')
    if parser.ext or _re.search(r'@import|url\(\s*["\']?https?:', raw):
        problems.append(f'external assets: {parser.ext[:3]}')
    if not parser.json_ok:
        problems.append('invalid JSON script block')
    shown = {r[0]: r[1] for r in parser.rows if len(r) == 2}
    for k, v in metrics.items():
        if k not in shown:
            problems.append(f'audit row missing: {k}')
            continue
        try:
            if _json.loads(shown[k]) != v or type(_json.loads(shown[k])) is not type(v) and not (_is_num(v) and _is_num(_json.loads(shown[k]))):
                problems.append(f'audit value mismatch for {k}: {shown[k]}')
        except ValueError:
            problems.append(f'audit value not JSON for {k}: {shown[k]}')
    missing = [i for i in ids if i not in parser.ids]
    if missing:
        problems.append(f'missing ids: {missing}')
    visible = _re.sub(r'\s+', ' ', ' '.join(parser.texts))
    absent = [t for t in text if t not in visible]
    if absent:
        problems.append(f'missing text: {absent}')
    if problems:
        raise AssertionError('; '.join(problems))
    return {'path': str(path), 'audit_metrics': sorted(metrics), 'ids': sorted(ids), 'text_checked': len(text),
            'self_contained': True, 'browser_check': 'not performed'}
