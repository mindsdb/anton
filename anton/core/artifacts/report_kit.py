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
           "filter_table", "filter_preview", "page", "save", "check",
           "rel_link", "md_table", "md_audit", "md_check", "link", "inline", "Html",
           "update_html", "find_values"]

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


def _num_text(value):
    # Same text as JavaScript String(): integral floats drop ".0", so the
    # pre-rendered (no-JavaScript) view matches the live view exactly.
    if type(value) is float and _math.isfinite(value) and value.is_integer() and abs(value) < 1e16:
        return str(int(value))
    return str(value)


class Html(str):
    """Markup produced by kit helpers (link, inline); passed through unescaped."""


def link(href, text) -> Html:
    """Safe anchor for p/ul/table cells, e.g. link(rel_link(out, 'source.json'), 'source.json')."""
    href = str(href)
    if _re.match(r'^\s*(javascript|vbscript|data):', href, _re.I):
        raise ValueError('unsafe link scheme')
    return Html(f'<a href="{_html.escape(href, quote=True)}">{esc(text)}</a>')


def inline(*parts) -> Html:
    """One run of escaped text and kit markup: p(inline('Source: ', link(...), ' (read-only)'))."""
    return Html(''.join(esc(x) for x in parts))


def esc(value) -> str:
    """Escape a value for HTML text/attributes; dict/list/bool/None render as JSON; kit Html passes through."""
    if isinstance(value, Html):
        return str(value)
    if isinstance(value, (dict, list, bool)) or value is None:
        value = _json.dumps(value, ensure_ascii=False)
    return _html.escape(_num_text(value), quote=True)


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


def _keys(cols, key):
    keys = [key] if isinstance(key, str) else list(key)
    if not keys:
        raise ValueError('key must name at least one column')
    return keys, [_col(cols, k, 'key') for k in keys]


def _select(rows, kidx, v, selection, positive_only, all_label):
    return [i for i, r in enumerate(rows)
            if all(selection.get(n, all_label) == all_label or str(r[ki]) == str(selection.get(n)) for n, ki in kidx)
            and (not positive_only or r[v] > 0)]


def filter_preview(rows, *, columns, key, value, positive_only=True, all_label='All', selection=None) -> dict:
    """What filter_table shows, computed with the same rules as its JavaScript.

    key is one column name or a list of names (one select per key, combined
    with AND). With selection={column: option, ...} returns {'rows', 'total'}
    for that combination (unspecified keys = all_label). Otherwise, for a
    single key returns {option: {'rows', 'total'}}; for several keys returns
    {key: {option: {'rows', 'total'}}} with the other selects at all_label.
    Rows come back in the caller's shape (dicts stay dicts).
    """
    cols = list(columns)
    keys, kix = _keys(cols, key)
    v = _col(cols, value, 'value')
    original = list(rows)
    rows = _rows(cols, original)
    bad = [i for i, r in enumerate(rows) if not _is_num(r[v])]
    if bad:
        raise ValueError(f'value column {value!r} must be numbers; rows {bad[:3]} have {[rows[i][v] for i in bad[:3]]}')
    kidx = list(zip(keys, kix))

    def result(sel):
        idx = _select(rows, kidx, v, sel, positive_only, all_label)
        total = sum(rows[i][v] for i in idx)
        return {'rows': [dict(original[i]) if isinstance(original[i], dict) else list(rows[i]) for i in idx],
                'total': round(total, 10) if isinstance(total, float) else total}

    if selection is not None:
        unknown = [n for n in selection if n not in keys]
        if unknown:
            raise ValueError(f'selection keys {unknown} are not filter keys {keys}')
        return result(selection)
    per_key = {n: {o: result({n: o}) for o in [all_label] + sorted({str(r[ki]) for r in rows})} for n, ki in kidx}
    return per_key[keys[0]] if isinstance(key, str) else per_key


_FILTER_JS = """(function(){var root=document.getElementById(%(id)s);if(!root)return;
var cfg=JSON.parse(document.getElementById(%(data)s).textContent),sels=cfg.selects.map(function(i){return document.getElementById(i);}),
body=root.querySelector('tbody'),total=document.getElementById(%(total)s),empty=document.getElementById(%(empty)s);
function cell(v){var td=document.createElement('td');td.textContent=(v!==null&&typeof v==='object')?JSON.stringify(v):String(v);
if(typeof v==='number')td.className='n';return td;}
function render(){var rows=cfg.rows.filter(function(r){return sels.every(function(s,j){return s.value===cfg.all||String(r[cfg.keys[j]])===s.value;})&&(!cfg.positive||r[cfg.value]>0);});
body.replaceChildren();rows.forEach(function(r){var tr=document.createElement('tr');r.forEach(function(v){tr.appendChild(cell(v));});body.appendChild(tr);});
var t=rows.reduce(function(s,r){return s+r[cfg.value];},0);total.textContent=cfg.label+': '+String(Math.round(t*1e10)/1e10);
empty.hidden=rows.length>0;}
sels.forEach(function(s){s.addEventListener('change',render);});render();})();"""


def filter_table(*, label, rows, columns, key, value, results_name, total_label, positive_only=True,
                 all_label='All', empty_text='No rows match this selection.', id='filter') -> str:
    """Labelled select(s) plus an accessible results region with a live total.

    key: one column name, or a list of names for several selects combined with
    AND; label: one label or a list matching key. value: numeric column to
    total. Options are all_label plus each distinct value (sorted). The
    all_label view is pre-rendered so the page works without JavaScript;
    JavaScript re-renders rows, the total and the empty state on change. Use
    filter_preview() with the same arguments to verify each option.
    """
    cols = list(columns)
    keys, kix = _keys(cols, key)
    labels = [label] if isinstance(label, str) else list(label)
    if len(labels) != len(keys):
        raise ValueError(f'label needs one entry per key ({len(keys)}), got {len(labels)}')
    v = _col(cols, value, 'value')
    rows = _rows(cols, rows)
    bad = [i for i, r in enumerate(rows) if not _is_num(r[v])]
    if bad:
        raise ValueError(f'value column {value!r} must be numbers; rows {bad[:3]} have {[rows[i][v] for i in bad[:3]]}')
    first = filter_preview(rows, columns=cols, key=keys, value=value, positive_only=positive_only,
                           all_label=all_label, selection={})
    ids = {k2: f'{id}-{k2}' for k2 in ('total', 'empty', 'data', 'region')}
    sel_ids = [f'{id}-select' if len(keys) == 1 else f'{id}-select-{j}' for j in range(len(keys))]
    controls = ''
    for lab, sid, ki in zip(labels, sel_ids, kix):
        opts = ''.join(f'<option value="{esc(o)}">{esc(o)}</option>' for o in [all_label] + sorted({str(r[ki]) for r in rows}))
        controls += f'<label for="{sid}">{esc(lab)}</label><select id="{sid}">{opts}</select> '
    cfg = {'rows': rows, 'keys': kix, 'selects': sel_ids, 'value': v, 'positive': bool(positive_only),
           'all': all_label, 'label': total_label}
    js = _FILTER_JS % {n: _json.dumps(ids[n2]) for n, n2 in (('id', 'region'), ('data', 'data'), ('total', 'total'), ('empty', 'empty'))}
    return (f'<div class="filter">{controls}</div>'
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
            self.local, self.plain, self.code = [], [], 0
            self.in_script = None
            self.tables, self.stack, self.cur, self.cell = [], [], None, None
            self.audit_ctx = None   # [tag, depth] while inside the element with id=audit_id
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
                elif a.get(attr):
                    self.local.append(a[attr])
            if tag in ('code', 'pre'):
                self.code += 1
            if tag == 'script':
                self.in_script = a.get('type', '')
                self.buf = ''
            if self.audit_ctx is not None and tag == self.audit_ctx[0]:
                self.audit_ctx[1] += 1
            elif a.get('id') == audit_id and tag != 'table' and self.audit_ctx is None:
                self.audit_ctx = [tag, 1]
            if tag == 'table':
                self.tables.append({'rows': [], 'own_id': a.get('id') == audit_id, 'inside': self.audit_ctx is not None})
                self.stack.append(len(self.tables) - 1)
            if self.stack and tag == 'tr':
                self.cur = []
            if self.stack and tag in ('td', 'th') and self.cur is not None:
                self.cell = ''

        def handle_endtag(self, tag):
            if tag in ('code', 'pre') and self.code:
                self.code -= 1
            if tag == 'script':
                if self.in_script == 'application/json':
                    try:
                        _json.loads(self.buf)
                    except ValueError:
                        self.json_ok = False
                self.in_script = None
            if self.stack and tag in ('td', 'th') and self.cell is not None:
                self.cur.append(self.cell.strip())
                self.cell = None
            if self.stack and tag == 'tr' and self.cur is not None:
                self.tables[self.stack[-1]]['rows'].append(self.cur)
                self.cur = None
            if tag == 'table' and self.stack:
                self.stack.pop()
            if self.audit_ctx is not None and tag == self.audit_ctx[0]:
                self.audit_ctx[1] -= 1
                if self.audit_ctx[1] == 0:
                    self.audit_ctx = None

        def handle_data(self, d):
            if self.in_script is not None:
                self.buf += d
                return
            if self.cell is not None:
                self.cell += d
            self.texts.append(d)
            if not self.code:
                self.plain.append(d)

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
    shown_markup = _re.findall(r'<\s*/?\s*(?:a|p|div|span|br|table|tr|td|th|ul|ol|li|strong|em|b|i|section|details|summary|h[1-6]|img)\b[^<>]{0,80}>',
                               ' '.join(parser.plain), _re.I)
    if shown_markup:
        problems.append(f'escaped HTML markup is visible as text (pass kit markup such as link()/inline(), not HTML strings): {shown_markup[:3]}')
    broken = _broken_links(path, parser.local, parser.ids)
    if broken:
        problems.append(f'broken local links: {broken[:5]}')
    # The audit table: the element with audit_id itself, else the first table
    # inside it, else the first table headed Metric | Value (pages not built
    # with the kit often put the id on an enclosing section).
    audit = (next((t for t in parser.tables if t['own_id']), None)
             or next((t for t in parser.tables if t['inside']), None)
             or next((t for t in parser.tables if t['rows'] and [c.strip().lower() for c in t['rows'][0][:2]] == ['metric', 'value']), None))
    shown = {r[0]: r[1] for r in (audit or {'rows': []})['rows'] if len(r) == 2}
    if metrics and audit is None:
        problems.append(f'no Metric/Value audit table found (#{audit_id}, a table inside it, or a table headed Metric | Value)')
    for k, v in metrics.items():
        if audit is None:
            break
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
            'local_links_checked': len(parser.local), 'self_contained': True, 'browser_check': 'not performed'}


# --- links and Markdown -----------------------------------------------------

def rel_link(from_file, target) -> str:
    """URL-encoded relative link from the folder of ``from_file`` to ``target``.

    Use for evidence links from an artifact to project files so the link
    resolves wherever the folder tree is opened (e.g. ../../../source.json).
    """
    import os as _os
    from urllib.parse import quote as _quote
    rel = _os.path.relpath(_Path(target).resolve(), _Path(from_file).resolve().parent)
    return _quote(_Path(rel).as_posix(), safe='/._-~')


def _broken_links(path, targets, ids=()):
    """Relative link targets that do not resolve to an existing file/anchor."""
    from urllib.parse import unquote as _unquote, urlsplit as _urlsplit
    base = _Path(path).resolve().parent
    broken = []
    for t in targets:
        t = t.strip().strip('<>')
        if not t or _re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', t):  # data:, mailto:, javascript:, http: ...
            continue
        if t.startswith('#'):
            if ids is not None and t[1:] and t[1:] not in ids:
                broken.append(t)
            continue
        target = _unquote(_urlsplit(t).path)
        if target and not (base / target).exists():
            broken.append(t)
    return broken


def _md_cell(value) -> str:
    if isinstance(value, (dict, list, bool)) or value is None:
        value = _json.dumps(value, ensure_ascii=False)
    text = _num_text(value)
    return _re.sub(r'\s*\n\s*', ' ', text).replace('\\', '\\\\').replace('|', '\\|')


def md_table(columns, rows) -> str:
    """GitHub-flavoured Markdown table: one header, a delimiter row with the
    same cell count, escaped pipes, no line breaks inside cells; numeric
    columns right-aligned. Rows are lists or dicts keyed by column name."""
    cols = [str(c) for c in columns]
    body = _rows(cols, rows)
    num = [bool(body) and all(_is_num(r[i]) for r in body) for i in range(len(cols))]
    out = ['| ' + ' | '.join(_md_cell(c) for c in cols) + ' |',
           '| ' + ' | '.join('---:' if n else '---' for n in num) + ' |']
    out += ['| ' + ' | '.join(_md_cell(v) for v in r) + ' |' for r in body]
    return '\n'.join(out) + '\n'


def md_audit(metrics: dict) -> str:
    """Markdown Metric/Value table showing each metric's exact JSON value."""
    return md_table(['Metric', 'Value'], [[k, _json.dumps(v, ensure_ascii=False)] for k, v in metrics.items()])


_MD_DELIM = _re.compile(r'^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$')


def _md_cells(line):
    """Split a table row on unescaped pipes; unescape \\| and \\\\ as a renderer does."""
    s = line.strip()
    if s.startswith('|'):
        s = s[1:]
    cells, cur, i, closed = [], '', 0, False
    while i < len(s):
        ch = s[i]
        if ch == '\\' and i + 1 < len(s) and s[i + 1] in '\\|':
            cur += s[i + 1]; i += 2; closed = False; continue
        if ch == '|':
            cells.append(cur.strip()); cur = ''; i += 1; closed = True; continue
        cur += ch; i += 1
        if not ch.isspace():
            closed = False
    if cur.strip() or not closed:
        cells.append(cur.strip())
    return cells


def md_check(path, metrics=None, *, text=(), audit_header=('Metric', 'Value')) -> dict:
    """Re-read a saved Markdown file and verify its structure.

    Raises AssertionError listing every problem; returns a receipt otherwise.
    Checks: every table's delimiter row and body rows have the header's cell
    count; with ``metrics``, a Metric/Value table shows each metric's exact
    JSON value; relative links and images resolve to existing files; no
    external images; required text fragments are present. Not a renderer.
    """
    raw = _Path(path).read_text(encoding='utf-8')
    lines = raw.splitlines()
    problems, tables, fence = [], [], False
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.lstrip().startswith(('```', '~~~')):
            fence = not fence
        if not fence and i + 1 < len(lines) and '|' in ln and _MD_DELIM.match(lines[i + 1]):
            header = _md_cells(ln)
            delim = _md_cells(lines[i + 1])
            if len(delim) != len(header):
                problems.append(f'table at line {i + 1}: delimiter has {len(delim)} cells, header has {len(header)}')
            body, j = [], i + 2
            while j < len(lines) and lines[j].strip() and '|' in lines[j]:
                cells = _md_cells(lines[j])
                if len(cells) != len(header):
                    problems.append(f'table at line {i + 1}: row {j + 1} has {len(cells)} cells, header has {len(header)}')
                body.append(cells)
                j += 1
            tables.append((header, body))
            i = j
            continue
        if not fence and _MD_DELIM.match(ln) and '|' in ln and (i == 0 or '|' not in lines[i - 1]):
            problems.append(f'delimiter row without a header at line {i + 1}')
        i += 1
    if metrics:
        strip = lambda c: c.strip().strip('`').strip()
        audit_rows = {}
        for header, body in tables:
            if [strip(h).lower() for h in header[:2]] == [h.lower() for h in audit_header]:
                audit_rows.update({strip(r[0]): strip(r[1]) for r in body if len(r) >= 2})
        if not audit_rows:
            problems.append('no Metric/Value audit table')
        for k, v in metrics.items():
            if k not in audit_rows:
                problems.append(f'audit row missing: {k}')
                continue
            try:
                got = _json.loads(audit_rows[k])
                if got != v or type(got) is not type(v) and not (_is_num(v) and _is_num(got)):
                    problems.append(f'audit value mismatch for {k}: {audit_rows[k]}')
            except ValueError:
                problems.append(f'audit value not JSON for {k}: {audit_rows[k]}')
    prose = _re.sub(r'```.*?```', '', raw, flags=_re.S)
    links = _re.findall(r'(!?)\[[^\]]*\]\(\s*(<[^>]*>|[^)\s]+)', prose)
    external_images = [t for bang, t in links if bang and _re.match(r'^(https?:)?//', t.strip('<>'))]
    if external_images:
        problems.append(f'external images: {external_images[:3]}')
    anchors = None  # Markdown heading anchors vary by renderer; not checked
    broken = _broken_links(path, [t for _, t in links], anchors)
    if broken:
        problems.append(f'broken local links: {broken[:5]}')
    flat = _re.sub(r'\s+', ' ', raw)
    absent = [t for t in text if _re.sub(r'\s+', ' ', t) not in flat]
    if absent:
        problems.append(f'missing text: {absent}')
    if problems:
        raise AssertionError('; '.join(problems))
    return {'path': str(path), 'tables': len(tables), 'audit_metrics': sorted(metrics or {}),
            'links_checked': len(links), 'text_checked': len(text), 'renderer_check': 'not performed'}


# --- editing existing reports ---------------------------------------------------

_VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'source', 'track', 'wbr'}


def _id_spans(raw: str, wanted: set) -> dict:
    """Inner-HTML offsets (start, end) for elements whose id is in ``wanted``."""
    from html.parser import HTMLParser
    starts = [0] + [i + 1 for i, ch in enumerate(raw) if ch == '\n']   # HTMLParser counts lines by \n only

    class P(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=False)
            self.spans, self.open = {}, []   # open: [id, tag, depth, inner_start]

        def _off(self):
            line, col = self.getpos()
            return starts[line - 1] + col

        def handle_starttag(self, tag, attrs):
            for o in self.open:
                if o[1] == tag:
                    o[2] += 1
            ident = dict(attrs).get('id')
            if ident in wanted and ident not in self.spans and tag not in _VOID:
                self.open.append([ident, tag, 1, self._off() + len(self.get_starttag_text())])
            elif ident in wanted and tag in _VOID:
                raise ValueError(f'#{ident} is a void <{tag}> element with no content to replace')

        def handle_endtag(self, tag):
            for o in list(self.open):
                if o[1] == tag:
                    o[2] -= 1
                    if o[2] == 0:
                        self.spans[o[0]] = (o[3], self._off())
                        self.open.remove(o)

    parser = P()
    parser.feed(raw)
    parser.close()
    if parser.open:
        raise ValueError(f'could not find the end tag of {["#" + o[0] for o in parser.open]}; rebuild the page instead')
    return parser.spans


def update_html(path, replacements: dict) -> dict:
    """Replace the inner HTML of elements by id in a saved report; every other byte stays identical.

    ``replacements`` maps an element id to an HTML fragment (e.g. from k.table, k.p, k.audit
    rows, or k.esc(text) for plain text); for an embedded <script type="application/json">
    block pass the new data as a dict/list and it is serialised safely. Raises ValueError for a missing id, a void element
    or an unclosed element, so nothing is half-written. Use it to refresh an existing report
    while keeping its theme, layout and wording; then run k.find_values and k.check.
    """
    path = _Path(path)
    if path.is_symlink():
        raise ValueError('refusing to write through a symlink')
    raw = path.read_bytes().decode('utf-8')   # bytes: keep CRLF and every other byte exactly
    wanted = {str(k) for k in replacements}
    spans = _id_spans(raw, wanted)
    missing = sorted(wanted - set(spans))
    if missing:
        raise ValueError(f'ids not found in {path.name}: {missing}')
    for ident, (a, b) in sorted(spans.items(), key=lambda kv: -kv[1][0]):
        value = replacements[ident]
        raw = raw[:a] + (value if isinstance(value, str) else _json_text(value)) + raw[b:]
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_bytes(raw.encode('utf-8'))
    tmp.replace(path)
    return {'path': str(path), 'replaced': sorted(spans), 'bytes': path.stat().st_size}


def find_values(path, values) -> dict:
    """Where each value still appears in a saved HTML/Markdown file, as whole tokens.

    Returns {value: [short context, ...]} only for values that still appear ({} when
    none do, so ``assert not k.find_values(...)`` works), covering visible text and embedded JSON
    (``80`` does not match ``1860`` or ``80.5``). After a data refresh, pass the
    superseded figures from the snapshot's change notes and update any occurrence that
    still describes the current state (a "previous value" note may legitimately keep it).
    """
    raw = _Path(path).read_bytes().decode('utf-8')
    text = _re.sub(r'<style.*?</style>', ' ', raw, flags=_re.S | _re.I)
    text = _re.sub(r'<script(?![^>]*application/json)[^>]*>.*?</script>', ' ', text, flags=_re.S | _re.I)
    text = _html.unescape(_re.sub(r'<[^>]+>', ' ', text))
    text = _re.sub(r'\s+', ' ', text)
    out = {}
    for v in values:
        needle = v if isinstance(v, str) else _json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, bool)) or v is None else _num_text(v)
        pat = _re.compile(r'(?<![\w.])' + _re.escape(needle) + r'(?![\w]|\.\d)')
        hits = [text[max(0, m.start() - 40):m.end() + 40].strip() for m in pat.finditer(text)][:10]
        if hits:
            out[needle] = hits
    return out
