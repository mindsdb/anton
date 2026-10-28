"""Compile a document to portable HTML. Only resolved bindings are published."""
from __future__ import annotations

from dataclasses import dataclass
from html import escape
from importlib.resources import files
import math
import re

from .views import checks, validate_ids
from .data import DataError, Evidence, calculate, execute, json_value, number
from .document import Callout, Chart, Document, Html, Metric, Prose, Table, resolve_facts, resolve_query


@dataclass(frozen=True)
class Rendered:
    html: str
    manifest: dict


def _asset(name):
    return files(__package__).joinpath(name).read_text(encoding="utf-8")


def _script_json(value):
    return json_value(value).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _inject_head(html, payload):
    """Keep UTF-8 within the sniffing window, before even a large data script."""
    head = re.search(r'<head(?:\s[^>]*)?>', html, re.I)
    if head is None:
        raise ValueError('An HTML head is required for data initialization')
    charset = re.match(r'\s*<meta\s+charset\s*=\s*([\"\']?)utf-8\1\s*/?>', html[head.end():], re.I)
    position = head.end() + charset.end() if charset else head.end()
    prefix = '' if charset else '<meta charset="utf-8">'
    if len((html[:position] + prefix).encode('utf-8')) > 1024:
        raise ValueError('Move the HTML head and UTF-8 declaration into the first 1024 bytes')
    return html[:position] + prefix + payload + html[position:]


def render(document: Document, evidence: Evidence, *, resource_sources: dict | None = None) -> Rendered:
    values = resolve_facts(document, evidence)
    bindings = {name: execute(evidence, query) for name, query in document.bindings.items()}
    sources, published_facts = {}, {}
    for name, metadata in (resource_sources or {}).items():
        sources[f'reference:{name}'] = {'label': metadata['label'], 'url': metadata['url'], 'scope': 'Supporting visual resource'}

    def text(value):
        def substitute(match):
            name = match.group(1)
            if name not in values:
                raise DataError(f"Unknown fact in text: {name}")
            published_facts[name] = values[name]
            return document.facts[name].format.display(values[name])
        return escape(re.sub(r"\{\{([A-Za-z][A-Za-z0-9_-]*)\}\}", substitute, value))

    def source_name(src):
        label = escape(src['label'])
        return f'<a href="{escape(src["url"])}" target="_blank" rel="noopener noreferrer">{label}</a>' if re.match(r'^https?://', src['url']) else label

    def source(query):
        src = evidence.datasets[query.dataset].source
        sources[query.dataset] = src.model_dump()
        return f'<p class="source">Source: {source_name(src.model_dump())}' + (f' · {escape(src.scope)}' if src.scope else '') + '</p>'

    def table(rows, columns):
        heads = ''.join(f'<th scope="col">{escape(c.label)}</th>' for c in columns)
        body = []
        for row in rows:
            cells = []
            for c in columns:
                value = row[c.field]
                content = c.format.display(value) if c.format else str(value) if value is not None else "Not available"
                numeric = ' class="numeric"' if isinstance(value, (int, float)) or c.format else ''
                shown = escape(content)
                if re.search(r'</?[A-Za-z][^>]*>', content):
                    shown = '<code>' + shown + '</code>'
                cells.append(f'<td{numeric} data-label="{escape(c.label)}">{shown}</td>')
            body.append('<tr>' + ''.join(cells) + '</tr>')
        empty = f'<tr><td colspan="{len(columns)}">No records match this selection.</td></tr>'
        wide = ' wide-table' if len(columns) > 4 else ''
        return f'<div class="table-scroll{wide}" tabindex="0" role="region" aria-label="Data table"><table role="table"><thead><tr>{heads}</tr></thead><tbody>{"".join(body) or empty}</tbody></table></div>'

    def queried(block, index):
        fields = [c.field for c in block.columns] if isinstance(block, Table) else [block.x, *(s.field for s in block.series)]
        query = resolve_query(document, evidence, block.query, fields)
        rows = execute(evidence, query)
        available = set(query.columns or query.group_by + [m.name for m in query.measures] + [c.name for c in query.calculations])
        if set(fields) - available:
            raise DataError(f"{block.title}: fields are not selected by its query: {sorted(set(fields) - available)}")
        if isinstance(block, Chart) and block.kind == 'line' and any(order.field == block.x for order in query.order_by):
            labels = [row[block.x] for row in rows if row[block.x] is not None]
            if any(isinstance(label, str) and not re.fullmatch(r"\d{4}-\d{2}(?:-\d{2}(?:[T ].*)?)?", label) for label in labels):
                raise DataError('Line charts must not lexicographically sort human-readable labels. Preserve source order, or order by an explicit numeric/ISO date key.')
        # Extra query fields may drive sorting; only displayed columns leave the host.
        public_rows = [{key: row[key] for key in dict.fromkeys(fields)} for row in rows]
        bindings[index] = public_rows
        title = f'<figcaption><h3>{text(block.title)}</h3>' + (f'<p>{text(block.caption)}</p>' if block.caption else '') + '</figcaption>'
        if isinstance(block, Table):
            content = table(public_rows, block.columns)
        else:
            from .document import Column
            if len(rows) > 5000:
                raise DataError(f"{block.title}: aggregate more than 5000 chart rows; the full data table can remain")
            columns = [Column(field=block.x, label=block.x_label or block.x),
                       *[Column(field=s.field, label=s.label, format=s.format) for s in block.series]]
            numeric_fields = [(s.field,100 if s.format.percent == 'ratio' else 1) for s in block.series] + ([(block.x,1)] if block.kind == 'scatter' else [])
            for row in public_rows:
                for field, scale in numeric_fields:
                    if row[field] is None and not (block.kind == 'scatter' and field == block.x):
                        continue
                    exact = calculate('multiply',number(row[field]),scale)
                    approximate = float(exact)
                    if not math.isfinite(approximate) or (exact and approximate == 0):
                        raise DataError(f'{block.title}: rescale {field} into representable chart units; the browser would overflow or turn a nonzero value into zero')
            values=[float(calculate('multiply',row[series.field],100 if series.format.percent == 'ratio' else 1)) for row in public_rows for series in block.series if row[series.field] is not None]
            low,high=min([0,*values]),max([0,*values])
            rough=((high-low) or 1)/4
            power=10.0**math.floor(math.log10(rough)) if rough>0 and math.isfinite(rough) else 0
            if not power:
                raise DataError(f'{block.title}: rescale the chart units; the browser cannot represent the axis extent')
            tick=next(n*power for n in (1,2,5,10) if n*power>=rough)
            limits=(math.floor(low/tick)*tick, math.ceil(high/tick)*tick)
            if not all(math.isfinite(value) for value in (*limits,limits[1]-limits[0])):
                raise DataError(f'{block.title}: rescale the chart units; the browser cannot represent the axis ticks')
            if block.kind == 'scatter' and public_rows:
                xs=[float(number(row[block.x])) for row in public_rows]
                if not math.isfinite(max(xs)-min(xs)):
                    raise DataError(f'{block.title}: rescale {block.x}; the browser cannot represent the scatter extent')
            chart = {"id": index, "kind": block.kind, "x": block.x, "xLabel": block.x_label,
                     "yLabel": block.y_label, "display": [[s.format.display(row[s.field]) for s in block.series] for row in public_rows], "series": [s.model_dump() for s in block.series], "rows": public_rows}
            content = f'<div class="chart" data-chart="{index}" aria-label="{text(block.title)}"></div>'
            content += f'<script type="application/json" class="chart-spec">{_script_json(chart)}</script>'
            content += '<details open class="chart-data"><summary>View data</summary>' + table(public_rows, columns) + '</details>'
        return '<figure>' + title + content + source(query) + '</figure>'

    for query in document.bindings.values():
        source(query)
    sections = []
    for section in document.sections:
        blocks = []
        for block in section.blocks:
            index = f'_view{len(bindings)}'
            if isinstance(block, Prose):
                blocks.append('<div class="prose">' + ''.join(f'<p>{text(p)}</p>' for p in block.text.split('\n\n')) + '</div>')
            elif isinstance(block, Callout):
                blocks.append(f'<aside class="callout {block.tone}"><h3>{text(block.title)}</h3><p>{text(block.text)}</p></aside>')
            elif isinstance(block, Metric):
                if block.fact not in values:
                    raise DataError(f"Unknown metric fact: {block.fact}")
                published_facts[block.fact] = values[block.fact]
                value = document.facts[block.fact].format.display(values[block.fact])
                blocks.append(f'<div class="metric"><h3>{text(block.label)}</h3><p class="metric-value">{escape(value)}</p><p class="metric-detail">{text(block.detail)}</p></div>')
            elif isinstance(block, (Table, Chart)):
                blocks.append(queried(block, index))
            elif isinstance(block, Html):
                blocks.append(f'<div class="custom">{block.html}</div>')
        heading = (f'<p class="eyebrow">{text(section.eyebrow)}</p>' if section.eyebrow else '')
        heading += f'<h2>{text(section.title)}</h2>' if section.title else ''
        heading += f'<p class="section-summary">{text(section.summary)}</p>' if section.summary else ''
        notes = f'<details class="speaker-notes"><summary>Speaker notes</summary><p>{text(section.notes)}</p></details>' if section.notes else ''
        sections.append(f'<section id="{section.id}" class="section" aria-label="{text(section.title or section.id)}">'
                        f'<header>{heading}</header><div class="blocks {section.layout}">{"".join(blocks)}</div>{notes}</section>')
    title, subtitle, eyebrow = text(document.title), text(document.subtitle), text(document.eyebrow)
    def fact_sources(name):
        fact = document.facts[name]
        if hasattr(fact, 'query'):
            source(resolve_query(document, evidence, fact.query, [fact.field]))
        else:
            for dependency in fact.inputs:
                fact_sources(dependency)
    for name in published_facts:
        fact_sources(name)
    outline = ''.join(f'<a href="#{s.id}">{text(s.title or s.id)}</a>' for s in document.sections)
    source_list = ''.join('<li>' + source_name(s) + (f' — {escape(s["scope"])}' if s['scope'] else '') + '</li>' for s in sources.values())
    manifest = {"version": 1, "evidence_sha256": evidence.digest(), "sources": sources,
                "facts": published_facts, "bindings": bindings, "view_checks": checks({name: bindings[name] for name in document.bindings}, document.views, covered={block.query for section in document.sections for block in section.blocks if isinstance(block, (Table, Chart)) and isinstance(block.query, str)})}
    theme = document.theme
    accent = "#75d1b2" if theme.mode == "dark" and theme.accent == "#12685c" else theme.accent
    classes = f'{document.kind} {theme.mode} {theme.typography} {theme.density}'
    html = f'''<!doctype html><html lang="{document.language}" class="{classes}"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><script type="application/json" id="_artifact-bindings">{_script_json(manifest)}</script><script>window.ARTIFACT_DATA=JSON.parse(document.getElementById("_artifact-bindings").textContent).bindings;</script><style>:root{{--accent:{accent}}}{_asset('style.css')}</style></head><body>
<a class="skip" href="#_artifact-content">Skip to content</a>
<div class="toolbar"><span class="document-label">{'Presentation' if document.kind == 'slides' else 'Report'}</span><div>
<button type="button" data-action="outline" aria-expanded="false" aria-controls="_artifact-outline">Contents</button>
<button type="button" data-action="present" hidden>Present</button><button type="button" data-action="print">Print / PDF</button></div></div>
<div class="reading-layout"><nav id="_artifact-outline" aria-label="Contents"><p class="eyebrow">Contents</p>{outline}</nav>
<main id="_artifact-content"><header class="document-header" id="_artifact-title"><p class="eyebrow">{eyebrow}</p><h1>{title}</h1><p class="subtitle">{subtitle}</p></header>
{''.join(sections)}<footer id="_artifact-sources"><h2>Sources</h2>{'<ul>' + source_list + '</ul>' if source_list else '<p>No external data sources.</p>'}</footer></main></div>
<script>{_asset('runtime.js') + _asset('slides.js') + _asset('check-views.js')}</script></body></html>'''
    validate_ids(html)
    return Rendered(html=html, manifest=manifest)
