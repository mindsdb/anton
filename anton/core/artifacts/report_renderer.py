"""Portable offline report components; no business rules or external packages.

The caller computes a JSON-compatible document from its actual data. All
visible content and embedded state use that one document. This module is
bundled into registered artifact folders so local scratchpads can use it.
"""
from copy import deepcopy
import html
import json
import math
from pathlib import Path
from urllib.parse import urlsplit


def _text(value):
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, bool)) or value is None else str(value)


def _escape(value):
    return html.escape(_text(value), quote=True)


def _json(value):
    # An untrusted </script> must not escape an inert JSON script block.
    return json.dumps(value, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate(document):
    document = deepcopy(document)
    if not isinstance(document, dict):
        raise ValueError('Report document must be an object')
    for field in ('title', 'columns', 'rows', 'source', 'analysis', 'summary', 'sources'):
        if field not in document:
            raise ValueError('Missing document field: '+field)
    if not isinstance(document['title'], str) or not document['title'].strip():
        raise ValueError('A report title is required')
    if document.get('theme', 'light') not in ('light', 'dark'):
        raise ValueError('Theme must be light or dark')
    columns, rows = document['columns'], document['rows']
    if not isinstance(columns, list) or not columns or not all(isinstance(c, str) for c in columns):
        raise ValueError('columns must be a nonempty list of labels')
    if not isinstance(rows, list) or not all(isinstance(r, list) and len(r) == len(columns) for r in rows):
        raise ValueError('Every detail row must match the columns')
    analysis = document['analysis']
    if not isinstance(analysis, dict) or not isinstance(analysis.get('metrics'), dict):
        raise ValueError('analysis.metrics must be an object')
    for values in (document['summary'], analysis.get('assumptions'), analysis.get('sources')):
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise ValueError('Summary, assumptions and analysis sources must be text lists')
    if not isinstance(document['sources'], list):
        raise ValueError('sources must be a list of evidence links')
    for link in document['sources']:
        if not isinstance(link, dict) or not all(isinstance(link.get(k), str) for k in ('label', 'href')):
            raise ValueError('Each evidence link needs label and href')
        href = link['href']
        if any(ord(c) < 32 for c in href) or urlsplit(href).scheme.lower() not in ('', 'http', 'https', 'file'):
            raise ValueError('Unsupported evidence-link scheme')
    chart = document.get('chart')
    if chart is not None:
        if not isinstance(chart, dict) or not isinstance(chart.get('title'), str):
            raise ValueError('Chart needs a title')
        labels, values = chart.get('labels'), chart.get('values')
        if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels) or not labels:
            raise ValueError('Chart labels must be a nonempty text list')
        if not isinstance(values, list) or len(labels) != len(values) or not all(_number(v) and v >= 0 for v in values):
            raise ValueError('Bar values must be finite nonnegative numbers matching labels')
        if chart.get('threshold') is not None and not (_number(chart['threshold']) and chart['threshold'] >= 0):
            raise ValueError('Threshold must be a finite nonnegative number')
    control = document.get('filter')
    if control is not None:
        if not isinstance(control, dict):
            raise ValueError('filter must be an object')
        for field in ('label', 'results_label', 'total_label'):
            if not isinstance(control.get(field), str) or not control[field].strip():
                raise ValueError('Filter requires '+field)
        for field in ('column', 'value_column'):
            if type(control.get(field)) is not int or not 0 <= control[field] < len(columns):
                raise ValueError('Filter column is out of range')
        if not all(_number(r[control['value_column']]) for r in rows):
            raise ValueError('Filtered totals need finite numeric cells')
        if 'positive_only' in control and type(control['positive_only']) is not bool:
            raise ValueError('positive_only must be boolean')
    _json(document)
    return document


def _table(columns, rows, identity):
    headings=''.join('<th scope="col">'+_escape(c)+'</th>' for c in columns)
    body=''.join('<tr>'+''.join('<td>'+_escape(v)+'</td>' for v in r)+'</tr>' for r in rows)
    return f'<div class="table-wrap"><table id="{identity}"><thead><tr>{headings}</tr></thead><tbody>{body}</tbody></table></div>'


def _chart(chart):
    if chart is None:
        return ''
    values, labels = chart['values'], chart['labels']
    maximum = max([1, *values, chart.get('threshold') or 0])
    height = 80 + 48*len(labels)
    items=[]
    for i, (label, value) in enumerate(zip(labels, values)):
        y=30+i*48
        items += [f'<text x="8" y="{y+21}" fill="currentColor">{_escape(label)}</text>',
                  f'<rect x="155" y="{y}" width="{480*value/maximum:.4f}" height="30" fill="var(--accent)"/>',
                  f'<text x="{165+480*value/maximum:.4f}" y="{y+21}" fill="currentColor">{_escape(value)}</text>']
    if chart.get('threshold') is not None:
        x=155+480*chart['threshold']/maximum
        items.append(f'<line x1="{x}" x2="{x}" y1="20" y2="{height-38}" stroke="var(--warning)" stroke-width="2" stroke-dasharray="5 4"/>')
    title = _escape(chart['title'])
    threshold = ('<p>Threshold: '+_escape(chart['threshold'])+'</p>') if chart.get('threshold') is not None else ''
    return (f'<section id="chart-section"><h2>{title}</h2><svg id="report-chart" role="img" aria-label="{title}" '
            f'viewBox="0 0 740 {height}"><title>{title}</title>'+''.join(items)+'</svg>'
            + threshold + _table(['Category', chart.get('value_label', 'Value')], list(map(list, zip(labels, values))), 'chart-data')+'</section>')


STYLE = '''
:root{--bg:#f4f6fa;--panel:#fff;--ink:#162336;--muted:#526478;--line:#d9e1eb;--accent:#2763bc;--warning:#ad4b12}
html[data-theme="dark"]{--bg:#101820;--panel:#182534;--ink:#f0f5fc;--muted:#b5c7dc;--line:#3a4c60;--accent:#79abff;--warning:#ffb77b}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,sans-serif}
main{max-width:1120px;margin:auto;padding:32px 24px 64px}h1{font-size:32px;line-height:1.2;letter-spacing:-.03em}h2{font-size:20px;margin:0 0 16px}
h1,h2,p,li,a{overflow-wrap:anywhere}
header{padding:12px 0 22px}section{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:24px;margin:18px 0}
.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse}th,td{padding:10px 14px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{color:var(--muted);font-size:14px}thead th{background:var(--bg)}
label{font-weight:600}select{font:inherit;margin:0 0 18px 12px;padding:8px 16px;background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:6px}
a{color:var(--accent)}svg{width:100%;height:auto;max-height:640px}svg text{font:15px system-ui,sans-serif}.eyebrow{color:var(--muted);font-size:13px;letter-spacing:.1em;text-transform:uppercase}
@media(max-width:600px){main{padding:18px 12px}section{padding:16px}h1{font-size:26px}}
@media print{body{background:white}main{max-width:none;padding:0}section{break-inside:avoid}select{display:none}}
'''

FILTER_SCRIPT = '''
const state=JSON.parse(document.getElementById('report-state').textContent);
const config=state.filter;
if(config){
 const select=document.getElementById('report-filter');
 const values=[...new Set(state.rows.map(row=>String(row[config.column])))].sort();
 if(select.children.length<=1){values.forEach((value,index)=>{const o=document.createElement('option');o.value=String(index);o.textContent=value;select.append(o)});}
 function refresh(){
  const selected=select.value;
  const rows=state.rows.filter(row=>(selected==='all'||String(row[config.column])===values[Number(selected)])&&(!config.positive_only||row[config.value_column]>0));
  const body=document.getElementById('filtered-table').tBodies[0];body.replaceChildren();
  rows.forEach(row=>{const tr=document.createElement('tr');row.forEach(value=>{const td=document.createElement('td');td.textContent=typeof value==='object'?JSON.stringify(value):String(value);tr.append(td)});body.append(tr)});
  document.getElementById('filtered-total').textContent=config.total_label+': '+rows.reduce((sum,row)=>sum+row[config.value_column],0);
  document.getElementById('empty-results').textContent=rows.length===0?'No matching results.':'';
  document.getElementById('empty-results').hidden=rows.length!==0;
 }
 select.addEventListener('change',refresh);refresh();
}
'''



def _filter_text(value):
    """Match JavaScript String for JSON-compatible category cells."""
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, dict):
        return '[object Object]'
    if isinstance(value, list):
        return ','.join('' if item is None else _filter_text(item) for item in value)
    if isinstance(value, (int, float)):
        from decimal import Decimal
        number=float(value)
        if number==0:
            return '0'
        text=repr(number)
        if 1e-6 <= abs(number) < 1e21:
            return format(Decimal(text), 'f').rstrip('0').rstrip('.') if '.' in format(Decimal(text),'f') else format(Decimal(text),'f')
        coefficient, exponent=text.lower().split('e')
        coefficient=coefficient.removesuffix('.0')
        exponent=int(exponent)
        return coefficient+'e'+('+' if exponent>=0 else '')+str(exponent)
    return str(value)


def render(path, document):
    """Write a complete HTML report and its reusable data snapshot.

    Structural checks are not verification of business calculations or claims.
    The caller is responsible for correctness, registration and user permissions.
    """
    path = Path(path)
    document = validate(document)
    if path.suffix.lower() not in ('.html', '.htm'):
        raise ValueError('HTML output filename required')
    state_path=path.with_name(path.name+'.data.json')
    if path.is_symlink() or state_path.is_symlink():
        raise ValueError('Refuse to replace a symlink')
    analysis=document['analysis']
    metrics=_table(['Metric', 'Value'], [[k,json.dumps(v,ensure_ascii=False,allow_nan=False)] for k,v in analysis['metrics'].items()], 'metrics-table')
    summary=''.join('<p>'+_escape(s)+'</p>' for s in document['summary'])
    sources=''.join('<li><a href="'+html.escape(s['href'],quote=True)+'">'+_escape(s['label'])+'</a></li>' for s in document['sources'])
    assumptions=''.join('<li>'+_escape(s)+'</li>' for s in analysis['assumptions'])
    detail=_table(document['columns'],document['rows'],'detail-table')
    control=document.get('filter')
    filtered=''
    if control:
        # The default All view must be useful before JavaScript runs.
        values=sorted({_filter_text(row[control['column']]) for row in document['rows']},
                      key=lambda text:text.encode('utf-16-be','surrogatepass'))
        options=''.join(f'<option value="{index}">{_escape(value)}</option>'
                        for index,value in enumerate(values))
        rows=[row for row in document['rows']
              if not control.get('positive_only') or row[control['value_column']]>0]
        total=sum(row[control['value_column']] for row in rows)
        empty='' if rows else 'No matching results.'
        hidden=' hidden' if rows else ''
        filtered=(f'<section id="filter-section"><label for="report-filter">{_escape(control["label"])}</label>'
                  '<select id="report-filter"><option value="all">All</option>'+options+'</select>'
                  f'<section id="filter-results" role="region" aria-label="{_escape(control["results_label"])}">'
                  f'<h2>{_escape(control["results_label"])}</h2><p id="filtered-total" aria-live="polite">'
                  +_escape(control['total_label']+': '+_filter_text(total))+'</p>'
                  f'<p id="empty-results"{hidden}>{empty}</p>'
                  +_table(document['columns'],rows,'filtered-table')+'</section></section>')
    text=(f'<!doctype html><html lang="en" data-theme="{document.get("theme","light")}"><head>'
          '<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
          f'<title>{_escape(document["title"])}</title><style>{STYLE}</style></head><body><main id="report-main">'
          f'<header><p class="eyebrow">Planning report</p><h1>{_escape(document["title"])}</h1></header>'
          f'<section id="decision-summary"><h2>Decision summary</h2>{summary}</section>'
          f'<section id="metrics-section"><h2>Metrics</h2>{metrics}</section>'
          +_chart(document.get('chart'))+filtered+
          f'<section id="detail-section"><h2>Detail</h2>{detail}</section>'
          f'<section id="sources-section"><h2>Sources and assumptions</h2><ul>{sources}</ul><ul>{assumptions}</ul></section>'
          f'<script type="application/json" id="report-state">{_json(document)}</script>'
          f'<script type="application/json" id="source-data">{_json(document["source"])}</script>'
          f'<script type="application/json" id="analysis-data">{_json(analysis)}</script>'
          f'<script>{FILTER_SCRIPT}</script></main></body></html>')
    path.parent.mkdir(parents=True,exist_ok=True)
    # Generate both serializations before writing; invalid input leaves the old
    # report intact. HTML is published last and embeds its own complete snapshot.
    state_text=json.dumps(document,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    for target,body in ((state_path,state_text),(path,text)):
        temporary=target.with_name(target.name+'.tmp')
        if temporary.is_symlink():raise ValueError('Refuse a symlink temporary file')
        temporary.write_text(body,encoding='utf-8');temporary.replace(target)
    return {'path':str(path),'snapshot':str(state_path),'structural_validation':'passed',
            'business_correctness':'caller must verify','browser_check':'not performed'}


def install(folder, artifact_type):
    from anton.core.artifacts.internal_files import REPORT_RENDERER_FILENAME

    """Bundle the trusted renderer into an HTML artifact; never overwrite edits."""
    if artifact_type != 'html-app':
        return
    target=Path(folder)/'_report_renderer.py'
    body=Path(__file__).read_bytes()
    if target.exists() or target.is_symlink():
        if target.is_symlink() or target.read_bytes()!=body:
            raise ValueError('An existing renderer file differs from the bundled version')
        return
    target.write_bytes(body)


"""Generic saved-file reconciliation for the documented local report schema.

This checks rendering consistency and executes saved filter JavaScript using a
small DOM model. It does not verify domain calculations or browser appearance.
It writes nothing and does not supply business data or expected task answers.
"""

def verify_saved(path, document, *, source_path, source_bytes, analysis_path=None,
                 project_root=None):
    import hashlib
    import subprocess
    from urllib.parse import unquote, urlsplit
    from bs4 import BeautifulSoup

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    root = Path(project_root or Path(source_path).parent).resolve()

    def local_file(value):
        candidate = Path(value)
        require(not candidate.is_symlink(), 'A checked file must not be a symlink')
        candidate = candidate.resolve()
        require(candidate.is_relative_to(root) and candidate.is_file(),
                'A checked file must exist inside the project')
        return candidate

    path = local_file(path)
    source_path = local_file(source_path)
    require(isinstance(source_bytes, bytes), 'source_bytes must be the original read bytes')
    require(source_path.read_bytes() == source_bytes, 'The original source changed')
    expected = validate(document)
    require(json.loads(source_bytes) == expected['source'], 'Report source differs from actual source')
    html_text = path.read_text(encoding='utf-8')
    soup = BeautifulSoup(html_text, 'html.parser')
    require(soup.html and soup.body and soup.html.get('lang') and
            soup.find('meta', attrs={'name': 'viewport'}), 'Missing HTML accessibility structure')
    require(not soup.select('script[src], link[rel="stylesheet"]'), 'External script/style dependency')
    ids = [item['id'] for item in soup.select('[id]')]
    require(len(ids) == len(set(ids)), 'Duplicate element IDs')

    def element(identity):
        node = soup.find(id=identity)
        require(node is not None, 'Missing saved element: ' + identity)
        return node

    state = element('report-state').get_text()
    require(json.loads(state) == expected, 'Embedded report-state differs')
    require(json.loads(element('source-data').get_text()) == expected['source'], 'Embedded source differs')
    require(json.loads(element('analysis-data').get_text()) == expected['analysis'], 'Embedded analysis differs')
    sidecar = local_file(path.with_name(path.name + '.data.json'))
    require(json.loads(sidecar.read_text()) == expected, 'Saved sidecar differs')
    if analysis_path is not None:
        require(json.loads(local_file(analysis_path).read_text()) == expected['analysis'], 'Root analysis differs')
    require(soup.title and soup.title.get_text() == expected['title'], 'Saved title differs')
    require(soup.h1 and soup.h1.get_text() == expected['title'], 'Saved heading differs')
    require(soup.html.get('data-theme') == expected.get('theme', 'light'), 'Saved theme differs')
    require([p.get_text() for p in element('decision-summary').find_all('p')] == expected['summary'],
            'Saved decision summary differs')

    def table_rows(identity):
        body = element(identity).find('tbody')
        require(body is not None, 'Missing table body: ' + identity)
        return [[cell.get_text() for cell in row.find_all('td', recursive=False)]
                for row in body.find_all('tr', recursive=False)]

    def table_headings(identity):
        head = element(identity).find('thead')
        require(head is not None, 'Missing table headings: ' + identity)
        cells = head.find_all('th')
        require(cells and all(cell.get('scope') == 'col' for cell in cells),
                'Missing accessible column headings: ' + identity)
        return [cell.get_text() for cell in cells]

    audit_rows = table_rows('metrics-table')
    require(all(len(row) == 2 for row in audit_rows), 'Malformed metric audit')
    require(len({row[0] for row in audit_rows}) == len(audit_rows), 'Duplicate metric audit keys')
    require({row[0]: json.loads(row[1]) for row in audit_rows} == expected['analysis']['metrics'],
            'Actual metric audit differs')
    require(table_headings('metrics-table') == ['Metric', 'Value'], 'Saved audit headings differ')
    require(table_headings('detail-table') == expected['columns'], 'Saved detail headings differ')
    require(table_rows('detail-table') == [[_text(cell) for cell in row] for row in expected['rows']],
            'Saved detail rows differ')
    sources = element('sources-section')
    links = sources.find_all('a', href=True)
    require([{'label': link.get_text(), 'href': link['href']} for link in links] == expected['sources'],
            'Saved evidence citations differ')
    lists = sources.find_all('ul', recursive=False)
    require(len(lists) == 2 and [li.get_text() for li in lists[1].find_all('li', recursive=False)]
            == expected['analysis']['assumptions'], 'Saved assumptions differ')
    for link in links:
        href = urlsplit(link['href'])
        require(href.scheme in ('', 'file') and not href.netloc and not href.query,
                'Local checker requires local project evidence')
        if not href.path:
            require(href.fragment and soup.find(id=unquote(href.fragment)), 'Unresolved internal citation')
        else:
            target = Path(unquote(href.path)) if href.scheme == 'file' else path.parent / unquote(href.path)
            target = local_file(target)
            if href.fragment:
                require(BeautifulSoup(target.read_text(encoding='utf-8'), 'html.parser').find(
                    id=unquote(href.fragment)) is not None, 'Unresolved evidence-file fragment')

    chart = expected.get('chart')
    if chart:
        svg = element('report-chart')
        require(svg.name == 'svg' and svg.get('role') == 'img' and
                svg.get('aria-label') == chart['title'] and svg.find('title') and
                svg.find('title').get_text() == chart['title'], 'Saved chart accessible name differs')
        require(table_headings('chart-data') == ['Category', chart.get('value_label', 'Value')],
                'Saved chart headings differ')
        require(table_rows('chart-data') == [[label, _text(value)] for label, value in
                zip(chart['labels'], chart['values'])], 'Saved chart data differs')
        require([node.get_text() for node in svg.find_all('text')] ==
                [text for label, value in zip(chart['labels'], chart['values'])
                 for text in (label, _text(value))], 'Saved chart labels/values differ')
        maximum = max([1, *chart['values'], chart.get('threshold') or 0])
        bars = svg.find_all('rect')
        require(len(bars) == len(chart['values']) and all(
            math.isclose(float(bar.get('width', 'nan')), 480 * value / maximum, abs_tol=0.00006)
            for bar, value in zip(bars, chart['values'])), 'Saved bar proportions differ')
        lines = svg.find_all('line')
        threshold = chart.get('threshold')
        require(len(lines) == (threshold is not None), 'Saved chart threshold presence differs')
        if threshold is not None:
            position = 155 + 480 * threshold / maximum
            require(all(math.isclose(float(lines[0].get(key, 'nan')), position, abs_tol=1e-8)
                        for key in ('x1', 'x2')), 'Saved chart threshold position differs')
            paragraph = element('chart-section').find('p')
            require(paragraph and paragraph.get_text() == 'Threshold: ' + _text(threshold),
                    'Saved chart threshold text differs')

    interaction = {'status': 'not applicable', 'browser': 'not performed'}
    control = expected.get('filter')
    if control:
        select = element('report-filter')
        label = soup.find('label', attrs={'for': 'report-filter'})
        require(label and label.get_text() == control['label'],
                'Saved filter label differs')
        region = element('filter-results')
        require(region.get('role') == 'region' and region.get('aria-label') == control['results_label'],
                'Missing accessible filtered results region')
        require(element('metrics-section') not in region.descendants, 'Global metrics inside filtered results')
        require(all(region.find(id=identity) is not None for identity in
                    ('filtered-table', 'filtered-total', 'empty-results')), 'Filter results outside region')
        options = [{'value': option.get('value', option.get_text()), 'text': option.get_text()}
                   for option in select.find_all('option')]
        require(table_headings('filtered-table') == expected['columns'], 'Saved filtered headings differ')
        initial_rows = table_rows('filtered-table')
        scripts = [tag.get_text() for tag in soup.find_all('script')
                   if tag.get('type') in (None, '', 'text/javascript', 'application/javascript')]
        require(scripts, 'No saved filter JavaScript')
        initial_expected = [[_text(value) for value in row] for row in expected['rows']
                            if not control.get('positive_only') or row[control['value_column']] > 0]
        payload = {'state': state, 'expected': expected, 'options': options, 'rows': initial_rows,
                   'initial_expected': initial_expected,
                   'total': element('filtered-total').get_text(),
                   'hidden': element('empty-results').has_attr('hidden'),
                   'empty': element('empty-results').get_text(), 'scripts': scripts}
        result = subprocess.run(['node', '-e', _SAVED_FILTER_CHECK], input=json.dumps(payload),
                                text=True, capture_output=True, timeout=5)
        require(result.returncode == 0, 'Saved filter JavaScript checks failed: ' + result.stderr[-1200:])
        interaction = json.loads(result.stdout)
    return {'saved_file_consistency': 'passed', 'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
            'detail_rows': len(expected['rows']), 'metrics': expected['analysis']['metrics'],
            'interaction': interaction, 'business_correctness': 'caller must verify',
            'browser': 'not performed', 'path': str(path)}


_SAVED_FILTER_CHECK = r'''
const fs=require('fs'), vm=require('vm'), assert=require('assert');
const p=JSON.parse(fs.readFileSync(0,'utf8')), expected=p.expected, cfg=expected.filter;
class Element {
 constructor(text=''){this.textContent=text;this.children=[];this.value='all';this.hidden=false;this.listeners={}}
 append(...values){this.children.push(...values)}
 replaceChildren(...values){this.children=[...values]}
 addEventListener(name,callback){this.listeners[name]=callback}
}
const makeRow=row=>{const tr=new Element();tr.children=row.map(value=>new Element(value));return tr};
const select=new Element();select.children=p.options.map(o=>Object.assign(new Element(o.text),{value:o.value}));
const body=new Element();body.children=p.rows.map(makeRow);
const els={'report-state':new Element(p.state),'report-filter':select,
 'filtered-table':{tBodies:[body]},'filtered-total':new Element(p.total),
 'empty-results':Object.assign(new Element(p.empty),{hidden:p.hidden})};
const categories=[...new Set(expected.rows.map(row=>String(row[cfg.column])))].sort();
const expectedOptions=[{value:'all',text:'All'},...categories.map((text,i)=>({value:String(i),text}))];
function checkOptions(){assert.deepStrictEqual(select.children.map(o=>({value:o.value,text:o.textContent})),expectedOptions)}
function check(value, initial=false){
 const rows=expected.rows.filter(row=>(value==='all'||String(row[cfg.column])===categories[Number(value)])&&(!cfg.positive_only||row[cfg.value_column]>0));
 const rendered=initial?p.initial_expected:rows.map(row=>row.map(v=>typeof v==='object'?JSON.stringify(v):String(v)));
 assert.deepStrictEqual(body.children.map(tr=>tr.children.map(td=>td.textContent)),rendered);
 const total=rows.reduce((sum,row)=>sum+row[cfg.value_column],0);
 assert.strictEqual(els['filtered-total'].textContent,cfg.total_label+': '+total);
 assert.strictEqual(els['empty-results'].hidden,rows.length!==0);
 if(rows.length===0){
  assert.strictEqual(typeof els['empty-results'].textContent,'string');
  assert.ok(els['empty-results'].textContent.trim().length>0,'Missing empty-results message');
 }else{assert.strictEqual(els['empty-results'].textContent,'');}
 return {selection:value,total,rows:rows.length};
}
checkOptions();check('all',true);
const document={getElementById:id=>els[id],createElement:()=>new Element()};
vm.runInNewContext(p.scripts.join('\n'),{document},{timeout:500,contextCodeGeneration:{strings:false,wasm:false}});
checkOptions();const observations=[check('all')];
assert.strictEqual(typeof select.listeners.change,'function');
for(const value of [...categories.map((_,i)=>String(i)),'all']){
 select.value=value;select.listeners.change();observations.push(check(value));checkOptions();
}
console.log(JSON.stringify({status:'passed',execution:'actual saved JavaScript with DOM model',browser:'not performed',observations}));
'''
