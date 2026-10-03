"""Generic, selector-based edits to an existing HTML report.

This helper supplies DOM operations, never business calculations or answers.
The caller must read current sources, choose targets, and verify the result.
"""
from pathlib import Path
import hashlib
import json
import os
import tempfile

from bs4 import BeautifulSoup


def _outside_targets(soup, selectors):
    """Canonical DOM with only explicitly selected contents masked."""
    masked = BeautifulSoup(str(soup), 'html.parser')
    for selector in selectors:
        matches = masked.select(selector)
        if len(matches) != 1:
            raise ValueError('An edit changed selector identity')
        matches[0].clear()
        matches[0].string = '[selected contents]'
    return str(masked).encode('utf-8')


def install(folder, artifact_type):
    from anton.core.artifacts.internal_files import REPORT_EDITOR_FILENAME

    if artifact_type != 'html-app':
        return
    destination = Path(folder) / REPORT_EDITOR_FILENAME
    bundled = Path(__file__).read_bytes()
    if not destination.exists():
        destination.write_bytes(bundled)
    elif destination.is_symlink() or destination.read_bytes() != bundled:
        raise ValueError('Existing report editor differs from bundled helper; no overwrite performed')


def update(path, *, texts=None, sections=None, tables=None, json_scripts=None):
    """Replace explicitly selected text, section contents, tables and JSON.

    Every selector must match exactly one node. Targets may not overlap.
    texts: {selector: plain_text}
    sections: {selector: {heading: plain_text, paragraphs: [plain_text, ...]}}
    tables: {selector: {columns: [plain_text, ...], rows: [[cell, ...], ...]}}
    json_scripts: {selector: JSON_serializable_object}
    Cells are converted with str; use json.dumps for exact JSON audit cells.
    Only this report is written; root analysis and sources remain caller-owned.
    Returns saved semantic values, changed selectors and a checked preservation
    signature for the DOM outside selected contents, not a task grade.
    Serialization may normalize whitespace or attributes; this is not a claim
    that unselected HTML bytes are identical.
    """
    path = Path(path).absolute()
    if path.suffix.lower() not in {'.html', '.htm'} or not path.is_file():
        raise ValueError('An existing HTML report is required')
    if any(p.is_symlink() for p in [path, *path.parents]):
        raise ValueError('Symlink report paths are not supported')
    original = path.read_bytes()
    soup = BeautifulSoup(original.decode('utf-8'), 'html.parser')
    operations = []
    for kind, mapping in [('text', texts), ('section', sections),
                          ('table', tables), ('json', json_scripts)]:
        if mapping is not None and not isinstance(mapping, dict):
            raise ValueError('Operations must be selector mappings')
        for selector, value in (mapping or {}).items():
            matches = soup.select(selector)
            if len(matches) != 1:
                raise ValueError(f'Selector must match exactly once: {selector}')
            node = matches[0]
            for _, _, prior, _ in operations:
                if node is prior or any(p is prior for p in node.parents) or any(p is node for p in prior.parents):
                    raise ValueError('Edit targets overlap')
            if kind == 'text':
                if not isinstance(value, str) or node.name in {'script', 'style', 'table'} or node.find():
                    raise ValueError('Text edits require a plain-text leaf element')
            elif kind == 'section':
                if node.name not in {'section', 'div', 'article'} or set(value) != {'heading', 'paragraphs'}:
                    raise ValueError('Sections require a heading and paragraphs')
                if not isinstance(value['heading'], str) or not isinstance(value['paragraphs'], list) or not all(isinstance(x, str) for x in value['paragraphs']):
                    raise ValueError('Section values must be plain text')
                # Replacing a containing section must never quietly delete
                # evidence, a control or an executable component.
                if node.find(['table', 'script', 'a', 'input', 'select', 'button', 'form']):
                    raise ValueError('Section contains protected structure')
            elif kind == 'table':
                if node.name != 'table' or set(value) != {'columns', 'rows'}:
                    raise ValueError('Tables require columns and rows')
                if node.find(['script', 'a', 'input', 'select', 'button']):
                    raise ValueError('Table contains protected structure')
                columns, rows = value['columns'], value['rows']
                if not isinstance(columns, list) or not columns or not all(isinstance(x, str) for x in columns):
                    raise ValueError('Columns must be a nonempty string list')
                if not isinstance(rows, list) or any(not isinstance(r, list) or len(r) != len(columns) for r in rows):
                    raise ValueError('Every row must match the column count')
            elif node.name != 'script' or node.get('type') != 'application/json':
                raise ValueError('JSON targets must be application/json scripts')
            if kind == 'json':
                json.dumps(value, ensure_ascii=False, allow_nan=False)
            operations.append((kind, selector, node, value))
    if not operations:
        raise ValueError('At least one explicit edit is required')
    selectors = [selector for _, selector, _, _ in operations]
    outside_before = _outside_targets(soup, selectors)
    for kind, selector, node, value in operations:
        caption = node.find('caption', recursive=False) if kind == 'table' else None
        caption = caption.extract() if caption else None
        node.clear()
        if kind == 'text':
            node.string = value
        elif kind == 'json':
            node.string = json.dumps(value, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
        elif kind == 'section':
            heading = soup.new_tag('h2'); heading.string = value['heading']; node.append(heading)
            for paragraph in value['paragraphs']:
                p = soup.new_tag('p'); p.string = paragraph; node.append(p)
        else:
            if caption:
                node.append(caption)
            head = soup.new_tag('thead'); tr = soup.new_tag('tr')
            for label in value['columns']:
                th = soup.new_tag('th', scope='col'); th.string = label; tr.append(th)
            head.append(tr); node.append(head)
            body = soup.new_tag('tbody')
            for row in value['rows']:
                tr = soup.new_tag('tr')
                for cell in row:
                    td = soup.new_tag('td'); td.string = str(cell); tr.append(td)
                body.append(tr)
            node.append(body)
    rendered = str(soup)
    checked = BeautifulSoup(rendered, 'html.parser')
    outside_after = _outside_targets(checked, selectors)
    if outside_after != outside_before:
        raise ValueError('Unselected report structure changed; no write performed')
    receipts = {}
    for kind, selector, _, value in operations:
        matches = checked.select(selector)
        if len(matches) != 1:
            raise ValueError('An edit changed selector identity')
        node = matches[0]
        if kind == 'json':
            observed = json.loads(node.string)
            assert observed == value
        elif kind == 'table':
            observed = [[td.get_text() for td in tr.find_all('td')] for tr in node.select('tbody tr')]
            assert observed == [[str(c) for c in row] for row in value['rows']]
            assert [th.get_text() for th in node.select('thead th')] == value['columns']
        elif kind == 'section':
            observed = {'heading': node.h2.get_text(), 'paragraphs': [p.get_text() for p in node.find_all('p', recursive=False)]}
            assert observed == value
        else:
            observed = node.get_text(); assert observed == value
        receipts[selector] = observed
    if path.read_bytes() != original:
        raise ValueError('Report changed during edit; no write performed')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.report-edit-', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(rendered.encode('utf-8'))
        temporary.chmod(path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()
    if path.read_text() != rendered:
        raise ValueError('Saved report differs from verified rendered content')
    return {'changed_selectors': selectors,
            'saved_values': receipts,
            'preservation': {'scope': 'DOM outside explicitly selected contents',
                             'verified': True,
                             'before_sha256': hashlib.sha256(outside_before).hexdigest(),
                             'after_sha256': hashlib.sha256(outside_after).hexdigest()},
            'before_sha256': hashlib.sha256(original).hexdigest(),
            'after_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
