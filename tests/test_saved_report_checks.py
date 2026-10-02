"""Generic saved-file checker controls; no scored or holdout task data."""
from copy import deepcopy
import json
import os
from pathlib import Path
import runpy

import pytest
from bs4 import BeautifulSoup
from anton.core.artifacts import report_renderer


def checker():
    return report_renderer.verify_saved


def example():
    rows = [['Item X', 'East', 5], ['Item Y', 'West', 0], ['Item Z', 'East', 2]]
    return {'title': 'Regional coverage', 'theme': 'light', 'summary': ['East has seven units.'],
            'columns': ['Item', 'Region', 'Units'], 'rows': rows, 'source': {'items': rows},
            'analysis': {'metrics': {'units': 7, 'by_region': {'East': 7, 'West': 0}},
                         'sources': ['inputs.json'], 'assumptions': ['All values are units.']},
            'sources': [{'label': 'Inputs', 'href': '../inputs.json'}],
            'filter': {'label': 'Region', 'results_label': 'Matching items',
                       'total_label': 'Filtered units', 'column': 1,
                       'value_column': 2, 'positive_only': True},
            'chart': {'title': 'Coverage', 'labels': ['East', 'West'], 'values': [7, 0],
                      'value_label': 'Units', 'threshold': 4}}


def save(tmp_path, document=None):
    document = deepcopy(document or example())
    source = tmp_path / 'inputs.json'
    raw = (json.dumps(document['source'], ensure_ascii=False) + '\n').encode()
    source.write_bytes(raw)
    analysis = tmp_path / 'analysis.json'
    analysis.write_text(json.dumps(document['analysis']))
    path = tmp_path / 'artifact/report.html'
    report_renderer.render(path, document)
    return path, document, {'source_path': source, 'source_bytes': raw,
                            'analysis_path': analysis, 'project_root': tmp_path}


def mutate(path, action):
    soup = BeautifulSoup(path.read_text(), 'html.parser')
    action(soup)
    path.write_text(str(soup))


def test_correct_filter_chart_and_read_only_receipt(tmp_path):
    path, doc, arguments = save(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    receipt = checker()(path, doc, **arguments)
    assert receipt['saved_file_consistency'] == 'passed'
    assert receipt['business_correctness'] == 'caller must verify'
    assert receipt['browser'] == 'not performed'
    assert receipt['interaction']['execution'] == 'actual saved JavaScript with DOM model'
    assert [(r['total'], r['rows']) for r in receipt['interaction']['observations']] == [
        (7, 2), (7, 2), (0, 0), (7, 2)]
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('change', [
    lambda s: setattr(s.select_one('#metrics-table tbody td:nth-child(2)'), 'string', '9'),
    lambda s: setattr(s.select_one('#detail-table tbody td'), 'string', 'wrong'),
    lambda s: s.select_one('#detail-table thead').decompose(),
    lambda s: s.select_one('#filtered-table th').attrs.pop('scope'),
    lambda s: setattr(s.select_one('#report-state'), 'string', '{}'),
    lambda s: setattr(s.select_one('#source-data'), 'string', '{}'),
    lambda s: setattr(s.select_one('#analysis-data'), 'string', '{}'),
    lambda s: setattr(s.h1, 'string', 'Wrong title'),
    lambda s: s.html.attrs.update({'data-theme': 'dark'}),
    lambda s: setattr(s.select_one('#decision-summary p'), 'string', 'Wrong summary'),
    lambda s: s.select_one('#sources-section a').attrs.update({'href': 'missing.json'}),
    lambda s: s.select_one('#metrics-table').attrs.update({'id': 'detail-table'}),
    lambda s: s.select_one('label').decompose(),
    lambda s: s.select_one('#filter-results').attrs.pop('role'),
    lambda s: s.select_one('#report-filter option:last-child').decompose(),
    lambda s: setattr(s.select_one('#filtered-total'), 'string', 'Filtered units: 99'),
    lambda s: s.select_one('#report-chart').attrs.update({'aria-label': 'Wrong chart'}),
    lambda s: s.select_one('#report-chart rect').attrs.update({'width': '1'}),
    lambda s: s.select_one('#report-chart line').attrs.update({'x1': '1'}),
    lambda s: setattr(s.select_one('#chart-data tbody td:nth-child(2)'), 'string', '9'),
    lambda s: setattr(s.select_one('#chart-section p'), 'string', 'Threshold: 9'),
])
def test_corrupted_saved_elements_fail(tmp_path, change):
    path, doc, arguments = save(tmp_path)
    mutate(path, change)
    with pytest.raises(ValueError):
        checker()(path, doc, **arguments)


@pytest.mark.parametrize('target', ['source', 'analysis', 'sidecar'])
def test_saved_files_must_reconcile(tmp_path, target):
    path, doc, arguments = save(tmp_path)
    selected = {'source': arguments['source_path'], 'analysis': arguments['analysis_path'],
                'sidecar': path.with_name(path.name + '.data.json')}[target]
    selected.write_text('{}')
    with pytest.raises(ValueError):
        checker()(path, doc, **arguments)


def test_actual_saved_script_error_is_not_a_pass(tmp_path):
    path, doc, arguments = save(tmp_path)
    mutate(path, lambda s: setattr(s.find_all('script')[-1], 'string', "throw Error('bad saved script')"))
    with pytest.raises(ValueError, match='JavaScript checks failed'):
        checker()(path, doc, **arguments)


def test_actual_saved_script_wrong_total_is_not_a_pass(tmp_path):
    path, doc, arguments = save(tmp_path)
    path.write_text(path.read_text().replace('sum+row[config.value_column]', 'sum+999'))
    with pytest.raises(ValueError, match='JavaScript checks failed'):
        checker()(path, doc, **arguments)


def test_missing_node_fails_instead_of_claiming_execution(tmp_path, monkeypatch):
    path, doc, arguments = save(tmp_path)
    monkeypatch.setenv('PATH', '')
    with pytest.raises(FileNotFoundError):
        checker()(path, doc, **arguments)


def test_nonfilter_report_does_not_require_node(tmp_path, monkeypatch):
    doc = example()
    del doc['filter']
    path, doc, arguments = save(tmp_path, doc)
    monkeypatch.setenv('PATH', '')
    assert checker()(path, doc, **arguments)['interaction']['status'] == 'not applicable'


def test_correct_json_cells_escaping_and_multiple_scripts(tmp_path):
    doc = example()
    doc['rows'][0][0] = {'name': 'X', 'details': [True, None]}
    doc['rows'][0][1] = '</option><script>bad()</script>'
    path, doc, arguments = save(tmp_path, doc)
    path.write_text(path.read_text().replace('</main>', '<script>const extra=1;</script></main>'))
    assert checker()(path, doc, **arguments)['interaction']['status'] == 'passed'


def test_empty_and_zero_negative_configurations(tmp_path):
    doc = example()
    doc['rows'] = [['Item Y', 'West', 0]]
    path, doc, arguments = save(tmp_path, doc)
    receipt = checker()(path, doc, **arguments)
    assert all(r['rows'] == 0 for r in receipt['interaction']['observations'])
    doc['filter']['positive_only'] = False
    doc['rows'].append(['Item Q', 'North', -3])
    report_renderer.render(path, doc)
    assert checker()(path, doc, **arguments)['interaction']['status'] == 'passed'


def test_links_are_project_scoped_and_fragments_resolve(tmp_path):
    doc = example()
    doc['sources'][0]['href'] = '#source-data'
    path, doc, arguments = save(tmp_path, doc)
    assert checker()(path, doc, **arguments)['saved_file_consistency'] == 'passed'
    doc['sources'][0]['href'] = '#missing'
    report_renderer.render(path, doc)
    with pytest.raises(ValueError, match='Unresolved internal'):
        checker()(path, doc, **arguments)
    doc['sources'][0]['href'] = '../inputs.json#missing'
    report_renderer.render(path, doc)
    with pytest.raises(ValueError, match='fragment'):
        checker()(path, doc, **arguments)
    doc['sources'][0]['href'] = '../../outside.json'
    report_renderer.render(path, doc)
    with pytest.raises(ValueError, match='inside the project'):
        checker()(path, doc, **arguments)


def test_leaf_symlink_is_rejected(tmp_path):
    path, doc, arguments = save(tmp_path)
    actual = path.with_name('actual.html')
    path.rename(actual)
    path.symlink_to(actual)
    with pytest.raises(ValueError, match='symlink'):
        checker()(path, doc, **arguments)


def test_self_consistent_wrong_business_value_not_domain_certification(tmp_path):
    doc = example()
    doc['analysis']['metrics']['units'] = 999
    path, doc, arguments = save(tmp_path, doc)
    receipt = checker()(path, doc, **arguments)
    assert receipt['business_correctness'] == 'caller must verify'
    assert receipt['metrics']['units'] == 999


"""Actual saved filter checks must accept legitimate localized empty messages."""
import pytest


@pytest.mark.parametrize('message', [
    'No items in this region.', 'Aucun résultat pour cette sélection.',
    'Keine passenden Einträge.', '一致する項目はありません。',
])
def test_custom_empty_state_keeps_filter_checks(tmp_path, message):
    path, document, arguments = save(tmp_path)
    path.write_text(path.read_text().replace('No matching results.', message))
    receipt = checker()(path, document, **arguments)
    assert [(o['total'], o['rows']) for o in receipt['interaction']['observations']] == [
        (7, 2), (7, 2), (0, 0), (7, 2)]


@pytest.mark.parametrize('message', ['', '   ', '\\t'])
def test_missing_empty_message_fails(tmp_path, message):
    path, document, arguments = save(tmp_path)
    path.write_text(path.read_text().replace('No matching results.', message))
    with pytest.raises(ValueError, match='JavaScript checks failed'):
        checker()(path, document, **arguments)


def test_visible_empty_state_with_rows_fails(tmp_path):
    path, document, arguments = save(tmp_path)
    path.write_text(path.read_text().replace(
        "document.getElementById('empty-results').hidden=rows.length!==0;",
        "document.getElementById('empty-results').hidden=false;"))
    with pytest.raises(ValueError, match='JavaScript checks failed'):
        checker()(path, document, **arguments)
