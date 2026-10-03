import asyncio
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

from bs4 import BeautifulSoup
import pytest

from anton.core.artifacts.report_editor import update
from anton.core.artifacts import report_editor
from anton.core.tools.tool_handlers import handle_create_artifact, handle_open_artifact


DOCUMENT = '''<!doctype html><html><head><title>Regional review</title><style>body{color:#172536}</style></head><body><h1>Regional review</h1><section id="summary"><h2>Conclusion</h2><p>Old view.</p></section><table id="detail"><caption>Regional data</caption><tr><td>Old</td><td>2</td></tr></table><table id="audit"><tr><td>Old metric</td><td>2</td></tr></table><p id="source-note">Input snapshot</p><a href="inputs.json">Evidence</a><label for="region">Region</label><select id="region"><option>North</option></select><script id="input" type="application/json">{"version":"old"}</script><script id="calculation" type="application/json">{"value":2}</script><script>window.kept=true;</script></body></html>'''


def report(tmp_path):
    p = tmp_path / 'report.html'; p.write_text(DOCUMENT); return p


def test_refresh_retains_unedited_structure_and_exact_json(tmp_path):
    p = report(tmp_path)
    source = tmp_path / 'inputs.json'; source.write_bytes(b'{"version":"new","units":31}\n')
    protected = source.read_bytes()
    actual = {'version': 'new', 'units': 31}
    analysis = {'value': 31, 'active': True, 'unit': 'items'}
    receipt = update(p,
        sections={'#summary': {'heading': 'Current decision', 'paragraphs': ['North requires 31 items.', 'Supply timing is unknown.']}},
        tables={'#detail': {'columns': ['Region', 'Units'], 'rows': [['North', 31], ['West', 0]]},
                '#audit': {'columns': ['Metric', 'Value'], 'rows': [[k, json.dumps(v)] for k,v in analysis.items()]}},
        texts={'#source-note': 'Current input snapshot'},
        json_scripts={'#input': actual, '#calculation': analysis})
    saved = BeautifulSoup(p.read_text(), 'html.parser')
    original = BeautifulSoup(DOCUMENT, 'html.parser')
    assert source.read_bytes() == protected
    for selector in ['title', 'style', 'h1', 'a', 'label', 'select', 'script:not([type])']:
        assert str(saved.select_one(selector)) == str(original.select_one(selector))
    assert saved.select_one('#detail caption').text == 'Regional data'
    assert json.loads(saved.select_one('#input').string) == actual
    assert json.loads(saved.select_one('#calculation').string) == analysis
    assert {tr.find_all('td')[0].text: json.loads(tr.find_all('td')[1].text) for tr in saved.select('#audit tbody tr')} == analysis
    assert receipt['saved_values']['#detail'] == [['North', '31'], ['West', '0']]
    assert receipt['before_sha256'] != receipt['after_sha256']
    assert receipt['preservation']['verified'] is True
    assert receipt['preservation']['before_sha256'] == receipt['preservation']['after_sha256']


def test_preservation_receipt_keeps_unrelated_semantics_and_target_attributes(tmp_path):
    p = report(tmp_path)
    p.write_text(DOCUMENT.replace('<p id="source-note">', '<!-- retained note -->\n<p title="A &amp; B" id="source-note">'))
    before = BeautifulSoup(p.read_text(), 'html.parser')
    receipt = update(p, texts={'#source-note': 'A < B & C'},
                     tables={'#detail': {'columns': ['Place', 'Quantity'], 'rows': [['East', 17]]}},
                     json_scripts={'#input': {'note': 'new </script> value'}})
    after = BeautifulSoup(p.read_text(), 'html.parser')
    for selector in ['#audit', '#summary', '#region', 'label', 'style', 'script:not([type])', '#calculation', 'a']:
        assert str(after.select_one(selector)) == str(before.select_one(selector))
    assert after.select_one('#source-note')['title'] == 'A & B'
    assert after.select_one('#source-note').get_text() == 'A < B & C'
    assert after.select_one('#detail caption').get_text() == 'Regional data'
    assert '<!-- retained note -->' in p.read_text()
    assert receipt['preservation']['before_sha256'] == receipt['preservation']['after_sha256']
    assert receipt['preservation']['scope'] == 'DOM outside explicitly selected contents'


def test_unselected_serialization_change_is_rejected_before_write(tmp_path, monkeypatch):
    p = report(tmp_path)
    original = p.read_bytes()
    parser = report_editor.BeautifulSoup
    calls = 0

    def damaged_serialization(*args, **kwargs):
        nonlocal calls
        calls += 1
        parsed = parser(*args, **kwargs)
        if calls == 3:  # The reparsed edited document, before any file write.
            parsed.select_one('a')['href'] = 'unintended-evidence.json'
        return parsed

    monkeypatch.setattr(report_editor, 'BeautifulSoup', damaged_serialization)
    with pytest.raises(ValueError, match='Unselected report structure changed'):
        update(p, texts={'#source-note': 'Current snapshot'})
    assert p.read_bytes() == original


@pytest.mark.parametrize('kwargs', [
    {'texts': {'#missing': 'bad'}},
    {'texts': {'p': 'ambiguous'}},
    {'texts': {'#summary': 'deletes structure'}},
    {'texts': {'style': 'bad'}},
    {'sections': {'body': {'heading': 'bad', 'paragraphs': ['bad']}}},
    {'tables': {'#detail': {'columns': ['one'], 'rows': [['a', 'b']]}}},
    {'json_scripts': {'#source-note': {'value': 4}}},
    {'json_scripts': {'#input': {'value': float('nan')}}},
    {'texts': {'#summary p': 'x'}, 'sections': {'#summary': {'heading': 'Overlap', 'paragraphs': ['x']}}},
    {},
])
def test_invalid_edit_cannot_partially_write(tmp_path, kwargs):
    p = report(tmp_path); before = p.read_bytes()
    with pytest.raises((ValueError, TypeError)):
        update(p, **kwargs)
    assert p.read_bytes() == before


def test_injected_markup_stays_data(tmp_path):
    p = report(tmp_path)
    hostile = '</script><script>alert(1)</script>'
    update(p, texts={'#source-note': '<img src=x onerror=alert(1)>'}, json_scripts={'#input': {'note': hostile}})
    saved = BeautifulSoup(p.read_text(), 'html.parser')
    assert not saved.find('img')
    assert len(saved.find_all('script')) == 3
    assert json.loads(saved.select_one('#input').string)['note'] == hostile
    assert saved.select_one('#source-note').text == '<img src=x onerror=alert(1)>'


def test_symlink_report_rejected_without_touching_target(tmp_path):
    target = report(tmp_path); before = target.read_bytes()
    link = tmp_path / 'linked.html'; link.symlink_to(target)
    with pytest.raises(ValueError):
        update(link, texts={'#source-note': 'new'})
    assert target.read_bytes() == before


def test_native_open_installs_usable_helper_and_keeps_identity(tmp_path):
    session = SimpleNamespace(_workspace=SimpleNamespace(artifacts_dir=tmp_path/'.anton/artifacts'), _artifacts_touched=set(), _session_id='')
    created = asyncio.run(handle_create_artifact(session, {'name': 'Regional review', 'description': 'Control', 'type': 'html-app', 'primary': 'report.html'}))
    details = json.loads(created.content)['details']; folder = Path(details['path'])
    p = folder/'report.html'; p.write_text(DOCUMENT)
    opened = asyncio.run(handle_open_artifact(session, {'slug': details['slug']}))
    opened_details = json.loads(opened.content)
    assert opened.ok and opened_details['id'] and opened_details['slug'] == details['slug']
    assert details['slug'] in session._artifacts_touched
    helper = folder/'_report_editor.py'; assert helper.is_file()
    runpy.run_path(str(helper))['update'](p, texts={'#source-note': 'Current snapshot'})
    assert 'Current snapshot' in p.read_text()
    assert json.loads(asyncio.run(handle_open_artifact(session, {'slug': details['slug']})).content)['id'] == opened_details['id']
    helper.write_text('# caller-owned helper\n')
    with pytest.raises(ValueError):
        asyncio.run(handle_open_artifact(session, {'slug': details['slug']}))
    assert helper.read_text() == '# caller-owned helper\n'
