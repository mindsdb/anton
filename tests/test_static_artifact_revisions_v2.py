import copy
import json

import pytest

from anton.core.artifacts.static.data import Evidence
from anton.core.llm.provider import LLMResponse, StreamComplete, ToolCall
from anton.core.tools.generate_artifact.state import BrowserCheck
from anton.core.tools.generate_artifact.static_v2 import revisions
from anton.core.tools.generate_artifact.static_v2.compose import compose
from tests.test_static_artifact_pipeline_v2 import state, VALID, REVIEW_SYSTEM, review_response


def revise(base, **kwargs):
    return revisions.apply(base, revisions.Revision.model_validate(kwargs))


def test_targeted_document_edits_preserve_unrelated_content_and_original():
    base = {'mode': 'document', **copy.deepcopy(VALID)}
    updated = revise(base, changes=[
        {'path': ['document', 'title'], 'value': 'Updated title'},
        {'op': 'add', 'path': ['document', 'sections', 1], 'value': {'id': 'extra', 'blocks': [{'type': 'prose', 'text': 'New content'}]}},
        {'op': 'remove', 'path': ['document', 'sections', 0, 'blocks', 0]}])
    assert base == {'mode': 'document', **VALID}
    assert updated['document']['title'] == 'Updated title'
    assert updated['document']['sections'][0] == {'id': 'body', 'blocks': []}
    assert updated['document']['sections'][1]['blocks'][0]['text'] == 'New content'


@pytest.mark.parametrize('change', [
    {'path': ['mode'], 'value': 'html'},
    {'path': ['document', 'missing'], 'value': None},
    {'path': ['document', 'title', 'x'], 'value': None},
    {'path': ['document', 'sections', -1], 'value': None},
    {'path': ['document', 'sections', '0'], 'value': None},
    {'op': 'add', 'path': ['document', 'title'], 'value': 'duplicate'},
    {'path': ['document', 'title']},
])
def test_invalid_revision_cannot_change_the_base(change):
    base = {'mode': 'document', **copy.deepcopy(VALID)}
    with pytest.raises(ValueError):
        revise(base, changes=[{'path': ['document', 'title'], 'value': 'Temporary'}, change])
    assert base == {'mode': 'document', **VALID}


def test_html_edits_require_unique_matches_and_are_atomic():
    base = {'mode': 'html', 'html': '<p>Keep</p><h1>Old</h1><p>Keep</p>', 'bindings': {'rows': {'dataset': 'rows'}}}
    updated = revise(base, text_edits=[{'find': '<h1>Old</h1>', 'replace': '<h1>New</h1>'}])
    assert updated['bindings'] == base['bindings'] and '<h1>New</h1>' in updated['html']
    for find in ['Keep', 'absent']:
        with pytest.raises(ValueError, match='exactly once'):
            revise(base, text_edits=[{'find': 'Old', 'replace': 'New'}, {'find': find, 'replace': 'Bad'}])
        assert '<h1>Old</h1>' in base['html']
    with pytest.raises(ValueError, match='custom HTML'):
        revise({'mode': 'document', **VALID}, text_edits=[{'find': 'report', 'replace': 'brief'}])


async def test_saved_document_revision_still_runs_browser_and_review(tmp_path, monkeypatch):
    checked = []
    def check(path, **kwargs):
        checked.append(path.read_text())
        return BrowserCheck(status='passed')
    monkeypatch.setattr('anton.core.tools.generate_artifact.verifiers.verify_frontend_live', check)
    assert (await compose(state(tmp_path, VALID), Evidence()))['status'] == 'generated'
    before = (tmp_path/'index.html').read_text()
    reviews = []
    async def stream(**kwargs):
        assert (tmp_path/'index.html').read_text() == before
        if kwargs['system'] == REVIEW_SYSTEM:
            reviews.append(kwargs)
            yield review_response(); return
        assert 'revise_static_artifact' in {tool['name'] for tool in kwargs['tools']}
        yield StreamComplete(response=LLMResponse(content='', tool_calls=[ToolCall('revise', 'revise_static_artifact', {'changes': [{'path': ['document', 'title'], 'value': 'A revised report'}]})]))
    st = state(tmp_path, VALID); st.session._llm.code_stream = stream
    assert (await compose(st, Evidence()))['status'] == 'generated'
    after = (tmp_path/'index.html').read_text()
    assert 'A revised report' in after and 'A result.' in after
    assert len(checked) == 2 and len(reviews) == 1
    recipe = json.loads((tmp_path/'.static/composition.json').read_text())
    assert recipe['composition']['document']['title'] == 'A revised report'


async def test_browser_failure_can_be_repaired_without_rewriting_the_candidate(tmp_path, monkeypatch):
    checked = []
    def check(path, **kwargs):
        checked.append(path.read_text())
        return BrowserCheck(status='failed', reason='Please repair') if len(checked) == 1 else BrowserCheck(status='passed')
    monkeypatch.setattr('anton.core.tools.generate_artifact.verifiers.verify_frontend_live', check)
    calls = []
    async def stream(**kwargs):
        if kwargs['system'] == REVIEW_SYSTEM:
            yield review_response(); return
        calls.append(kwargs)
        if len(calls) == 1:
            assert 'revise_static_artifact' not in {tool['name'] for tool in kwargs['tools']}
            call = ToolCall('initial', 'compose_static_artifact', VALID)
        else:
            assert not (tmp_path/'index.html').exists()
            call = ToolCall('repair', 'revise_static_artifact', {'changes': [{'path': ['document', 'title'], 'value': 'Repaired title'}]})
        yield StreamComplete(response=LLMResponse(content='', tool_calls=[call]))
    st = state(tmp_path, VALID); st.session._llm.code_stream = stream
    assert (await compose(st, Evidence()))['status'] == 'generated'
    assert len(calls) == 2 and 'Repaired title' in checked[-1] and 'A result.' in checked[-1]


async def test_custom_html_revision_retains_exact_bindings_and_privacy(tmp_path, monkeypatch):
    evidence = Evidence.model_validate({'datasets': {'items': {'source': {'label': 'User'}, 'rows': [{'id': '007', 'amount': 12, 'private': 'PRIVATE_ONLY'}]}}})
    html = '<html><head><meta name="viewport" content="width=device-width,initial-scale=1"></head><body><h1>Old title</h1><div id="rows"></div><script>document.querySelector("#rows").textContent=JSON.stringify(window.ARTIFACT_DATA.items)</script></body></html>'
    (tmp_path/'index.html').write_text(html)
    monkeypatch.setattr('anton.core.tools.generate_artifact.verifiers.verify_frontend_live', lambda *a, **k: BrowserCheck(status='passed'))
    async def initial(**kwargs):
        if kwargs['system'] == REVIEW_SYSTEM:
            yield review_response(); return
        yield StreamComplete(response=LLMResponse(content='', tool_calls=[ToolCall('initial', 'compose_custom_html', {'html': html, 'bindings': {'items': {'dataset': 'items', 'columns': ['id', 'amount']}}, 'views': [{'selector': '#rows', 'binding': 'items', 'minimum': 'one'}]})]))
    st = state(tmp_path, VALID); st.session._llm.code_stream = initial
    assert (await compose(st, evidence))['status'] == 'generated'
    before = json.loads((tmp_path/'.static/composition.json').read_text())['composition']
    async def edit(**kwargs):
        if kwargs['system'] == REVIEW_SYSTEM:
            yield review_response(); return
        yield StreamComplete(response=LLMResponse(content='', tool_calls=[ToolCall('edit', 'revise_static_artifact', {'text_edits': [{'find': '<h1>Old title</h1>', 'replace': '<h1>New title</h1>'}]})]))
    st.session._llm.code_stream = edit
    assert (await compose(st, evidence))['status'] == 'generated'
    after = json.loads((tmp_path/'.static/composition.json').read_text())['composition']
    assert after['bindings'] == before['bindings'] and after['views'] == before['views']
    public = (tmp_path/'index.html').read_text()
    assert '<h1>New title</h1>' in public and 'PRIVATE_ONLY' not in public
