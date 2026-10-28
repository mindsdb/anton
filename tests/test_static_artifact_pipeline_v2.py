from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from anton.core.artifacts.static.data import Evidence
from anton.core.llm.provider import LLMResponse, StreamComplete, ToolCall
from anton.core.tools.generate_artifact.state import GenState, BrowserCheck
from anton.core.tools.generate_artifact.static_v2.compose import compose, compile_output, Custom
from anton.core.tools.generate_artifact.static_v2 import collect as collect_function


def state(tmp_path, payload):
    async def stream(**kwargs):
        yield StreamComplete(response=LLMResponse(content='', tool_calls=[ToolCall('c','compose_static_artifact',payload)]))
    session=SimpleNamespace(_scratchpads=SimpleNamespace(cell_sandbox=None),_llm=SimpleNamespace(code_stream=stream),history=[])
    return GenState(session=session,artifact_type='html-app',artifact_path=tmp_path,slug='test',user_request='Make a report',agent_understanding='Use data',act_first=True)


VALID={'document':{'title':'A report','sections':[{'id':'body','blocks':[{'type':'prose','text':'A result.'}]}]}}


async def test_publication_replaces_page_only_after_candidate_browser_check(tmp_path,monkeypatch):
    old=tmp_path/'index.html';old.write_text('OLD PAGE')
    def check(path, **kwargs):
        assert old.read_text()=='OLD PAGE'
        assert path.parent.name=='.static'
        assert 'A result.' in path.read_text()
        return BrowserCheck(status='passed')
    monkeypatch.setattr('anton.core.tools.generate_artifact.verifiers.verify_frontend_live',check)
    result=await compose(state(tmp_path,VALID),Evidence())
    assert result['status']=='generated' and result['verified_in_browser']
    assert 'A result.' in old.read_text()


async def test_failed_composition_preserves_prior_page_and_recipe(tmp_path,monkeypatch):
    old=tmp_path/'index.html';old.write_text('OLD PAGE')
    recipe=tmp_path/'.static/composition.json';recipe.parent.mkdir();recipe.write_text('OLD RECIPE')
    result=await compose(state(tmp_path,{'document':{'title':'Bad','sections':[]}}),Evidence())
    assert isinstance(result,str) and 'failed validation' in result
    assert old.read_text()=='OLD PAGE' and recipe.read_text()=='OLD RECIPE'


async def test_cancel_during_browser_check_prevents_late_publish(tmp_path,monkeypatch):
    old=tmp_path/'index.html';old.write_text('OLD PAGE');st=state(tmp_path,VALID)
    def check(*args,**kwargs):
        st.abandoned.set()
        return BrowserCheck(status='passed')
    monkeypatch.setattr('anton.core.tools.generate_artifact.verifiers.verify_frontend_live',check)
    assert 'cancelled' in await compose(st,Evidence())
    assert old.read_text()=='OLD PAGE'


async def test_unavailable_browser_is_reported_without_claiming_pass(tmp_path,monkeypatch):
    monkeypatch.setattr('anton.core.tools.generate_artifact.verifiers.verify_frontend_live',lambda *a,**k:BrowserCheck(status='unverified',reason='Browser unavailable'))
    result=await compose(state(tmp_path,VALID),Evidence())
    assert result['status']=='generated' and not result['verified_in_browser']
    assert result['checks_skipped']==['Browser unavailable']


def test_custom_html_projection_is_bound_before_first_author_script():
    evidence=Evidence.model_validate({'datasets':{'items':{'source':{'label':'Source'},'rows':[{'name':'a','private':'SECRET'}]}}})
    output=Custom.model_validate({'mode':'html','html':'<!doctype html><html><head><script>window.result=window.ARTIFACT_DATA.items;</script></head><body>Hi</body></html>',
                                 'bindings':{'items':{'dataset':'items','columns':['name']}},'views':[{'selector':'body','binding':'items','minimum':'one'}]})
    html,manifest=compile_output(output,evidence)
    assert html.index('window.ARTIFACT_DATA=')<html.index('window.result=')
    assert 'SECRET' not in html


def test_internal_evidence_is_never_artifact_content(tmp_path):
    from anton.core.artifacts.store import iter_content_files
    (tmp_path/'.static').mkdir();(tmp_path/'.static/evidence.json').write_text('SECRET')
    (tmp_path/'index.html').write_text('Public')
    assert [p.relative_to(tmp_path).as_posix() for p, _ in iter_content_files(tmp_path)]==['index.html']


async def test_collector_rejects_failed_source_and_does_not_accept_stale_evidence(tmp_path,monkeypatch):
    import importlib
    from anton.core.tools.registry import ToolOutcome
    module=importlib.import_module('anton.core.tools.generate_artifact.static_v2.collect')
    finished={'ready':True,'needs_data':True,'reason':'Done'}
    responses=[LLMResponse(content='',tool_calls=[ToolCall('exec','scratchpad',{'action':'exec','name':'data','code':'fetch()'}), ToolCall('finish','finish_static_evidence',finished)]),
               LLMResponse(content='',tool_calls=[ToolCall('stop','finish_static_evidence',{'ready':False,'needs_data':True,'reason':'The designated source is missing'})])]
    async def stream(**kwargs):
        yield StreamComplete(response=responses.pop(0))
    async def failed(*args):
        return ToolOutcome(content='Error: source missing',ok=False)
    monkeypatch.setattr('anton.core.tools.tool_handlers.handle_scratchpad',failed)
    st=state(tmp_path,VALID);st.session._llm.plan_stream=stream;st.session._llm.code_stream=stream
    result=await module.collect(st)
    assert 'designated source is missing' in result
    assert not (tmp_path/'index.html').exists()


async def test_collector_finish_before_source_call_is_not_accepted(tmp_path,monkeypatch):
    import importlib
    module=importlib.import_module('anton.core.tools.generate_artifact.static_v2.collect')
    responses=[LLMResponse(content='',tool_calls=[ToolCall('finish','finish_static_evidence',{'ready':True,'needs_data':False,'reason':'No data'}),ToolCall('later','read_text_file',{'path':'/unused'})]),
               LLMResponse(content='',tool_calls=[ToolCall('stop','finish_static_evidence',{'ready':False,'needs_data':False,'reason':'Source work is not finished'})])]
    async def stream(**kwargs):
        yield StreamComplete(response=responses.pop(0))
    handler=Mock(side_effect=AssertionError('A source after finish must not run'))
    monkeypatch.setattr('anton.core.tools.tool_handlers.handle_read_text_file',handler)
    st=state(tmp_path,VALID);st.session._llm.plan_stream=stream;st.session._llm.code_stream=stream
    assert 'not finished' in await module.collect(st)
    handler.assert_not_called()


async def test_large_evidence_uses_catalog_and_exact_inspection(tmp_path,monkeypatch):
    from decimal import Decimal
    large=Evidence.model_validate({'datasets':{'items':{'source':{'label':'Generated QA'},'rows':[{'n':i,'text':'x'*120} for i in range(1000)]}}})
    calls=[]
    async def stream(**kwargs):
        calls.append(kwargs)
        if len(calls)==1:
            yield StreamComplete(response=LLMResponse(content='',tool_calls=[ToolCall('inspect','inspect_static_data',{'dataset':'items','measures':[{'name':'total','op':'sum','field':'n'}]})]))
        else:
            yield StreamComplete(response=LLMResponse(content='',tool_calls=[ToolCall('compose','compose_static_artifact',VALID)]))
    st=state(tmp_path,VALID);st.session._llm.code_stream=stream
    monkeypatch.setattr('anton.core.tools.generate_artifact.verifiers.verify_frontend_live',lambda *a,**k:BrowserCheck(status='passed'))
    assert (await compose(st,large))['status']=='generated'
    assert 'schema catalog' in calls[0]['messages'][0]['content'].lower()
    assert len(calls[0]['messages'][0]['content'])<10000
    assert any('499500' in str(m) for m in calls[1]['messages'])


async def test_source_procedure_reader_uses_trusted_full_body_even_after_parent_recall():
    from anton.core.tools.generate_artifact.static_v2.source_skills import catalog, recall
    store=SimpleNamespace(list_summaries=lambda:[{'label':'sensor-observations','description':'Read observations using the documented endpoint'}],
        load_for_artifacts=Mock(return_value=SimpleNamespace(declarative_md='Read /v7/observations with the documented time basis.')))
    session=SimpleNamespace(_skill_store=store,_recalled_skills={'sensor-observations'})
    assert 'sensor-observations' in catalog(session)
    result=await recall(session,{'label':'sensor-observations'})
    assert result.ok and '/v7/observations' in result.content
    store.load_for_artifacts.assert_called_once_with('sensor-observations')
    store.load_for_artifacts.return_value=None
    assert not (await recall(session,{'label':'different-source'})).ok


async def test_collector_offers_source_procedures_and_never_conversation_memory(tmp_path):
    import importlib
    module=importlib.import_module('anton.core.tools.generate_artifact.static_v2.collect')
    seen=[]
    async def stream(**kwargs):
        seen.append(kwargs)
        yield StreamComplete(response=LLMResponse(content='',tool_calls=[ToolCall('finish','finish_static_evidence',{'ready':True,'needs_data':False,'reason':'Qualitative artifact'})]))
    st=state(tmp_path,VALID);st.session._llm.plan_stream=stream
    result=await module.collect(st)
    assert isinstance(result,Evidence)
    names={tool['name'] for tool in seen[0]['tools']}
    assert 'recall_skill' in names and 'recall' not in names


async def test_loaded_source_cannot_be_declared_unavailable_without_a_read_attempt(tmp_path,monkeypatch):
    import importlib
    from anton.core.tools.registry import ToolOutcome
    module=importlib.import_module('anton.core.tools.generate_artifact.static_v2.collect')
    stop={'ready':False,'needs_data':True,'reason':'Source failed'}
    responses=[LLMResponse(content='',tool_calls=[ToolCall('procedure','recall_skill',{'label':'sensor'})]),
        LLMResponse(content='',tool_calls=[ToolCall('premature','finish_static_evidence',stop)]),
        LLMResponse(content='',tool_calls=[ToolCall('read','scratchpad',{'action':'exec','name':'source','code':'read_documented_source()'}),ToolCall('stop','finish_static_evidence',stop)])]
    calls=[]
    async def stream(**kwargs):
        calls.append(kwargs)
        yield StreamComplete(response=responses.pop(0))
    async def failed(*args):return ToolOutcome(content='The actual source read failed',ok=False)
    monkeypatch.setattr('anton.core.tools.tool_handlers.handle_scratchpad',failed)
    st=state(tmp_path,VALID);st.session._skill_store=SimpleNamespace(list_summaries=lambda:[{'label':'sensor'}],load_for_artifacts=lambda label:SimpleNamespace(declarative_md='Read the documented endpoint.'))
    st.session._llm.plan_stream=stream;st.session._llm.code_stream=stream
    assert 'Source failed' in await module.collect(st)
    assert len(calls)==3
    assert 'no source read has run yet' in str(calls[-1]['messages'])


async def test_large_text_cells_do_not_escape_the_bounded_catalog(tmp_path,monkeypatch):
    large=Evidence.model_validate({'datasets':{'documents':{'source':{'label':'Long text QA'},'rows':[{'body':'x'*300000,'count':2}]}}})
    seen=[]
    async def stream(**kwargs):
        seen.append(kwargs)
        yield StreamComplete(response=LLMResponse(content='',tool_calls=[ToolCall('compose','compose_static_artifact',VALID)]))
    st=state(tmp_path,VALID);st.session._llm.code_stream=stream
    monkeypatch.setattr('anton.core.tools.generate_artifact.verifiers.verify_frontend_live',lambda *a,**k:BrowserCheck(status='passed'))
    assert (await compose(st,large))['status']=='generated'
    assert len(seen[0]['messages'][0]['content'])<10000
    assert '"truncated": true' in seen[0]['messages'][0]['content']
