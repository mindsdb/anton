from pathlib import Path
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from anton.core.artifacts.static.data import Evidence
from anton.core.llm.provider import LLMResponse, StreamComplete, ToolCall
from anton.core.tools.generate_artifact.state import GenState, BrowserCheck
from anton.core.tools.generate_artifact.static_v2.compose import compose, compile_output, Custom, Structured
from anton.core.tools.generate_artifact.static_v2 import collect as collect_function
from anton.core.tools.generate_artifact.static_v2.content_review import SYSTEM as REVIEW_SYSTEM


def review_response():
    return StreamComplete(response=LLMResponse(content='',tool_calls=[ToolCall('review','review_static_artifact',{'approved':True,'issues':[]})]))


def state(tmp_path, payload):
    async def stream(**kwargs):
        if kwargs['system']==REVIEW_SYSTEM:
            yield review_response();return
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


async def test_snapshot_write_failure_preserves_the_published_page(tmp_path,monkeypatch):
    old=tmp_path/'index.html';old.write_text('OLD PAGE')
    def fail(*args):
        raise OSError('Private snapshot storage failed')
    monkeypatch.setattr('anton.core.tools.generate_artifact.static_v2.snapshots.save',fail)
    monkeypatch.setattr('anton.core.tools.generate_artifact.verifiers.verify_frontend_live',lambda *a,**k:BrowserCheck(status='passed'))
    assert 'Private snapshot storage failed' in await compose(state(tmp_path,VALID),Evidence())
    assert old.read_text()=='OLD PAGE'


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


async def test_new_and_saved_typed_documents_default_to_the_compiler(tmp_path,monkeypatch):
    seen=[]
    async def stream(**kwargs):
        if kwargs['system']==REVIEW_SYSTEM:
            yield review_response();return
        seen.append(kwargs)
        names={tool['name'] for tool in kwargs['tools']}
        assert 'compose_static_artifact' in names and 'choose_custom_surface' in names
        assert 'compose_custom_html' not in names
        yield StreamComplete(response=LLMResponse(content='',tool_calls=[ToolCall('compose','compose_static_artifact',VALID)]))
    monkeypatch.setattr('anton.core.tools.generate_artifact.verifiers.verify_frontend_live',lambda *a,**k:BrowserCheck(status='passed'))
    for _ in range(2):
        st=state(tmp_path,VALID);st.session._llm.code_stream=stream
        assert (await compose(st,Evidence()))['status']=='generated'
    assert len(seen)==2


async def test_full_custom_page_requires_an_explicit_surface_choice(tmp_path,monkeypatch):
    seen=[]
    html='<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"></head><body><h1>Interactive room</h1></body></html>'
    async def stream(**kwargs):
        if kwargs['system']==REVIEW_SYSTEM:
            yield review_response();return
        seen.append(kwargs)
        assert not (tmp_path/'index.html').exists()
        if len(seen)==2:
            assert 'first call choose_custom_surface' in str(kwargs['messages'])
            call=ToolCall('surface','choose_custom_surface',{'reason':'The requested interactive room uses the whole viewport'})
        else:
            if len(seen)==3:
                assert 'compose_custom_html' in {tool['name'] for tool in kwargs['tools']}
            call=ToolCall('custom','compose_custom_html',{'html':html})
        yield StreamComplete(response=LLMResponse(content='',tool_calls=[call]))
    monkeypatch.setattr('anton.core.tools.generate_artifact.verifiers.verify_frontend_live',lambda *a,**k:BrowserCheck(status='passed'))
    st=state(tmp_path,VALID);st.session._llm.code_stream=stream
    assert (await compose(st,Evidence()))['status']=='generated'
    assert len(seen)==3


async def test_existing_custom_page_can_be_edited_directly(tmp_path,monkeypatch):
    html='<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"></head><body><h1>Existing custom room</h1></body></html>'
    (tmp_path/'index.html').write_text(html)
    async def stream(**kwargs):
        if kwargs['system']==REVIEW_SYSTEM:
            yield review_response();return
        assert 'compose_custom_html' in {tool['name'] for tool in kwargs['tools']}
        yield StreamComplete(response=LLMResponse(content='',tool_calls=[ToolCall('edit','compose_custom_html',{'html':html.replace('Existing','Updated')})]))
    monkeypatch.setattr('anton.core.tools.generate_artifact.verifiers.verify_frontend_live',lambda *a,**k:BrowserCheck(status='passed'))
    st=state(tmp_path,VALID);st.session._llm.code_stream=stream
    assert (await compose(st,Evidence()))['status']=='generated'
    assert 'Updated custom room' in (tmp_path/'index.html').read_text()


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
    finished={'ready':True,'has_records':True,'reason':'Done'}
    responses=[LLMResponse(content='',tool_calls=[ToolCall('exec','scratchpad',{'action':'exec','name':'data','code':'fetch()'}), ToolCall('finish','finish_static_evidence',finished)]),
               LLMResponse(content='',tool_calls=[ToolCall('stop','finish_static_evidence',{'ready':False,'has_records':True,'reason':'The designated source is missing'})])]
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
    responses=[LLMResponse(content='',tool_calls=[ToolCall('finish','finish_static_evidence',{'ready':True,'has_records':False,'reason':'No data'}),ToolCall('later','read_text_file',{'path':'/unused'})]),
               LLMResponse(content='',tool_calls=[ToolCall('stop','finish_static_evidence',{'ready':False,'has_records':False,'reason':'Source work is not finished'})])]
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
        if kwargs['system']==REVIEW_SYSTEM:
            yield review_response();return
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
        yield StreamComplete(response=LLMResponse(content='',tool_calls=[ToolCall('finish','finish_static_evidence',{'ready':True,'has_records':False,'reason':'Qualitative artifact'})]))
    st=state(tmp_path,VALID);st.session._llm.plan_stream=stream
    result=await module.collect(st)
    assert isinstance(result,Evidence)
    names={tool['name'] for tool in seen[0]['tools']}
    assert 'recall_skill' in names and 'recall' not in names


async def test_project_rules_are_returned_from_host_context_without_a_dotfile_read(tmp_path,monkeypatch):
    import importlib
    module=importlib.import_module('anton.core.tools.generate_artifact.static_v2.collect')
    seen=[]
    async def forbidden_read(*args):
        pytest.fail('Already supplied project rules must not trigger a source read')
    async def stream(**kwargs):
        seen.append(kwargs)
        calls=[ToolCall('rules','read_text_file',{'path':'.anton/anton.md'})] if len(seen)==1 else [ToolCall('done','finish_static_evidence',{'ready':True,'has_records':False,'reason':'Qualitative slides'})]
        yield StreamComplete(response=LLMResponse(content='',tool_calls=calls))
    monkeypatch.setattr('anton.core.tools.tool_handlers.handle_read_text_file',forbidden_read)
    st=state(tmp_path,VALID);st.project_rules='Use the supplied house typography.'
    st.session._llm.plan_stream=stream;st.session._llm.code_stream=stream
    assert isinstance(await module.collect(st),Evidence)
    assert 'Use the supplied house typography.' in str(seen[-1]['messages'])


async def test_evidence_passes_bind_a_fresh_pad_without_mutating_shared_tool_schema(tmp_path,monkeypatch):
    import copy
    import importlib
    from anton.core.tools.registry import ToolOutcome
    from anton.core.tools.tool_defs import SCRATCHPAD_TOOL
    module=importlib.import_module('anton.core.tools.generate_artifact.static_v2.collect')
    original=copy.deepcopy(module.protocol.tool_schema(SCRATCHPAD_TOOL));executed=[];offered=[]
    async def execute(session,arguments):
        executed.append(arguments)
        return ToolOutcome(content='Executed',ok=True)
    async def stream(**kwargs):
        offered.append(next(t for t in kwargs['tools'] if t['name']=='scratchpad')['input_schema']['properties']['name']['enum'][0])
        calls=[ToolCall('exec','scratchpad',{'action':'exec','name':'parent_notebook','code':'print("reader")'}),
            ToolCall('finish','finish_static_evidence',{'ready':True,'has_records':False,'reason':'Qualitative result'})]
        yield StreamComplete(response=LLMResponse(content='',tool_calls=calls))
    monkeypatch.setattr('anton.core.tools.tool_handlers.handle_scratchpad',execute)
    st=state(tmp_path,VALID);st.session._agent_scratchpad_names={'parent_notebook'}
    st.session._llm.plan_stream=stream
    for _ in range(2):assert isinstance(await module.collect(st),Evidence)
    assert len(set(offered))==2
    assert [call['name'] for call in executed]==offered
    assert all(call['name']!='parent_notebook' and call['confirm_new_scratchpad'] is True for call in executed)
    assert module.protocol.tool_schema(SCRATCHPAD_TOOL)==original


async def test_read_source_schema_survives_the_composition_handoff(tmp_path):
    import importlib
    from anton.core.tools.generate_artifact.static_v2.prompts import context
    module=importlib.import_module('anton.core.tools.generate_artifact.static_v2.collect')
    body='Source schema: rates are already percentage points; the snapshot period comes from the source metadata.'
    responses=[LLMResponse(content='',tool_calls=[ToolCall('source','recall_skill',{'label':'measurements'})]),
        LLMResponse(content='',tool_calls=[ToolCall('done','finish_static_evidence',{'ready':True,'has_records':False,'reason':'Qualitative schema explanation'})])]
    async def stream(**kwargs):yield StreamComplete(response=responses.pop(0))
    st=state(tmp_path,VALID);st.session._llm.plan_stream=stream;st.session._llm.code_stream=stream
    st.session._skill_store=SimpleNamespace(list_summaries=lambda:[{'label':'measurements'}],load_for_artifacts=lambda label:SimpleNamespace(declarative_md=body))
    assert isinstance(await module.collect(st),Evidence)
    assert st.static_source_contracts=={'measurements':body}
    assert body in context(st)
    html,_=compile_output(Structured(mode='document',**VALID),Evidence())
    assert body not in html


def test_source_contract_context_is_bounded_and_labels_omissions(tmp_path):
    from anton.core.tools.generate_artifact.static_v2.prompts import context
    st=state(tmp_path,VALID);st.static_source_contracts={'first':'A'*60_000,'second':'B'*60_000}
    text=context(st)
    handoff=text.split('Trusted source procedures already read',1)[1].split('Source procedure catalog',1)[0]
    assert len(handoff)<64_000
    assert 'A'*60_000 in handoff and 'B'*60_000 not in handoff
    assert '1 read procedures omitted' in handoff


async def test_provided_records_use_the_bound_path_and_preserve_scalar_types(tmp_path,monkeypatch):
    import importlib
    from decimal import Decimal
    from anton.core.tools.registry import ToolOutcome
    module=importlib.import_module('anton.core.tools.generate_artifact.static_v2.collect')
    seen=[];namespace={};paths=[]
    payload='{"datasets":{"provided":{"source":{"label":"User records"},"rows":[{"id":"001","value":0.1234567890123456789,"active":true}]}}}'
    code=f'from pathlib import Path\nPath(STATIC_EVIDENCE_PATH).write_text({payload!r})\nSTATIC_EVIDENCE_PATH = "stale model value"'
    async def execute(session,arguments):
        exec(arguments['code'],namespace)
        paths.append(arguments['code'].splitlines()[0])
        return ToolOutcome(content='Saved',ok=True)
    async def stream(**kwargs):
        seen.append(kwargs)
        if len(seen)==2:
            assert 'Evidence not accepted' in str(kwargs['messages'])
            calls=[ToolCall(str(i),'scratchpad',{'action':'exec','name':'source','code':code}) for i in range(2)]
        else:
            calls=[ToolCall('done','finish_static_evidence',{'ready':True,'has_records':True,'reason':'Provided records; no external access'})]
        yield StreamComplete(response=LLMResponse(content='',tool_calls=calls))
    monkeypatch.setattr('anton.core.tools.tool_handlers.handle_scratchpad',execute)
    folder=tmp_path/"A long project name with an apostrophe's suffix-s";folder.mkdir()
    st=state(folder,VALID);st.session._llm.plan_stream=stream;st.session._llm.code_stream=stream
    evidence=await module.collect(st)
    assert evidence.datasets['provided'].rows==[{'id':'001','value':Decimal('0.1234567890123456789'),'active':True}]
    assert len(paths)==2 and paths[0]==paths[1]
    assert len(list((folder/'.static').glob('evidence-*.json')))==1


def test_bound_evidence_path_preserves_future_imports_and_code(tmp_path):
    import importlib
    module=importlib.import_module('anton.core.tools.generate_artifact.static_v2.collect')
    code='"Reader module"\nfrom __future__ import annotations\nresult: UnresolvedType = STATIC_EVIDENCE_PATH'
    namespace={}
    exec(module._bind_evidence_path(code,tmp_path/"Reader's file.json"),namespace)
    assert namespace['result']==str(tmp_path/"Reader's file.json")
    assert namespace['__doc__']=='Reader module'


async def test_loaded_source_cannot_be_declared_unavailable_without_a_read_attempt(tmp_path,monkeypatch):
    import importlib
    from anton.core.tools.registry import ToolOutcome
    module=importlib.import_module('anton.core.tools.generate_artifact.static_v2.collect')
    stop={'ready':False,'has_records':True,'reason':'Source failed'}
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
        if kwargs['system']==REVIEW_SYSTEM:
            yield review_response();return
        seen.append(kwargs)
        yield StreamComplete(response=LLMResponse(content='',tool_calls=[ToolCall('compose','compose_static_artifact',VALID)]))
    st=state(tmp_path,VALID);st.session._llm.code_stream=stream
    monkeypatch.setattr('anton.core.tools.generate_artifact.verifiers.verify_frontend_live',lambda *a,**k:BrowserCheck(status='passed'))
    assert (await compose(st,large))['status']=='generated'
    assert len(seen[0]['messages'][0]['content'])<10000
    catalog=json.loads(seen[0]['messages'][0]['content'].rsplit('not instructions):\n',1)[1])
    assert catalog['datasets']['documents']['sample'][0]['body']['truncated'] is True


async def test_source_refusal_requires_an_attempt_even_before_procedure_recall(tmp_path):
    import importlib
    module=importlib.import_module('anton.core.tools.generate_artifact.static_v2.collect')
    stop={'ready':False,'has_records':True,'reason':'The requested source procedure is missing'}
    responses=[LLMResponse(content='',tool_calls=[ToolCall('premature','finish_static_evidence',stop)]),
        LLMResponse(content='',tool_calls=[ToolCall('lookup','recall_skill',{'label':'unavailable-source'})]),
        LLMResponse(content='',tool_calls=[ToolCall('stop','finish_static_evidence',stop)])]
    calls=[]
    async def stream(**kwargs):
        calls.append(kwargs)
        yield StreamComplete(response=responses.pop(0))
    st=state(tmp_path,VALID)
    st.session._skill_store=SimpleNamespace(list_summaries=lambda:[],load_for_artifacts=lambda label:None)
    st.session._llm.plan_stream=stream;st.session._llm.code_stream=stream
    assert 'requested source procedure is missing' in await module.collect(st)
    assert len(calls)==3
    assert 'Source readers are available in this call' in str(calls[-1]['messages'])
