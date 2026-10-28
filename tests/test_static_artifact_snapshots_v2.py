import importlib
import json
from decimal import Decimal


from anton.core.artifacts.static.data import Evidence
from anton.core.llm.provider import LLMResponse, StreamComplete, ToolCall
from anton.core.tools.registry import ToolOutcome
from anton.core.tools.generate_artifact.static_v2 import snapshots
from tests.test_static_artifact_pipeline_v2 import state, VALID


def published(tmp_path):
    st=state(tmp_path,VALID)
    evidence=Evidence.model_validate({'datasets':{'items':{'source':{'label':'User records'},
        'rows':[{'id':'001','amount':Decimal('0.1234567890123456789'),'tiny':1e-7,'private':'PRIVATE_ONLY'}]}}})
    st.static_evidence_origin={'mode':'acquired','acquired_at':'2026-10-28T10:00:00+00:00'}
    snapshots.save(st,'PUBLISHED PAGE',evidence)
    (tmp_path/'index.html').write_text('PUBLISHED PAGE')
    return st,evidence


def test_published_snapshot_preserves_exact_records_and_original_acquisition_time(tmp_path):
    st,evidence=published(tmp_path)
    recovered,origin=snapshots.load(st)
    assert recovered.datasets['items'].rows==evidence.datasets['items'].rows
    assert type(recovered.datasets['items'].rows[0]['amount']) is Decimal
    assert origin['mode']=='reused' and origin['acquired_at']==st.static_evidence_origin['acquired_at']
    from anton.core.artifacts.store import iter_content_files
    assert [p.name for p,_ in iter_content_files(tmp_path)]==['index.html']


def test_snapshot_for_an_unpublished_or_externally_changed_page_is_not_reused(tmp_path):
    st,evidence=published(tmp_path)
    snapshots.save(st,'CANDIDATE NEVER PUBLISHED',evidence)
    (tmp_path/'index.html').write_text('EXTERNAL EDIT')
    assert snapshots.load(st) is None


def test_changed_snapshot_payload_is_rejected(tmp_path):
    st,_=published(tmp_path)
    path=next((tmp_path/'.static').glob('published-*.json'))
    record=json.loads(path.read_text());record['evidence_json']=record['evidence_json'].replace('001','002')
    path.write_text(json.dumps(record))
    assert snapshots.load(st) is None


async def test_presentation_edit_can_explicitly_reuse_without_reading_the_source(tmp_path):
    module=importlib.import_module('anton.core.tools.generate_artifact.static_v2.collect')
    st,evidence=published(tmp_path)
    async def stream(**kwargs):
        assert 'reuse_static_evidence' in [t['name'] for t in kwargs['tools']]
        yield StreamComplete(response=LLMResponse(content='',tool_calls=[ToolCall('reuse','reuse_static_evidence',{'reason':'Only change typography; preserve data'})]))
    st.session._llm.plan_stream=stream
    result=await module.collect(st)
    assert result.datasets['items'].rows==evidence.datasets['items'].rows
    assert st.static_evidence_origin['mode']=='reused'


async def test_failed_fresh_read_cannot_fall_back_to_snapshot(tmp_path,monkeypatch):
    module=importlib.import_module('anton.core.tools.generate_artifact.static_v2.collect')
    st,_=published(tmp_path);seen=[]
    responses=[LLMResponse(content='',tool_calls=[ToolCall('source','scratchpad',{'action':'exec','name':'data','code':'read_source()'})]),
        LLMResponse(content='',tool_calls=[ToolCall('reuse','reuse_static_evidence',{'reason':'Source failed; use old records'})]),
        LLMResponse(content='',tool_calls=[ToolCall('stop','finish_static_evidence',{'ready':False,'has_records':True,'reason':'Fresh source is unavailable'})])]
    async def stream(**kwargs):
        seen.append(kwargs)
        yield StreamComplete(response=responses.pop(0))
    async def failed(*args):
        return ToolOutcome(content='Read failed',ok=False)
    monkeypatch.setattr('anton.core.tools.tool_handlers.handle_scratchpad',failed)
    st.session._llm.plan_stream=stream;st.session._llm.code_stream=stream
    assert 'Fresh source is unavailable' in await module.collect(st)
    assert 'Snapshot reuse is unavailable' in str(seen[-1]['messages'])


async def test_evidence_refresh_does_not_offer_or_accept_snapshot_reuse(tmp_path):
    module=importlib.import_module('anton.core.tools.generate_artifact.static_v2.collect')
    st,_=published(tmp_path);seen=[]
    async def stream(**kwargs):
        seen.append(kwargs)
        assert 'reuse_static_evidence' not in [t['name'] for t in kwargs['tools']]
        if len(seen)==1:
            call=ToolCall('reuse','reuse_static_evidence',{'reason':'Use cached data'})
        else:
            call=ToolCall('stop','finish_static_evidence',{'ready':False,'has_records':False,'reason':'Cannot satisfy refresh'})
        yield StreamComplete(response=LLMResponse(content='',tool_calls=[call]))
    st.session._llm.plan_stream=stream;st.session._llm.code_stream=stream
    assert 'Cannot satisfy refresh' in await module.collect(st,feedback='The user requested current values')
    assert 'Snapshot reuse is unavailable' in str(seen[-1]['messages'])
