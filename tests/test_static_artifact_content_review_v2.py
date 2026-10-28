import base64
from types import SimpleNamespace

import pytest

from anton.core.artifacts.static.data import Evidence
from anton.core.artifacts.render_check import RenderScreenshot
from anton.core.llm.provider import LLMResponse,StreamComplete,ToolCall
from anton.core.tools.generate_artifact.state import BrowserCheck
from anton.core.tools.generate_artifact.static_v2 import content_review
from anton.core.tools.generate_artifact.static_v2.compose import Structured,compose
from tests.test_static_artifact_pipeline_v2 import state,VALID


def response(approved,issues):
    return StreamComplete(response=LLMResponse(content='',tool_calls=[ToolCall('review','review_static_artifact',{'approved':approved,'issues':issues})]))


@pytest.mark.parametrize('with_screenshot',[False,True])
async def test_fresh_review_sees_request_evidence_previous_page_and_screenshot(tmp_path,with_screenshot):
    seen=[]
    async def stream(**kwargs):
        seen.append(kwargs)
        yield response(True,[])
    st=state(tmp_path,VALID);st.session._llm.code_stream=stream
    recorded=[]
    st.trace_log=SimpleNamespace(llm_call=lambda **kwargs:recorded.append(kwargs),node=lambda *a,**k:None)
    image=RenderScreenshot(media_type='image/png',data=base64.b64encode(b'synthetic image bytes').decode()) if with_screenshot else None
    result=await content_review.check(st,evidence='EXACT EVIDENCE',composition=Structured(mode='document',**VALID),
        manifest={'facts':{'total':12},'bindings':{'items':[{'amount':12}]}},previous='PREVIOUS COMPOSITION',screenshot=image)
    assert result=={'status':'no_issues_found','screenshot_provided':with_screenshot}
    messages=seen[0]['messages'][0]['content']
    assert 'Make a report' in messages[0]['text'] and 'EXACT EVIDENCE' in messages[0]['text']
    assert 'PREVIOUS COMPOSITION' in messages[0]['text']
    if with_screenshot:
        assert messages[1]['type']=='image'
        assert image.data not in str(recorded[0])
        assert 'Desktop screenshot attached' in str(recorded[0])
    else:
        assert len(messages)==1
    assert [tool['name'] for tool in seen[0]['tools']]==['review_static_artifact']


async def test_review_rejection_is_repaired_before_primary_publication(tmp_path,monkeypatch):
    prior=tmp_path/'index.html';prior.write_text('PRIOR PAGE')
    author_calls=[];reviews=[]
    async def stream(**kwargs):
        if kwargs['system']==content_review.SYSTEM:
            reviews.append(kwargs)
            assert prior.read_text()=='PRIOR PAGE'
            yield response(len(reviews)>1,[] if len(reviews)>1 else ['The requested conclusion is absent.'])
        else:
            author_calls.append(kwargs)
            text='Corrected conclusion.' if len(author_calls)>1 else 'Initial draft.'
            payload={'document':{'title':'Findings','sections':[{'id':'one','blocks':[{'type':'prose','text':text}]}]}}
            yield StreamComplete(response=LLMResponse(content='',tool_calls=[ToolCall('compose','compose_static_artifact',payload)]))
    monkeypatch.setattr('anton.core.tools.generate_artifact.verifiers.verify_frontend_live',lambda *a,**k:BrowserCheck(status='passed'))
    st=state(tmp_path,VALID);st.session._llm.code_stream=stream
    result=await compose(st,Evidence())
    assert result['status']=='generated' and len(reviews)==2 and len(author_calls)==2
    assert 'requested conclusion is absent' in str(author_calls[-1]['messages'])
    assert 'Corrected conclusion.' in prior.read_text()


async def test_incomplete_content_review_never_replaces_the_previous_page(tmp_path,monkeypatch):
    prior=tmp_path/'index.html';prior.write_text('PRIOR PAGE')
    async def stream(**kwargs):
        if kwargs['system']==content_review.SYSTEM:
            yield StreamComplete(response=LLMResponse(content='Looks fine',tool_calls=[]))
        else:
            yield StreamComplete(response=LLMResponse(content='',tool_calls=[ToolCall('compose','compose_static_artifact',VALID)]))
    monkeypatch.setattr('anton.core.tools.generate_artifact.verifiers.verify_frontend_live',lambda *a,**k:BrowserCheck(status='passed'))
    st=state(tmp_path,VALID);st.session._llm.code_stream=stream
    assert 'Content review did not return' in await compose(st,Evidence())
    assert prior.read_text()=='PRIOR PAGE'


async def test_cancellation_during_content_review_prevents_publication(tmp_path,monkeypatch):
    prior=tmp_path/'index.html';prior.write_text('PRIOR PAGE')
    st=state(tmp_path,VALID)
    async def stream(**kwargs):
        if kwargs['system']==content_review.SYSTEM:
            st.abandoned.set();yield response(True,[])
        else:
            yield StreamComplete(response=LLMResponse(content='',tool_calls=[ToolCall('compose','compose_static_artifact',VALID)]))
    st.session._llm.code_stream=stream
    monkeypatch.setattr('anton.core.tools.generate_artifact.verifiers.verify_frontend_live',lambda *a,**k:BrowserCheck(status='passed'))
    assert 'cancelled' in await compose(st,Evidence())
    assert prior.read_text()=='PRIOR PAGE'
