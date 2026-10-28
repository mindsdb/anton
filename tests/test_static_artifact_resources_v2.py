import hashlib
import json

import httpx2 as httpx
import pytest

from anton.core.artifacts.static.data import Evidence
from anton.core.llm.provider import LLMResponse, StreamComplete, ToolCall
from anton.core.tools.generate_artifact.state import BrowserCheck
from anton.core.tools.generate_artifact.static_v2 import resources
from anton.core.tools.generate_artifact.static_v2.compose import compose
from tests.test_static_artifact_pipeline_v2 import state,REVIEW_SYSTEM,review_response


def transport(monkeypatch,handler,guard=lambda url:None):
    client=httpx.AsyncClient
    monkeypatch.setattr(resources.httpx,'AsyncClient',lambda **kwargs:client(transport=httpx.MockTransport(handler),**kwargs))
    monkeypatch.setattr(resources,'_check_url_ssrf',guard)


async def test_resource_is_exact_script_safe_and_available_before_author_scripts(monkeypatch):
    raw=b'{"nodes":[{"label":"</script><script>bad()</script>"}],"edges":[]}'
    transport(monkeypatch,lambda request:httpx.Response(200,content=raw))
    item=await resources.load(resources.Resource(name='diagram',label='Public diagram',url='https://assets.example/diagram-v1.json'))
    assert item['data']==json.loads(raw)
    assert item['source']['sha256']==hashlib.sha256(raw).hexdigest()
    manifest={};html=resources.embed('<html><head><script>draw(window.ARTIFACT_RESOURCES.diagram)</script></head><body></body></html>',manifest,{'diagram':item})
    assert html.index('window.ARTIFACT_RESOURCES=')<html.index('draw(')
    assert '</script><script>bad()' not in html
    assert manifest['resources']['diagram']['label']=='Public diagram'


async def test_redirect_target_is_checked_before_any_internal_request(monkeypatch):
    seen=[]
    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(302,headers={'location':'https://127.0.0.1/private'})
    transport(monkeypatch,handler,lambda url:'Internal destination blocked' if '127.0.0.1' in url else None)
    with pytest.raises(ValueError,match='Internal destination blocked'):
        await resources.load(resources.Resource(name='x',label='Reference',url='https://assets.example/v1.json'))
    assert seen==['https://assets.example/v1.json']


async def test_resource_size_limit_never_returns_a_truncated_object(monkeypatch):
    monkeypatch.setattr(resources,'MAX_BYTES',24)
    transport(monkeypatch,lambda request:httpx.Response(200,content=b'{"label":"'+b'x'*100+b'"}'))
    with pytest.raises(ValueError,match='exceeds 2 MiB'):
        await resources.load(resources.Resource(name='x',label='Reference',url='https://assets.example/v1.json'))


@pytest.mark.parametrize('body',[b'<html>Not JSON</html>',b'{"bad":NaN}',b'"only a string"'])
async def test_non_json_or_nonfinite_resources_are_rejected(monkeypatch,body):
    transport(monkeypatch,lambda request:httpx.Response(200,content=body))
    with pytest.raises(ValueError):
        await resources.load(resources.Resource(name='x',label='Reference',url='https://assets.example/v1.json'))


@pytest.mark.parametrize('url',['http://assets.example/v1.json','https://user:password@assets.example/v1.json','https://assets.example/v1.json?token=secret'])
async def test_credentials_and_non_public_resource_shapes_are_refused(monkeypatch,url):
    transport(monkeypatch,lambda request:pytest.fail('URL must be refused before a request'))
    with pytest.raises(ValueError,match='public HTTPS URL'):
        await resources.load(resources.Resource(name='x',label='Reference',url=url))


@pytest.mark.parametrize('large_key',[False,True])
async def test_resource_tool_embeds_data_and_saves_only_reload_references_in_recipe(tmp_path,monkeypatch,large_key):
    calls=[]
    async def stream(**kwargs):
        if kwargs['system']==REVIEW_SYSTEM:
            yield review_response();return
        calls.append(kwargs)
        if len(calls)==1:
            call=ToolCall('resource','load_static_resource',{'name':'diagram','label':'Public diagram','url':'https://assets.example/diagram-v1.json'})
        else:
            call=ToolCall('compose','compose_custom_html',{'html':'<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"></head><body><h1>Diagram</h1><script>const nodes=window.ARTIFACT_RESOURCES.diagram.nodes;</script></body></html>'})
        yield StreamComplete(response=LLMResponse(content='',tool_calls=[call]))
    async def load(resource):
        data={'x'*300000:'Complete asset'} if large_key else {'nodes':[{'id':'A'},{'id':'B'}]}
        return {'data':data,'source':{'label':resource.label,'url':resource.url,'sha256':'sha','bytes':len(json.dumps(data))}}
    def check(path,**kwargs):
        assert 'window.ARTIFACT_RESOURCES=' in path.read_text()
        return BrowserCheck(status='passed')
    monkeypatch.setattr(resources,'load',load)
    monkeypatch.setattr('anton.core.tools.generate_artifact.verifiers.verify_frontend_live',check)
    st=state(tmp_path,{})
    st.session._llm.code_stream=stream
    result=await compose(st,Evidence())
    assert isinstance(result,dict),result
    assert result['status']=='generated'
    recipe=json.loads((tmp_path/'.static/composition.json').read_text())
    assert recipe['resources']['diagram']['url']=='https://assets.example/diagram-v1.json'
    assert 'nodes' not in recipe['resources']['diagram']
    if large_key:
        assert len(str(calls[-1]['messages']))<10000
        assert 'keys_are_previews' in str(calls[-1]['messages'])
        assert 'x'*300000 in (tmp_path/'index.html').read_text()


async def test_pinned_browser_library_is_bundled_before_author_code(monkeypatch):
    code=b'window.DIAGRAM_LIBRARY={ready:true};'
    transport(monkeypatch,lambda request:httpx.Response(200,content=code,headers={'content-type':'application/javascript; charset=utf-8'}))
    item=await resources.load(resources.Resource(name='library',label='Diagram library 1.2.3',kind='script',url='https://cdn.jsdelivr.net/npm/diagram@1.2.3/dist/browser.js'))
    html=resources.embed('<html><head></head><body><script>draw(DIAGRAM_LIBRARY)</script></body></html>',{}, {'library':item})
    assert html.index('window.DIAGRAM_LIBRARY=')<html.index('draw(')
    assert '<script src=' not in html


async def test_failed_candidate_can_release_a_broken_library_before_retrying(tmp_path,monkeypatch):
    author_calls=[];checked=[]
    html='<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"></head><body><h1>Corrected diagram</h1></body></html>'
    sequence=[ToolCall('load','load_static_resource',{'name':'unused','label':'Partial library','kind':'script','url':'https://cdn.jsdelivr.net/npm/diagram@1.2.3/partial.js'}),
        ToolCall('compose','compose_custom_html',{'html':html}),
        ToolCall('release','release_static_resource',{'name':'unused'}),
        ToolCall('compose2','compose_custom_html',{'html':html})]
    async def stream(**kwargs):
        if kwargs['system']==REVIEW_SYSTEM:
            yield review_response();return
        author_calls.append(kwargs)
        yield StreamComplete(response=LLMResponse(content='',tool_calls=[sequence.pop(0)]))
    async def load(resource):
        return {'data':'throw new Error("UNUSED_BUNDLE_FAILURE");','source':{'label':resource.label,'url':resource.url,'sha256':'sha','bytes':40,'kind':'script'}}
    def check(path,**kwargs):
        broken='UNUSED_BUNDLE_FAILURE' in path.read_text();checked.append(broken)
        return BrowserCheck(status='failed',reason='Broken bundled script') if broken else BrowserCheck(status='passed')
    monkeypatch.setattr(resources,'load',load)
    monkeypatch.setattr('anton.core.tools.generate_artifact.verifiers.verify_frontend_live',check)
    st=state(tmp_path,{});st.session._llm.code_stream=stream
    assert (await compose(st,Evidence()))['status']=='generated'
    assert checked==[True,False]
    assert 'Released unused' in str(author_calls[-1]['messages'])
    assert 'UNUSED_BUNDLE_FAILURE' not in (tmp_path/'index.html').read_text()
    assert json.loads((tmp_path/'.static/composition.json').read_text())['resources']=={}


@pytest.mark.parametrize('url',['https://unapproved.example/npm/diagram@1.2.3/browser.js','https://cdn.jsdelivr.net/npm/diagram@latest/browser.js'])
async def test_script_requires_an_exact_version_on_the_approved_host(monkeypatch,url):
    transport(monkeypatch,lambda request:pytest.fail('Unapproved script must not be requested'))
    with pytest.raises(ValueError,match='pinned npm package version'):
        await resources.load(resources.Resource(name='library',label='Library',kind='script',url=url))


@pytest.mark.parametrize('body,mime',[(b'console.log("</script>")','application/javascript'),(b'<html>error</html>','text/html')])
async def test_unsafe_or_wrong_type_script_is_rejected_instead_of_rewritten(monkeypatch,body,mime):
    transport(monkeypatch,lambda request:httpx.Response(200,content=body,headers={'content-type':mime}))
    with pytest.raises(ValueError):
        await resources.load(resources.Resource(name='library',label='Library',kind='script',url='https://cdn.jsdelivr.net/npm/diagram@1.2.3/dist/browser.js'))
