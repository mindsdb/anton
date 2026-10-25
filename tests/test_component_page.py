import json
from types import SimpleNamespace

import pytest

from anton.core.llm.provider import LLMResponse, StreamComplete
from anton.core.tools.generate_artifact import component_page, fast_static, observed_data
from anton.core.tools.generate_artifact.state import BrowserCheck, GenState, VerifyResult

PROGRAM = "await CoworkPage({title:'Revenue',lang:'en',sections:[{type:'text',title:'Observed',text:String(window.ARTIFACT_DATA)}]});"


def test_program_round_trip_preserves_source_and_passes_existing_static_checks():
    html = component_page.document(PROGRAM)
    assert component_page.existing_program(html) == PROGRAM
    assert fast_static.verifiers.verify_frontend(html, is_fullstack=False).ok
    value = {"label": "</script><img src=x onerror=alert(1)>", "value": 42}
    bound = observed_data.bind(html, {observed_data.identity(value): value})
    assert list(observed_data.previous_data(bound).values()) == [value]
    assert component_page.existing_program(bound) == PROGRAM


@pytest.mark.parametrize("bad", ["<html></html>", "await CoworkPage({title:'</script>'});", "42"])
def test_non_program_or_script_termination_requires_repair(bad):
    with pytest.raises(ValueError):
        component_page.document(bad)


async def test_unsupported_request_keeps_both_normal_html_repair_attempts(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTON_STATIC_COMPONENT_WRITER", "on")
    good = '<html><head><meta name="viewport" content="width=device-width"></head><body>Custom UI</body></html>'
    responses = iter([component_page.FALLBACK, good, good])
    calls = []
    async def stream(**kwargs):
        calls.append(kwargs)
        yield StreamComplete(LLMResponse(content=next(responses)))
    results = iter([BrowserCheck(status="failed",verdict=VerifyResult(errors=["repair a real failure"])), BrowserCheck(status="passed")])
    monkeypatch.setattr(fast_static.verifiers,"verify_frontend_live",lambda *a,**kw:next(results))
    state=GenState(session=SimpleNamespace(history=[],_llm=SimpleNamespace(code_stream=stream)),artifact_type="html-app",
                   artifact_path=tmp_path,slug="custom",act_first=True)
    result=await fast_static.run(state,mode="direct")
    assert result["status"]=="generated"
    assert len(calls)==3
    assert calls[0]["system"]==component_page.SYSTEM
    assert "complete HTML document" in calls[1]["system"]
    assert (tmp_path/"index.html").read_text()==good


async def test_component_repair_receives_the_program_and_retains_bound_data(tmp_path,monkeypatch):
    monkeypatch.setenv("ANTON_STATIC_COMPONENT_WRITER","on")
    monkeypatch.setenv("ANTON_STATIC_DATA_BINDING","on")
    monkeypatch.setenv("ANTON_STATIC_CANONICAL_ONLY","on")
    data={"revenue":42};(tmp_path/"observed-data.json").write_text(json.dumps(data))
    calls=[]
    async def stream(**kwargs):
        calls.append(kwargs)
        yield StreamComplete(LLMResponse(content=PROGRAM))
    results=iter([BrowserCheck(status="failed",verdict=VerifyResult(errors=["repair"])),BrowserCheck(status="passed")])
    monkeypatch.setattr(fast_static.verifiers,"verify_frontend_live",lambda *a,**kw:next(results))
    state=GenState(session=SimpleNamespace(history=[],_llm=SimpleNamespace(code_stream=stream)),artifact_type="html-app",artifact_path=tmp_path,slug="page",act_first=True)
    result=await fast_static.run(state,mode="direct")
    assert result["status"]=="generated"
    assert calls[1]["messages"][1]["content"]==PROGRAM
    assert list(observed_data.previous_data((tmp_path/"index.html").read_text()).values())==[data]
    assert (tmp_path/"presentation-program.js").read_text()==PROGRAM


async def test_semantic_html_saves_exact_checked_bytes_without_an_extra_model_call(tmp_path,monkeypatch):
    monkeypatch.setenv('ANTON_STATIC_COMPONENT_WRITER','on')
    monkeypatch.setenv('ANTON_STATIC_SEMANTIC_HTML','on')
    calls=[];checked=[]
    async def stream(**kwargs):
        calls.append(kwargs)
        yield StreamComplete(LLMResponse(content=PROGRAM))
    def browser(entry,**kwargs):
        assert kwargs['materialize_main']=='cowork-page'
        original=entry.read_text()
        final=original.replace('<main id="cowork-page"></main>','<main id="cowork-page"><table><tr><td>Observed</td></tr></table></main>')
        checked.append(final)
        return BrowserCheck(status='passed',materialized_html=final)
    monkeypatch.setattr(fast_static.verifiers,'verify_frontend_live',browser)
    state=GenState(session=SimpleNamespace(history=[],_llm=SimpleNamespace(code_stream=stream)),
        artifact_type='html-app',artifact_path=tmp_path,slug='page',act_first=True)
    result=await fast_static.run(state,mode='direct')
    assert result['status']=='generated' and len(calls)==1
    saved=(tmp_path/'index.html').read_text()
    assert saved==checked[0] and '<table' in saved
    assert component_page.existing_program(saved)==PROGRAM
