import json
from pathlib import Path
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
    (tmp_path / "observed-data.json").write_text(json.dumps({"revenue": 42}))
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
    result=await fast_static.run(state)
    assert result["status"]=="generated"
    assert len(calls)==3
    assert calls[0]["system"].startswith(component_page.SYSTEM)
    assert "complete HTML document" in calls[1]["system"]
    assert observed_data.strip_binding((tmp_path/"index.html").read_text())==good


async def test_component_repair_receives_the_program_and_retains_bound_data(tmp_path,monkeypatch):
    monkeypatch.setenv("ANTON_STATIC_COMPONENT_WRITER","on")
    data={"revenue":42}
    (tmp_path/"observed-data.json").write_text(json.dumps(data))
    calls=[]
    async def stream(**kwargs):
        calls.append(kwargs)
        yield StreamComplete(LLMResponse(content=PROGRAM))
    results=iter([BrowserCheck(status="failed",verdict=VerifyResult(errors=["repair"])),BrowserCheck(status="passed")])
    monkeypatch.setattr(fast_static.verifiers,"verify_frontend_live",lambda *a,**kw:next(results))
    state=GenState(session=SimpleNamespace(history=[],_llm=SimpleNamespace(code_stream=stream)),artifact_type="html-app",artifact_path=tmp_path,slug="page",act_first=True)
    result=await fast_static.run(state)
    assert result["status"]=="generated"
    assert calls[1]["messages"][1]["content"]==PROGRAM
    assert list(observed_data.previous_data((tmp_path/"index.html").read_text()).values())==[data]
    assert (tmp_path/"presentation-program.js").read_text()==PROGRAM


@pytest.mark.parametrize("program,rejected", [
    ("await CoworkPage({title:'T',controls:[{id:'view',type:'tabs',value:'a',options:[]}],"
     "render: selection => selection === 'a' ? {sections:[]} : {sections:[]}});", True),
    ("const render = s => { switch (s) { default: return {sections:[]}; } };\n"
     "await CoworkPage({title:'T',controls:[],render});", True),
    ("await CoworkPage({title:'T',controls:[{id:'view',type:'tabs',value:'a',options:[]}],"
     "render: ({view}) => view === 'a' ? {sections:[]} : {sections:[]}});", False),
    ("await CoworkPage({title:'T',controls:[],render: selection => selection.view === 'a' ? {sections:[]} : {sections:[]}});", False),
    ("const open = rows.filter(s => s === 'Open');\n"
     "await CoworkPage({title:'T',controls:[],render: s => ({sections:[]})});", False),
])
def test_a_render_that_treats_the_selection_as_a_string_is_sent_back_for_repair(program, rejected):
    if rejected:
        with pytest.raises(ValueError, match="keyed by control ids"):
            component_page.document(program)
    else:
        assert component_page.existing_program(component_page.document(program)) == program


async def test_the_component_writer_is_told_how_render_receives_controls(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTON_STATIC_COMPONENT_WRITER", "on")
    (tmp_path / "observed-data.json").write_text(json.dumps({"revenue": 42}))
    systems = []
    async def stream(**kwargs):
        systems.append(kwargs["system"])
        yield StreamComplete(LLMResponse(content=PROGRAM))
    monkeypatch.setattr(fast_static.verifiers, "verify_frontend_live", lambda *a, **kw: BrowserCheck(status="passed"))
    state = GenState(session=SimpleNamespace(history=[], _llm=SimpleNamespace(code_stream=stream)),
                     artifact_type="html-app", artifact_path=tmp_path, slug="page", act_first=True)
    assert (await fast_static.run(state))["status"] == "generated"
    assert "object keyed by control IDs" in systems[0]


def test_maps_get_label_placement_and_legends_and_motion_only_on_request():
    page = component_page.document("await CoworkPage({title:'T',sections:[]});")
    assert "CoworkMapLabels" in page and "CoworkGeoLegend" in page
    assert "Only when the user asks for motion" in Path(fast_static.__file__).read_text()


@pytest.mark.parametrize("section,bound", [
    ("{type:'map',title:'M',points:[{name:'a',place:'Lyon'}]}", True),
    ("{type:'text',title:'T',text:'Lyon'}", False)])
async def test_places_are_bound_only_for_a_page_with_a_map(tmp_path, monkeypatch, section, bound):
    monkeypatch.setenv("ANTON_STATIC_COMPONENT_WRITER", "on")
    (tmp_path / "observed-data.json").write_text(json.dumps({"rows": [{"at": "Lyon"}]}))
    program = "await CoworkPage({title:'T',lang:'en',sections:[" + section + "]});"
    async def stream(**kwargs):
        yield StreamComplete(LLMResponse(content=program))
    monkeypatch.setattr(fast_static.verifiers, "verify_frontend_live", lambda *a, **kw: BrowserCheck(status="passed"))
    state = GenState(session=SimpleNamespace(history=[], _llm=SimpleNamespace(code_stream=stream)),
                     artifact_type="html-app", artifact_path=tmp_path, slug="page", act_first=True)
    assert (await fast_static.run(state))["status"] == "generated"
    assert ("window.ARTIFACT_PLACES=" in (tmp_path / "index.html").read_text()) is bound
