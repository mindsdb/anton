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


@pytest.mark.parametrize("bad", [
    "<html></html>", "await CoworkPage({title:'</script>'});", "await CoworkPage({title:'</script foo>'});",
    "await CoworkPage({title:'</script\t'});", "42",
])
def test_non_program_or_script_termination_requires_repair(bad):
    with pytest.raises(ValueError):
        component_page.document(bad)


async def test_unsupported_request_keeps_both_normal_html_repair_attempts(tmp_path, monkeypatch):
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


async def test_maps_get_label_placement_and_legends_and_motion_only_on_request(tmp_path, monkeypatch):
    page = component_page.document("await CoworkPage({title:'T',sections:[]});")
    assert "CoworkMapLabels" in page and "CoworkGeoLegend" in page
    (tmp_path / "observed-data.json").write_text(json.dumps({"revenue": 42}))
    systems = []
    async def stream(**kwargs):
        systems.append(kwargs["system"])
        yield StreamComplete(LLMResponse(content=PROGRAM))
    monkeypatch.setattr(fast_static.verifiers, "verify_frontend_live", lambda *a, **kw: BrowserCheck(status="passed"))
    state = GenState(session=SimpleNamespace(history=[], _llm=SimpleNamespace(code_stream=stream)),
                     artifact_type="html-app", artifact_path=tmp_path, slug="page", act_first=True)
    assert (await fast_static.run(state))["status"] == "generated"
    assert "Only when the user asks for motion" in systems[0]


@pytest.mark.parametrize("section,bound", [
    ("{type:'map',title:'M',points:[{name:'a',place:'Lyon'}]}", True),
    ("{type:'text',title:'T',text:'Lyon'}", False)])
async def test_places_are_bound_only_for_a_page_with_a_map(tmp_path, monkeypatch, section, bound):
    (tmp_path / "observed-data.json").write_text(json.dumps({"rows": [{"at": "Lyon"}]}))
    program = "await CoworkPage({title:'T',lang:'en',sections:[" + section + "]});"
    async def stream(**kwargs):
        yield StreamComplete(LLMResponse(content=program))
    monkeypatch.setattr(fast_static.verifiers, "verify_frontend_live", lambda *a, **kw: BrowserCheck(status="passed"))
    state = GenState(session=SimpleNamespace(history=[], _llm=SimpleNamespace(code_stream=stream)),
                     artifact_type="html-app", artifact_path=tmp_path, slug="page", act_first=True)
    assert (await fast_static.run(state))["status"] == "generated"
    assert ("window.ARTIFACT_PLACES=" in (tmp_path / "index.html").read_text()) is bound


async def test_the_writer_is_told_that_place_names_can_be_mapped(tmp_path, monkeypatch):
    (tmp_path / "observed-data.json").write_text(json.dumps({"rows": [{"at": "Lyon"}]}))
    prompts = []
    async def stream(**kwargs):
        prompts.append(kwargs["messages"][0]["content"])
        yield StreamComplete(LLMResponse(content=PROGRAM))
    monkeypatch.setattr(fast_static.verifiers, "verify_frontend_live", lambda *a, **kw: BrowserCheck(status="passed"))
    state = GenState(session=SimpleNamespace(history=[], _llm=SimpleNamespace(code_stream=stream)),
                     artifact_type="html-app", artifact_path=tmp_path, slug="page", act_first=True)
    assert (await fast_static.run(state))["status"] == "generated"
    assert "{place: name}" in prompts[0] and "Lyon" not in prompts[0].split("Place names")[1]


@pytest.mark.parametrize("reply,checked", [(PROGRAM, True), (component_page.FALLBACK, False)])
async def test_component_pages_carry_the_figure_checks_and_html_pages_do_not(tmp_path, monkeypatch, reply, checked):
    (tmp_path / "observed-data.json").write_text(json.dumps({"revenue": [{"month": "m1", "amount": 42}]}))
    html = '<html><head><meta name="viewport" content="width=device-width"></head><body>Revenue</body></html>'
    replies = iter([reply, html])
    async def stream(**kwargs):
        yield StreamComplete(LLMResponse(content=next(replies)))
    monkeypatch.setattr(fast_static.verifiers, "verify_frontend_live", lambda *a, **kw: BrowserCheck(status="passed"))
    state = GenState(session=SimpleNamespace(history=[], _llm=SimpleNamespace(code_stream=stream)),
                     artifact_type="html-app", artifact_path=tmp_path, slug="page", act_first=True)
    assert (await fast_static.run(state))["status"] == "generated"
    page = (tmp_path / "index.html").read_text()
    assert ('id="cowork-figure-checks"' in page) is checked
    assert ("Figure check: " in page) is checked
    assert "zeroReason" in component_page.SYSTEM


async def test_fast_static_catches_inline_js_syntax_error_and_repairs(tmp_path, monkeypatch):
    import shutil

    if not shutil.which("node"):
        pytest.skip("node not available")
    broken = (
        '<!doctype html><html><head><meta name="viewport" content="width=device-width"></head>'
        '<body><div id="app">ok</div><script>function broken(a, b { return a + b; }</script></body></html>'
    )
    fixed = (
        '<!doctype html><html><head><meta name="viewport" content="width=device-width"></head>'
        '<body><div id="app">ok</div><script>function fixed(a, b) { return a + b; }</script></body></html>'
    )
    replies = iter([broken, fixed])
    seen = []

    async def stream(**kwargs):
        seen.append(kwargs["messages"])
        yield StreamComplete(LLMResponse(content=next(replies)))

    monkeypatch.setattr(fast_static.verifiers, "verify_frontend_live", lambda *a, **kw: BrowserCheck(status="passed"))
    state = GenState(session=SimpleNamespace(history=[], _llm=SimpleNamespace(code_stream=stream)),
                     artifact_type="html-app", artifact_path=tmp_path, slug="page", act_first=True)
    result = await fast_static.run(state)
    assert result["status"] == "generated"
    assert len(seen) == 2
    assert "JavaScript syntax error" in seen[1][-1]["content"]
    assert "SyntaxError:" in seen[1][-1]["content"]
    assert "node:internal" not in seen[1][-1]["content"]


def test_check_inline_js_syntax_edge_cases():
    import shutil
    from anton.core.tools.generate_artifact import verifiers

    if not shutil.which("node"):
        pytest.skip("node not available")
    # HTML comments, textarea, quoted '>' in attributes, and data-cowork-observed are handled cleanly.
    safe_html = (
        '<!doctype html><html><body>'
        '<!-- <script>function broken( {</script> -->'
        '<textarea><script>function also_broken( {</script></textarea>'
        '<script data-note="a > b" type="text/javascript; charset=utf-8">const x = 1 + 2;</script>'
        '<script type="application/json" data-cowork-observed>{"a":1}</script>'
        '</body></html>'
    )
    assert verifiers.check_inline_js_syntax(safe_html) == []

    # ES module syntax error preserves SyntaxError and strips Node internal frames.
    bad_esm = '<!doctype html><html><body><script type="module">export const x = (1 + ;</script></body></html>'
    esm_errs = verifiers.check_inline_js_syntax(bad_esm)
    assert len(esm_errs) == 1
    assert "SyntaxError:" in esm_errs[0]
    assert "node:internal" not in esm_errs[0]

    # Classic script with MIME charset parameter is checked and caught.
    bad_classic = '<!doctype html><html><body><script type="text/javascript; charset=utf-8">function f(a {</script></body></html>'
    classic_errs = verifiers.check_inline_js_syntax(bad_classic)
    assert len(classic_errs) == 1
    assert "SyntaxError:" in classic_errs[0]
    assert "node:internal" not in classic_errs[0]


def test_presentation_reconcile_js_zero_reason_and_per_cell_double_count_provenance():
    import json as _json
    import shutil
    import subprocess
    from anton.core.tools.generate_artifact import observed_data

    if not shutil.which("node"):
        pytest.skip("node not available")
    reconcile_js = (
        component_page.HERE / "presentation_reconcile.js"
    ).read_text(encoding="utf-8")
    data = {
        "key": {
            "activity": [
                {"week": "w1", "group": "A", "events": 70},
                {"week": "w1", "group": "B", "events": 47},
                {"week": "w1", "group": "All", "events": 117},
                {"week": "w2", "group": "A", "events": 30},
                {"week": "w2", "group": "B", "events": 20},
                {"week": "w2", "group": "All", "events": 50},
                {"week": "w3", "group": "A", "events": 15},
                {"week": "w3", "group": "B", "events": 10},
                {"week": "w3", "group": "All", "events": 25},
            ],
            "locations": [{"name": "North", "capacity": 234}],
        }
    }
    checks_json = _json.dumps({"doubleCounts": observed_data.double_counts(data)})
    data_json = _json.dumps(data)
    harness = f"""
const errors = [];
console.error = (...args) => errors.push(args.join(' '));
let drawn;
global.window = {{
  ARTIFACT_DATA: {data_json},
  CoworkNumber: v => String(v),
  CoworkSelection: s => s,
  CoworkPage: async cfg => {{ drawn = cfg; }},
}};
global.document = {{
  getElementById: id => id === 'cowork-figure-checks' ? {{ textContent: {_json.dumps(checks_json)} }} : null,
}};
{reconcile_js}
(async () => {{
  // 1. All-zero table honors section.zeroReason when EVERY numeric column is zero.
  await window.CoworkPage({{
    title: 'T',
    sections: [{{
      type: 'table', title: 'S', zeroReason: 'Baseline has zero delays',
      columns: [{{ key: 'n', label: 'Name' }}, {{ key: 'd', label: 'Delay' }}],
      rows: [{{ n: 'A', d: 0 }}, {{ n: 'B', d: 0 }}],
    }}],
  }});
  if (errors.length !== 0) {{ console.log('CASE1:' + JSON.stringify(errors)); process.exit(1); }}

  // 2. Mixed table: section.zeroReason must NOT excuse an unannotated all-zero column.
  await window.CoworkPage({{
    title: 'T',
    sections: [{{
      type: 'table', title: 'S', zeroReason: 'Baseline row',
      columns: [{{ key: 'n', label: 'Name' }}, {{ key: 'ok', label: 'Orders' }}, {{ key: 'broken', label: 'Revenue' }}],
      rows: [{{ n: 'A', ok: 5, broken: 0 }}, {{ n: 'B', ok: 7, broken: 0 }}],
    }}],
  }});
  if (errors.length !== 1 || !errors[0].includes('column "Revenue"')) {{ console.log('CASE2:' + JSON.stringify(errors)); process.exit(2); }}
  errors.length = 0;

  // 3. Whitespace-only zeroReason must NOT hide errors.
  await window.CoworkPage({{
    title: 'T',
    metrics: [{{ label: 'Unmatched', value: 0, zeroReason: '   ' }}],
  }});
  if (errors.length !== 1) {{ console.log('CASE3:' + JSON.stringify(errors)); process.exit(3); }}
  errors.length = 0;

  // 4. Per-cell doubleCount provenance: unrelated raw column `capacity: 234` is NOT flagged.
  await window.CoworkPage({{
    title: 'T',
    sections: [{{
      type: 'table', title: 'Locations',
      columns: [{{ key: 'name', label: 'Name' }}, {{ key: 'capacity', label: 'Capacity' }}],
      rows: window.ARTIFACT_DATA.key.locations,
    }}],
  }});
  if (errors.length !== 0) {{ console.log('CASE4:' + JSON.stringify(errors)); process.exit(4); }}

  // 5. Same numeric value 234 shown in a metric or related column (`total_events`) IS still caught.
  await window.CoworkPage({{
    title: 'T',
    metrics: [{{ label: 'Week 1 events', value: 234 }}],
    sections: [{{
      type: 'table', title: 'Summary',
      columns: [{{ key: 'name', label: 'Name' }}, {{ key: 'total_events', label: 'Total events' }}],
      rows: [{{ name: 'Week 1', total_events: 234 }}],
    }}],
  }});
  if (errors.length !== 1 || !errors[0].includes('234') || !errors[0].includes('counts rows twice')) {{
    console.log('CASE5:' + JSON.stringify(errors)); process.exit(5);
  }}
  console.log('ALL_OK');
}})();
"""
    proc = subprocess.run(["node", "-e", harness], capture_output=True, text=True, timeout=5)
    assert proc.returncode == 0, f"stdout={proc.stdout} stderr={proc.stderr}"
    assert "ALL_OK" in proc.stdout


def test_rejects_chat_user_field_ignores_role_ordering_errors():
    from types import SimpleNamespace
    from anton.core.llm.openai import _rejects_chat_user_field

    role_err = SimpleNamespace(body={"error": {"message": "Invalid role 'user' after assistant", "param": "messages"}})
    consec_err = SimpleNamespace(body={"error": {"message": 'Consecutive "user" messages are not allowed'}})
    real_param = SimpleNamespace(body={"error": {"message": "Invalid parameter", "param": "user"}})
    real_unknown = SimpleNamespace(body={"error": {"message": "Unknown field 'user' in request"}})
    real_422 = SimpleNamespace(body={"detail": [{"loc": ["body", "user"], "msg": "Extra inputs are not permitted"}]})
    other_422 = SimpleNamespace(body={"detail": [{"loc": ["body", "messages", "user", "content"], "msg": "bad"}]})

    assert _rejects_chat_user_field(role_err) is False
    assert _rejects_chat_user_field(consec_err) is False
    assert _rejects_chat_user_field(other_422) is False
    assert _rejects_chat_user_field(real_param) is True
    assert _rejects_chat_user_field(real_unknown) is True
    assert _rejects_chat_user_field(real_422) is True
