"""Real Chromium checks; optional where the project has no browser runtime."""
import os
import pytest

from anton.core.artifacts.static.data import Evidence
from anton.core.artifacts.static.document import Document
from anton.core.artifacts.static.render import render

pytestmark=pytest.mark.skipif(os.environ.get('ANTON_TEST_BROWSER') != '1', reason='Opt-in Chromium suite: ANTON_TEST_BROWSER=1')
playwright=pytest.importorskip('playwright.sync_api')


def specimen(kind='report'):
    evidence=Evidence.model_validate({'datasets':{'sample':{'source':{'label':'Synthetic QA records'},'rows':[
        {'name':'A very long category name that needs room on a phone','a':12,'b':None},
        {'name':'Negative','a':-5,'b':9}, {'name':'Zero','a':0,'b':3}]}}})
    sections=[{'id':'comparison','title':'Compare records','blocks':[{'type':'chart','title':'Amounts','x':'name','kind':chart_kind,
        'query':{'dataset':'sample','columns':['name','a','b']},'series':[{'field':'a','label':'A'},{'field':'b','label':'B'}]} for chart_kind in ['bar','line']]},
        {'id':'conclusion','title':'A conclusion','blocks':[{'type':'callout','title':'Interpretation','text':'Missing values are not zero.'}]}]
    return render(Document.model_validate({'kind':kind,'title':'A report with long labels','sections':sections}),evidence).html


@pytest.mark.parametrize('width,height',[(1440,900),(390,844),(640,900),(641,900),(1100,900),(1101,900),(1024,600)])
def test_charts_reflow_without_overflow_or_errors(width,height):
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch()
        page=browser.new_page(viewport={'width':width,'height':height})
        errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
        page.set_content(specimen())
        page.wait_for_function('document.querySelectorAll(".chart svg").length === 2')
        assert not errors
        assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
        assert page.locator('.chart-data[open]').count()==0
        page.locator('.chart-data summary').first.click()
        assert page.locator('.chart-data[open]').count()==1
        if width==390:
            label=page.locator('.chart svg text').filter(has_text='A very long category name').first
            assert label.locator('tspan').count()>=1
            assert 'A very long category name' in label.text_content()
        browser.close()


def test_slide_controls_keyboard_exit_and_print_restore_all_content():
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch();page=browser.new_page(viewport={'width':390,'height':844})
        errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
        page.set_content(specimen('slides'))
        page.get_by_role('button',name='Present',exact=True).click()
        assert page.locator('.presenting').count()==1
        assert page.locator('#comparison:visible').count()==1
        page.wait_for_function('document.querySelectorAll(".chart svg").length===2')
        page.get_by_role('button',name='Next slide').click()
        assert page.locator('#conclusion:visible').count()==1
        page.emulate_media(media='print')
        assert page.locator('#comparison').is_visible()
        page.emulate_media(media='screen')
        page.locator('h2').filter(has_text='A conclusion').click()
        page.keyboard.press('Escape')
        assert page.locator('.presenting').count()==0
        assert page.locator('.section:visible').count()==2
        assert not errors
        browser.close()


def test_custom_render_contract_detects_empty_data_view():
    data=Evidence.model_validate({'datasets':{'desks':{'source':{'label':'Synthetic QA'},'rows':[{'id':'D1'},{'id':'D2'}]}}})
    doc=Document.model_validate({'title':'Desks','bindings':{'desks':{'dataset':'desks','columns':['id']}},
        'views':[{'selector':'.desk','binding':'desks','minimum':'rows'}], 'sections':[{'id':'desks','blocks':[{'type':'html','html':'<div id="desk-grid"></div>'}]}]})
    html=render(doc,data).html
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch();page=browser.new_page();errors=[]
        page.on('console',lambda message:errors.append(message.text) if message.type=='error' else None)
        page.set_content(html)
        page.get_by_role('alert').wait_for()
        assert any('expected at least 2' in error for error in errors)
        browser.close()


def test_wide_tables_recompose_as_complete_labelled_records_on_phones():
    columns=['Record','Category','Site','Date','Amount','Explanation']
    data=Evidence.model_validate({'datasets':{'records':{'source':{'label':'Synthetic QA'},
        'rows':[dict(zip(columns,['A','Inspection','A long site name','2026-10-01',12,'A complete observation that should wrap without losing any words.']))]}}})
    doc=Document.model_validate({'title':'Wide data','sections':[{'id':'records','blocks':[{'type':'table','title':'Every field',
        'query':'records','columns':[{'field':key,'label':key} for key in columns]}]}]})
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch();page=browser.new_page(viewport={'width':390,'height':844})
        page.set_content(render(doc,data).html)
        assert page.get_by_role('table').count()==1
        assert page.locator('td').count()==len(columns)
        assert page.locator('td').evaluate_all('(cells)=>cells.every(cell=>getComputedStyle(cell).display==="grid" && cell.scrollWidth<=cell.clientWidth)')
        assert page.locator('td').first.get_attribute('data-label')=='Record'
        assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
        page.set_viewport_size({'width':1440,'height':900})
        assert page.locator('td').first.evaluate('(cell)=>getComputedStyle(cell).display')=='table-cell'
        browser.close()


@pytest.mark.parametrize('layer',['bindings','resources'])
def test_large_injected_payload_keeps_utf8_before_encoding_sniff_limit(tmp_path,layer):
    from anton.core.tools.generate_artifact.static_v2.compose import Custom,compile_output
    from anton.core.tools.generate_artifact.static_v2.resources import embed
    html='<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"></head><body><h1>0° · Проверка</h1><p>Complete Unicode text.</p></body></html>'
    if layer=='bindings':
        evidence=Evidence.model_validate({'datasets':{'records':{'source':{'label':'Synthetic QA'},'rows':[{'body':'x'*5000}]}}})
        html,_=compile_output(Custom(mode='html',html=html,bindings={'records':{'dataset':'records','columns':['body']}},views=[{'selector':'h1','binding':'records'}]),evidence)
    else:
        html=embed(html,{}, {'reference':{'data':{'body':'x'*5000},'source':{'label':'Synthetic QA','url':'https://assets.example/v1.json','bytes':5000,'sha256':'sha'}}})
    assert html.index('<meta charset="utf-8">')<100
    page_path=tmp_path/'unicode.html';page_path.write_text(html,encoding='utf-8')
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch();page=browser.new_page()
        page.goto(page_path.as_uri())
        assert page.locator('h1').inner_text()=='0° · Проверка'
        assert page.evaluate('document.characterSet')=='UTF-8'
        browser.close()


def test_embedded_reference_data_and_library_work_with_network_disabled(tmp_path):
    from anton.core.tools.generate_artifact.static_v2.resources import embed
    source={'label':'Synthetic resource QA','url':'https://assets.example/v1.json','bytes':40,'sha256':'sha'}
    assets={'diagram':{'data':{'nodes':[1,2,3]},'source':source},
        'library':{'data':'window.DIAGRAM_LIBRARY={count:window.ARTIFACT_RESOURCES.diagram.nodes.length};','source':{**source,'kind':'script'}}}
    html=embed('<!doctype html><html><head></head><body><h1 id="count"></h1><script>document.getElementById("count").textContent=DIAGRAM_LIBRARY.count+" nodes";</script></body></html>',{},assets)
    path=tmp_path/'offline.html';path.write_text(html,encoding='utf-8')
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch();context=browser.new_context(offline=True);page=context.new_page()
        errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(path.as_uri())
        assert page.locator('h1').inner_text()=='3 nodes'
        assert not errors
        browser.close()


@pytest.mark.parametrize('long_labels',[False,True])
def test_time_axis_endpoint_labels_do_not_overlap_on_phones(long_labels):
    from datetime import date,timedelta
    data=Evidence.model_validate({'datasets':{'periods':{'source':{'label':'Synthetic weekly QA'},'rows':[
        {'date':('W'*40+str(n)) if long_labels else str(date(2026,10,5)+timedelta(weeks=n)),'amount':n+1} for n in range(13)]}}})
    doc=Document.model_validate({'title':'Weekly record','sections':[{'id':'trend','blocks':[{'type':'chart','title':'Observed amounts','kind':'line',
        'query':'periods','x':'date','series':[{'field':'amount','label':'Amount'}]}]}]})
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch();page=browser.new_page(viewport={'width':390,'height':844})
        page.set_content(render(doc,data).html)
        page.wait_for_function('document.querySelectorAll(".x-tick").length>=2')
        boxes=page.locator('.x-tick').evaluate_all('(nodes)=>nodes.map(node=>{const r=node.getBoundingClientRect();return {left:r.left,right:r.right};})')
        assert all(right['left']-left['right']>=5 for left,right in zip(boxes,boxes[1:]))
        browser.close()
