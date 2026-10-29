"""Real Chromium checks; optional where the project has no browser runtime."""
import os
import pytest

from anton.core.artifacts.static.data import Evidence
from anton.core.artifacts.static.document import Document
from anton.core.artifacts.static.render import render

pytestmark=pytest.mark.skipif(os.environ.get('ANTON_TEST_BROWSER') != '1', reason='Opt-in Chromium suite: ANTON_TEST_BROWSER=1')
playwright=pytest.importorskip('playwright.sync_api')


def specimen(kind='report',dark=False):
    evidence=Evidence.model_validate({'datasets':{'sample':{'source':{'label':'Synthetic QA records'},'rows':[
        {'name':'A very long category name that needs room on a phone','a':12,'b':None},
        {'name':'Negative','a':-5,'b':9}, {'name':'Zero','a':0,'b':3}]}}})
    sections=[{'id':'comparison','title':'Compare records','blocks':[{'type':'chart','title':'Amounts','x':'name','kind':chart_kind,
        'query':{'dataset':'sample','columns':['name','a','b']},'series':[{'field':'a','label':'A'},{'field':'b','label':'B'}]} for chart_kind in ['bar','line']]},
        {'id':'conclusion','title':'A conclusion','blocks':[{'type':'callout','title':'Interpretation','text':'Missing values are not zero.'}]}]
    return render(Document.model_validate({'kind':kind,'theme':{'mode':'dark' if dark else 'light'},'title':'A report with long labels','sections':sections}),evidence).html


@pytest.mark.parametrize('width,height',[(1440,900),(390,844),(640,900),(641,900),(1100,900),(1101,900),(1024,600)])
def test_charts_reflow_without_overflow_or_errors(width,height):
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch()
        page=browser.new_page(viewport={'width':width,'height':height})
        errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
        page.set_content(specimen())
        assert page.locator('.chart rect').count()==5
        assert page.locator('.chart [data-point]').count()==5
        assert not errors
        assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
        assert page.locator('.chart-data[open]').count()==0
        page.locator('.chart-data summary').first.click()
        assert page.locator('.chart-data[open]').count()==1
        if width==390:
            label=page.locator('.chart .category').filter(has_text='A very long category name').first
            assert label.evaluate('(el)=>el.scrollWidth<=el.clientWidth')
            assert 'A very long category name' in label.text_content()
        if width>=641:
            label=page.locator('.chart .category').filter(has_text='A very long category name').first
            assert label.evaluate('(el)=>el.getBoundingClientRect().height>parseFloat(getComputedStyle(el).lineHeight)*1.5')
            assert label.bounding_box()['x']>=page.locator('.chart').first.bounding_box()['x']-1
        browser.close()


@pytest.mark.parametrize('width',[390,640,641,1024,1100,1101,1440])
def test_compiled_charts_and_complete_data_render_without_javascript(width):
    html=specimen()
    assert html.count('<rect ')==5 and html.count('data-point ')==5
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch();context=browser.new_context(java_script_enabled=False,viewport={'width':width,'height':900})
        page=context.new_page();requests=[]
        page.route('**/*',lambda route:(requests.append(route.request.url),route.abort()))
        page.set_content(html)
        assert page.locator('.chart rect').count()==5
        assert page.locator('.chart [data-point]').count()==5
        assert page.locator('.chart-data[open]').count()==2
        assert not page.locator('[data-action=outline]').is_visible()
        assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
        assert not requests
        browser.close()


def test_static_point_marks_are_actually_painted_without_javascript():
    from io import BytesIO
    from PIL import Image
    data=Evidence.model_validate({'datasets':{'point':{'source':{'label':'Synthetic mark QA'},'rows':[{'x':1,'y':1}]}}})
    doc=Document.model_validate({'title':'A point','sections':[{'id':'plot','blocks':[
        {'type':'chart','title':'Observed point','kind':'scatter','query':'point','x':'x','series':[{'field':'y','label':'Amount'}]}]}]})
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch();page=browser.new_page(java_script_enabled=False,viewport={'width':390,'height':900})
        page.set_content(render(doc,data).html)
        center=page.locator('[data-point]').evaluate('(el)=>{const p=new DOMPoint(Number(el.getAttribute("x1")),Number(el.getAttribute("y1"))).matrixTransform(el.getScreenCTM());return {x:p.x,y:p.y};}')
        pixels=Image.open(BytesIO(page.screenshot(full_page=True))).convert('RGB')
        assert pixels.getpixel((round(center['x']),round(center['y'])))==(18,104,92)
        browser.close()


def test_slide_controls_keyboard_exit_and_print_restore_all_content():
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch();page=browser.new_page(viewport={'width':390,'height':844})
        errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
        page.set_content(specimen('slides'))
        page.get_by_role('button',name='Present',exact=True).click()
        assert page.locator('.presenting').count()==1
        assert page.locator('#comparison:visible').count()==1
        assert page.locator('.chart [data-point]').count()==5
        page.get_by_role('button',name='Next slide').click()
        assert page.locator('#conclusion:visible').count()==1
        page.emulate_media(media='print')
        assert page.locator('#comparison').is_visible()
        page.emulate_media(media='screen')
        page.get_by_role('button',name='Previous slide').focus()
        page.keyboard.press('Escape')
        assert page.locator('.presenting').count()==0
        assert page.locator('.section:visible').count()==2
        assert not errors
        browser.close()


def test_slide_reading_has_one_cover_and_no_source_metadata_slide():
    document=Document.model_validate({'kind':'slides','title':'Workshop','sections':[
        {'id':'cover','title':'Workshop','blocks':[{'type':'prose','text':'The opening slide'}]},
        {'id':'close','title':'Conclusion','blocks':[{'type':'prose','text':'The final slide'}]}]})
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch();page=browser.new_page(viewport={'width':390,'height':844})
        page.set_content(render(document,Evidence()).html)
        assert page.get_by_role('heading',name='Workshop',exact=True).count()==1
        assert page.locator('#_artifact-content > .section').count()==2
        assert page.locator('#_artifact-sources').evaluate('(el)=>getComputedStyle(el).borderTopWidth')=='0px'
        page.get_by_role('button',name='Present',exact=True).click()
        assert page.get_by_role('status').inner_text()=='1 / 2'
        assert page.get_by_role('heading',name='Workshop',exact=True).is_visible()
        page.emulate_media(media='print')
        assert page.locator('#cover').is_visible() and page.locator('#close').is_visible()
        assert not page.locator('#_artifact-sources').is_visible()
        browser.close()


def test_zero_and_small_bar_values_keep_their_true_scaled_widths():
    evidence=Evidence.model_validate({'datasets':{'readings':{'source':{'label':'Synthetic scale QA'},'rows':[
        {'label':'Zero','value':0},{'label':'Small','value':0.001},{'label':'Large','value':1000}]}}})
    document=Document.model_validate({'title':'Scaled values','sections':[{'id':'chart','blocks':[
        {'type':'chart','title':'Observed amounts','kind':'bar','query':'readings','x':'label','series':[{'field':'value','label':'Value'}]}]}]})
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch();page=browser.new_page(viewport={'width':390,'height':844})
        page.set_content(render(document,evidence).html)
        page.wait_for_function('document.querySelectorAll(".chart svg rect").length===3')
        widths=page.locator('.chart svg rect').evaluate_all('(elements)=>elements.map(el=>Number(el.getAttribute("width")))')
        assert widths[0]==0
        assert widths[1]/widths[2]==pytest.approx(0.000001)
        assert page.locator('.chart-data table tbody tr').count()==3
        browser.close()


def test_custom_fragment_class_names_and_hidden_controls_do_not_become_host_views():
    document=Document.model_validate({'kind':'slides','title':'Mixed content','sections':[
        {'id':'one','blocks':[{'type':'html','html':'<section class="section" id="custom-section"><div class="chart" id="custom-chart">Custom diagram</div><button hidden id="custom-hidden">Hidden control</button></section>'}]},
        {'id':'two','blocks':[{'type':'prose','text':'Second slide'}]}]})
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch();page=browser.new_page();errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.set_content(render(document,Evidence()).html)
        assert page.locator('#custom-chart').inner_text()=='Custom diagram'
        assert page.locator('#custom-section').evaluate('(el)=>getComputedStyle(el).paddingTop')=='0px'
        page.get_by_role('button',name='Present',exact=True).click()
        assert page.get_by_role('status').inner_text()=='1 / 2'
        page.get_by_role('button',name='Next slide').click()
        assert page.get_by_role('status').inner_text()=='2 / 2'
        assert page.get_by_role('button',name='Next slide').is_disabled()
        page.emulate_media(media='print')
        assert page.locator('#one').is_visible() and page.locator('#two').is_visible()
        assert not page.locator('#custom-hidden').is_visible()
        assert not errors
        browser.close()


@pytest.mark.parametrize('kind',['bar','line','scatter'])
def test_percentage_geometry_and_axis_use_the_same_units(kind):
    data=Evidence.model_validate({'datasets':{'sample':{'source':{'label':'Synthetic percentage scales'},'rows':[
        {'x':1,'fraction':.5,'points':50},{'x':2,'fraction':.8,'points':80}]}}})
    doc=Document.model_validate({'title':'Percentages','sections':[{'id':'one','blocks':[
        {'type':'chart','title':'Equivalent scales','kind':kind,'query':'sample','x':'x','series':[
            {'field':'fraction','label':'Fraction','format':{'percent':'ratio'}},
            {'field':'points','label':'Points','format':{'percent':'points'}}]}]}]})
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch();page=browser.new_page(viewport={'width':390,'height':844});errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.set_content(render(doc,data).html)
        assert any('%' in text for text in page.locator('.chart .x-ticks span' if kind=='bar' else '.chart .y-ticks span').all_text_contents())
        marks=page.locator('.chart rect' if kind=='bar' else '.chart [data-point]')
        dimensions=marks.evaluate_all('(els)=>els.map(el=>el.getAttribute("width") || el.getAttribute("y1"))')
        if kind=='bar':
            assert dimensions[0]==dimensions[1] and dimensions[2]==dimensions[3]
        else:
            assert dimensions[0]==dimensions[2] and dimensions[1]==dimensions[3]
        assert not errors and not page.evaluate('document.documentElement.scrollWidth>innerWidth')
        browser.close()


def test_dark_report_prints_with_legible_text_and_chart_colors():
    def luminance(rgb):
        channels=[int(n)/255 for n in rgb.removeprefix('rgb(').removesuffix(')').split(',')]
        return sum(weight*(value/12.92 if value<=.04045 else ((value+.055)/1.055)**2.4) for weight,value in zip([.2126,.7152,.0722],channels))
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch();page=browser.new_page(viewport={'width':1440,'height':900})
        page.set_content(specimen(dark=True))
        assert page.locator('.chart rect').count()==5
        page.emulate_media(media='print')
        color=page.locator('.source').first.evaluate('(el)=>getComputedStyle(el).color')
        assert 1.05/(luminance(color)+.05)>=4.5
        assert page.locator('.chart rect').first.evaluate('(el)=>getComputedStyle(el).fill')=='rgb(18, 104, 92)'
        assert page.locator('body').evaluate('(el)=>getComputedStyle(el).backgroundColor')=='rgb(255, 255, 255)'
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


def test_composite_view_uses_lookup_data_without_extra_hidden_records():
    data=Evidence.model_validate({'datasets':{
        'items':{'source':{'label':'Synthetic items'},'rows':[{'id':'a','category':'x'},{'id':'b','category':'z'}]},
        'categories':{'source':{'label':'Synthetic lookup'},'rows':[{'id':'x','label':'North'},{'id':'y','label':'Unused'},{'id':'z','label':'South'}]}}})
    doc=Document.model_validate({'title':'Composite view','bindings':{
        'items':{'dataset':'items','columns':['id','category']},'categories':{'dataset':'categories','columns':['id','label']}},
        'views':[{'selector':'.item','binding':'items','minimum':'rows','inputs':['categories']}],
        'sections':[{'id':'items','blocks':[{'type':'html','html':'<div id="items-list"></div><script>for(const row of ARTIFACT_DATA.items){const el=document.createElement("p");el.className="item";el.textContent=row.id+": "+ARTIFACT_DATA.categories.find(c=>c.id===row.category).label;document.getElementById("items-list").append(el);}</script>'}]}]})
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch();page=browser.new_page();errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.on('console',lambda message:errors.append(message.text) if message.type=='error' else None)
        page.set_content(render(doc,data).html);page.wait_for_timeout(450)
        assert page.locator('.item').all_text_contents()==['a: North','b: South']
        assert not page.get_by_role('alert').count() and not errors
        assert 'Unused' not in page.locator('body').inner_text()
        browser.close()


@pytest.mark.parametrize('field_count',[4,6])
def test_wide_tables_recompose_as_complete_labelled_records_on_phones(field_count):
    columns=['Record','Category','Site','Date','Amount','Explanation'][:field_count]
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


@pytest.mark.parametrize('width',[390,641,1440])
def test_tabs_switch_exact_compiled_views_with_keyboard_and_print_all_panes(width):
    from tests.test_static_artifact_render_v2 import tab_specimen
    doc,data=tab_specimen();html=render(doc,data).html
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch();page=browser.new_page(viewport={'width':width,'height':900})
        errors=[];requests=[];page.on('pageerror',lambda error:errors.append(str(error)))
        page.route('**/*',lambda route:(requests.append(route.request.url),route.abort()))
        page.set_content(html)
        assert page.get_by_role('tabpanel').count()==1
        assert page.get_by_role('tab',name='Alternative',exact=True).get_attribute('aria-selected')=='true'
        assert page.locator('.tab-panel:visible .metric-value').inner_text()=='9'
        page.get_by_role('tab',name='Observed',exact=True).click()
        assert page.locator('.tab-panel:visible .metric-value').inner_text()=='3'
        page.keyboard.press('End')
        assert page.get_by_role('tab',name='Alternative',exact=True).evaluate('(el)=>el===document.activeElement')
        assert page.locator('.tab-panel:visible .chart rect').count()==2
        page.keyboard.press('ArrowRight')
        assert page.locator('.tab-panel:visible .metric-value').inner_text()=='3'
        assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
        page.emulate_media(media='print')
        assert page.locator('.tab-panel:visible').count()==2
        assert page.locator('.tab-heading:visible').count()==2
        assert not page.get_by_role('tablist').is_visible()
        assert not errors and not requests
        browser.close()


def test_tabs_without_javascript_expose_every_compiled_pane_and_no_inert_controls():
    from tests.test_static_artifact_render_v2 import tab_specimen
    doc,data=tab_specimen()
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch();page=browser.new_page(java_script_enabled=False,viewport={'width':390,'height':900})
        page.set_content(render(doc,data).html)
        assert page.locator('.tab-panel:visible').count()==2
        assert page.locator('.tab-heading:visible').count()==2
        assert not page.locator('.tab-list').is_visible()
        assert page.locator('.chart rect').count()==4
        assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
        browser.close()


def test_french_reader_keeps_slide_navigation_and_missing_data_localized():
    doc=Document.model_validate({'kind':'slides','language':'fr','title':'Présentation','sections':[
        {'id':'opening','title':'Début','blocks':[{'type':'prose','text':'Une observation.'}]},
        {'id':'conclusion','title':'Conclusion','blocks':[{'type':'prose','text':'La suite.'}]}]})
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch();page=browser.new_page(viewport={'width':390,'height':844})
        page.set_content(render(doc,Evidence()).html)
        page.get_by_role('button',name='Présenter',exact=True).click()
        page.get_by_role('button',name='Diapositive suivante',exact=True).click()
        assert page.locator('#conclusion').is_visible() and not page.locator('#opening').is_visible()
        page.get_by_role('button',name='Lire',exact=True).click()
        assert page.locator('#opening').is_visible()
        assert page.get_by_role('button',name='Imprimer / PDF',exact=True).is_visible()
        assert 'Aucune source de données externe.' in page.locator('body').inner_text()
        browser.close()


def test_dense_chart_has_one_keyboard_stop_then_exact_data_inspection():
    data=Evidence.model_validate({'datasets':{'sample':{'source':{'label':'Synthetic keyboard QA'},'rows':[
        {'name':str(n),'amount':n} for n in range(200)]}}})
    doc=Document.model_validate({'title':'Dense chart','sections':[{'id':'plot','blocks':[
        {'type':'chart','kind':'line','title':'Observed amounts','query':'sample','x':'name','series':[{'field':'amount','label':'Amount'}]}]}]})
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch();page=browser.new_page(viewport={'width':390,'height':900})
        page.set_content(render(doc,data).html)
        assert page.locator('[data-point]').count()==200
        assert page.locator('.chart [tabindex]').count()==0
        page.get_by_role('img',name='Observed amounts',exact=True).focus()
        page.keyboard.press('Tab')
        assert page.locator('.chart-data summary').evaluate('(el)=>el===document.activeElement')
        page.keyboard.press('Enter');page.keyboard.press('Tab')
        assert page.get_by_role('region',name='Data table',exact=True).evaluate('(el)=>el===document.activeElement')
        assert page.locator('.chart-data tbody tr').count()==200
        browser.close()
