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
        assert page.locator('.section:visible').count()==0
        page.get_by_role('button',name='Next slide').click()
        page.wait_for_function('document.querySelectorAll(".chart svg").length===2')
        assert page.locator('#comparison:visible').count()==1
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
        'views':[{'selector':'.desk','binding':'desks'}], 'sections':[{'id':'desks','blocks':[{'type':'html','html':'<div id="desk-grid"></div>'}]}]})
    html=render(doc,data).html
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch();page=browser.new_page();errors=[]
        page.on('console',lambda message:errors.append(message.text) if message.type=='error' else None)
        page.set_content(html)
        page.get_by_role('alert').wait_for()
        assert any('expected at least 2' in error for error in errors)
        browser.close()
