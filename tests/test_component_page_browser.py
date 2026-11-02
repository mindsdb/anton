"""Browser checks for the component page runtime without external assets."""

import runpy
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")
COMPONENT = Path(__file__).resolve().parents[1] / "anton/core/tools/generate_artifact/component_page.py"


@pytest.mark.parametrize("width", [390, 1440])
def test_overview_survives_control_changes_and_is_readable(width):
    document = runpy.run_path(str(COMPONENT))["document"]
    html = document("""await CoworkPage({
      title:'Hospital capacity', lang:'en',
      overview:{lead:{text:'All wards',value:'42 beds'},metrics:[{label:'Occupied',value:31}]},
      controls:[{id:'ward',label:'Ward',type:'tabs',value:'north',options:[
        {value:'north',label:'North'},{value:'south',label:'South'}]}],
      render:({ward})=>({sections:[{type:'text',title:'Details',text:ward==='north'?'North ward':'South ward'}]})
    });""")
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": width, "height": 844})
            page.set_content(html)
            assert page.locator("#overview-finding").inner_text().find("42 beds") >= 0
            assert "31" in page.locator("#overview-metrics").inner_text()
            page.get_by_role("button", name="South").click()
            assert "South ward" in page.locator("#page-content").inner_text()
            assert "42 beds" in page.locator("#overview-finding").inner_text()
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        finally:
            browser.close()


def test_failed_chart_library_can_load_after_a_control_change():
    document = runpy.run_path(str(COMPONENT))["document"]
    html = document("""await CoworkPage({title:'Hospital activity',controls:[
      {id:'ward',label:'Ward',type:'tabs',value:'north',options:[
        {value:'north',label:'North'},{value:'south',label:'South'}]}],
      render:({ward})=>({sections:[{type:'chart',title:ward==='north'?'North chart':'South chart',option:{}}]})
    });""")
    library = """window.echarts={init(node){node.append(document.createElement('canvas'));return {
      setOption(){},resize(){},dispose(){},on(){},off(){},
      getModel(){return {eachSeries(){}}}
    }}};"""
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page()
            attempts = []

            def serve_library(route):
                attempts.append(route.request.url)
                if len(attempts) == 1:
                    route.abort()
                else:
                    route.fulfill(status=200, content_type="application/javascript", body=library)

            page.route("https://cdn.jsdelivr.net/npm/echarts@5.5.1/**", serve_library)
            page.set_content(html)
            assert "Could not load echarts" in page.get_by_role("alert").inner_text()
            page.get_by_role("button", name="South").click()
            assert page.locator("#page-content canvas").count() == 1
            assert "South chart" in page.locator("#page-content").inner_text()
            assert len(attempts) == 2
        finally:
            browser.close()


def test_one_failed_chart_keeps_other_sections_visible():
    document = runpy.run_path(str(COMPONENT))["document"]
    html = document("""await CoworkPage({title:'Clinic activity',sections:[
      {type:'chart',title:'Visits chart',option:{}},
      {type:'text',title:'Summary',text:'37 visits were recorded'}
    ]});""")
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page()
            page.route("https://cdn.jsdelivr.net/npm/echarts@5.5.1/**", lambda route: route.abort())
            page.set_content(html)
            assert "Could not load echarts" in page.get_by_role("alert").inner_text()
            assert "37 visits were recorded" in page.locator("#page-content").inner_text()
            assert "Visits chart" in page.locator("#page-content").inner_text()
        finally:
            browser.close()


def test_superseded_chart_error_does_not_erase_newer_tab():
    document = runpy.run_path(str(COMPONENT))["document"]
    html = document("""await CoworkPage({title:'Clinic overview',controls:[
      {id:'view',label:'View',type:'tabs',value:'chart',options:[
        {value:'chart',label:'Chart'},{value:'notes',label:'Notes'}]}],
      render:({view})=>({sections:[view==='chart'
        ?{type:'chart',title:'Chart',option:{}}
        :{type:'text',title:'Notes',text:'Current clinic notes'}]})
    });""")
    # Hold the chart script request so the user can switch views before it fails.
    intercept = """<script>
      const append = document.head.append.bind(document.head);
      document.head.append = (...nodes) => {
        if (nodes[0]?.src?.includes('echarts@')) {window.pendingChartScript=nodes[0];return;}
        return append(...nodes);
      };
    </script>"""
    html = html.replace("<head>", "<head>" + intercept, 1)
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page()
            page.set_content(html)
            page.get_by_role("button", name="Notes").click()
            assert "Current clinic notes" in page.locator("#page-content").inner_text()
            page.evaluate("window.pendingChartScript.onerror()")
            assert "Current clinic notes" in page.locator("#page-content").inner_text()
            assert page.get_by_role("alert").count() == 0
        finally:
            browser.close()


def test_metric_cards_round_for_readability_without_changing_source_values_on_phone():
    document = runpy.run_path(str(COMPONENT))["document"]
    html = document("""window.sourceValues=[0.006,523533.3333333333,73.80952380952381];
      await CoworkPage({title:'Lab measurements',metrics:[
      {label:'Concentration',value:window.sourceValues[0],unit:'mg/L'},
      {label:'Unit cost',value:window.sourceValues[1],unit:'USD'},
      {label:'Rate',value:window.sourceValues[2],unit:'%'}],sections:[]});""")
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 390, "height": 844})
            page.set_content(html)
            values = page.locator(".metric-value").all_inner_texts()
            assert values == ["0.006 mg/L", "$523,533.33", "73.81%"]
            assert page.evaluate("window.sourceValues") == [0.006, 523533.3333333333, 73.80952380952381]
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        finally:
            browser.close()
