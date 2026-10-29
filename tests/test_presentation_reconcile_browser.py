"""Shown figures are reconciled with the bound data before the page is returned."""

import pytest

from anton.core.tools.generate_artifact import component_page, observed_data

playwright = pytest.importorskip("playwright.sync_api")

CHARTS = """window.echarts={init(node){node.append(document.createElement('canvas'));return {
  setOption(){},resize(){},dispose(){},on(){},off(){},getModel(){return {eachSeries(){}}}}}};"""
ORDERS = {"orders": [{"store": "North", "amount": 120}, {"store": "South", "amount": 80}]}


def render(program, data=ORDERS):
    html = observed_data.bind(component_page.document(program), {"key": data}, checks=True)
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page()
            errors = []
            page.on("console", lambda message: message.type == "error" and errors.append(message.text))
            page.route("https://cdn.jsdelivr.net/npm/echarts@5.5.1/**",
                       lambda route: route.fulfill(status=200, content_type="application/javascript", body=CHARTS))
            page.set_content(html)
            page.wait_for_function("document.querySelector('#page-content')")
            page.wait_for_timeout(200)
            return [e for e in errors if e.startswith("Figure check:")], page.locator("#cowork-page").inner_text()
        finally:
            browser.close()


def test_a_zero_metric_needs_a_reason_and_the_reason_is_shown():
    errors, _ = render("await CoworkPage({title:'Sales',metrics:[{label:'Unmatched sales',value:CoworkNumber(0,'units')}]});")
    assert len(errors) == 1 and 'metric "Unmatched sales" shows 0 units while the bound data has rows' in errors[0]
    assert "add zeroReason" in errors[0]
    errors, text = render("await CoworkPage({title:'Sales',metrics:["
                          "{label:'Late orders',value:0,zeroReason:'No order shipped late'},{label:'Orders',value:2,zeroReason:'unused'}]});")
    assert errors == []
    assert "No order shipped late" in text and "unused" not in text


def test_nothing_is_flagged_without_source_rows():
    errors, _ = render("await CoworkPage({title:'Sales',metrics:[{label:'Orders',value:0}]});", {"orders": []})
    assert errors == []


def test_a_table_column_of_zeros_is_flagged_but_a_single_zero_cell_is_not():
    errors, _ = render("""await CoworkPage({title:'Sales',sections:[{type:'table',title:'By store',
      columns:[{key:'store',label:'Store'},{key:'amount',label:'Amount'}],
      rows:[{store:'North',amount:0},{store:'South',amount:0}]}]});""")
    assert len(errors) == 1 and 'table "By store" shows 0 in every row of column "Amount"' in errors[0]
    errors, _ = render("""await CoworkPage({title:'Sales',sections:[{type:'table',title:'By store',
      columns:[{key:'store',label:'Store'},{key:'amount',label:'Amount'}],
      rows:[{store:'North',amount:0},{store:'South',amount:80}]}]});""")
    assert errors == []
    errors, text = render("""await CoworkPage({title:'Sales',sections:[{type:'table',title:'By store',
      columns:[{key:'store',label:'Store'},{key:'refund',label:'Refunds',zeroReason:'No refunds were issued'}],
      rows:[{store:'North',refund:0},{store:'South',refund:0}]}]});""")
    assert errors == [] and "No refunds were issued" in text


def test_a_row_of_zeros_across_period_columns_is_flagged():
    program = """const weeks=['w1','w2','w3'];
    await CoworkPage({title:'Sales',sections:[{type:'table',title:'Weekly',
      columns:[{key:'store',label:'Store'},{key:'days',label:'Lead time',unit:'days'},
        ...weeks.map(w=>({key:w,label:w,unit:'units'}))],
      rows:[{store:'North',days:3,w1:4,w2:1,w3:5},{store:'South',days:2,w1:SOUTH,w2:0,w3:0}]}]});"""
    errors, _ = render(program.replace("SOUTH", "0"))
    assert len(errors) == 1 and 'shows 0 in all 3 "units" columns for row "South"' in errors[0]
    errors, _ = render(program.replace("SOUTH", "6"))
    assert errors == []


def test_empty_views_are_flagged_and_views_with_values_are_not():
    errors, _ = render("""await CoworkPage({title:'Sales',sections:[
      {type:'table',title:'Stores',columns:[{key:'store',label:'Store'}],rows:[]},
      {type:'chart',title:'Trend',option:{xAxis:{type:'category',data:['a','b']},yAxis:{type:'value'},series:[{type:'bar',data:[0,0]}]}},
      {type:'timeline',title:'Events',items:[]}]});""")
    assert [e.split(" while ")[0] for e in errors] == [
        'Figure check: table "Stores" has no rows', 'Figure check: chart "Trend" plots only 0', 'Figure check: timeline "Events" has no items']
    errors, _ = render("""await CoworkPage({title:'Sales',sections:[
      {type:'table',title:'Stores',columns:[{key:'store',label:'Store'}],rows:[{store:'North'}]},
      {type:'chart',title:'Trend',option:{xAxis:{type:'category',data:['a','b']},yAxis:{type:'value'},series:[{type:'bar',data:[0,4]}]}},
      {type:'chart',title:'Links',option:{series:[{type:'graph',data:[{name:'a'},{name:'b'}],links:[{source:'a',target:'b'}]}]}}]});""")
    assert errors == []


def test_every_control_state_is_checked():
    errors, _ = render("""await CoworkPage({title:'Sales',controls:[{id:'store',label:'Store',type:'tabs',value:'North',
      options:[{value:'North',label:'North'},{value:'South',label:'South'}]}],
      render:({store})=>({metrics:[{label:'Orders',value:store==='North'?2:0}]})});""")
    assert len(errors) == 1 and 'metric "Orders" shows 0 (Store: South)' in errors[0]


def test_a_figure_that_counts_total_rows_twice_is_flagged_with_the_correct_value():
    rows = [{"store": store, "category": category, "sales": sales}
            for store in ["North", "South", "East"]
            for category, sales in [("Books", 120), ("Games", 75), ("Music", 40), ("Total", 235)]]
    program = """const rows=window.ARTIFACT_DATA.key.sales;
    await CoworkPage({title:'Sales',metrics:[{label:'Sales',value:CoworkNumber(rows.filter(KEEP).reduce((s,r)=>s+r.sales,0))}]});"""
    errors, _ = render(program.replace("KEEP", "r=>true"), {"sales": rows})
    assert len(errors) == 1
    assert 'Figure check: 1,410 (shown in metric "Sales") counts rows twice' in errors[0]
    assert 'including elements where category = "Total"' in errors[0] and "Counted once it is 705" in errors[0]
    errors, _ = render(program.replace("KEEP", "r=>r.category!=='Total'"), {"sales": rows})
    assert errors == []
