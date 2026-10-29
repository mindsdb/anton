import json
import re
from decimal import Decimal

import pytest

from anton.core.artifacts.static.data import DataError, Evidence
from anton.core.artifacts.static.document import Document
from anton.core.artifacts.static.render import render
from anton.core.tools.generate_artifact.verifiers import verify_frontend


def document(blocks, **kwargs):
    return Document.model_validate({'title':'A report', 'sections':[{'id':'one', 'blocks':blocks}], **kwargs})


@pytest.mark.parametrize('percent',['ratio','points'])
@pytest.mark.parametrize('extra',[{'prefix':'%'},{'suffix':'%'}])
def test_percentage_format_cannot_repeat_its_symbol(percent,extra):
    from anton.core.artifacts.static.document import Format
    with pytest.raises(ValueError,match='already append'):
        Format(percent=percent,**extra)


def test_percentage_fact_cannot_gain_a_literal_percentage_symbol_in_text():
    source=Evidence.model_validate({'datasets':{'sample':{'source':{'label':'Synthetic percentage'},'rows':[{'rate':72}]}}})
    facts={'rate':{'type':'query','query':{'dataset':'sample','columns':['rate']},'field':'rate','format':{'percent':'points'}}}
    doc=document([{'type':'prose','text':'Observed: {{rate}}%.'}],facts=facts)
    with pytest.raises(DataError,match='already includes'):
        render(doc,source)
    doc.sections[0].blocks[0].text='Observed: {{rate}}; target: 75%.'
    assert 'Observed: 72%; target: 75%.' in render(doc,source).html


@pytest.mark.parametrize('value',[Decimal('1e500'),Decimal('1e-500')])
def test_chart_rejects_values_the_browser_would_overflow_or_turn_into_zero(value):
    evidence=Evidence.model_validate({'datasets':{'sample':{'source':{'label':'Synthetic numeric QA'},'rows':[{'name':'A','value':value}]}}})
    doc=document([{'type':'chart','title':'Amounts','query':'sample','x':'name','series':[{'field':'value','label':'Amount'}]}])
    with pytest.raises(DataError,match='rescale'):
        render(doc,evidence)


@pytest.mark.parametrize('values',[[Decimal('1e-323')],[-1e308,1e308],[1.79e308]])
def test_chart_rejects_an_unrepresentable_tick_extent(values):
    source=Evidence.model_validate({'datasets':{'sample':{'source':{'label':'Extreme numeric QA'},'rows':[{'name':str(i),'value':v} for i,v in enumerate(values)]}}})
    doc=document([{'type':'chart','title':'Extreme range','query':'sample','x':'name','series':[{'field':'value','label':'Value'}]}])
    with pytest.raises(DataError,match='rescale'):
        render(doc,source)


def test_shared_chart_axis_rejects_incompatible_units():
    with pytest.raises(ValueError,match='compatible units'):
        document([{'type':'chart','title':'Incompatible','query':'sample','x':'name','series':[
            {'field':'amount','label':'Amount'}, {'field':'rate','label':'Rate','format':{'percent':'ratio'}}]}])


def evidence():
    return Evidence.model_validate({'datasets':{'rows':{'source':{'label':'Uploaded file'}, 'rows':[
        {'name':'</script><script>alert(1)</script>', 'amount':9007199254740993, 'private':'DO_NOT_EXPORT'}]}}})


def test_portable_html_escapes_text_and_scripts_and_projects_private_columns():
    doc=document([{'type':'table', 'title':'<Unsafe title>', 'query':{'dataset':'rows','columns':['name','amount']},
                   'columns':[{'field':'name','label':'Name'},{'field':'amount','label':'Amount','format':{}}]}])
    result=render(doc,evidence())
    assert 'DO_NOT_EXPORT' not in result.html
    assert '<script>alert(1)</script>' not in result.html
    assert '&lt;Unsafe title&gt;' in result.html
    assert '9,007,199,254,740,993' in result.html
    assert verify_frontend(result.html, is_fullstack=False).ok
    match=re.search(r'id="_artifact-bindings">(.*?)</script>',result.html,re.S)
    assert json.loads(match[1])['bindings']['_view0'][0]['amount']==9007199254740993


def test_calculated_public_rate_does_not_publish_input_only_fields():
    source=Evidence.model_validate({'datasets':{'rows':{'source':{'label':'Synthetic rates'},'rows':[
        {'name':'A','private_actual':1,'private_plan':2,'private_note':'DO_NOT_EXPORT'}]}}})
    query={'dataset':'rows','columns':['name','rate'],'calculations':[{'name':'rate','op':'divide','fields':['private_actual','private_plan']}]}
    doc=document([{'type':'table','title':'Rates','query':query,'columns':[
        {'field':'name','label':'Name'},{'field':'rate','label':'Rate','format':{'percent':'ratio','decimals':1}}]}])
    result=render(doc,source)
    assert '50.0%' in result.html
    assert all(value not in result.html for value in ['private_actual','private_plan','DO_NOT_EXPORT'])
    match=re.search(r'id="_artifact-bindings">(.*?)</script>',result.html,re.S)
    assert json.loads(match[1])['bindings']['_view0']==[{'name':'A','rate':'0.5'}]


def test_aggregated_calculation_can_render_without_publishing_measure_inputs():
    source=Evidence.model_validate({'datasets':{'rows':{'source':{'label':'Synthetic grouped rates'},'rows':[
        {'name':'A','actual':1,'plan':2},{'name':'A','actual':2,'plan':4}]}}})
    query={'dataset':'rows','group_by':['name'],'measures':[
        {'name':'done','op':'sum','field':'actual'},{'name':'target','op':'sum','field':'plan'}],
        'calculations':[{'name':'rate','op':'divide','fields':['done','target']}]}
    doc=document([{'type':'chart','title':'Grouped rates','query':query,'x':'name','series':[
        {'field':'rate','label':'Rate','format':{'percent':'ratio'}}]}])
    result=render(doc,source)
    match=re.search(r'id="_artifact-bindings">(.*?)</script>',result.html,re.S)
    assert json.loads(match[1])['bindings']['_view0']==[{'name':'A','rate':'0.5'}]


def test_flow_groups_adjacent_metrics_without_crossing_other_content_or_losing_facts():
    facts={name:{'type':'query','query':{'dataset':'rows','columns':['amount']},'field':'amount'} for name in ['one','two','three']}
    doc=document([{'type':'metric','label':'One','fact':'one'},{'type':'metric','label':'Two','fact':'two'},
        {'type':'prose','text':'Keep this between the metric groups.'},{'type':'metric','label':'Three','fact':'three'}],facts=facts)
    result=render(doc,evidence())
    assert result.html.count('<div class="metric-row">')==1
    assert result.html.count('<div class="metric">')==3
    assert '</div></div><div class="prose"><p>Keep this between' in result.html
    assert set(result.manifest['facts'])==set(facts)
    doc.sections[0].layout='split'
    assert '<div class="metric-row">' not in render(doc,evidence()).html


def test_custom_binding_is_available_before_fragment_scripts_execute():
    doc=document([{'type':'html','html':'<script>window.test=JSON.parse(document.getElementById("_artifact-bindings").textContent).bindings.choices;</script>'}],
                 bindings={'choices':{'dataset':'rows','columns':['amount']}},views=[{'selector':'body','binding':'choices','minimum':'one'}])
    result=render(doc,evidence())
    assert result.html.index('id="_artifact-bindings"') < result.html.index('window.test=')
    assert 'DO_NOT_EXPORT' not in result.html


def test_chart_rejects_numeric_strings_and_unselected_fields():
    block={'type':'chart','title':'Amount','query':{'dataset':'rows','columns':['name','amount']},'x':'name','series':[{'field':'amount','label':'Amount'}]}
    bad=evidence();bad.datasets['rows'].rows[0]['amount']='100'
    with pytest.raises(DataError,match='Expected a number'):
        render(document([block]),bad)
    block['series'][0]['field']='private'
    with pytest.raises(DataError,match='not selected'):
        render(document([block]),evidence())


def test_unreferenced_facts_are_not_exported():
    doc=document([{'type':'prose','text':'Nothing quantitative here.'}], facts={
        'secret':{'type':'query','query':{'dataset':'rows','columns':['private']},'field':'private'}})
    result=render(doc,evidence())
    assert 'DO_NOT_EXPORT' not in result.html
    assert result.manifest['facts']=={}
    assert result.manifest['sources']=={}


def test_missing_binding_fails_instead_of_shipping_placeholder():
    with pytest.raises(DataError,match='Unknown fact'):
        render(document([{'type':'prose','text':'Total: {{unknown}}'}]),Evidence())


def test_null_chart_gap_and_exact_chart_labels():
    block={'type':'chart','title':'Amount','query':{'dataset':'rows','columns':['name','amount']},'x':'name','series':[{'field':'amount','label':'Amount'}]}
    result=render(document([block]),evidence())
    spec=json.loads(re.search(r'class="chart-spec">(.*?)</script>',result.html,re.S)[1])
    assert spec['display']==[['9,007,199,254,740,993']]
    assert 'src="https://' not in result.html and '@import' not in result.html


def test_visuals_can_reference_dataset_or_private_reusable_query():
    doc=document([{'type':'table','title':'Amounts','query':'public_rows','columns':[{'field':'amount','label':'Amount'}]}],
                 queries={'public_rows':{'dataset':'rows','columns':['amount']}})
    result=render(doc,evidence())
    assert '9007199254740993' in result.html and 'DO_NOT_EXPORT' not in result.html
    doc.sections[0].blocks[0].query='rows'
    assert render(doc,evidence()).manifest['bindings']['_view0']==[{'amount':9007199254740993}]


def test_host_ids_do_not_reserve_natural_section_names_and_duplicate_custom_ids_fail():
    doc=Document.model_validate({'title':'Title','sections':[{'id':'title','blocks':[{'type':'prose','text':'A title section.'}]}]})
    assert 'id="_artifact-title"' in render(doc,Evidence()).html
    from anton.core.artifacts.static.document import Html
    doc.sections[0].blocks=[Html(type='html',html='<div id="title"></div>')]
    with pytest.raises(DataError,match='Duplicate HTML id'):
        render(doc,Evidence())


def test_line_chart_rejects_ambiguous_text_sorting_instead_of_drawing_false_chronology():
    block={'type':'chart','kind':'line','title':'Trend','x':'name','query':{'dataset':'rows','columns':['name','amount'],'order_by':[{'field':'name'}]},'series':[{'field':'amount','label':'Amount'}]}
    with pytest.raises(DataError,match='Preserve source order'):
        render(document([block]),evidence())


def test_custom_views_default_to_one_component_and_can_require_all_records():
    from anton.core.artifacts.static.views import View, checks
    rows={'observations':[{'id':'a'},{'id':'b'}]}
    assert checks(rows,[View(selector='.chart',binding='observations')])[0]['minimum']==1
    assert checks(rows,[View(selector='.record',binding='observations',minimum='rows')])[0]['minimum']==2


def test_composite_views_check_observations_without_inventing_lookup_marks():
    from anton.core.artifacts.static.views import View, checks
    rows={'items':[{'id':'a'},{'id':'b'}],'categories':[{'id':n} for n in range(10)]}
    assert checks(rows,[View(selector='.item',binding='items',minimum='rows',inputs=['categories'])])==[{'selector':'.item','minimum':2}]
    with pytest.raises(DataError,match='Unknown view inputs'):
        checks(rows,[View(selector='.item',binding='items',inputs=['categories','missing'])])
    with pytest.raises(DataError,match='Custom bindings need a render check'):
        checks({**rows,'unused':[]},[View(selector='.item',binding='items',inputs=['categories'])])


def tab_specimen():
    data=Evidence.model_validate({'datasets':{'observations':{'source':{'label':'Synthetic tab QA'},'rows':[
        {'condition':'Observed','name':'A','amount':1,'private':'KEEP_PRIVATE'},
        {'condition':'Observed','name':'B','amount':2,'private':'KEEP_PRIVATE'},
        {'condition':'Alternative','name':'A','amount':4,'private':'KEEP_PRIVATE'},
        {'condition':'Alternative','name':'B','amount':5,'private':'KEEP_PRIVATE'}]}}})
    bindings={key:{'dataset':'observations','columns':['name','amount'],'where':[{'field':'condition','value':label}]} for key,label in [('observed','Observed'),('alternative','Alternative')]}
    facts={key:{'type':'query','query':{'dataset':'observations','where':[{'field':'condition','value':label}],'measures':[{'name':'total','op':'sum','field':'amount'}]},'field':'total'} for key,label in [('observed','Observed'),('alternative','Alternative')]}
    panes=[{'label':label,'blocks':[{'type':'metric','label':'Total','fact':key},
        {'type':'chart','title':label+' amounts','query':key,'x':'name','series':[{'field':'amount','label':'Amount'}]},
        {'type':'table','title':label+' records','query':key,'columns':[{'field':'name','label':'Name'},{'field':'amount','label':'Amount'}]}]} for key,label in [('observed','Observed'),('alternative','Alternative')]]
    doc=document([{'type':'tabs','title':'Compare conditions','initial':1,'panes':panes}],bindings=bindings,facts=facts)
    return doc,data


def test_tabs_publish_exact_filtered_views_and_project_private_input_fields():
    doc,data=tab_specimen();result=render(doc,data)
    assert result.manifest['facts']=={'observed':Decimal(3),'alternative':Decimal(9)}
    assert result.manifest['bindings']['alternative']==[{'name':'A','amount':4},{'name':'B','amount':5}]
    assert not result.manifest['view_checks']  # Native blocks cover their named bindings.
    assert 'KEEP_PRIVATE' not in result.html
    assert result.html.count('class="tab-panel"')==2
    assert not re.search(r'class="tab-panel"[^>]*hidden',result.html)
    assert re.search(r'class="tab-list"[^>]*hidden',result.html)
    assert verify_frontend(result.html,is_fullstack=False).ok


@pytest.mark.parametrize('change',[{'initial':2},{'panes':[{'label':'Duplicate','blocks':[{'type':'prose','text':'One'}]},{'label':'Duplicate','blocks':[{'type':'prose','text':'Two'}]}]}])
def test_tabs_reject_missing_initial_pane_or_ambiguous_labels(change):
    doc,data=tab_specimen();payload=doc.model_dump();payload['sections'][0]['blocks'][0].update(change)
    with pytest.raises(ValueError):Document.model_validate(payload)


@pytest.mark.parametrize('language,expected',[('fr-CA',['Sommaire','Voir les données','Non disponible']),('es',['Contenido','Ver datos','No disponible'])])
def test_reader_labels_follow_document_language_including_missing_values(language,expected):
    data=Evidence.model_validate({'datasets':{'sample':{'source':{'label':'Synthetic localization QA'},'rows':[{'category':'A','value':None}]}}})
    doc=document([{'type':'chart','title':'Observations','query':'sample','x':'category','series':[{'field':'value','label':'Value'}]}],language=language)
    result=render(doc,data)
    assert all(value in result.html for value in expected)
    assert 'No observed numeric values.' not in result.html
    assert '>View data<' not in result.html and '>Contents<' not in result.html


def test_reader_label_overrides_are_escaped_and_reject_unknown_keys():
    from anton.core.artifacts.static.labels import EN,TRANSLATIONS
    assert all(set(labels)==set(EN) for labels in TRANSLATIONS.values())
    doc=document([{'type':'prose','text':'An observation'}],language='de',labels={'contents':'Inhalt <script>','print':'Drucken'})
    result=render(doc,Evidence())
    assert 'Inhalt &lt;script&gt;' in result.html and '>Drucken<' in result.html
    assert '>Inhalt <script>' not in result.html
    with pytest.raises(ValueError):document([{'type':'prose','text':'Text'}],labels={'typo':'Label'})


def test_compiled_document_preserves_required_source_credits():
    source=Evidence.model_validate({'datasets':{'sample':{'source':{'label':'Open records','attribution':'Credit <provider>'},'rows':[{'name':'A','value':1}]}}})
    doc=document([{'type':'table','title':'Records','query':'sample','columns':[{'field':'name','label':'Name'},{'field':'value','label':'Value'}]}])
    result=render(doc,source)
    assert 'Credit &lt;provider&gt;' in result.html
    assert result.manifest['sources']['sample']['attribution']=='Credit <provider>'
