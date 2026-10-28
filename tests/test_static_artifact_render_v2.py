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
    match=re.search(r'id="artifact-bindings">(.*?)</script>',result.html,re.S)
    assert json.loads(match[1])['bindings']['_view0'][0]['amount']==9007199254740993


def test_custom_binding_is_available_before_fragment_scripts_execute():
    doc=document([{'type':'html','html':'<script>window.test=JSON.parse(document.getElementById("artifact-bindings").textContent).bindings.choices;</script>'}],
                 bindings={'choices':{'dataset':'rows','columns':['amount']}})
    result=render(doc,evidence())
    assert result.html.index('id="artifact-bindings"') < result.html.index('window.test=')
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
