from decimal import Decimal
import json

import pytest
from pydantic import ValidationError

from anton.core.artifacts.static.data import DataError, Evidence, Query, execute


def evidence(rows):
    return Evidence.model_validate({"datasets": {"orders": {"source": {"label": "Uploaded orders"}, "rows": rows}}})


def test_authoring_json_preserves_number_types_and_exact_decimal_digits():
    from anton.core.artifacts.static.data import exact_json,json_value
    value=Decimal('0.12345678901234567890123456789')
    encoded=exact_json({'amount':value,'label':'0.12345678901234567890123456789','flag':True,'none':None})
    decoded=json.loads(encoded,parse_float=Decimal)
    assert decoded['amount']==value and isinstance(decoded['amount'],Decimal)
    assert isinstance(decoded['label'],str) and decoded['flag'] is True and decoded['none'] is None
    assert isinstance(json.loads(json_value({'amount':value}))['amount'],str)


def test_evidence_digest_distinguishes_decimal_values_from_literal_strings():
    assert evidence([{'value':Decimal('1.25')}]).digest()!=evidence([{'value':'1.25'}]).digest()


def test_decimal_arithmetic_and_grouping_preserve_missing_values():
    data = evidence([{"region": "a", "net": .1}, {"region": "a", "net": .2}, {"region": "b", "net": None}])
    q = Query.model_validate({"dataset": "orders", "group_by": ["region"],
                             "measures": [{"name": "net", "field": "net", "op": "sum"}]})
    assert execute(data, q) == [{"region": "a", "net": Decimal("0.3")}, {"region": "b", "net": None}]


def test_field_validation_precedes_filter_and_never_drops_incomplete_rows():
    data = evidence([{"region": "a", "net": 2}, {"region": "b"}])
    q = Query.model_validate({"dataset": "orders", "columns": ["net"], "where": [{"field": "region", "value": "a"}]})
    with pytest.raises(DataError, match="missing fields"):
        execute(data, q)


def test_projection_does_not_export_unselected_fields():
    data = evidence([{"name": "one", "private": "secret", "amount": 3}])
    q = Query(dataset="orders", columns=["name", "amount"])
    assert execute(data, q) == [{"name": "one", "amount": 3}]
    assert data.datasets["orders"].rows[0]["private"] == "secret"


@pytest.mark.parametrize('op,value,expected',[
    ('eq',.1,['decimal','float']),('ne',.1,['other','text','flag']),
    ('in',[.1],['decimal','float']),('eq','0.1',['text']),('in',[True],['flag'])])
def test_numeric_equality_filters_compare_decimal_values_without_string_or_boolean_coercion(op,value,expected):
    data=evidence([{'id':'decimal','n':Decimal('0.1')},{'id':'float','n':.1},{'id':'other','n':.2},
        {'id':'text','n':'0.1'},{'id':'flag','n':True}])
    query=Query(dataset='orders',columns=['id'],where=[{'field':'n','op':op,'value':value}])
    assert [row['id'] for row in execute(data,query)]==expected


def test_explicit_decimal_casts_preserve_source_text_and_exact_aggregate_precision():
    data=evidence([{'id':'0007','reading':'1234567890.1234567890123456789'},
        {'id':'0012','reading':'0.0000000000000000001'}]);before=data.digest()
    casts=[{'name':'numeric_reading','field':'reading','type':'decimal'}]
    rows=execute(data,Query(dataset='orders',columns=['id','reading','numeric_reading'],casts=casts))
    assert rows[0]=={'id':'0007','reading':'1234567890.1234567890123456789','numeric_reading':Decimal('1234567890.1234567890123456789')}
    assert execute(data,Query(dataset='orders',casts=casts,measures=[{'name':'total','op':'sum','field':'numeric_reading'}]))==[{'total':Decimal('1234567890.1234567890123456790')}]
    assert data.digest()==before


def test_casts_run_before_numeric_filters_and_keep_unselected_fields_private():
    data=evidence([{'id':'a','raw':'.10'},{'id':'b','raw':'0.2'}])
    query=Query(dataset='orders',columns=['id'],casts=[{'name':'n','field':'raw'}],where=[{'field':'n','value':.1}])
    assert execute(data,query)==[{'id':'a'}]
    query=Query(dataset='orders',casts=[{'name':'n','field':'raw'}],group_by=['n'],measures=[{'name':'count','op':'count'}])
    assert execute(data,query)==[{'n':Decimal('.1'),'count':1},{'n':Decimal('.2'),'count':1}]


@pytest.mark.parametrize('value',['12,5','$12','NaN','Infinity','not measured',True])
def test_explicit_cast_rejects_ambiguous_text_and_booleans(value):
    with pytest.raises(DataError):
        execute(evidence([{'raw':value}]),Query(dataset='orders',columns=['n'],casts=[{'name':'n','field':'raw'}]))


def test_cast_nulls_and_empty_results_keep_existing_missingness_semantics():
    query=Query(dataset='orders',casts=[{'name':'n','field':'raw'}],measures=[{'name':'total','op':'sum','field':'n'}])
    assert execute(evidence([{'raw':None},{'raw':'2'}]),query)==[{'total':None}]
    assert execute(evidence([]),query)==[{'total':None}]


@pytest.mark.parametrize('casts',[
    [{'name':'raw','field':'raw'}],
    [{'name':'other','field':'raw'}],
    [{'name':'first','field':'later'},{'name':'later','field':'raw'}],
])
def test_casts_do_not_overwrite_source_fields_or_read_future_casts(casts):
    with pytest.raises(DataError):
        execute(evidence([{'raw':'1','other':'2'}]),Query(dataset='orders',columns=['raw'],casts=casts))


def test_cast_field_coverage_is_checked_before_filtering():
    query=Query(dataset='orders',columns=['id'],casts=[{'name':'n','field':'raw'}],where=[{'field':'id','value':'a'}])
    with pytest.raises(DataError,match='missing field'):
        execute(evidence([{'id':'a','raw':'1'},{'id':'b'}]),query)


def test_query_calculations_resolve_rates_and_variances_before_public_projection():
    data=evidence([{'name':'A','actual':Decimal('0.3'),'private_plan':Decimal('0.2')},
        {'name':'B','actual':4,'private_plan':0}, {'name':'C','actual':None,'private_plan':3}])
    before=data.model_dump()
    query=Query.model_validate({'dataset':'orders','columns':['name','rate','variance'],
        'calculations':[{'name':'rate','op':'divide','fields':['actual','private_plan']},
            {'name':'variance','op':'subtract','fields':['actual','private_plan']}],
        'order_by':[{'field':'rate','descending':True}]})
    assert execute(data,query)==[{'name':'A','rate':Decimal('1.5'),'variance':Decimal('0.1')},
        {'name':'B','rate':None,'variance':Decimal(4)}, {'name':'C','rate':None,'variance':None}]
    assert data.model_dump()==before


def test_calculation_can_divide_grouped_totals_and_chain_earlier_outputs():
    data=evidence([{'team':'a','actual':1,'plan':2},{'team':'a','actual':4,'plan':8},
        {'team':'b','actual':9,'plan':10}])
    query=Query.model_validate({'dataset':'orders','group_by':['team'],
        'measures':[{'name':'done','op':'sum','field':'actual'},{'name':'target','op':'sum','field':'plan'}],
        'calculations':[{'name':'rate','op':'divide','fields':['done','target']},
            {'name':'squared','op':'multiply','fields':['rate','rate']}],
        'order_by':[{'field':'rate','descending':True}]})
    assert execute(data,query)==[{'team':'b','done':Decimal(9),'target':Decimal(10),'rate':Decimal('.9'),'squared':Decimal('.81')},
        {'team':'a','done':Decimal(5),'target':Decimal(10),'rate':Decimal('.5'),'squared':Decimal('.25')}]


def test_calculation_input_coverage_is_checked_before_source_filtering():
    query=Query.model_validate({'dataset':'orders','columns':['rate'],'where':[{'field':'name','value':'A'}],
        'calculations':[{'name':'rate','op':'divide','fields':['actual','plan']}]})
    with pytest.raises(DataError,match='missing fields'):
        execute(evidence([{'name':'A','actual':1,'plan':2},{'name':'B','actual':3}]),query)


@pytest.mark.parametrize('calculations',[
    [{'name':'actual','op':'add','fields':['plan','plan']}],
    [{'name':'one','op':'add','fields':['two','actual']},{'name':'two','op':'add','fields':['actual','plan']}],
    [{'name':'one','op':'divide','fields':['one','actual']}],
])
def test_query_calculations_reject_shadowing_and_forward_or_cyclic_references(calculations):
    query=Query(dataset='orders',columns=['actual'],calculations=calculations)
    with pytest.raises(DataError):
        execute(evidence([{'actual':1,'plan':2}]),query)


@pytest.mark.parametrize('value',['4',True])
def test_query_calculation_does_not_coerce_nonnumeric_inputs(value):
    query=Query(dataset='orders',columns=['rate'],calculations=[{'name':'rate','op':'divide','fields':['actual','plan']}])
    with pytest.raises(DataError,match='Expected a number'):
        execute(evidence([{'actual':value,'plan':2}]),query)


def test_aggregate_calculation_cannot_read_unaggregated_source_fields():
    query=Query(dataset='orders',measures=[{'name':'total','op':'sum','field':'actual'}],
        calculations=[{'name':'rate','op':'divide','fields':['total','plan']}])
    with pytest.raises(DataError,match='query output fields'):
        execute(evidence([{'actual':1,'plan':2}]),query)


def test_empty_query_calculations_do_not_invent_records_or_zero_totals():
    calc=[{'name':'rate','op':'divide','fields':['actual','plan']}]
    assert execute(evidence([]),Query(dataset='orders',columns=['rate'],calculations=calc))==[]
    query=Query(dataset='orders',measures=[{'name':field,'op':'sum','field':field} for field in ['actual','plan']],calculations=calc)
    assert execute(evidence([]),query)==[{'actual':None,'plan':None,'rate':None}]


@pytest.mark.parametrize("value", ["12", True])
def test_numeric_fields_are_not_silently_coerced(value):
    q = Query.model_validate({"dataset": "orders", "measures": [{"name": "n", "op": "sum", "field": "net"}]})
    with pytest.raises(DataError, match="Expected a number"):
        execute(evidence([{"net": value}]), q)


def test_empty_sum_is_missing_but_count_is_zero():
    q = Query.model_validate({"dataset": "orders", "measures": [
        {"name": "net", "op": "sum", "field": "net"}, {"name": "rows", "op": "count"}]})
    assert execute(evidence([]), q) == [{"net": None, "rows": 0}]


def test_group_keys_do_not_merge_boolean_and_number():
    q = Query.model_validate({"dataset": "orders", "group_by": ["key"], "measures": [{"name": "n", "op": "count"}]})
    assert len(execute(evidence([{"key": True}, {"key": 1}]), q)) == 2


@pytest.mark.parametrize('value',[float('nan'),float('inf'),Decimal('Infinity'),Decimal('NaN')])
def test_nonfinite_evidence_rejected(value):
    with pytest.raises(ValidationError, match="non-finite"):
        evidence([{"net": value}])


def test_finite_decimal_evidence_does_not_overflow_a_float_conversion():
    value=Decimal('1e500')
    data=evidence([{'net':value}])
    query=Query(dataset='orders',measures=[{'name':'total','op':'sum','field':'net'}])
    assert execute(data,query)==[{'total':value}]


@pytest.mark.parametrize('value,scale,expected',[(Decimal('.125'),'ratio','12.5%'),(Decimal('12.5'),'points','12.5%'),
    (Decimal('1.5'),'ratio','150.0%'),(Decimal('150'),'points','150.0%'),(Decimal('-12.5'),'points','-12.5%'),
    (Decimal('0'),'ratio','0.0%'),(Decimal('12.5'),'none','12.5')])
def test_percentage_input_scale_is_explicit(value,scale,expected):
    from anton.core.artifacts.static.document import Format
    assert Format(percent=scale,decimals=1).display(value)==expected


def test_ambiguous_boolean_percent_format_is_rejected():
    from anton.core.artifacts.static.document import Format
    with pytest.raises(ValidationError):
        Format(percent=True)


@pytest.mark.parametrize('value,expected',[(Decimal('12.5'),'12.5'),(Decimal('1250.75'),'1,250.75'),
    (Decimal('42.00'),'42'),(Decimal('0.000123'),'0.000123'),(Decimal('0.333333333333'),'0.333333'),
    (Decimal('0.0000001234'),'1.234e-7'),(Decimal('-1e-500'),'-1e-500')])
def test_automatic_precision_keeps_fractional_values_without_forcing_decimals_on_counts(value,expected):
    from anton.core.artifacts.static.document import Format
    assert Format().display(value)==expected


def test_sorted_projection_filters_and_nulls():
    q = Query.model_validate({"dataset": "orders", "columns": ["net"], "where": [{"field": "region", "value": ["a"], "op": "in"}],
                             "order_by": [{"field": "net", "descending": True}]})
    assert execute(evidence([{"region": "a", "net": None}, {"region": "a", "net": 3}, {"region": "b", "net": 9}]), q) == [{"net": 3}, {"net": None}]


def test_fact_graph_formats_ratios_without_model_arithmetic():
    from anton.core.artifacts.static.document import Document, resolve_facts
    doc = Document.model_validate({"title": "Orders", "facts": {
        "total": {"type": "query", "query": {"dataset": "orders", "measures": [{"name": "n", "op": "sum", "field": "net"}]}, "field": "n"},
        "count": {"type": "query", "query": {"dataset": "orders", "measures": [{"name": "n", "op": "count"}]}, "field": "n"},
        "average": {"type": "calculation", "op": "divide", "inputs": ["total", "count"], "format": {"decimals": 2}}},
        "sections": [{"id": "summary", "blocks": [{"type": "metric", "label": "Average", "fact": "average"}]}]})
    values = resolve_facts(doc, evidence([{"net": .1}, {"net": .2}]))
    assert values["average"] == Decimal("0.15")
    assert doc.facts["average"].format.display(values["average"]) == "0.15"
    assert resolve_facts(doc, evidence([]))["average"] is None


def test_fact_cycles_and_unknown_references_fail_explicitly():
    from anton.core.artifacts.static.document import Document, resolve_facts
    doc = Document.model_validate({"title": "Test", "facts": {
        "a": {"type": "calculation", "op": "add", "inputs": ["a", "a"]}},
        "sections": [{"id": "one", "blocks": [{"type": "prose", "text": "Test"}]}]})
    with pytest.raises(DataError, match="Cyclic fact"):
        resolve_facts(doc, evidence([]))


def test_partial_missing_aggregate_requires_explicit_skip():
    q=Query.model_validate({'dataset':'orders','measures':[{'name':'total','op':'sum','field':'net'}]})
    data=evidence([{'net':3},{'net':None}])
    assert execute(data,q)==[{'total':None}]
    q.measures[0].missing='skip'
    assert execute(data,q)==[{'total':Decimal(3)}]


def test_numeric_group_keys_are_value_based_but_booleans_are_distinct():
    q=Query.model_validate({'dataset':'orders','group_by':['key'],'measures':[{'name':'n','op':'count'}]})
    assert execute(evidence([{'key':1},{'key':1.0},{'key':True}]),q)==[{'key':1,'n':2},{'key':True,'n':1}]


def test_json_decimal_precision_can_be_preserved_at_acquisition():
    import json
    data=Evidence.model_validate(json.loads('{"datasets":{"orders":{"source":{"label":"Exact JSON"},"rows":[{"net":0.1234567890123456789}]}}}',parse_float=Decimal))
    q=Query.model_validate({'dataset':'orders','measures':[{'name':'total','op':'sum','field':'net'}]})
    assert execute(data,q)==[{'total':Decimal('0.1234567890123456789')}]


def test_large_integer_sum_and_formatted_value_do_not_round_away_units():
    from anton.core.artifacts.static.document import Format
    value=10**35+1
    q=Query.model_validate({'dataset':'orders','measures':[{'name':'total','op':'sum','field':'net'}]})
    result=execute(evidence([{'net':10**35},{'net':1}]),q)[0]['total']
    assert result==Decimal(value)
    assert Format().display(result).replace(',','')==str(value)


@pytest.mark.parametrize('kind,expected',[('date','2024-02-29'),('week_start','2024-02-26'),('month_start','2024-02-01'),('quarter_start','2024-01-01'),('year_start','2024-01-01')])
def test_calendar_casts_handle_leap_days_preserve_null_and_leave_original_fields(kind,expected):
    data=evidence([{'day':'2024-02-29'},{'day':None}])
    query=Query(dataset='orders',columns=['day','period'],casts=[{'name':'period','field':'day','type':kind}])
    assert execute(data,query)==[{'day':'2024-02-29','period':expected},{'day':None,'period':None}]
    assert data.datasets['orders'].rows==[{'day':'2024-02-29'},{'day':None}]


@pytest.mark.parametrize('invalid',['2023-02-29','02/29/2024','2024-02-29T23:00:00Z','20240229',123,True])
def test_calendar_casts_reject_invalid_dates_and_implicit_timezones(invalid):
    with pytest.raises(DataError):
        execute(evidence([{'day':invalid}]),Query(dataset='orders',columns=['period'],casts=[{'name':'period','field':'day','type':'week_start'}]))


def test_calendar_grouping_and_ordered_limit_cover_full_buckets_across_year_boundary():
    rows=[{'day':'2024-12-29','amount':Decimal('.1')},{'day':'2024-12-30','amount':Decimal('.2')},
          {'day':'2025-01-05','amount':Decimal('.3')},{'day':'2025-01-06','amount':None}]
    data=evidence(rows)
    query=Query(dataset='orders',casts=[{'name':'week','field':'day','type':'week_start'}],group_by=['week'],
        measures=[{'name':'total','field':'amount','op':'sum'}],order_by=[{'field':'total','descending':True}],limit=2)
    assert execute(data,query)==[{'week':'2024-12-30','total':Decimal('.5')},{'week':'2024-12-23','total':Decimal('.1')}]
    assert data.datasets['orders'].rows==rows
    query.limit=None
    assert execute(data,query)[-1]=={'week':'2025-01-06','total':None}


@pytest.mark.parametrize('invalid',[0,-1,True,1.5,'2',100001])
def test_ordered_limits_require_a_bounded_positive_integer(invalid):
    with pytest.raises(ValidationError):Query(dataset='orders',columns=['amount'],limit=invalid)


def test_limit_does_not_hide_invalid_later_rows():
    with pytest.raises(DataError,match='missing fields'):
        execute(evidence([{'amount':1},{'other':2}]),Query(dataset='orders',columns=['amount'],limit=1))


def test_optional_source_metadata_preserves_legacy_hashes_and_classifies_reference_data():
    from hashlib import sha256
    from anton.core.artifacts.static.data import exact_json
    legacy={'datasets':{'orders':{'source':{'label':'Legacy records','url':'','scope':''},'rows':[{'amount':2}]}}}
    data=Evidence.model_validate(legacy)
    assert data.digest()==sha256(exact_json(legacy).encode()).hexdigest()
    data.datasets['orders'].source.kind='reference'
    data.datasets['orders'].source.attribution='Source and contributors, CC BY 4.0'
    assert data.digest()!=sha256(exact_json(legacy).encode()).hexdigest()
