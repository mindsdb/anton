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
