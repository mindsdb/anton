import copy
import json

import pytest

from anton.core.tools.generate_artifact.source_profile import profile_tables


def test_profile_preserves_observed_states_missing_null_and_numeric_ranges():
    data = {"appointments": [
        {"state": "Booked", "fee": 4.5, "day": "2031-02-07", "confirmed": True},
        {"state": "Waiting", "fee": None, "day": "2031-01-06", "confirmed": False},
        {"state": "Waiting", "day": None},
        {"state": "Complete", "fee": 8, "day": "2031-01-09", "confirmed": True},
    ]}
    original = copy.deepcopy(data)
    profile = json.loads(profile_tables(data))["tables"]["appointments"]
    assert data == original
    assert profile["rows"] == 4
    cols = profile["columns"]
    assert cols["state"]["values"] == [["Booked", 1], ["Waiting", 2], ["Complete", 1]]
    assert cols["fee"]["types"] == {"number": 2, "null": 1}
    assert cols["fee"]["missing"] == 1
    assert cols["fee"]["range"] == [4.5, 8]
    assert cols["day"]["date_range"] == ["2031-01-06", "2031-02-07"]
    assert cols["confirmed"]["types"] == {"boolean": 3}
    assert "range" not in cols["confirmed"]


def test_profile_never_treats_first_row_as_the_whole_schema_or_guesses_dates():
    p = json.loads(profile_tables({"x": [{"x": "2031-02-30"}, {"y": [1, 2]}, {"x": 3}]}))["tables"]["x"]
    assert set(p["columns"]) == {"x", "y"}
    assert p["columns"]["x"]["missing"] == 1
    assert p["columns"]["x"]["types"] == {"string": 1, "number": 1}
    assert "date_range" not in p["columns"]["x"]
    assert p["columns"]["y"]["types"] == {"array": 1}
    assert "distinct" not in p["columns"]["y"]
    assert json.loads(profile_tables({"empty": []}))["tables"] == {"empty": {"rows": 0, "columns": {}}}


def test_budget_is_valid_json_and_omissions_are_explicit():
    rows = [{"code": "v" * 25 + str(i), "state": "Waiting"} for i in range(12)]
    text = profile_tables({"items": rows}, max_chars=350)
    assert len(text) <= 350
    profile = json.loads(text)
    p = profile["tables"]["items"]
    assert p["rows"] == len(rows)
    assert p["columns"]["code"]["distinct"] == len(rows)
    assert profile["value_lists_omitted"] == 1
    assert "values" not in p["columns"]["code"]
    assert p["columns"]["state"]["values"] == [["Waiting", 12]]
    with pytest.raises(ValueError, match="fewer tables"):
        profile_tables({"wide": [{"c" + str(i): i for i in range(50)}]}, max_chars=256)


@pytest.mark.parametrize("data", [[], {"x": {}}, {"x": [3]}, {"x": [{1: 3}]}, {3: []}])
def test_invalid_shapes_are_rejected(data):
    with pytest.raises(ValueError):
        profile_tables(data)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_values_are_rejected(value):
    with pytest.raises(ValueError):
        profile_tables({"x": [{"amount": value}]})


def test_credentials_are_rejected_before_any_output_is_returned():
    with pytest.raises(ValueError, match="credentials") as error:
        profile_tables({"x": [{"nested": {"client_secret": "private-example"}}]})
    assert "private-example" not in str(error.value)
