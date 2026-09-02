"""Generated data files, and why their schemas cannot drift.

The design question this settles: how do you keep a `.data.js` and its
`.schema.json` in step when anything might rewrite either?

The answer is not to try. The schema is *derived* from the bytes on disk
and can be re-derived at any moment, so nothing is ever asked to keep
them in step — `reconcile()` just recomputes. That removes the need for
one blessed way of producing data, which was the alternative.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anton.core.yolo import Workspace
from anton.core.yolo.data import (
    DataError,
    derive_schema,
    global_name,
    read_data,
    reconcile,
    write_data,
)

ROWS = [
    {"date": "2026-08-31", "price": 41.22, "volume": 19773, "flagged": False},
    {"date": "2026-09-01", "price": 41.90, "volume": None, "flagged": True},
]


def schema_of(workspace: Workspace, name: str = "prices") -> dict:
    return json.loads(workspace.read(f"{name}.schema.json"))


# ─── The file is two things at once ─────────────────────────────────────


def test_a_data_file_is_loadable_by_a_script_tag(tmp_path: Path):
    """The reason it is .js and not .json: an artifact opened from disk
    cannot fetch() a sibling file, so the data has to arrive as a global.
    The same file works unchanged when published over HTTP."""
    workspace = Workspace(tmp_path)
    write_data(workspace, "prices", ROWS)
    text = workspace.read("prices.data.js")
    assert text.startswith("window.ANTON_DATA_prices = ")
    assert text.rstrip().endswith(";")


def test_the_same_file_is_readable_without_a_javascript_engine(tmp_path: Path):
    """The reason it is not awkward: the value is a JSON literal behind a
    fixed prefix, so anything can read it back."""
    workspace = Workspace(tmp_path)
    write_data(workspace, "prices", ROWS)
    assert read_data(workspace, "prices.data.js") == ROWS


def test_a_data_file_in_the_wrong_shape_says_so(tmp_path: Path):
    workspace = Workspace(tmp_path)
    workspace.write("bad.data.js", "const x = [1,2,3];\n")
    with pytest.raises(DataError, match="expected"):
        read_data(workspace, "bad.data.js")


def test_a_data_file_holding_broken_json_says_so(tmp_path: Path):
    workspace = Workspace(tmp_path)
    workspace.write("bad.data.js", "window.ANTON_DATA_bad = [1,2,;\n")
    with pytest.raises(DataError, match="valid JSON"):
        read_data(workspace, "bad.data.js")


def test_awkward_names_still_make_a_legal_global(tmp_path: Path):
    assert global_name("q3-sales report") == "ANTON_DATA_q3_sales_report"


# ─── The schema is derived, never stated ────────────────────────────────


def test_types_come_from_the_values_not_from_a_description(tmp_path: Path):
    """The failure this prevents: a hand-written {"price": "number"} over
    rows that actually hold "1,234.00" strings. Code written against that
    renders a broken chart rather than raising."""
    schema = derive_schema("prices", [{"price": "1,234.00"}])
    assert schema["fields"]["price"]["type"] == "string"


def test_a_boolean_is_not_reported_as_a_number(tmp_path: Path):
    """True is an int in Python. Calling it a number sends whoever reads
    the schema looking for arithmetic on a flag."""
    assert derive_schema("f", [{"ok": True}])["fields"]["ok"]["type"] == "boolean"


def test_nullability_is_observed(tmp_path: Path):
    fields = derive_schema("prices", ROWS)["fields"]
    assert fields["volume"]["nullable"] is True
    assert "nullable" not in fields["price"]


def test_a_column_with_mixed_types_admits_it(tmp_path: Path):
    schema = derive_schema("m", [{"x": 1}, {"x": "two"}])
    assert schema["fields"]["x"]["type"] == "number | string"


def test_a_column_missing_from_some_rows_is_still_described(tmp_path: Path):
    schema = derive_schema("m", [{"a": 1}, {"a": 2, "b": "late"}])
    assert set(schema["fields"]) == {"a", "b"}
    assert schema["fields"]["b"]["nullable"] is True


def test_the_schema_names_the_global(tmp_path: Path):
    """Perfect knowledge of the columns is useless if the chart cannot
    find the array."""
    workspace = Workspace(tmp_path)
    write_data(workspace, "prices", ROWS)
    assert schema_of(workspace)["global"] == "ANTON_DATA_prices"


def test_notes_are_the_only_thing_asked_for(tmp_path: Path):
    """Units, gaps and timezones cannot be inferred from the values, and
    they are what produce plausible-looking wrong charts."""
    workspace = Workspace(tmp_path)
    write_data(workspace, "prices", ROWS, notes="Daily close. Gaps on weekends.")
    assert "Gaps on weekends" in schema_of(workspace)["notes"]


# ─── Reconciliation: drift is impossible, not merely discouraged ────────


def test_a_missing_sidecar_is_written(tmp_path: Path):
    """Data produced some other way — by hand, by an older cell — still
    ends up described. This is what removes the need for one blessed
    way of writing a data file."""
    workspace = Workspace(tmp_path)
    workspace.write("prices.data.js", 'window.ANTON_DATA_prices = [{"a": 1}];')

    report = reconcile(workspace)
    assert any("written" in line for line in report)
    assert schema_of(workspace)["fields"]["a"]["type"] == "number"


def test_a_stale_sidecar_is_refreshed(tmp_path: Path):
    workspace = Workspace(tmp_path)
    write_data(workspace, "prices", ROWS)
    # Something else rewrites the data, with a different shape entirely.
    workspace.write("prices.data.js", 'window.ANTON_DATA_prices = [{"date":"x","price":"1,234.00"}];')

    report = reconcile(workspace)
    assert any("refreshed" in line for line in report)
    assert schema_of(workspace)["fields"]["price"]["type"] == "string"
    assert schema_of(workspace)["rows"] == 1


def test_reconciling_preserves_the_one_field_nobody_can_derive(tmp_path: Path):
    workspace = Workspace(tmp_path)
    write_data(workspace, "prices", ROWS, notes="Daily close. Gaps on weekends.")
    workspace.write("prices.data.js", 'window.ANTON_DATA_prices = [{"date":"x"}];')

    reconcile(workspace)
    assert "Gaps on weekends" in schema_of(workspace)["notes"]


def test_an_unchanged_sidecar_is_left_alone(tmp_path: Path):
    """Rewriting it every time would churn the file and its timestamp for
    no reason, and make every reconcile look like a change."""
    workspace = Workspace(tmp_path)
    write_data(workspace, "prices", ROWS)
    before = workspace.read("prices.schema.json")
    assert reconcile(workspace) == []
    assert workspace.read("prices.schema.json") == before


def test_unreadable_data_is_reported_rather_than_skipped(tmp_path: Path):
    workspace = Workspace(tmp_path)
    workspace.write("mystery.data.js", "this is not a data file at all\n")
    [line] = reconcile(workspace)
    assert "cannot read it back" in line


def test_a_corrupt_sidecar_is_simply_rewritten(tmp_path: Path):
    workspace = Workspace(tmp_path)
    write_data(workspace, "prices", ROWS)
    workspace.write("prices.schema.json", "{ not json")
    assert any("written" in line for line in reconcile(workspace))
    assert schema_of(workspace)["global"] == "ANTON_DATA_prices"


def test_an_orphan_sidecar_is_flagged(tmp_path: Path):
    workspace = Workspace(tmp_path)
    workspace.write("gone.schema.json", '{"global": "ANTON_DATA_gone"}')
    [line] = reconcile(workspace)
    assert "not there" in line
