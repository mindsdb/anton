from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from anton.core.artifacts.internal_files import DATA_DIRNAME
from anton.core.tools.generate_artifact import data_files
from anton.core.tools.generate_artifact.data_files import DataFile


def _write(artifact: Path, file_id: str, text: str) -> Path:
    folder = artifact / DATA_DIRNAME
    folder.mkdir(exist_ok=True)
    path = folder / f"{file_id}.json"
    path.write_text(text, encoding="utf-8")
    return path


def test_valid_file_passes(tmp_path):
    _write(tmp_path, "1", json.dumps([{"Id": "SO-5001", "Value": 60950.0}]))
    assert data_files.validate(tmp_path, "1") is None


@pytest.mark.parametrize("file_id, text, fragment", [
    ("x", "[]", "not a data file id"),
    ("2", None, "does not exist"),
    ("1", "[{", "not valid JSON"),
    ("1", '[{"v": NaN}]', "not valid JSON"),
    ("1", '"just a string"', "object or array"),
])
def test_invalid_files_are_reported(tmp_path, file_id, text, fragment):
    if text is not None:
        _write(tmp_path, "1", text)
    error = data_files.validate(tmp_path, file_id)
    assert error is not None and fragment in error


def test_a_symlinked_file_is_refused(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text("[]")
    (tmp_path / DATA_DIRNAME).mkdir()
    os.symlink(outside, tmp_path / DATA_DIRNAME / "1.json")
    assert "could not be read" in data_files.validate(tmp_path, "1")


def test_a_changed_file_from_an_earlier_run_is_refused(tmp_path):
    path = _write(tmp_path, "1", "[1]")
    protected = data_files.snapshot(tmp_path)
    path.write_text("[1, 2, 3]")
    os.utime(path, ns=(1, 1))
    error = data_files.validate(tmp_path, "1", protected=protected)
    assert error is not None and "earlier run" in error and "id 2" in error
    assert data_files.validate(tmp_path, "1") is None


def test_unchanged_file_from_an_earlier_run_can_be_declared_again(tmp_path):
    _write(tmp_path, "1", "[1]")
    protected = data_files.snapshot(tmp_path)
    assert data_files.validate(tmp_path, "1", protected=protected) is None


def test_serialize_is_compact_and_escapes_less_than(tmp_path):
    text = data_files.serialize({"a": "</script>", "b": [1, 2]})
    assert "<" not in text
    assert json.loads(text) == {"a": "</script>", "b": [1, 2]}
    assert " " not in text.replace("</script>", "")


def test_total_size_counts_the_escaped_form(tmp_path, monkeypatch):
    monkeypatch.setattr(data_files, "MAX_PAGE_BYTES", 100)
    _write(tmp_path, "1", json.dumps(["<" * 30]))
    error = data_files.check_total_size(tmp_path, [DataFile("1", ["s"])])
    assert error is not None and "page limit" in error


def test_parse_declared_tolerates_shapes():
    parsed = data_files.parse_declared([
        {"id": 1, "sources": "orders", "description": "rows"},
        {"id": "2", "sources": ["a", "b"]},
        {"no_id": True},
        "junk",
    ])
    assert parsed == [DataFile("1", ["orders"], "rows"), DataFile("2", ["a", "b"], "")]
    assert data_files.parse_declared(None) == []


def test_check_declared_matches_sources_by_key(tmp_path):
    _write(tmp_path, "1", "[1]")
    _write(tmp_path, "2", "[2]")
    valid, errors = data_files.check_declared(
        tmp_path,
        [DataFile("1", ["Orders  Table"]), DataFile("2", ["unknown"])],
        ["orders table"],
    )
    assert [f.id for f in valid] == ["1"]
    assert len(errors) == 1 and "id 2" in errors[0] and "data_sources" in errors[0]


def test_check_declared_requires_sources(tmp_path):
    _write(tmp_path, "1", "[1]")
    valid, errors = data_files.check_declared(tmp_path, [DataFile("1", [])], ["orders"])
    assert valid == [] and "sources" in errors[0]


def test_snapshot_written_since_next_id_and_prune(tmp_path):
    assert data_files.next_free_id(tmp_path) == "1"
    _write(tmp_path, "1", "[1]")
    before = data_files.snapshot(tmp_path)
    _write(tmp_path, "3", "[3]")
    (tmp_path / DATA_DIRNAME / "notes.txt").write_text("ignored")
    assert data_files.written_since(tmp_path, before) == ["3"]
    assert data_files.next_free_id(tmp_path) == "4"
    data_files.prune(tmp_path, {"3"})
    assert sorted(p.name for p in (tmp_path / DATA_DIRNAME).iterdir()) == ["3.json", "notes.txt"]


def test_restorable_keeps_only_valid_files(tmp_path):
    _write(tmp_path, "1", "[1]")
    raw = [{"id": "1", "sources": ["a"]}, {"id": "2", "sources": ["b"]}, {"id": "bad"}]
    assert data_files.restorable(tmp_path, raw) == [DataFile("1", ["a"], "")]


def test_data_file_round_trips_through_dict():
    item = DataFile("7", ["orders"], "rows")
    assert DataFile.from_dict(item.to_dict()) == item
    assert DataFile.from_dict({"id": "0"}) is None
