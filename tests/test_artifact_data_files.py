from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from anton.core.artifacts.internal_files import DATA_DIRNAME
from anton.core.tools.generate_artifact import data_files, verifiers
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


def test_a_symlinked_data_folder_is_not_followed(tmp_path):
    artifact = tmp_path / "art"
    other = tmp_path / "other"
    artifact.mkdir()
    other.mkdir()
    (other / "5.json").write_text("[5]")
    os.symlink(other, artifact / DATA_DIRNAME)
    assert data_files.snapshot(artifact) == {}
    data_files.prune(artifact, set())
    assert (other / "5.json").read_text() == "[5]"


_FRAMES = {
    "Base Plan": [
        {"Id": "SO-5001", "Value": 60950.0, "Status": "On Time"},
        {"Id": "SO-5002", "Value": 49290.0, "Status": "On Time"},
    ],
    "Supplier Delay": [
        {"Id": "SO-5001", "Value": 60950.0, "Status": "Late"},
        {"Id": "SO-5002", "Value": 49290.0, "Status": "On Time"},
        {"Id": "SO-5003", "Value": 34000.0, "Status": "On Time"},
        {"Id": "SO-5004", "Value": 66700.0, "Status": "On Time"},
    ],
}


def test_summary_reaches_the_records_inside_a_dict(tmp_path):
    _write(tmp_path, "1", json.dumps(_FRAMES))
    text = data_files.summarize(tmp_path, [DataFile("1", ["orders"], "order lines")])
    assert ".anton-data/1.json" in text and "orders" in text and "order lines" in text
    assert 'data["Supplier Delay"]: array of 4 records' in text
    assert "Id (string)" in text and "Value (number)" in text
    assert '"Late"' in text
    # Two records in one array, the first three of four in the other.
    assert text.count("sample:") == 5
    # Ids are unique, so they are not listed as values; only samples show them.
    assert "SO-5004" not in text


def test_summary_is_bounded(tmp_path):
    rows = [{f"field_{i}": "x" * 50 for i in range(80)}]
    _write(tmp_path, "1", json.dumps(rows))
    text = data_files.summarize(tmp_path, [DataFile("1", ["s"])])
    assert len(text) <= data_files.SUMMARY_CHARS + 200


def test_sections_have_their_headers_and_are_empty_without_files(tmp_path):
    _write(tmp_path, "1", "[1]")
    files = [DataFile("1", ["s"])]
    assert data_files.files_section(tmp_path, []) == ""
    assert data_files.files_section(tmp_path, files).startswith(data_files.DATA_FILES_HEADER)
    assert data_files.READ_CALL in data_files.files_section(tmp_path, files)
    assert data_files.saved_files_section(tmp_path, files).startswith(data_files.SAVED_DATA_FILES_HEADER)


def test_state_sections_do_not_touch_the_session_without_files(tmp_path):
    class Exploding:
        def __getattr__(self, name):
            raise AssertionError("the session must not be read without data files")

    state = SimpleNamespace(artifact_path=tmp_path, data_files=[], previous_data_files=[], session=Exploding())
    assert data_files.state_files_section(state) == ""
    assert data_files.state_saved_files_section(state) == ""


def test_state_sections_are_empty_once_the_type_is_not_an_html_app(tmp_path):
    _write(tmp_path, "1", "[1]")
    files = [DataFile("1", ["s"])]
    state = SimpleNamespace(
        artifact_path=tmp_path, data_files=files, previous_data_files=files, session=None,
        artifact_type="html-app", final_artifact_type="fullstack-stateless-app",
    )
    assert data_files.state_files_section(state) == ""
    assert data_files.state_saved_files_section(state) == ""
    state.final_artifact_type = "html-app"
    assert data_files.state_files_section(state).startswith(data_files.DATA_FILES_HEADER)
    assert data_files.state_saved_files_section(state).startswith(data_files.SAVED_DATA_FILES_HEADER)


def test_lone_surrogates_are_not_valid_for_a_page(tmp_path):
    _write(tmp_path, "1", '["\\ud800"]')
    assert "unreadable" in data_files.summarize(tmp_path, [DataFile("1", ["s"])])
    error = data_files.validate(tmp_path, "1")
    assert error is not None and "not valid JSON" in error
    assert data_files.restorable(tmp_path, [{"id": "1", "sources": ["s"]}]) == []


_PAGE = (
    "<!DOCTYPE html><html><head><title>t</title>"
    "<script data-anton-data='1' type=\"application/json\"></script>"
    "<!-- <script type=\"application/json\" data-anton-data=\"9\"></script> -->"
    "</head><body><div id=\"k\"></div>"
    "<script>const d = JSON.parse(document.querySelector('script[data-anton-data=\"1\"]').textContent);</script>"
    "</body></html>"
)


def test_blocks_are_found_with_any_attribute_order_and_not_in_comments():
    blocks = data_files.find_blocks(_PAGE)
    assert [b.file_id for b in blocks] == ["1"]
    assert blocks[0].attrs["type"] == "application/json"
    assert _PAGE[blocks[0].body_start:blocks[0].body_end] == ""
    assert data_files.reads_blocks(_PAGE, blocks) is True


def test_a_block_nobody_reads_is_detected():
    page = _PAGE.replace(
        "document.querySelector('script[data-anton-data=\"1\"]')", "document.getElementById('x')"
    )
    assert data_files.reads_blocks(page, data_files.find_blocks(page)) is False


def test_fill_inserts_compact_escaped_data_on_its_own_line(tmp_path):
    _write(tmp_path, "1", json.dumps({"rows": [{"t": "</script>"}]}, indent=4))
    result = data_files.fill_page(_PAGE, tmp_path, [DataFile("1", ["s"])])
    assert result.fatal is None
    block = data_files.find_blocks(result.html)[0]
    body = result.html[block.body_start:block.body_end]
    assert body.startswith("\n") and body.endswith("\n") and body.count("\n") == 2
    assert "</script>" not in body
    assert json.loads(body) == {"rows": [{"t": "</script>"}]}


def test_fill_is_repeatable_and_leaves_unknown_ids(tmp_path):
    _write(tmp_path, "1", "[1]")
    files = [DataFile("1", ["s"])]
    once = data_files.fill_page(_PAGE, tmp_path, files).html
    _write(tmp_path, "1", "[2]")
    twice = data_files.fill_page(once, tmp_path, files).html
    block = data_files.find_blocks(twice)[0]
    assert json.loads(twice[block.body_start:block.body_end]) == [2]
    assert data_files.fill_page(_PAGE, tmp_path, []).html == _PAGE


def test_fill_reports_an_unreadable_file_and_an_oversized_page(tmp_path, monkeypatch):
    missing = data_files.fill_page(_PAGE, tmp_path, [DataFile("1", ["s"])])
    assert missing.fatal is not None and ".anton-data/1.json" in missing.fatal
    # The file itself fits the limit (it is read with it); the page with it does not.
    _write(tmp_path, "1", json.dumps(["x" * 100]))
    monkeypatch.setattr(data_files, "MAX_PAGE_BYTES", len(_PAGE) + 50)
    big = data_files.fill_page(_PAGE, tmp_path, [DataFile("1", ["s"])])
    assert big.fatal is not None and "page limit" in big.fatal


def test_fill_leaves_a_real_block_with_an_unknown_id_as_it_is(tmp_path):
    _write(tmp_path, "1", "[1]")
    page = _PAGE.replace("</head>", "<script data-anton-data=\"8\" type=\"application/json\">[8]</script></head>")
    filled = data_files.fill_page(page, tmp_path, [DataFile("1", ["s"])]).html
    blocks = {b.file_id: b for b in data_files.find_blocks(filled)}
    assert filled[blocks["8"].body_start:blocks["8"].body_end] == "[8]"
    assert filled[blocks["1"].body_start:blocks["1"].body_end] == "\n[1]\n"


def test_fill_reports_a_file_that_cannot_be_encoded_for_the_page(tmp_path):
    _write(tmp_path, "1", '["\\ud800"]')
    result = data_files.fill_page(_PAGE, tmp_path, [DataFile("1", ["s"])])
    assert result.fatal is not None and "could not be read" in result.fatal


_HEAD = '<!DOCTYPE html><html><head><meta name="viewport" content="width=device-width, initial-scale=1.0"></head><body><div id="a"></div>'
_READ = "<script>const d=JSON.parse(document.querySelector('script[data-anton-data=\"1\"]').textContent);</script>"


def _errors(body: str, ids=frozenset({"1"})) -> list[str]:
    return verifiers.verify_frontend(_HEAD + body + "</body></html>", is_fullstack=False, data_ids=set(ids)).errors


def test_a_page_that_follows_the_block_protocol_passes():
    assert _errors('<script type="application/json" data-anton-data="1"></script>' + _READ) == []


@pytest.mark.parametrize("body, fragment", [
    ('<script type="application/json" data-anton-data="7"></script>' + _READ, "does not exist"),
    ('<script type="application/json" data-anton-data="1"></script>' * 2 + _READ, "more than once"),
    ('<script data-anton-data="1"></script>' + _READ, "not a JSON script"),
    ("<script>const rows=[1,2];</script>", "no data block"),
    ('<script type="application/json" data-anton-data="1"></script><script>go()</script>', "No script reads"),
])
def test_block_protocol_errors(body, fragment):
    errors = _errors(body)
    assert any(fragment in e for e in errors), errors


def test_no_block_checks_without_data_ids():
    page = _HEAD + "<script>const rows=[1];</script></body></html>"
    assert verifiers.verify_frontend(page, is_fullstack=False).errors == []
