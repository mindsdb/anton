import json
import pytest

from anton.core.tools.generate_artifact import observed_data


def test_complete_source_records_survive_without_model_transcription():
    original = {"columns": ["sku", "revenue"], "rows": [[f"SKU-{i}", i * 7.25] for i in range(70)]}
    data, notes = observed_data.collect(["[output]\n" + json.dumps(original)])
    key = observed_data.identity(original)
    assert data[key] == original
    assert "window.ARTIFACT_DATA" in notes
    assert 'is an ARRAY with 70 elements' in notes
    assert f'window.ARTIFACT_DATA["{key}"]["rows"]' in notes


def test_host_binding_is_lossless_and_does_not_allow_data_to_close_a_script():
    value = {"records": [{"label": '</script><script>throw Error("injected")</script>', "amount": 73.25}]}
    data = {observed_data.identity(value): value}
    page = observed_data.bind("<!doctype html><html><head></head><body></body></html>", data)
    assert '</script><script>throw Error' not in page
    assert observed_data.previous_data(page) == data
    assert observed_data.previous_data(observed_data.bind(page, data)) == data
    assert page.count('id="cowork-observed-json"') == 1
    assert observed_data.strip_binding(page) == "<!doctype html><html><head></head><body></body></html>"


def test_binding_ignores_comment_tags_and_respects_quoted_tag_delimiters():
    value = {"rows": [1, 2, 3]}
    data = {observed_data.identity(value): value}
    source = '<html><!-- <head> \u2028 -->\n<head data-label="a > b"><title>Data</title></head><body></body></html>'
    result = observed_data.bind(source, data)
    assert '<!-- <head> \u2028 -->' in result
    assert '<head data-label="a > b"><meta charset="utf-8" data-cowork-observed><script' in result
    assert observed_data.previous_data(result) == data


def test_incomplete_json_and_credentials_do_not_become_page_data():
    data, notes = observed_data.collect(['{"access_token":"test-credential"}\n{"rows":[[1],'])
    assert data == {}
    assert "test-credential" not in notes
    assert list(observed_data.values('{"rows":[NaN]}')) == []


def test_edit_keeps_existing_observed_data_when_no_new_query_is_needed():
    original = {"rows": [["A", 3.5], ["B", 8.25]]}
    data = {observed_data.identity(original): original}
    page = observed_data.bind("<html><body>Existing chart</body></html>", data)
    restored, notes = observed_data.collect([], page)
    assert restored == data
    assert "Existing chart" in observed_data.strip_binding(page)


def test_large_handoff_bypasses_the_console_cap_but_refuses_symlinks(tmp_path):
    folder = tmp_path / "artifact"
    folder.mkdir()
    value = {"rows": [[f"record-{i}", i * 9] for i in range(1500)]}
    handoff = folder / "observed-data.json"
    handoff.write_text(json.dumps(value))
    assert handoff.stat().st_size > 10000
    assert observed_data.read_handoff(folder) == value
    handoff.unlink()
    outside = tmp_path / "outside.json"
    outside.write_text('{"private": "outside"}')
    handoff.symlink_to(outside)
    with pytest.raises(OSError):
        observed_data.read_handoff(folder)
def test_compact_summary_preserves_exact_objects_and_real_array_contracts():
    import json
    from anton.core.tools.generate_artifact import observed_data
    value = {"scenarios": {str(i): {"amount": i * 17.25, "active": "No"} for i in range(12)},
             "records": [{"key": str(i), "value": i * 3} for i in range(30)]}
    data, text = observed_data.collect([json.dumps(value)], compact=True)
    original, expanded = observed_data.collect([json.dumps(value)])
    assert data == original == {observed_data.identity(value): value}
    assert len(text) < len(expanded) * .6
    assert '"active":"No"' in text
    assert "is an ARRAY with 30 elements; iterate this array directly" in text
    assert "is an OBJECT with these exact observed values" in text
