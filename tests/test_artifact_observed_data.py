import json
import pytest

from anton.core.tools.generate_artifact import observed_data


def test_complete_source_records_survive_without_model_transcription():
    original = {"columns": ["sku", "revenue"], "rows": [[f"SKU-{i}", i * 7.25] for i in range(70)]}
    data, notes = observed_data.collect(original)
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


@pytest.mark.parametrize("key", ["access_token", "apiKey", "x-api-key", "private_key", "aws_secret_access_key", "dsn"])
def test_credential_keys_are_recognized_in_common_spellings(key):
    assert observed_data.credential_bearing({"source": {key: "value"}})
    assert not observed_data.credential_bearing({"source": {"rows": [1, 2]}})


@pytest.mark.parametrize("key", ["supply_api_key", "MY_API_TOKEN", "X-Auth-Token"])
def test_credentials_with_prefixes_are_rejected(key):
    assert observed_data.credential_bearing({"source": {key: "dummy-secret"}})


def test_credential_url_is_rejected_but_normal_urls_and_keys_are_allowed():
    assert observed_data.credential_bearing({"source": "postgres://user:pass@host/db"})
    assert not observed_data.credential_bearing({"source": "postgres://host/db", "primary_key": 42})


def test_handoff_passes_through_output_scrubber(monkeypatch):
    monkeypatch.setattr(observed_data, "scrub_output", lambda text: text.replace("dummy-secret", "[REDACTED]"))
    assert observed_data.credential_bearing({"rows": [{"note": "dummy-secret"}]})


def test_edit_keeps_existing_observed_data_when_no_new_query_is_needed():
    original = {"rows": [["A", 3.5], ["B", 8.25]]}
    data = {observed_data.identity(original): original}
    page = observed_data.bind("<html><body>Existing chart</body></html>", data)
    restored, notes = observed_data.collect(None, page)
    assert restored == data
    replaced, _ = observed_data.collect({"rows": [["C", 1.0]]}, page)
    assert list(replaced.values()) == [{"rows": [["C", 1.0]]}]
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
def test_contract_keeps_small_objects_exact_and_describes_arrays():
    value = {"segments": {str(i): {"amount": i * 17.25, "open": "no"} for i in range(12)},
             "records": [{"key": str(i), "value": i * 3} for i in range(30)]}
    data, text = observed_data.collect(value)
    assert data == {observed_data.identity(value): value}
    assert '"open":"no"' in text
    assert "is an ARRAY with 30 elements; iterate this array directly" in text
    assert "is an OBJECT with these exact observed values" in text


@pytest.mark.parametrize("key", ["key", "primary_key", "sort_key", "token_count", "token_type", "auth_status",
                                 "auth_method", "password_changed_at", "input_tokens", "author"])
def test_ordinary_data_columns_are_not_credentials(key):
    assert not observed_data.credential_bearing({"records": [{key: 3}]})


@pytest.mark.parametrize("key", ["supply_api_key", "MY_API_TOKEN", "X-Auth-Token", "db_password", "clientSecret"])
def test_credential_names_are_recognized_in_common_spellings(key):
    assert observed_data.credential_bearing({"source": {key: "value"}})


def _sales_with_totals():
    rows = []
    for month in ["2030-01", "2030-02", "2030-03"]:
        for store in ["North", "South"]:
            parts = [{"month": month, "store": store, "category": name, "sales": sales, "returns": returns, "margin_pct": 31.5}
                     for name, sales, returns in [("Books", 120, 4), ("Games", 75, 2), ("Music", 40, 1)]]
            rows += parts + [{"month": month, "store": store, "category": "Total", "sales": 235, "returns": 7, "margin_pct": 30.0}]
    return rows


def test_rows_that_sum_their_group_are_marked_in_the_contract():
    rows = _sales_with_totals()
    assert observed_data.total_rows(rows) == [
        {"field": "category", "value": "Total", "by": ["month", "store"], "measures": ["sales", "returns"]}]
    _, text = observed_data.collect({"sales": rows})
    assert 'TOTAL ROWS: elements where "category" = "Total" hold the sum of the other elements with the same month, store for sales, returns.' in text


def test_ordinary_rows_and_a_lone_nonzero_part_are_not_totals():
    plain = [r for r in _sales_with_totals() if r["category"] != "Total"]
    assert observed_data.total_rows(plain) == []
    # Only one category has returns, so its row equals "the sum" of the others without being a total.
    lone = [dict(r, returns=4 if r["category"] in {"Books", "Total"} else 0) for r in _sales_with_totals()]
    assert [m["value"] for m in observed_data.total_rows(lone)] == ["Total"]
    _, text = observed_data.collect({"sales": plain})
    assert "TOTAL ROWS" not in text
    # The third line number equals 1 + 2 in every order, but quantities and amounts do not add up.
    lines = [{"order": order, "line": line, "item": item, "quantity": quantity, "amount": quantity * 4.5}
             for order in ["A-1", "A-2", "A-3"]
             for line, item, quantity in [(1, "bolt", 30), (2, "nut", 12), (3, "washer", 50)]]
    assert observed_data.total_rows(lines) == []


def test_a_sum_that_includes_total_rows_is_reported_with_its_parts():
    data = {"key": {"sales": _sales_with_totals()}}
    found = {item["value"]: item for item in observed_data.double_counts(data)}
    whole = found[2 * 6 * 235]
    assert whole["parts"] == 6 * 235 and whole["measure"] == "sales" and whole["scope"] == ""
    assert whole["path"] == 'window.ARTIFACT_DATA["key"]["sales"]'
    assert (whole["field"], whole["marker"]) == ("category", "Total")
    assert found[2 * 2 * 235]["scope"] == 'month = "2030-01"'
    # Correct sums, and sums too small to tell apart from other counts, are never reported.
    assert 6 * 235 not in found and all(abs(value) >= 10 for value in found)
    assert observed_data.double_counts({"key": {"sales": [r for r in _sales_with_totals() if r["category"] != "Total"]}}) == []


def test_figure_facts_are_embedded_only_when_asked_and_stripped_with_the_data():
    value = {"sales": _sales_with_totals()}
    data = {observed_data.identity(value): value}
    page = "<!doctype html><html><head></head><body></body></html>"
    assert 'id="cowork-figure-checks"' not in observed_data.bind(page, data)
    checked = observed_data.bind(page, data, checks=True)
    facts = json.loads(checked.split('id="cowork-figure-checks" type="application/json" data-cowork-observed>')[1].split("</script>")[0])
    assert 2 * 6 * 235 in [item["value"] for item in facts["doubleCounts"]]
    assert observed_data.previous_data(checked) == data
    assert observed_data.strip_binding(checked) == page
