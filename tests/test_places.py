import pytest

from anton.core.tools.generate_artifact import observed_data, places


def near(place, lat, lon):
    return place is not None and abs(place["lat"] - lat) < 0.5 and abs(place["lon"] - lon) < 0.5


def test_a_town_resolves_with_or_without_its_country():
    assert near(places.resolve("Lyon"), 45.75, 4.85)
    assert places.resolve("Lyon, France")["country"] == "France"


def test_a_region_or_state_code_picks_one_of_several_towns():
    assert places.resolve("Springfield") is None
    assert near(places.resolve("Springfield, Illinois"), 39.80, -89.64)
    assert near(places.resolve("Portland, ME"), 43.66, -70.26)


def test_a_qualifier_that_contradicts_the_town_resolves_nothing():
    assert places.resolve("Lyon, Japan") is None


@pytest.mark.parametrize("value", ["Pending", "ORD", "SHP-1001", "none", "", "x" * 200, 42])
def test_values_that_name_no_place_stay_unresolved(value):
    assert places.resolve(value) is None


def test_a_record_naming_its_region_overrides_the_better_known_town():
    found = places.index({"sites": [{"city": "Perth", "region": "Scotland"}]})
    assert near(found["Perth, Scotland"], 56.40, -3.43)
    assert "Perth" not in found  # alone it would be the Australian city


@pytest.mark.parametrize("code,wanted", [
    ("sections:[{type:'map',points:[]}]", True), ('{type: "globe"}', True),
    ("const p = window.ARTIFACT_PLACES[name];", True), ("sections:[{type:'chart'}]", False), ("", False)])
def test_only_a_page_with_a_map_or_globe_needs_the_places(code, wanted):
    assert places.needed(code) is wanted


def test_places_are_bound_beside_the_data_and_stripped_on_the_next_build():
    html = observed_data.bind("<html><head></head><body></body></html>", {"k": [{"at": "Lyon"}]},
                              places.index({"rows": [{"at": "Lyon"}]}))
    assert 'id="cowork-places-json"' in html and "window.ARTIFACT_PLACES=" in html
    stripped = observed_data.strip_binding(html)
    assert "ARTIFACT_PLACES" not in stripped and "ARTIFACT_DATA" not in stripped
    assert "cowork-places-json" not in observed_data.bind("<html><head></head></html>", {"k": 1})


def test_columns_of_array_rows_are_judged_one_by_one():
    rows = {"columns": ["id", "city", "mode"],
            "rows": [["A-1", "Lyon", "Rail"], ["A-2", "Porto", "Air"], ["A-3", "Gdansk", "Road"]]}
    found = places.index(rows)
    assert {"Lyon", "Porto", "Gdansk"} <= set(found) and "Rail" not in found


def test_a_field_that_is_mostly_not_places_is_left_out():
    found = places.index({"lanes": [{"mode": m} for m in ("Rail", "Air", "Road", "Sea")]})
    assert found == {}
