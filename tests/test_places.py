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


def test_uppercase_iso_region_codes_resolve_globally():
    # Diverse ISO 3166-2 subdivisions across continents (Europe, Americas, Asia, Africa, Oceania).
    assert near(places.resolve("Munich, BY"), 48.14, 11.58)
    assert near(places.resolve("Zurich, ZH"), 47.37, 8.54)
    assert near(places.resolve("Rotterdam, ZH"), 51.92, 4.48)
    assert near(places.resolve("Edinburgh, SCT"), 55.95, -3.19)
    assert near(places.resolve("Campinas, SP"), -22.91, -47.06)
    assert near(places.resolve("Guadalajara, JAL"), 20.67, -103.35)
    assert near(places.resolve("Mumbai, MH"), 19.08, 72.88)
    assert near(places.resolve("Guangzhou, GD"), 23.13, 113.26)
    assert near(places.resolve("Johannesburg, GP"), -26.20, 28.05)
    assert near(places.resolve("Auckland, AUK"), -36.85, 174.76)
    assert near(places.resolve("London, ON"), 42.98, -81.23)
    assert near(places.resolve("Sydney, NSW"), -33.87, 151.21)
    assert near(places.resolve("Perth, WA"), -31.95, 115.86)
    # Existing country-code and US state-code lookups remain intact without shadowing.
    assert places.resolve("Amsterdam, NL")["country"] == "The Netherlands"
    assert places.resolve("St. John's, NL")["country"] == "Canada"
    assert places.resolve("Seattle, WA")["country"] == "United States"
    assert places.resolve("Midland, WA")["country"] == "United States"
    assert places.resolve("San Marcos, CA")["country"] == "United States"
    assert places.resolve("Victoria, AR")["country"] == "Argentina"
    assert places.resolve("Vitoria, ES")["country"] == "Spain"
    # Non-ISO GeoNames internal codes (e.g., TO.EU) are not loaded as region codes.
    _, _, region_codes, _ = places._index()
    assert ("TO", "EU") not in region_codes.get("EU", set())


def test_status_or_flag_columns_never_hijack_cities_in_index():
    # Lowercase/titlecase words never match ISO region codes.
    found = places.index({
        "sites": [
            {"city": "London", "alerts": "on", "action": "Act"},
            {"city": "Rotterdam", "alerts": "On", "action": "act"},
            {"city": "Auckland", "alerts": "on", "action": "Act"},
        ]
    })
    assert found["London"]["country"] == "United Kingdom"
    assert found["Rotterdam"]["country"] == "The Netherlands"
    assert found["Auckland"]["country"] == "New Zealand"
    assert "London, on" not in found

    # Uppercase non-region columns (ON/OFF, all-ON, 3-value ON/OK/OFF) never hijack cities.
    for rows in (
        [{"city": "London", "power": "ON"}, {"city": "Lyon", "power": "OFF"}, {"city": "Nice", "power": "OFF"}],
        [{"city": "London", "power": "ON"}, {"city": "Bristol", "power": "ON"}, {"city": "Leeds", "power": "ON"}],
        [{"city": "London", "status": "ON"}, {"city": "Bristol", "status": "OK"}, {"city": "Leeds", "status": "OFF"}],
    ):
        flagged = places.index({"sites": rows})
        assert flagged["London"]["country"] == "United Kingdom"
        assert "London, ON" not in flagged

    # Non-region uppercase flag columns on large tables do not exhaust MAX_STRINGS or drop real places.
    large = {
        "events": [
            {"id": f"REC-{i}", "a": f"val-{i}", "b": f"tag-{i}", "c": f"grp-{i}", "power": "ON"}
            for i in range(600)
        ],
        "sites": [{"city": "Lyon"}, {"city": "Porto"}, {"city": "Gdansk"}],
    }
    indexed = places.index(large)
    assert {"Lyon", "Porto", "Gdansk"} <= set(indexed)

    # Genuine uppercase ISO region columns resolve and disambiguate globally.
    regions = places.index({
        "sites": [
            {"city": "London", "state": "ON"},
            {"city": "Santa Maria", "state": "RS"},
            {"city": "Albany", "state": "WA"},
            {"city": "San Antonio", "state": "VS"},
            {"city": "Campinas", "state": "SP"},
        ]
    })
    assert regions["London, ON"]["country"] == "Canada"
    assert regions["Santa Maria, RS"]["country"] == "Brazil"
    assert regions["Albany, WA"]["country"] == "Australia"
    assert regions["San Antonio, VS"]["country"] == "Chile"
    assert regions["Campinas"]["country"] == "Brazil"


def test_a_large_dataset_still_scopes_a_town_to_its_region():
    ids = [{"order": f"order-{i:05d}"} for i in range(places.MAX_STRINGS + 50)]
    found = places.index([*ids, {"city": "London", "state": "Ontario"}, {"city": "Toronto", "state": "Ontario"}])
    assert found["London, Ontario"]["country"] == "Canada"
    assert "London" not in found


def test_index_resolves_each_code_pair_once(monkeypatch):
    calls = []
    real = places.resolve
    monkeypatch.setattr(places, "resolve", lambda text: calls.append(text) or real(text))
    places.index([
        {"city": "London", "state": "ON"}, {"city": "Toronto", "state": "ON"},
        {"city": "Ottawa", "state": "ON"},
    ])
    assert len(calls) == len(set(calls))
