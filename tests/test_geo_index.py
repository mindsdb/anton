import copy
import sys
import json
from types import SimpleNamespace

import pytest

from anton.core.tools.generate_artifact import geo_index


@pytest.fixture
def gazetteer(monkeypatch):
    class Data:
        def get_countries(self):
            return {"AA":{"name":"Northland","iso3":"AAX"},"BB":{"name":"Southland","iso3":"BBX"}}
        def get_cities(self):
            return {i:{"geonameid":i,"name":name,"countrycode":country,"latitude":i,"longitude":i,
                       "alternatenames":aliases} for i,name,country,aliases in [
                           (1,"Harbor Town","AA",[]),(2,"Harbor Town","BB",[]),
                           (3,"Hillview","AA",["Híll-view"]),(4,"Northland","BB",[]),
                           (5,"Harbor Town Island","AA",["Harbor Town"]),
                           (6,"Milltown","AA",["Spring Hill"]),(7,"Spring Hill","BB",[])]}
    monkeypatch.setitem(sys.modules,"geonamescache",SimpleNamespace(GeonamesCache=Data))
    geo_index._gazetteer.cache_clear()
    yield
    geo_index._gazetteer.cache_clear()


def test_exact_names_keep_ambiguity_and_country_disambiguation_without_mutating_data(gazetteer):
    data={"rows":[{"location":s} for s in ["Harbor Town","Harbor Town, Northland","Hillview","Híll-view","Unknown"]]}
    original=copy.deepcopy(data);result=geo_index.build(data)
    assert len(result["Harbor Town"])==2
    assert [p["country"] for p in result["Harbor Town, Northland"]]==["AA"]
    assert result["Hillview"]==result["Híll-view"]
    assert "Unknown" not in result
    assert data==original


def test_country_name_does_not_resolve_to_an_unrelated_town(gazetteer):
    assert "Northland" not in geo_index.build({"location":"Northland"})


def test_neither_substring_guessing_nor_credential_values_are_geocoded(gazetteer):
    assert not geo_index.build({"description":"Somewhere near Hillview"})
    assert not geo_index.build({"password":"Hillview"})


def test_collection_resolves_only_the_requested_place_and_never_chooses_an_ambiguous_one(gazetteer):
    with pytest.raises(ValueError, match="2 matches"):
        geo_index.resolve_place("Harbor Town")
    point=geo_index.resolve_place("Harbor Town","Northland")
    assert point["country"]=="AA" and point["location"]=="Harbor Town"
    with pytest.raises(ValueError, match="0 matches"):
        geo_index.resolve_place("Unknown")


def test_country_scope_precedes_exact_name_preference_and_aliases_remain_available(gazetteer):
    # Exact Harbor Town names remain ambiguous; an island alias adds no ambiguity.
    assert len(geo_index.build(["Harbor Town"])["Harbor Town"]) == 2
    assert geo_index.resolve_place("Harbor Town", "Northland")["geonameid"] == 1
    assert geo_index.resolve_place("Spring Hill")["geonameid"] == 7
    assert geo_index.resolve_place("Spring Hill", "Northland")["geonameid"] == 6
    with pytest.raises(ValueError, match="0 matches"):
        geo_index.resolve_place("Hillview, Northland", "Southland")
    with pytest.raises(ValueError, match="0 matches"):
        geo_index.resolve_place("Hillview", "Unknown country")


def test_resolved_place_supports_mapping_and_attributes_without_changing_json_or_cached_data(gazetteer):
    point=geo_index.resolve_place('Hillview')
    assert point.lat==point['lat'] and point.lon==point['lon']
    assert point.country=='AA' and point.location=='Hillview'
    assert json.loads(json.dumps(point))==dict(point)
    with pytest.raises(AttributeError):
        _=point.nonexistent
    point['lat']=99
    assert geo_index.resolve_place('Hillview').lat==3


def test_conflicting_country_hint_identifies_position_role_error_without_guessing(gazetteer):
    with pytest.raises(ValueError,match='country hint conflicts'):
        geo_index.resolve_place('Hillview, Northland','Southland')
