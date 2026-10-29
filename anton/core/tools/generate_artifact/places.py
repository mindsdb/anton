"""Offline lookup of place names found in source data, so a map can show them.

The index (data/places.tsv.gz, built by scripts/build_places.py from GeoNames, CC BY 4.0) holds
towns of 5,000 people or more with their countries and regions. A name resolves only to one
place: "Stratford, Ontario" picks the Canadian town, a bare "Stratford" stays unresolved.
"""
from __future__ import annotations

import gzip
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).parent / "data" / "places.tsv.gz"
# A bare name that several towns share still resolves when one is this many times larger
DOMINANT = 10
MAX_STRINGS = 4096


def normalized(value):
    return "".join(c for c in unicodedata.normalize("NFKD", str(value)).casefold() if c.isalnum())


@lru_cache(maxsize=1)
def _index():
    countries, regions, names = {}, {}, {}
    with gzip.open(DATA, "rt", encoding="utf-8") as rows:
        for row in rows:
            kind, *f = row.rstrip("\n").split("\t")
            if kind == "C":
                code, iso3, name = f
                countries[code] = name
                for key in (code, iso3, name):
                    regions.setdefault(normalized(key), set()).add((code, None))
            elif kind == "A":
                code, name, ascii_name = f
                country, region = code.split(".", 1)
                keys = {name, ascii_name} | ({region} if country == "US" else set())  # US.CA is "CA"
                for key in keys:
                    regions.setdefault(normalized(key), set()).add((country, region))
            else:
                name, lat, lon, country, region, population, alternates = f
                place = {"name": name, "lat": float(lat), "lon": float(lon), "country": country,
                         "region": region, "population": int(population or 0)}
                names.setdefault(normalized(name), []).append((place, True))
                for alt in filter(None, alternates.split("|")):
                    names.setdefault(normalized(alt), []).append((place, False))
    return countries, regions, names


def resolve(text):
    """The one place a source string names, or None when it names none or several."""
    # A place name has a capital; a short all-capitals word is a code ("APAC", "ORD")
    if (not isinstance(text, str) or not 2 <= len(text) <= 120 or text == text.lower()
            or re.fullmatch(r"[A-Z]{2,4}", text.strip())):
        return None
    countries, regions, names = _index()
    town, *qualifiers = [part.strip() for part in text.split(",")]
    found = names.get(normalized(town), [])
    # A bare country name is no town ("Canada"), except a city-state's own city ("Hong Kong"). A
    # bare region name names its own city ("Busan", "Berlin"), not a town elsewhere ("Ontario")
    areas = regions.get(normalized(town), set()) if not qualifiers else set()
    if any(region is None for _, region in areas):
        found = [(p, own) for p, own in found if own and (p["country"], None) in areas]
    for qualifier in qualifiers:
        scopes = regions.get(normalized(qualifier), set())
        found = [(p, own) for p, own in found
                 if (p["country"], None) in scopes or (p["country"], p["region"]) in scopes]
    # A town's own name outranks another town's alternate name
    own = [p for p, is_own in found if is_own] or [p for p, _ in found]
    places = sorted({id(p): p for p in own}.values(), key=lambda p: -p["population"])
    if not places or len(places) > 1 and places[0]["population"] < DOMINANT * places[1]["population"]:
        return None
    place = places[0]
    if areas and (place["country"], place["region"]) not in areas and (place["country"], None) not in areas:
        return None
    return {"lat": place["lat"], "lon": place["lon"], "name": place["name"],
            "country": countries.get(place["country"], place["country"]), "source": "GeoNames"}


# The writer sees no place list (most pages draw no map), only that place names can be mapped
CONTRACT = ("Place names in this data (towns, ports, cities, or a record's town and region joined as "
            "\"town, region\") can go on a map without coordinates: in a map or globe section give a "
            "point {place: name} and a route end the name; in plain HTML read "
            "window.ARTIFACT_PLACES[name]. Names with one known match are placed and the rest are "
            "listed under the map, so never leave a requested place off the map for lack of coordinates.")
_GEOGRAPHY = re.compile(r"""\btype\s*:\s*['"](?:map|globe)['"]|ARTIFACT_PLACES""")


def needed(code):
    """Whether the written page draws a map or globe, or reads the places itself."""
    return bool(_GEOGRAPHY.search(code or ""))


def index(data):
    """Every string in the data that names exactly one place, mapped to that place.

    A field counts only when most of its values are places, so a Mode of "Rail" or a Region
    of "APAC" is not a town. A record that also names a country or region adds "<value>,
    <region>" ("London, Ontario"), and then the bare value no longer resolves elsewhere
    ("London" alone would be the English city)."""
    _, regions, _ = _index()
    keys, scoped, values_seen = {}, {}, set()

    def add(text, key):
        if len(keys) < MAX_STRINGS:
            keys.setdefault(text, set()).add(key)

    def record(items):
        fields = {k: v for k, v in items.items() if isinstance(v, str) and 2 <= len(v) <= 60}
        for qualifier in (v for v in fields.values() if normalized(v) in regions):
            for k, text in fields.items():
                if text != qualifier and "," not in text:
                    add(f"{text}, {qualifier}", k)
                    scoped.setdefault(text, set()).add(f"{text}, {qualifier}")
        for k, item in items.items():
            visit(item, k)

    def visit(value, key=None):
        if isinstance(value, str):
            values_seen.add(value)
            add(value, key)
        elif isinstance(value, dict):
            record(value)
        elif isinstance(value, list) and value and all(isinstance(row, list) for row in value):
            for row in value:  # rows given as arrays: each column is a field
                record({(key, i): item for i, item in enumerate(row)})
        elif isinstance(value, list):
            for item in value:
                visit(item, key)

    visit(data)
    found = {text: place for text in sorted(keys) if (place := resolve(text))}
    made = {name for names in scoped.values() for name in names} - values_seen
    values = {}
    for text, ks in keys.items():
        if text not in made:
            for k in ks:
                values.setdefault(k, set()).add(text)
    placed = {k: sum(v in found or any(n in found for n in scoped.get(v, ())) for v in vs) / len(vs)
              for k, vs in values.items()}
    for text in list(found):
        base = text.rsplit(", ", 1)[0] if text in made else text
        if not any(k is None or placed.get(k, 0) > 0.5 for k in keys.get(base, ()) | keys[text]):
            del found[text]
    for text, names in scoped.items():
        here = [found[name] for name in names if name in found]
        if text in found and here and all(place != found[text] for place in here):
            del found[text]
        elif text in found:  # "Lyon, FR" adds nothing when "Lyon" already names the same town
            for name in names & made:
                found.pop(name, None)
        else:  # one made-up name per town: "Busan, South Korea" rather than also "Busan, KR"
            for name in sorted((n for n in names & made if n in found), key=len)[:-1]:
                if any(found[name] == found[o] for o in names if o in found and o != name):
                    del found[name]
    return found
