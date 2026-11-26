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

# Uppercase-only postal codes for Canadian provinces/territories and non-US-colliding
# Australian states (WA is omitted so US Washington lookups remain unambiguous).
# Matched case-sensitively so ordinary words like "on" or "act" never hijack cities.
_UPPERCASE_REGION_CODES: dict[str, tuple[str, ...]] = {
    "CA.01": ("AB",), "CA.02": ("BC",), "CA.03": ("MB",), "CA.04": ("NB",),
    "CA.05": ("NL",), "CA.07": ("NS",), "CA.08": ("ON",), "CA.09": ("PE",),
    "CA.10": ("QC",), "CA.11": ("SK",), "CA.12": ("YT",), "CA.13": ("NT",), "CA.14": ("NU",),
    "AU.01": ("ACT",), "AU.02": ("NSW",), "AU.03": ("NT",), "AU.04": ("QLD",),
    "AU.05": ("SA",), "AU.06": ("TAS",), "AU.07": ("VIC",),
}


def normalized(value):
    return "".join(c for c in unicodedata.normalize("NFKD", str(value)).casefold() if c.isalnum())


@lru_cache(maxsize=1)
def _index():
    countries, regions, region_codes, names = {}, {}, {}, {}
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
                for abbr in _UPPERCASE_REGION_CODES.get(code, ()):
                    region_codes.setdefault(abbr, set()).add((country, region))
            else:
                name, lat, lon, country, region, population, alternates = f
                place = {"name": name, "lat": float(lat), "lon": float(lon), "country": country,
                         "region": region, "population": int(population or 0)}
                names.setdefault(normalized(name), []).append((place, True))
                for alt in filter(None, alternates.split("|")):
                    names.setdefault(normalized(alt), []).append((place, False))
    return countries, regions, region_codes, names


def _region_scopes(qualifier: str, regions: dict, region_codes: dict) -> set:
    scopes = set(regions.get(normalized(qualifier), set()))
    stripped = qualifier.strip()
    if stripped.isupper():
        scopes |= region_codes.get(stripped, set())
    return scopes


def resolve(text):
    """The one place a source string names, or None when it names none or several."""
    # A place name has a capital; a short all-capitals word is a code ("APAC", "ORD")
    if (not isinstance(text, str) or not 2 <= len(text) <= 120 or text == text.lower()
            or re.fullmatch(r"[A-Z]{2,4}", text.strip())):
        return None
    countries, regions, region_codes, names = _index()
    town, *qualifiers = [part.strip() for part in text.split(",")]
    found = names.get(normalized(town), [])
    # A bare country name is no town ("Canada"), except a city-state's own city ("Hong Kong"). A
    # bare region name names its own city ("Busan", "Berlin"), not a town elsewhere ("Ontario")
    areas = regions.get(normalized(town), set()) if not qualifiers else set()
    if any(region is None for _, region in areas):
        found = [(p, own) for p, own in found if own and (p["country"], None) in areas]
    for qualifier in qualifiers:
        scopes = _region_scopes(qualifier, regions, region_codes)
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
    _, regions, region_codes, _ = _index()
    keys, scoped, values_seen = {}, {}, set()

    def add(text, key):
        if len(keys) < MAX_STRINGS:
            keys.setdefault(text, set()).add(key)

    def record(items):
        fields = {k: v for k, v in items.items() if isinstance(v, str) and 2 <= len(v) <= 60}
        for qualifier in (
            v for v in fields.values()
            if normalized(v) in regions or (v.strip().isupper() and v.strip() in region_codes)
        ):
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
