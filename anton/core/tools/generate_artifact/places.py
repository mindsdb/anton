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
# ISO 3166-2 alphabetic admin1 codes across all countries (built by scripts/build_places.py).
# Matched uppercase-only so ordinary words like "on" or "act" never hijack city lookups.
REGION_CODES = Path(__file__).parent / "data" / "region_codes.tsv"
# A bare name that several towns share still resolves when one is this many times larger
DOMINANT = 10
MAX_STRINGS = 4096


def normalized(value):
    return "".join(c for c in unicodedata.normalize("NFKD", str(value)).casefold() if c.isalnum())


def _load_region_codes() -> dict[str, tuple[str, ...]]:
    if not REGION_CODES.exists():
        return {}
    mapping: dict[str, list[str]] = {}
    with REGION_CODES.open("r", encoding="utf-8") as rows:
        for row in rows:
            line = row.strip()
            if not line or line.startswith("#"):
                continue
            code, abbr = line.split("\t", 1)
            mapping.setdefault(code, []).append(abbr)
    return {code: tuple(abbrs) for code, abbrs in mapping.items()}


@lru_cache(maxsize=1)
def _index():
    countries, regions, region_codes, names = {}, {}, {}, {}
    iso_codes = _load_region_codes()
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
                for abbr in iso_codes.get(code, ()):
                    region_codes.setdefault(abbr, set()).add((country, region))
            else:
                name, lat, lon, country, region, population, alternates = f
                place = {"name": name, "lat": float(lat), "lon": float(lon), "country": country,
                         "region": region, "population": int(population or 0)}
                names.setdefault(normalized(name), []).append((place, True))
                for alt in filter(None, alternates.split("|")):
                    names.setdefault(normalized(alt), []).append((place, False))
    return countries, regions, region_codes, names


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
        base_scopes = regions.get(normalized(qualifier), set())
        base_found = [
            (p, own) for p, own in found
            if (p["country"], None) in base_scopes or (p["country"], p["region"]) in base_scopes
        ]
        if base_found:
            found = base_found
        else:
            stripped = qualifier.strip()
            code_scopes = region_codes.get(stripped, set()) if stripped.isupper() else set()
            found = [
                (p, own) for p, own in found
                if (p["country"], None) in code_scopes or (p["country"], p["region"]) in code_scopes
            ]
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
    made_keys: set[str] = set()
    lookups: dict[str, object] = {}

    def lookup(text):  # one resolve per distinct text
        if text not in lookups:
            lookups[text] = resolve(text)
        return lookups[text]
    table_values: dict[tuple[int, object], set[str]] = {}
    records: list[tuple[int, dict]] = []

    def add(text, key, made=False):
        # Raw strings and made-up "<town>, <region>" names are capped apart, so a
        # large dataset cannot crowd the pairs out and leave "London, Ontario" as England.
        if made:
            if text in made_keys or len(made_keys) < MAX_STRINGS:
                made_keys.add(text)
                keys.setdefault(text, set()).add(key)
        elif len(keys) < MAX_STRINGS:  # raw strings only arrive before any made name
            keys.setdefault(text, set()).add(key)

    def record(items, table_id):
        fields = {k: v for k, v in items.items() if isinstance(v, str) and 2 <= len(v) <= 60}
        if fields:
            records.append((table_id, fields))
            for k, v in fields.items():
                table_values.setdefault((table_id, k), set()).add(v)
        for k, item in items.items():
            visit(item, k)

    def visit(value, key=None, table_id=0):
        if isinstance(value, str):
            values_seen.add(value)
            add(value, key)
        elif isinstance(value, dict):
            record(value, table_id or id(value))
        elif isinstance(value, list) and value and all(isinstance(row, list) for row in value):
            tid = id(value)
            for row in value:  # rows given as arrays: each column is a field
                record({(key, i): item for i, item in enumerate(row)}, tid)
        elif isinstance(value, list):
            tid = id(value)
            for item in value:
                visit(item, key, tid)

    visit(data)

    # First synthesize pairs from full region/country names (never blocked by code heuristics),
    # and collect candidate uppercase ISO-code pairs separately without consuming MAX_STRINGS.
    code_candidates: dict[tuple[int, object, object], set[tuple[str, str]]] = {}
    for table_id, fields in records:
        for qk, qualifier in fields.items():
            in_regions = normalized(qualifier) in regions
            in_codes = not in_regions and qualifier.strip().isupper() and qualifier.strip() in region_codes
            if not (in_regions or in_codes):
                continue
            for k, text in fields.items():
                if text == qualifier or "," in text:
                    continue
                if in_regions:
                    made_name = f"{text}, {qualifier}"
                    add(made_name, k, made=True)
                    scoped.setdefault(text, set()).add(made_name)
                else:
                    code_candidates.setdefault((table_id, k, qk), set()).add((text, qualifier))

    # Admit an uppercase ISO-code qualifier column only when (1) most distinct values in that
    # table's column are valid regions/codes AND (2) >50% of ALL distinct (text, col_val) pairs
    # across rows of that table actually resolve to known places — preventing status/flag columns
    # ("ON"/"OK"/"UP") from saturating MAX_STRINGS or hijacking cities.
    passing_qks: set[tuple[int, object]] = set()
    for table_id, _, qk in code_candidates:
        if (table_id, qk) in passing_qks:
            continue
        qvals = table_values.get((table_id, qk), ())
        if qvals and (
            sum(normalized(v) in regions or (v.strip().isupper() and v.strip() in region_codes)
                for v in qvals) / len(qvals) > 0.5
        ):
            passing_qks.add((table_id, qk))

    if passing_qks:
        full_col_pairs: dict[tuple[int, object, object], set[tuple[str, str]]] = {}
        for table_id, fields in records:
            for qk, qualifier in fields.items():
                if (table_id, qk) not in passing_qks:
                    continue
                for k, text in fields.items():
                    if (table_id, k, qk) in code_candidates and text != qualifier and "," not in text:
                        full_col_pairs.setdefault((table_id, k, qk), set()).add((text, qualifier))
        for (table_id, k, qk), pairs in code_candidates.items():
            if (table_id, qk) not in passing_qks:
                continue
            # At most MAX_STRINGS pairs a column, as `add` bounds the raw strings.
            total_pairs = sorted(full_col_pairs.get((table_id, k, qk), pairs))[:MAX_STRINGS]
            resolved_all = sum(1 for text, qual in total_pairs if lookup(f"{text}, {qual}"))
            if total_pairs and resolved_all / len(total_pairs) > 0.5:
                for text, qual in sorted(pairs)[:MAX_STRINGS]:
                    if lookup(f"{text}, {qual}"):
                        made_name = f"{text}, {qual}"
                        add(made_name, k, made=True)
                        scoped.setdefault(text, set()).add(made_name)

    found = {text: place for text in sorted(keys) if (place := lookup(text))}
    made = {name for names in scoped.values() for name in names} - values_seen
    values = {}
    for text, ks in keys.items():
        if text in made:
            # Counts for its town when the cap kept the bare town out of the keys.
            text = text.rsplit(", ", 1)[0]
            if text in keys:
                continue
        for k in ks:
            values.setdefault(k, set()).add(text)
    placed = {k: sum(v in found or any(n in found for n in scoped.get(v, ())) for v in vs) / len(vs)
              for k, vs in values.items()}
    for text in list(found):
        base = text.rsplit(", ", 1)[0] if text in made else text
        if not any(k is None or placed.get(k, 0) > 0.5 for k in keys.get(base, set()) | keys[text]):
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
