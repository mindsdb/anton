"""Offline place lookup for maps: "City", "City, Country" or "City, Region, Country" to (lat, lon).

Uses the GeoNames cities table shipped with `geonamescache` (towns over 15,000 people, CC BY 4.0
GeoNames). Returns None rather than guessing when a name is unknown or ambiguous in the country
given, so a map never shows an invented place.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

_PREFIX = re.compile(r"^(port of|port|puerto de|puerto|porto de|port de|hafen|city of|ciudad de|greater)\s+", re.I)
_SUFFIX = re.compile(r"\s+(port|harbou?r|terminal|airport|city|metro|area)$", re.I)
_COUNTRY_ALIASES = {
    "usa": "US", "us": "US", "u.s.": "US", "u.s.a.": "US", "united states of america": "US", "america": "US",
    "uk": "GB", "u.k.": "GB", "england": "GB", "scotland": "GB", "wales": "GB", "great britain": "GB", "britain": "GB",
    "south korea": "KR", "korea": "KR", "republic of korea": "KR", "north korea": "KP", "russia": "RU",
    "uae": "AE", "emirates": "AE", "vietnam": "VN", "viet nam": "VN", "taiwan": "TW", "czech republic": "CZ",
    "czechia": "CZ", "holland": "NL", "the netherlands": "NL", "turkiye": "TR", "türkiye": "TR", "ivory coast": "CI",
    "hong kong": "HK", "macau": "MO", "bolivia": "BO", "venezuela": "VE", "iran": "IR", "syria": "SY", "laos": "LA",
    "tanzania": "TZ", "moldova": "MD", "brunei": "BN", "burma": "MM", "myanmar": "MM", "dr congo": "CD", "drc": "CD",
}
_US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO",
    "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
    "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY",
}


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


@lru_cache(maxsize=1)
def _index():
    try:
        import geonamescache  # type: ignore
    except ImportError:
        return None
    gc = geonamescache.GeonamesCache(min_city_population=15000)
    countries = gc.get_countries()
    by_name: dict[str, list[dict]] = {}
    for city in gc.get_cities().values():
        names = {_fold(city["name"])}
        for alt in city.get("alternatenames") or []:
            if alt and alt.isascii() and len(alt) > 2:
                names.add(_fold(alt))
        for n in names:
            if n:
                by_name.setdefault(n, []).append(city)
    cnames: dict[str, str] = {}
    for iso, c in countries.items():
        cnames[_fold(c["name"])] = iso
        cnames[iso.lower()] = iso
        cnames[c["iso3"].lower()] = iso
    for k, v in _COUNTRY_ALIASES.items():
        cnames[_fold(k)] = v
    return by_name, cnames


def _country(text: str, cnames: dict[str, str]) -> str | None:
    f = _fold(text)
    if f in cnames:
        return cnames[f]
    if f in _US_STATES or text.strip().upper() in _US_STATES.values():
        return "US"
    return None


def geocode(place: str, country: str | None = None) -> tuple[float, float] | None:
    """(lat, lon) of a town or city, or None when it isn't known with confidence.

    Accepts "Rotterdam", "Rotterdam, Netherlands", "Austin, TX, USA", "Port of Santos, Brazil".
    With a country (in the text or as `country=`), only towns there match; otherwise the most
    populous town of that name wins, unless two towns of similar size share it (then None).
    """
    idx = _index()
    if idx is None or not place:
        return None
    by_name, cnames = idx
    parts = [p.strip() for p in re.split(r"[,(/)]", str(place)) if p.strip()]
    if not parts:
        return None
    iso = _country(country, cnames) if country else None
    for p in reversed(parts[1:]):
        iso = iso or _country(p, cnames)
    city = parts[0]
    candidates = []
    for name in (city, _PREFIX.sub("", city), _SUFFIX.sub("", _PREFIX.sub("", city))):
        key = _fold(name)
        if key in by_name:
            candidates = by_name[key]
            break
    if not candidates:
        return None
    if iso:
        candidates = [c for c in candidates if c["countrycode"] == iso]
        if not candidates:
            return None
    candidates = sorted(candidates, key=lambda c: -c["population"])
    if not iso and len(candidates) > 1:
        a, b = candidates[0], candidates[1]
        if b["population"] > 0.5 * a["population"] and b["countrycode"] != a["countrycode"]:
            return None  # e.g. "Valencia" alone: Spain or Venezuela
    best = candidates[0]
    return round(float(best["latitude"]), 4), round(float(best["longitude"]), 4)


def geocode_many(places) -> dict[str, tuple[float, float] | None]:
    """{place: (lat, lon) or None} for each distinct place."""
    return {p: geocode(p) for p in dict.fromkeys(places) if p is not None}
