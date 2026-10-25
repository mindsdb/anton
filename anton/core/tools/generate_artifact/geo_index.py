"""Exact local place-name lookup. Ambiguous names stay ambiguous; no network or guesses."""
from functools import lru_cache
import unicodedata

from .observed_data import credential_bearing


class PlaceRecord(dict):
    """A JSON-compatible mapping whose named fields also support attribute access."""
    __slots__ = ()

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def normalized(value):
    return "".join(c for c in unicodedata.normalize("NFKD", value).casefold() if c.isalnum())


@lru_cache(maxsize=1)
def _gazetteer():
    from geonamescache import GeonamesCache

    cache = GeonamesCache()
    countries = cache.get_countries()
    country_names = {}
    for code, country in countries.items():
        for name in (code, country.get("iso3", ""), country["name"]):
            if name:
                country_names[normalized(name)] = code
    cities = {}
    for city in cache.get_cities().values():
        country = countries[city["countrycode"]]
        point = {"lat": city["latitude"], "lon": city["longitude"], "country": city["countrycode"],
                 "countryName": country["name"], "countryISO3": country.get("iso3", ""),
                 "name": city["name"], "geonameid": city["geonameid"], "source": "GeoNames"}
        for name in {normalized(n) for n in [city["name"], *city.get("alternatenames", [])]}:
            if name:
                cities.setdefault(name, {})[city["geonameid"]] = point
    return country_names, cities


def _place_scope(text, country_names):
    city, embedded = text, None
    if "," in text:
        prefix, suffix = text.rsplit(",", 1)
        embedded = country_names.get(normalized(suffix))
        if embedded:
            city = prefix
    return city, embedded or country_names.get(normalized(text))


def build(observed, *, country=None):
    if credential_bearing(observed):
        return {}
    strings = set()
    def visit(value):
        if len(strings) >= 4096:
            return
        if isinstance(value, str) and 2 <= len(value) <= 120:
            strings.add(value)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(observed)
    if not strings:
        return {}
    country_names, cities = _gazetteer()
    requested_country = country_names.get(normalized(str(country))) if country else None
    if country and not requested_country:
        return {}
    index = {}
    for text in sorted(strings):
        city_name, embedded_country = _place_scope(text, country_names)
        # A country name must not resolve to a same-named town in another
        # country. This still permits city-states' actual city records.
        if requested_country and embedded_country and requested_country != embedded_country:
            continue
        scoped_country = requested_country or embedded_country
        candidates = list(cities.get(normalized(city_name), {}).values())
        if scoped_country:
            candidates = [point for point in candidates if point["country"] == scoped_country]
        canonical = [point for point in candidates if normalized(point["name"]) == normalized(city_name)]
        candidates = canonical or candidates
        if 0 < len(candidates) <= 64:
            index[text] = candidates
    return index


def resolve_place(place, country=None):
    """Return a JSON mapping (also allowing .lat/.lon) for one unambiguous source place."""
    if not isinstance(place, str) or not place.strip():
        raise ValueError("A non-empty observed place name is required.")
    if country:
        country_names, _ = _gazetteer()
        requested = country_names.get(normalized(str(country)))
        _, embedded = _place_scope(place, country_names)
        if requested and embedded and requested != embedded:
            raise ValueError(f"Place {place!r} has 0 matches: country hint conflicts with the observed place. "
                             "Use only the country of this position, not an origin, destination or organization.")
    matches = build([place], country=country).get(place, [])
    if len(matches) != 1:
        raise ValueError(f"Place {place!r} has {len(matches)} matches; provide its source country or report it unresolved.")
    return PlaceRecord({**matches[0], "location": place})
