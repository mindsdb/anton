"""Offline geographic references. Candidates are not observed asset positions."""
from collections import defaultdict
import csv
from decimal import Decimal
from functools import lru_cache
import gzip
from hashlib import sha256
from importlib.resources import files
from io import BytesIO, TextIOWrapper
import json
import unicodedata
from zipfile import ZipFile


def normalize(value):
    return ''.join(c for c in unicodedata.normalize('NFKD', value).casefold() if c.isalnum())


@lru_cache(maxsize=1)
def manifest():
    return json.loads(files(__package__).joinpath('reference_data/manifest.json').read_text())


def asset(name):
    raw = files(__package__).joinpath('reference_data', name).read_bytes()
    expected = manifest()['outputs'][name]
    if len(raw) != expected['bytes'] or sha256(raw).hexdigest() != expected['sha256']:
        raise ValueError('The pinned geographic reference failed its integrity check')
    return raw


class Gazetteer:
    def __init__(self, countries, regions, rows):
        self.countries = countries
        self.country_codes = {}
        for code, info in countries.items():
            for alias in [code, info['alpha3'], info['name']]:
                if alias:
                    self.country_codes[normalize(alias)] = code
        self.regions = regions
        self.rows, self.index = [], defaultdict(list)
        for row in rows:
            record = {'place_id': int(row[0]), 'name': row[1], 'latitude': row[4],
                'longitude': row[5], 'country_code': row[8], 'admin1': row[10],
                'feature': row[7], 'population': int(row[14]), 'modified': row[18]}
            index = len(self.rows)
            self.rows.append(record)
            aliases = {normalize(value) for value in [row[1], row[2], *row[3].split(',')]}
            for alias in aliases - {''}:
                self.index[alias].append(index)

    def lookup(self, place, country='', region='', limit=8):
        code = self.country_codes.get(normalize(country)) if country else None
        if country and code is None:
            raise ValueError(f'Unknown country context {country!r}; use its source-supported ISO alpha-2/alpha-3 code or canonical country name. No global fallback was used.')
        records = [self.rows[index] for index in self.index.get(normalize(place), [])]
        if code is not None:
            records = [row for row in records if row['country_code'] == code]
        if region:
            wanted = normalize(region)
            records = [row for row in records if wanted in {normalize(value) for value in
                [row['admin1'], *self.regions.get(row['country_code']+'.'+row['admin1'], [])]}]
        records.sort(key=lambda row: (-row['population'], row['place_id']))
        candidates = []
        for row in records[:limit]:
            names = self.regions.get(row['country_code']+'.'+row['admin1'], [])
            candidates.append({**row, 'latitude': Decimal(row['latitude']), 'longitude': Decimal(row['longitude']),
                'country': self.countries.get(row['country_code'], {}).get('name', row['country_code']),
                'region': names[0] if names else '', 'coordinate_kind': 'place_centroid',
                'reference_url': f'https://www.geonames.org/{row["place_id"]}'})
        return {'place': place, 'country_context': country, 'region_context': region,
            'total_candidates': len(records), 'returned_candidates': len(candidates),
            'complete_candidates': len(records) <= limit, 'candidates': candidates}


@lru_cache(maxsize=1)
def gazetteer():
    with ZipFile(BytesIO(asset('geonames-cities5000.zip'))) as archive:
        countries = {}
        for row in csv.reader(archive.read('countryInfo.txt').decode().splitlines(), delimiter='\t'):
            if row and not row[0].startswith('#'):
                countries[row[0]] = {'alpha3': row[1], 'name': row[4]}
        regions = {row[0]: row[1:3] for row in csv.reader(archive.read('admin1CodesASCII.txt').decode().splitlines(), delimiter='\t') if row}
        with archive.open('cities5000.txt') as stream:
            return Gazetteer(countries, regions, csv.reader(TextIOWrapper(stream, encoding='utf-8'), delimiter='\t'))


def world_reference():
    raw = gzip.decompress(asset('world-countries.geojson.gz'))
    data = json.loads(raw)
    return {'data': data, 'source': {'label': 'Natural Earth country boundaries (1:110m)',
        'url': manifest()['world_input']['url'], 'kind': 'json', 'sha256': sha256(raw).hexdigest(),
        'bytes': len(raw), 'snapshot': manifest()['snapshot'], 'builtin': 'world-countries-110m'}}
