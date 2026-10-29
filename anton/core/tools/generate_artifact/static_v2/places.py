"""Bounded offline place candidates, merged without model-transcribed coordinates."""
import asyncio

from pydantic import Field, model_validator

from anton.core.artifacts.static.data import Evidence, Model, Source, json_value
from anton.core.artifacts.static.geography import gazetteer, manifest

DATASET = 'geonames_places'
CANDIDATE_FIELDS = ('place_id', 'name', 'latitude', 'longitude', 'country_code', 'admin1',
    'feature', 'population', 'modified', 'country', 'region', 'coordinate_kind', 'reference_url')


class Place(Model):
    key: str = Field(min_length=1, max_length=200)
    place: str = Field(min_length=1, max_length=200)
    country: str = Field(default='', max_length=100)
    region: str = Field(default='', max_length=100)


class Lookup(Model):
    places: list[Place] = Field(min_length=1, max_length=30)

    @model_validator(mode='after')
    def distinct(self):
        if len({place.key for place in self.places}) != len(self.places):
            raise ValueError('Place request keys must be distinct')
        return self


TOOL = {'name': 'lookup_static_places', 'description': 'Look up city/place-name candidates in a pinned offline GeoNames reference. Supply source-supported country and region context separately, and a stable key for joining back to the source. Returns at most eight candidates per place, with total counts. These are approximate city centroids, never observed asset positions. Aliases can be historical and coverage is incomplete: even one candidate needs contextual validation. Never automatically pick the first candidate, invent missing positions, or replace supplied coordinates (including zero). The host adds exact returned rows to the private geonames_places reference dataset; do not copy them into the evidence file. This supporting lookup does not access or replace the designated business source.', 'input_schema': Lookup.model_json_schema()}


def source():
    return Source(label='GeoNames offline place candidates', url='https://www.geonames.org/', kind='reference',
        attribution='GeoNames, CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).',
        scope=f"Snapshot {manifest()['snapshot']}. {manifest()['place_coverage']} Candidates require contextual validation; coordinates are approximate place centroids, not observed entity positions.")


class Places:
    def __init__(self):
        self.requests = {}

    async def lookup(self, arguments):
        request = Lookup.model_validate(arguments)
        if len(set(self.requests) | {place.key for place in request.places}) > 100:
            raise ValueError('At most 100 distinct place keys can be looked up in one evidence pass')

        def query():
            index = gazetteer()
            return {place.key: index.lookup(place.place, place.country, place.region) for place in request.places}

        results = await asyncio.to_thread(query)
        self.requests.update(results)
        return json_value({'dataset': DATASET, 'source': source().model_dump(), 'requests': results})

    def enrich(self, evidence: Evidence, *, reused=False):
        if not self.requests:
            return evidence
        existing = evidence.datasets.get(DATASET)
        if existing and (not reused or existing.source != source()):
            raise ValueError(f'{DATASET} is host-owned; rename that business dataset and omit reference rows from the evidence file')
        rows = [row for row in existing.rows if row.get('request_key') not in self.requests] if existing else []
        for key, result in self.requests.items():
            context = {'request_key': key, 'query_place': result['place'], 'query_country': result['country_context'],
                'query_region': result['region_context'], 'candidate_count': result['total_candidates'],
                'candidates_complete': result['complete_candidates']}
            rows.extend({**context, 'status': 'candidate', **candidate} for candidate in result['candidates'])
            if not result['candidates']:
                rows.append({**context, 'status': 'unresolved', **dict.fromkeys(CANDIDATE_FIELDS)})
        payload = evidence.model_dump()
        payload['datasets'][DATASET] = {'source': source().model_dump(), 'rows': rows}
        return Evidence.model_validate(payload)
