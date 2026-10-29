from decimal import Decimal
import importlib

import pytest

from anton.core.artifacts.static.data import Evidence
from anton.core.artifacts.static.geography import Gazetteer, gazetteer, world_reference
from anton.core.llm.provider import LLMResponse, StreamComplete, ToolCall
from anton.core.tools.generate_artifact.static_v2 import places
from tests.test_static_artifact_pipeline_v2 import state, VALID


def city(number, name='Áurea', country='AA', region='01', population=100, lat='0', lon='0'):
    return [str(number), name, 'Aurea', 'Golden Town,Old Aurea', lat, lon, 'P', 'PPL', country, '', region, '', '', '', str(population), '', '', 'Etc/UTC', '2026-01-01']


@pytest.fixture
def index(monkeypatch):
    result = Gazetteer({'AA': {'alpha3': 'AAA', 'name': 'Alpha'}, 'BB': {'alpha3': 'BBB', 'name': 'Beta'}},
        {'AA.01': ['North', 'North'], 'AA.02': ['South', 'South']},
        [city(1), city(2, region='02'), city(3, country='BB', population=1000)])
    monkeypatch.setattr(places, 'gazetteer', lambda: result)
    return result


def test_ambiguity_context_and_zero_centroids(index):
    result = index.lookup('aurea')
    assert result['total_candidates'] == 3
    assert [row['place_id'] for row in result['candidates']] == [3, 1, 2]
    assert 'resolved' not in result
    narrowed = index.lookup('Golden Town', country='AAA', region='North')['candidates']
    assert [row['place_id'] for row in narrowed] == [1]
    assert narrowed[0]['latitude'] == narrowed[0]['longitude'] == Decimal(0)
    assert narrowed[0]['coordinate_kind'] == 'place_centroid'
    assert index.lookup('Áurea', country='Alpha', region='missing')['candidates'] == []
    assert index.lookup('missing')['candidates'] == []
    with pytest.raises(ValueError, match='No global fallback'):
        index.lookup('aurea', country='unrecognized')
    narrowed[0]['name'] = 'changed'
    assert index.lookup('aurea', country='AA', region='01')['candidates'][0]['name'] == 'Áurea'


def test_candidate_limit_is_explicit():
    index = Gazetteer({}, {}, [city(i) for i in range(20)])
    result = index.lookup('aurea')
    assert result['total_candidates'] == 20 and result['returned_candidates'] == 8
    assert not result['complete_candidates']


def test_global_reference_load_and_fresh_geojson():
    index = gazetteer()
    assert len(index.rows) == 69985 and len(index.countries) > 230
    world = world_reference()
    assert world['data']['type'] == 'FeatureCollection' and len(world['data']['features']) == 177
    world['data']['features'].clear()
    assert len(world_reference()['data']['features']) == 177


async def test_exact_reference_merge_preserves_business_records_and_missing(index):
    refs = places.Places()
    await refs.lookup({'places': [{'key': 'supplied-role', 'place': 'Aurea', 'country': 'AA', 'region': '01'}, {'key': 'unknown', 'place': 'NoSuchPlace'}]})
    evidence = Evidence.model_validate({'datasets': {'assets': {'source': {'label': 'User'}, 'rows': [{'id': '007', 'latitude': 0, 'longitude': 0}]}}})
    result = refs.enrich(evidence)
    assert list(evidence.datasets) == ['assets']
    assert result.datasets['assets'] == evidence.datasets['assets']
    reference = result.datasets[places.DATASET]
    assert reference.source.kind == 'reference' and reference.source.attribution.startswith('GeoNames')
    assert reference.rows[0]['latitude'] == Decimal(0)
    assert reference.rows[1]['request_key'] == 'unknown' and reference.rows[1]['status'] == 'unresolved'
    assert reference.rows[1]['candidate_count'] == 0 and reference.rows[1]['candidates_complete']
    assert all(reference.rows[1][field] is None for field in places.CANDIDATE_FIELDS)
    assert reference.rows[0].keys() == reference.rows[1].keys()
    await refs.lookup({'places': [{'key': 'supplied-role', 'place': 'Missing'}]})
    updated = refs.enrich(result, reused=True)
    assert len(updated.datasets[places.DATASET].rows) == 2
    assert updated.datasets[places.DATASET].rows[0]['status'] == 'unresolved'
    assert result.datasets[places.DATASET].rows[0]['status'] == 'candidate'
    with pytest.raises(ValueError, match='host-owned'):
        refs.enrich(result)


async def test_lookup_is_atomic_bounded_and_validates_distinct_keys(index):
    refs = places.Places()
    with pytest.raises(ValueError, match='distinct'):
        await refs.lookup({'places': [{'key': 'a', 'place': 'Aurea'}, {'key': 'a', 'place': 'Aurea'}]})
    with pytest.raises(ValueError, match='Unknown country'):
        await refs.lookup({'places': [{'key': 'a', 'place': 'Aurea'}, {'key': 'b', 'place': 'Aurea', 'country': 'bad'}]})
    assert not refs.requests
    refs.requests = {str(i): {} for i in range(100)}
    with pytest.raises(ValueError, match='100 distinct'):
        await refs.lookup({'places': [{'key': 'new', 'place': 'Aurea'}]})


async def test_reference_lookup_cannot_establish_business_source_unavailability(tmp_path, index):
    module = importlib.import_module('anton.core.tools.generate_artifact.static_v2.collect')
    seen = []
    async def stream(**kwargs):
        seen.append(kwargs)
        if len(seen) == 1:
            calls = [ToolCall('lookup', 'lookup_static_places', {'places': [{'key': 'a', 'place': 'Aurea'}]}), ToolCall('stop', 'finish_static_evidence', {'ready': False, 'has_records': True, 'reason': 'Business source unavailable'})]
        else:
            assert 'no source read has run yet' in str(kwargs['messages'])
            calls = [ToolCall('finish', 'finish_static_evidence', {'ready': True, 'has_records': False, 'reason': 'Qualitative map with references only'})]
        yield StreamComplete(response=LLMResponse(content='', tool_calls=calls))
    st = state(tmp_path, VALID)
    st.session._llm.plan_stream = st.session._llm.code_stream = stream
    result = await module.collect(st)
    assert isinstance(result, Evidence) and len(result.datasets[places.DATASET].rows) == 3
    assert not st.static_source_reads


async def test_rejected_evidence_is_never_returned_as_accepted(tmp_path, index):
    module = importlib.import_module('anton.core.tools.generate_artifact.static_v2.collect')
    async def stream(**kwargs):
        yield StreamComplete(response=LLMResponse(content='', tool_calls=[ToolCall('finish', 'finish_static_evidence', {'ready': True, 'has_records': True, 'reason': 'Done'})]))
    st = state(tmp_path, VALID)
    st.session._llm.plan_stream = st.session._llm.code_stream = stream
    # The collector uses a fresh bound path. Force a syntactically valid, empty payload.
    from unittest.mock import patch
    with patch.object(module, 'read_bytes_nofollow', return_value=b'{"datasets":{}}'):
        result = await module.collect(st)
    assert isinstance(result, str) and 'did not finish' in result


async def test_snapshot_enrichment_keeps_original_records_and_acquisition_time(tmp_path, index):
    from tests.test_static_artifact_snapshots_v2 import published
    module = importlib.import_module('anton.core.tools.generate_artifact.static_v2.collect')
    st, original = published(tmp_path)
    async def stream(**kwargs):
        yield StreamComplete(response=LLMResponse(content='', tool_calls=[
            ToolCall('lookup', 'lookup_static_places', {'places': [{'key': '007-current', 'place': 'Aurea', 'country': 'AA', 'region': '01'}]}),
            ToolCall('reuse', 'reuse_static_evidence', {'reason': 'Add approximate city context without refreshing records'})]))
    st.session._llm.plan_stream = stream
    enriched = await module.collect(st)
    assert enriched.datasets['items'] == original.datasets['items']
    assert len(enriched.datasets[places.DATASET].rows) == 1
    assert st.static_evidence_origin['mode'] == 'reused'
    assert st.static_evidence_origin['reference_enriched']
    assert st.static_evidence_origin['acquired_at'] == '2026-10-28T10:00:00+00:00'


async def test_place_lookup_cannot_clear_failed_business_read(tmp_path, monkeypatch, index):
    from anton.core.tools.registry import ToolOutcome
    module = importlib.import_module('anton.core.tools.generate_artifact.static_v2.collect')
    seen = []
    async def failed(*args):
        return ToolOutcome(content='Read failed', ok=False)
    monkeypatch.setattr('anton.core.tools.tool_handlers.handle_scratchpad', failed)
    async def stream(**kwargs):
        seen.append(kwargs)
        if len(seen) == 1:
            calls = [ToolCall('read', 'scratchpad', {'action': 'exec', 'code': 'read_source()'}),
                ToolCall('lookup', 'lookup_static_places', {'places': [{'key': 'a', 'place': 'Aurea'}]}),
                ToolCall('finish', 'finish_static_evidence', {'ready': True, 'has_records': False, 'reason': 'Reference succeeded'})]
        else:
            assert 'A source tool failed' in str(kwargs['messages'])
            calls = [ToolCall('stop', 'finish_static_evidence', {'ready': False, 'has_records': True, 'reason': 'Business read failed'})]
        yield StreamComplete(response=LLMResponse(content='', tool_calls=calls))
    st = state(tmp_path, VALID)
    st.session._llm.plan_stream = st.session._llm.code_stream = stream
    assert 'Business read failed' in await module.collect(st)
