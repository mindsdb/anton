"""Check complete public reference snapshots, not a task-specific place list."""
import csv
import gzip
import hashlib
from importlib.resources import files
from io import StringIO
import json
from decimal import Decimal
from zipfile import ZipFile

ROOT=files('anton.core.artifacts.static').joinpath('reference_data')


def test_geography_snapshot_bytes_match_the_recorded_provenance():
    manifest=json.loads(ROOT.joinpath('manifest.json').read_text())
    for name,expected in manifest['outputs'].items():
        raw=ROOT.joinpath(name).read_bytes()
        assert len(raw)==expected['bytes']
        assert hashlib.sha256(raw).hexdigest()==expected['sha256']
    with ROOT.joinpath('geonames-cities5000.zip').open('rb') as stream,ZipFile(stream) as archive:
        assert set(archive.namelist())==set(manifest['geonames_members'])
        for name,expected in manifest['geonames_members'].items():
            raw=archive.read(name)
            assert len(raw)==expected['bytes'] and hashlib.sha256(raw).hexdigest()==expected['sha256']
    notice=ROOT.joinpath('NOTICE.txt').read_text()
    assert 'CC BY 4.0' in notice and 'Public domain' in notice


def test_complete_global_city_extract_has_valid_unique_records_and_coordinates():
    with ROOT.joinpath('geonames-cities5000.zip').open('rb') as stream,ZipFile(stream) as archive:
        rows=list(csv.reader(StringIO(archive.read('cities5000.txt').decode()),delimiter='\t'))
    assert len(rows)>50_000 and len({row[0] for row in rows})==len(rows)
    for row in rows:
        assert len(row)==19 and row[1] and int(row[0])>0 and int(row[14])>=0
        assert -90<=Decimal(row[4])<=90 and -180<=Decimal(row[5])<=180
    with ROOT.joinpath('geonames-cities5000.zip').open('rb') as stream,ZipFile(stream) as archive:
        countries=[row for row in archive.read('countryInfo.txt').decode().splitlines() if row and not row.startswith('#')]
    assert len(countries)>230


def test_world_reference_has_named_polygon_features_and_no_thematic_business_fields():
    world=json.loads(gzip.decompress(ROOT.joinpath('world-countries.geojson.gz').read_bytes()))
    assert world['type']=='FeatureCollection' and len(world['features'])==177
    assert len({f['id'] for f in world['features']})==177
    for feature in world['features']:
        assert set(feature['properties'])=={'name','iso_a2'} and feature['properties']['name']
        assert feature['geometry']['type'] in ['Polygon','MultiPolygon']
