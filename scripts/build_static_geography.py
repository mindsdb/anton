"""Build the vendored geographic references from the exact recorded source bytes.

The GeoNames inputs are daily extracts: preserve the matching downloaded files
when reproducing a snapshot. Runtime lookups never contact a geocoding service.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED


def build(inputs: Path, target: Path):
    target.mkdir(parents=True, exist_ok=True)
    downloads = json.loads((inputs / 'downloads.json').read_text())
    world_source = json.loads((inputs / 'world-download.json').read_text())
    for item in downloads:
        assert hashlib.sha256((inputs / item['file']).read_bytes()).hexdigest() == item['sha256']
    raw = (inputs / 'world-countries.geojson').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == world_source['sha256']
    original = json.loads(raw)
    world = {'type': 'FeatureCollection', 'features': []}
    for feature in original['features']:
        props = feature['properties']
        code = props['ISO_A2']
        if code == '-99':
            code = props.get('ISO_A2_EH', code)
        world['features'].append({'type': 'Feature', 'id': props['ADM0_A3'],
            'properties': {'name': props['ADMIN'], 'iso_a2': code}, 'geometry': feature['geometry']})
    encoded = json.dumps(world, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
    (target / 'world-countries.geojson.gz').write_bytes(gzip.compress(encoded, mtime=0))
    with ZipFile(inputs / 'cities5000.zip') as original_cities:
        members = {'cities5000.txt': original_cities.read('cities5000.txt')}
    members.update({name: (inputs / name).read_bytes() for name in ['countryInfo.txt', 'admin1CodesASCII.txt', 'readme.txt']})
    with ZipFile(target / 'geonames-cities5000.zip', 'w', compression=ZIP_DEFLATED, compresslevel=9) as bundle:
        for name, content in members.items():
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            bundle.writestr(info, content)
    outputs = {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in target.iterdir() if p.is_file() and p.name not in ['manifest.json', 'NOTICE.txt']}
    manifest = {'version': 1, 'snapshot': '2026-10-29', 'geonames_inputs': downloads,
        'world_input': world_source, 'outputs': outputs,
        'geonames_members': {name: {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()} for name, content in members.items()}, 'world_features': len(world['features']),
        'world_processing': 'Only feature id/name/ISO country code and unchanged upstream geometry retained; gzip mtime=0.',
        'place_coverage': 'GeoNames cities5000: cities over 5,000 population or administrative seats. Includes multilingual/historical aliases. Not a complete gazetteer or observed GPS positions.'}
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('target', type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.inputs, args.target)))
