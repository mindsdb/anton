"""Private published evidence, paired with the exact page that used it."""
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
import json

from anton.core.artifacts.static.data import Evidence, exact_json
from anton.utils.nofollow import read_bytes_nofollow, replace_text_nofollow


def now():
    return datetime.now(timezone.utc).isoformat()


def load(state):
    """No orphaned, changed, or unvalidated snapshot is an editable source."""
    try:
        page = read_bytes_nofollow(state.artifact_path, state.artifact_path / (state.primary or 'index.html'), limit=8 * 1024 * 1024)
        page_hash = sha256(page).hexdigest()
        path = state.artifact_path / '.static' / f'published-{page_hash}.json'
        record = json.loads(read_bytes_nofollow(state.artifact_path, path, limit=32 * 1024 * 1024))
        payload = record['evidence_json']
        if not isinstance(payload, str) or not isinstance(record['float_fields'], list):
            return None
        if record['page_sha256'] != page_hash or record['payload_sha256'] != sha256(payload.encode()).hexdigest():
            return None
        data = json.loads(payload, parse_float=Decimal)
        for name, index, field in record['float_fields']:
            data['datasets'][name]['rows'][index][field] = float(data['datasets'][name]['rows'][index][field])
        evidence = Evidence.model_validate(data)
        if evidence.digest() != record['evidence_sha256']:
            return None
        acquired_at = record['acquired_at']
        datetime.fromisoformat(acquired_at)
        if not evidence.datasets:
            return None
        return evidence, {'mode': 'reused', 'acquired_at': acquired_at, 'page_sha256': page_hash}
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError):
        return None


def save(state, html, evidence):
    page_hash = sha256(html.encode()).hexdigest()
    origin = state.static_evidence_origin
    payload = exact_json(evidence.model_dump())
    record = {'page_sha256': page_hash, 'payload_sha256': sha256(payload.encode()).hexdigest(),
        'evidence_sha256': evidence.digest(),
        'float_fields': [(name, index, field) for name, dataset in evidence.datasets.items()
            for index, row in enumerate(dataset.rows) for field, value in row.items() if type(value) is float],
        'acquired_at': origin.get('acquired_at', now()), 'evidence_json': payload}
    # A failed final page swap leaves an unused snapshot, never a new source
    # for the previous page. This directory is excluded from artifact exports.
    replace_text_nofollow(state.artifact_path,
        state.artifact_path / '.static' / f'published-{page_hash}.json',
        exact_json(record), create_parents=True)
