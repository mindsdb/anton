"""Generate committed JSON Schemas for the V1 wire contracts.

The schemas under ``schemas/`` are GENERATED from the Pydantic models here — do
not hand-edit them. ``tests/test_cloud_turn_schemas.py`` regenerates and diffs
against the committed files so a protocol change can't land without updating the
schema (and whoever reviews it).

Run ``python -m anton.cloud_turn.schema_export`` to rewrite the committed files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from anton.cloud_turn.protocol import TurnRequestV1, TurnEventV1

#: Repo-root-relative directory holding the committed schemas + fixtures.
SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"
REQUEST_SCHEMA_PATH = SCHEMA_DIR / "cloud-turn-request-v1.json"
EVENT_SCHEMA_PATH = SCHEMA_DIR / "cloud-turn-event-v1.json"


def request_schema() -> dict[str, Any]:
    """JSON Schema for :class:`TurnRequestV1`."""
    return TurnRequestV1.model_json_schema()


def event_schema() -> dict[str, Any]:
    """JSON Schema for the :data:`TurnEventV1` discriminated union."""
    return TypeAdapter(TurnEventV1).json_schema()


def _dump(schema: dict[str, Any]) -> str:
    # Stable, sorted, newline-terminated so diffs are minimal + deterministic.
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def expected_files() -> dict[Path, str]:
    """Map of committed schema path → its expected serialized content."""
    return {
        REQUEST_SCHEMA_PATH: _dump(request_schema()),
        EVENT_SCHEMA_PATH: _dump(event_schema()),
    }


def write_schemas() -> list[Path]:
    """(Re)write the committed schema files. Returns the paths written."""
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for path, content in expected_files().items():
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


if __name__ == "__main__":
    for p in write_schemas():
        print(f"wrote {p}")
