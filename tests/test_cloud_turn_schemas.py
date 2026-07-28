"""Exported V1 JSON Schemas + fixtures stay in sync with the models (items 5/6)."""

from __future__ import annotations

import json

import pytest
from pydantic import TypeAdapter

from anton.cloud_turn import fixtures as fx
from anton.cloud_turn import schema_export as sx
from anton.cloud_turn.protocol import TurnEventV1, TurnRequestV1


# ── item 5: committed schemas match the models ──────────────────────────────

def test_committed_schemas_match_models():
    """Regenerate from the Pydantic models and diff the committed files, so a
    protocol change can't land without updating the checked-in schema."""
    for path, expected in sx.expected_files().items():
        assert path.exists(), f"missing committed schema {path} — run schema_export"
        assert path.read_text(encoding="utf-8") == expected, (
            f"{path.name} is stale — run `python -m anton.cloud_turn.schema_export`"
        )


def test_schemas_are_valid_json_with_expected_roots():
    req = json.loads(sx.REQUEST_SCHEMA_PATH.read_text())
    ev = json.loads(sx.EVENT_SCHEMA_PATH.read_text())
    assert req["title"] == "TurnRequestV1"
    # The event schema is the discriminated union of the three event kinds.
    assert "oneOf" in ev or "anyOf" in ev or "$defs" in ev


# ── item 6: committed fixtures match the models + validate ───────────────────

def test_committed_fixtures_match_models():
    for path, expected in fx.expected_files().items():
        assert path.exists(), f"missing fixture {path} — run fixtures"
        assert path.read_text(encoding="utf-8") == expected, (
            f"{path.name} is stale — run `python -m anton.cloud_turn.fixtures`"
        )


_event_adapter = TypeAdapter(TurnEventV1)


@pytest.mark.parametrize("name", list(fx.FIXTURES))
def test_each_fixture_validates_against_its_model(name):
    """Round-trip each committed fixture through the SAME model the schema is
    generated from — proving fixtures conform to the exported contract."""
    data = json.loads((fx.FIXTURE_DIR / f"{name}.json").read_text())
    if name.startswith("request-"):
        TurnRequestV1.model_validate(data)
    else:
        _event_adapter.validate_python(data)


def test_representative_request_and_event_fixtures():
    # A representative request and a representative event, explicitly.
    req = json.loads((fx.FIXTURE_DIR / "request-with-history.json").read_text())
    parsed = TurnRequestV1.model_validate(req)
    assert parsed.history[0].content == "What is 2 + 2?"

    ev = json.loads((fx.FIXTURE_DIR / "event-turn-completed-tool-using.json").read_text())
    completed = _event_adapter.validate_python(ev)
    assert completed.kind == "turn.completed"
    assert completed.output_messages[1].content[0].type == "tool_result"


def test_invalid_request_fixture_has_null_ids():
    ev = json.loads(
        (fx.FIXTURE_DIR / "event-turn-failed-invalid-request-null-ids.json").read_text()
    )
    assert ev["run_id"] is None and ev["attempt_id"] is None
    assert ev["error"]["code"] == "invalid_request"
