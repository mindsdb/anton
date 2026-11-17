"""The `custom_api` block of a cloud turn: parsed exactly, as cowork-server's producer sends it."""

import json
import logging
from pathlib import Path

import pytest

from anton.cloud_turn.contract import (
    MAX_CUSTOM_API_CONNECTIONS,
    CustomApiBlockV1,
    CustomApiConnectionRefV1,
    TurnRequestV1,
)

# Recorded from cowork-server's producer (cowork/turnqueue/producer.py) with auth faked.
_PRODUCED = json.loads((Path(__file__).parent / "fixtures" / "turn_custom_api_block.json").read_text())["custom_api"]
_REF = _PRODUCED["connections"][0]


def _request(custom_api) -> str:
    return json.dumps({"protocol_version": 1, "conversation_id": "c", "input": "hi", "custom_api": custom_api})


def test_the_producers_block_parses_to_its_connections():
    request = TurnRequestV1.from_json(_request(_PRODUCED))

    assert isinstance(request.custom_api, CustomApiBlockV1)
    assert request.custom_api.protocol_version == 1
    assert request.custom_api.connections == tuple(
        CustomApiConnectionRefV1(c["connection_id"], c["connector_id"], c["name"], c["credential_version"])
        for c in _PRODUCED["connections"]
    )


def test_a_real_looking_key_never_shows_in_the_requests_repr():
    key = "mdb_turnkey.0123456789abcdef"
    body = {
        "protocol_version": 1, "conversation_id": "c", "input": "hi",
        "llm": {"provider": "minds-cloud", "api_key": key, "base_url": "https://minds.internal/v1"},
        "oauth": {"turn_key": key, "connections": []},
        "custom_api": _PRODUCED,
    }

    request = TurnRequestV1.from_json(json.dumps(body))

    assert request.llm["api_key"] == key
    assert key not in repr(request)


def test_a_turn_without_the_block_parses_to_none():
    assert TurnRequestV1.from_json('{"protocol_version":1,"conversation_id":"c","input":"hi"}').custom_api is None


def test_the_cap_matches_cowork_servers():
    """cowork-server's cowork/turnqueue/models.py holds the same MAX_CUSTOM_API_CONNECTIONS."""
    assert MAX_CUSTOM_API_CONNECTIONS == 25


_BAD_BLOCKS = {
    "unknown_version": {"protocol_version": 2, "connections": [_REF]},
    "bool_version": {"protocol_version": True, "connections": [_REF]},
    "missing_version": {"connections": [_REF]},
    "missing_connections": {"protocol_version": 1},
    "extra_key": {"protocol_version": 1, "connections": [_REF], "fields": {"api_key": "planted-secret"}},
    "old_shape": {"turn_key": "planted-secret", "connections": [_REF]},
    "not_an_object": "planted-secret",
    "not_a_list": {"protocol_version": 1, "connections": "planted-secret"},
    "empty": {"protocol_version": 1, "connections": []},
    "over_the_cap": {"protocol_version": 1, "connections": [{**_REF, "connection_id": i} for i in range(1, 27)]},
    "bool_id": {"protocol_version": 1, "connections": [{**_REF, "connection_id": True}]},
    "zero_version": {"protocol_version": 1, "connections": [{**_REF, "credential_version": 0}]},
    "empty_name": {"protocol_version": 1, "connections": [{**_REF, "name": ""}]},
    "non_text_connector": {"protocol_version": 1, "connections": [{**_REF, "connector_id": 5}]},
    "secret_bearing_reference": {"protocol_version": 1, "connections": [{**_REF, "api_key": "planted-secret"}]},
    "not_a_connection": {"protocol_version": 1, "connections": ["planted-secret"]},
    "duplicate_id": {"protocol_version": 1, "connections": [_REF, _REF]},
    "connector_id_shape": {"protocol_version": 1, "connections": [{**_REF, "connector_id": "Acme-CRM"}]},
    "connector_id_too_short": {"protocol_version": 1, "connections": [{**_REF, "connector_id": "a"}]},
    "name_too_long": {"protocol_version": 1, "connections": [{**_REF, "name": "n" * 129}]},
}


@pytest.mark.parametrize("custom_api", _BAD_BLOCKS.values(), ids=_BAD_BLOCKS.keys())
def test_an_unusable_block_is_dropped_with_one_warning_and_the_turn_goes_on(custom_api, caplog):
    with caplog.at_level(logging.WARNING, logger="anton.cloud_turn.contract"):
        request = TurnRequestV1.from_json(_request(custom_api))

    assert request.custom_api is None
    assert request.conversation_id == "c"
    warnings = [r for r in caplog.records if r.name == "anton.cloud_turn.contract"]
    assert len(warnings) == 1
    assert warnings[0].getMessage().startswith("custom_api block dropped: ")
    assert warnings[0].getMessage().endswith("(conversation c)")
    assert "planted-secret" not in caplog.text
