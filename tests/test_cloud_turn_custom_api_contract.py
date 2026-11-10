"""The `custom_api` block of a cloud turn: parsed exactly, as cowork-server's producer sends it."""

import json
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


def test_the_producers_block_parses_to_its_connections_and_turn_key():
    request = TurnRequestV1.from_json(_request(_PRODUCED))

    assert isinstance(request.custom_api, CustomApiBlockV1)
    assert request.custom_api.turn_key == _PRODUCED["turn_key"]
    assert request.custom_api.connections == tuple(
        CustomApiConnectionRefV1(c["connection_id"], c["connector_id"], c["name"], c["credential_version"])
        for c in _PRODUCED["connections"]
    )


def test_the_turn_key_never_shows_in_the_requests_repr():
    request = TurnRequestV1.from_json(_request(_PRODUCED))

    assert _PRODUCED["turn_key"] not in repr(request.custom_api)


def test_a_turn_without_the_block_parses_to_none():
    assert TurnRequestV1.from_json('{"protocol_version":1,"conversation_id":"c","input":"hi"}').custom_api is None


def test_the_cap_matches_cowork_servers():
    """cowork-server's cowork/turnqueue/models.py holds the same MAX_CUSTOM_API_CONNECTIONS."""
    assert MAX_CUSTOM_API_CONNECTIONS == 25


@pytest.mark.parametrize("custom_api", [
    "tk",
    {"connections": [_REF]},
    {"turn_key": "", "connections": [_REF]},
    {"turn_key": "k", "connections": [_REF], "fields": {"api_key": "planted-secret"}},
    {"turn_key": "k", "connections": []},
    {"turn_key": "k", "connections": [{**_REF, "connection_id": i} for i in range(1, 27)]},
    {"turn_key": "k", "connections": [{**_REF, "connection_id": True}]},
    {"turn_key": "k", "connections": [{**_REF, "credential_version": 0}]},
    {"turn_key": "k", "connections": [{**_REF, "name": ""}]},
    {"turn_key": "k", "connections": [{**_REF, "connector_id": 5}]},
    {"turn_key": "k", "connections": [{**_REF, "api_key": "planted-secret"}]},
    {"turn_key": "k", "connections": [_REF, _REF]},
], ids=["not_an_object", "no_turn_key", "empty_turn_key", "extra_field", "empty", "over_the_cap", "bool_id",
        "zero_version", "empty_name", "non_text_connector", "secret_bearing_reference", "duplicate_id"])
def test_a_malformed_block_refuses_the_request_without_quoting_it(custom_api):
    with pytest.raises(ValueError) as refused:
        TurnRequestV1.from_json(_request(custom_api))

    assert "planted-secret" not in str(refused.value)
