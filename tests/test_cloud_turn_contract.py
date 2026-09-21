import json

import pytest

from anton.cloud_turn.contract import (
    DatasourceBlockV1,
    TurnRequestV1,
)


def test_turn_request_parses_versioned_datasource_refs():
    request = TurnRequestV1.from_json(json.dumps({
        "protocol_version": 1,
        "conversation_id": "c",
        "correlation_id": "r",
        "input": "query",
        "datasource": {
            "protocol_version": 1,
            "connections": [{"connection_id": 7, "credential_version": 3}],
        },
    }))
    assert isinstance(request.datasource, DatasourceBlockV1)
    assert request.correlation_id == "r"
    assert request.datasource.connections[0].connection_id == 7


@pytest.mark.parametrize("datasource", [
    {"protocol_version": 2, "connections": [{"connection_id": 7, "credential_version": 3}]},
    {"protocol_version": 1, "connections": [{"connection_id": 7, "credential_version": 3}], "gateway_url": "https://evil"},
    {"protocol_version": 1, "connections": [{"connection_id": 7, "credential_version": 3}], "capability": "secret"},
    {"protocol_version": 1, "connections": [{"connection_id": 7, "credential_version": 0}]},
])
def test_turn_request_rejects_unsupported_or_secret_bearing_datasource(datasource):
    raw = json.dumps({
        "protocol_version": 1,
        "conversation_id": "c",
        "input": "query",
        "datasource": datasource,
    })
    with pytest.raises(ValueError) as exc_info:
        TurnRequestV1.from_json(raw)
    assert "evil" not in str(exc_info.value)
    assert "secret" not in str(exc_info.value)


def test_turn_request_without_a_datasource_block_parses_to_none():
    request = TurnRequestV1.from_json('{"protocol_version":1,"conversation_id":"c","input":"hi"}')
    assert request.datasource is None


@pytest.mark.parametrize("version", ["2", "1.9", "true", '"1"', "null"])
def test_turn_request_rejects_a_version_that_is_not_the_integer_one(version):
    """A coercing guard accepts 1.9 and true as 1, then parses an envelope that
    is not v1 with the v1 schema and silently drops what else it carried."""
    with pytest.raises(ValueError, match="unsupported cloud turn protocol version"):
        TurnRequestV1.from_json(
            f'{{"protocol_version":{version},"conversation_id":"c","input":"hi"}}'
        )
