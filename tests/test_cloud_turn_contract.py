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
    {"protocol_version": True, "connections": [{"connection_id": 7, "credential_version": 3}]},
    {"protocol_version": 1.0, "connections": [{"connection_id": 7, "credential_version": 3}]},
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


def _interactive_request(value):
    body = {"protocol_version": 1, "conversation_id": "c", "input": "hi"}
    if value is not ...:
        body["interactive"] = value
    return TurnRequestV1.from_json(json.dumps(body))


def test_interactive_true_is_read():
    assert _interactive_request(True).interactive is True


@pytest.mark.parametrize("value", [..., False, None, 1, "true", {"on": True}])
def test_interactive_is_false_unless_a_real_true(value):
    assert _interactive_request(value).interactive is False
def test_turn_request_parses_the_connectors_block():
    connectors = {"google_drive": {"usage_notes": "Pass supportsAllDrives=true."}}
    request = TurnRequestV1.from_json(json.dumps({
        "protocol_version": 1, "conversation_id": "c", "input": "q",
        "connectors": connectors,
    }))
    assert request.connectors == connectors


@pytest.mark.parametrize("value", [None, "x", ["google_drive"], 3])
def test_a_missing_or_non_object_connectors_block_parses_to_none(value):
    body = {"protocol_version": 1, "conversation_id": "c", "input": "q"}
    if value is not None:
        body["connectors"] = value
    assert TurnRequestV1.from_json(json.dumps(body)).connectors is None


def _limits_request(value):
    body = {"protocol_version": 1, "conversation_id": "c", "input": "q"}
    if value is not ...:
        body["limits"] = value
    return TurnRequestV1.from_json(json.dumps(body))


def test_the_limits_block_is_read():
    limits = {"max_tool_rounds": 250, "max_continuations": 10, "max_turn_tokens": 25_000_000}
    assert _limits_request(limits).limits == limits


@pytest.mark.parametrize("value", [..., None, "x", [250], 25_000_000])
def test_a_missing_or_non_object_limits_block_reads_as_no_limits(value):
    assert _limits_request(value).limits == {}


def test_a_bad_limit_is_dropped_and_the_rest_are_kept():
    request = _limits_request({
        "max_tool_rounds": 250,
        # 0 means no ceiling to anton; a producer bug must not turn it off.
        "max_turn_tokens": 0,
        "max_continuations": True,
        "later_limit": 5,
    })
    assert request.limits == {"max_tool_rounds": 250}


@pytest.mark.parametrize("value", ["250", 2.5, -1, None])
def test_a_limit_that_is_not_a_usable_integer_is_dropped(value):
    assert _limits_request({"max_tool_rounds": value}).limits == {}


def test_zero_auto_continues_is_a_real_limit():
    assert _limits_request({"max_continuations": 0}).limits == {"max_continuations": 0}
