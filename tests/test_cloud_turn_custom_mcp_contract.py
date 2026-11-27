"""The `custom_mcp` block of a cloud turn: parsed exactly, each stored definition held to auth's published limits."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pytest

import anton.cloud_turn.contract as contract_mod
from anton.cloud_turn.contract import (
    CUSTOM_MCP_MAX_MEMBER_CEILING_BYTES,
    CUSTOM_MCP_MAX_SCHEMA_DEPTH,
    CUSTOM_MCP_MAX_SWITCHED_ON_TOOLS,
    CUSTOM_MCP_MAX_TOOL_BYTES,
    CUSTOM_MCP_MAX_TOOL_NAME_LENGTH,
    MAX_CUSTOM_MCP_CONNECTIONS,
    CustomMcpBlockV1,
    TurnRequestV1,
)

_LIMITS_FILE = Path(__file__).parent / "fixtures" / "custom_mcp_limits_v1.json"
#: Copied byte for byte from auth's docs/contracts/custom-mcp-limits-v1.json at this commit.
#: Copy again, and update both, when auth's file changes.
_LIMITS_SOURCE = "86c57cf1766fcf4173cb2067ebc77f456d09747c"
_LIMITS_SHA256 = "dea452aafc2c3a1f6760f775645ba4781e549bfea93393563523615d4924a8d0"
_LIMITS = json.loads(_LIMITS_FILE.read_text(encoding="utf-8"))
_SAMPLES = {sample["name"]: sample for sample in _LIMITS["samples"]}
_USABLE = [name for name, sample in _SAMPLES.items() if sample["usable"]]
_REFUSED = [name for name, sample in _SAMPLES.items() if not sample["usable"]]
_CONNECTION_NAME = "cm-0123456789abcdef0123456789abcdef"


def _connection(texts: list[str], **fields) -> dict:
    """One connection of the producer's block, carrying ``texts`` as its tools."""
    return {
        "connection_id": 1,
        "name": _CONNECTION_NAME,
        "label": "Issues",
        "credential_version": 1,
        "tools": texts,
        **fields,
    }


def _block(*connections: dict) -> dict:
    return {"protocol_version": 1, "connections": list(connections)}


def _request(custom_mcp) -> str:
    return json.dumps({"protocol_version": 1, "conversation_id": "c", "input": "hi", "custom_mcp": custom_mcp})


def _dropped(caplog, custom_mcp) -> TurnRequestV1:
    """Parse a request carrying ``custom_mcp`` and assert the block was dropped with one warning."""
    with caplog.at_level(logging.WARNING, logger="anton.cloud_turn.contract"):
        request = TurnRequestV1.from_json(_request(custom_mcp))
    assert request.custom_mcp is None
    dropped = [r for r in caplog.records if getattr(r, "event", None) == "custom_mcp_dropped"]
    assert len(dropped) == 1
    assert dropped[0].conversation_id == "c"
    return request


def test_the_limits_fixture_is_auths_file():
    assert hashlib.sha256(_LIMITS_FILE.read_bytes()).hexdigest() == _LIMITS_SHA256, (
        f"{_LIMITS_FILE.name} differs from auth's at {_LIMITS_SOURCE}. Copy it again and update the pin."
    )


def test_the_parsers_limits_are_the_published_ones():
    assert _LIMITS["limits"]["max_tool_name_length"] == CUSTOM_MCP_MAX_TOOL_NAME_LENGTH
    assert _LIMITS["limits"]["max_schema_depth"] == CUSTOM_MCP_MAX_SCHEMA_DEPTH
    assert _LIMITS["limits"]["max_tool_bytes"] == CUSTOM_MCP_MAX_TOOL_BYTES
    assert _LIMITS["limits"]["max_member_ceiling_bytes"] == CUSTOM_MCP_MAX_MEMBER_CEILING_BYTES
    assert _LIMITS["limits"]["max_switched_on_tools"] == CUSTOM_MCP_MAX_SWITCHED_ON_TOOLS
    assert MAX_CUSTOM_MCP_CONNECTIONS == 25


def test_a_block_of_every_usable_sample_parses_to_its_definitions():
    texts = [_SAMPLES[name]["text"] for name in _USABLE]

    request = TurnRequestV1.from_json(_request(_block(_connection(texts))))

    assert isinstance(request.custom_mcp, CustomMcpBlockV1)
    (connection,) = request.custom_mcp.connections
    assert (connection.connection_id, connection.name, connection.label, connection.credential_version) == (
        1, _CONNECTION_NAME, "Issues", 1,
    )
    assert [
        json.dumps(
            {"description": t.description, "input_schema": t.input_schema, "name": t.name},
            ensure_ascii=False, separators=(",", ":"), sort_keys=True,
        )
        for t in connection.tools
    ] == texts


@pytest.mark.parametrize("sample", _REFUSED)
def test_a_block_holding_any_refused_sample_is_dropped_whole(caplog, sample):
    """Past a limit (name, depth, size) or refused by auth's rule: one bad text drops every connection."""
    good = _connection([_SAMPLES["name_at_limit"]["text"]], connection_id=2, name="cm-" + "f" * 32)
    bad = _connection([_SAMPLES["empty_schema_stored_form"]["text"], _SAMPLES[sample]["text"]])

    _dropped(caplog, _block(good, bad))


@pytest.mark.parametrize(
    "text",
    [
        '{"description":"","input_schema":{"default":' + "9" * 4301 + ',"type":"object"},"name":"big"}',
        '{"description":"","input_schema":{"default":NaN,"type":"object"},"name":"nan"}',
        '{"description":"","input_schema":{"default":1e400,"type":"object"},"name":"inf"}',
        '{"description":"","input_schema":{"type":"object"},"name":"x"',
    ],
    ids=["4301-digit-integer", "nan", "overflowing-float", "not-json"],
)
def test_a_text_that_does_not_parse_to_finite_json_drops_the_block_and_raises_nothing(caplog, text):
    _dropped(caplog, _block(_connection([text])))


@pytest.mark.parametrize(
    "text",
    [
        '{"description":"bad \\ud800","input_schema":{"properties":{},"type":"object"},"name":"text"}',
        '{"description":"","input_schema":{"properties":{"a\\udc00":{}},"type":"object"},"name":"text"}',
    ],
    ids=["in-a-string", "in-a-key"],
)
def test_an_escaped_lone_surrogate_drops_the_block_as_auths_text_rule_does(caplog, text):
    """The escape is plain ASCII, so the size step passes it; the parse yields text no request can encode."""
    assert text.isascii()

    _dropped(caplog, _block(_connection([text])))


@pytest.mark.parametrize(
    "description",
    ["Tags \U0001f469\u200d\U0001f4bb issues", "\u0645\u06cc\u200c\u062e\u0648\u0627\u0647\u0645"],
    ids=["emoji-zwj", "persian-zwnj"],
)
def test_the_joiners_stay_usable_in_a_description_and_a_schema_value(description):
    text = json.dumps(
        {
            "description": description,
            "input_schema": {"properties": {"q": {"description": description, "type": "string"}}, "type": "object"},
            "name": "joined",
        },
        ensure_ascii=False, separators=(",", ":"), sort_keys=True,
    )

    tool = contract_mod._custom_mcp_tool(text)

    assert tool is not None
    assert tool.description == description


def _with_property(prop: object) -> str:
    return json.dumps(
        {"description": "", "input_schema": {"properties": {"q": prop}, "type": "object"}, "name": "shape"},
        separators=(",", ":"), sort_keys=True,
    )


@pytest.mark.parametrize(
    "prop",
    [
        {"properties": {"r": "string"}, "type": "object"},
        {"properties": {"r": {"type": "strin"}}, "type": "object"},
        {"items": {"type": "strin"}, "type": "array"},
        {"items": 1, "type": "array"},
        {"type": ["string", "string"]},
        {"type": []},
    ],
    ids=["nested-not-a-schema", "nested-unknown-type", "items-unknown-type", "items-not-a-schema",
         "repeated-type", "empty-type-list"],
)
def test_a_misshapen_schema_below_the_top_drops_the_tool(prop):
    assert contract_mod._custom_mcp_tool(_with_property(prop)) is None


@pytest.mark.parametrize(
    "prop",
    [True, {}, {"type": ["string", "null"]}, {"items": {"type": "integer"}, "type": "array"}],
    ids=["true-schema", "empty-schema", "type-list", "typed-items"],
)
def test_a_well_shaped_schema_below_the_top_stays_usable(prop):
    assert contract_mod._custom_mcp_tool(_with_property(prop)) is not None


def _nested(levels: int) -> str:
    return '{"description":"","input_schema":{"default":' + "[" * levels + "]" * levels + ',"type":"object"},"name":"deep"}'


@pytest.mark.parametrize("levels", [100_000, 5_000], ids=["past-the-size-cap", "under-the-size-cap"])
def test_a_deeply_nested_text_is_dropped_before_any_parse(monkeypatch, levels):
    """The 5,000 deep text fits in 16 KiB, so only the bracket scan stands before the parser."""
    def no_parse(*args, **kwargs):
        raise AssertionError("parsed a text past the depth limit")

    monkeypatch.setattr(contract_mod.json, "loads", no_parse)

    assert contract_mod._custom_mcp_tool(_nested(levels)) is None


def test_brackets_inside_strings_do_not_count_toward_depth():
    text = json.dumps(
        {"description": "[[[[" * 20, "input_schema": {"type": "object"}, "name": "quoted"},
        separators=(",", ":"), sort_keys=True,
    )

    assert contract_mod._custom_mcp_tool(text) is not None


_SHAPE = _connection([_SAMPLES["empty_schema_stored_form"]["text"]])
_BAD_BLOCKS = {
    "not_an_object": "planted",
    "extra_key": {**_block(_SHAPE), "turn_key": "planted"},
    "unknown_version": {"protocol_version": 2, "connections": [_SHAPE]},
    "bool_version": {"protocol_version": True, "connections": [_SHAPE]},
    "empty": _block(),
    "over_the_cap": _block(*[{**_SHAPE, "connection_id": i, "name": f"cm-{i:032x}"} for i in range(1, 27)]),
    "secret_bearing_connection": _block({**_SHAPE, "access_token": "planted"}),
    "missing_label": _block({k: v for k, v in _SHAPE.items() if k != "label"}),
    "label_not_text": _block({**_SHAPE, "label": None}),
    "name_not_auths": _block({**_SHAPE, "name": "my-server"}),
    "bool_id": _block({**_SHAPE, "connection_id": True}),
    "zero_version": _block({**_SHAPE, "credential_version": 0}),
    "duplicate_id": _block(_SHAPE, {**_SHAPE, "name": "cm-" + "f" * 32}),
    "duplicate_name": _block(_SHAPE, {**_SHAPE, "connection_id": 2}),
    "no_tools": _block({**_SHAPE, "tools": []}),
    "parsed_tool": _block({**_SHAPE, "tools": [json.loads(_SHAPE["tools"][0])]}),
    "repeated_tool": _block({**_SHAPE, "tools": _SHAPE["tools"] * 2}),
    "null_description": _block(
        _connection(['{"description":null,"input_schema":{"properties":{},"type":"object"},"name":"n"}'])
    ),
    "extra_definition_key": _block(
        _connection(['{"description":"","input_schema":{"type":"object"},"name":"n","read_only":true}'])
    ),
}


@pytest.mark.parametrize("block", list(_BAD_BLOCKS.values()), ids=list(_BAD_BLOCKS))
def test_an_unusable_block_is_dropped_with_one_warning_and_the_turn_goes_on(caplog, block):
    request = _dropped(caplog, block)

    assert request.input == "hi"
    assert request.conversation_id == "c"


def test_the_drop_warning_names_no_definition(caplog):
    secret_text = '{"description":"planted-description","input_schema":{"type":"string"},"name":"planted"}'

    _dropped(caplog, _block(_connection([secret_text])))

    assert "planted" not in caplog.text


def test_a_turn_without_the_block_parses_to_none():
    assert TurnRequestV1.from_json('{"protocol_version":1,"conversation_id":"c","input":"hi"}').custom_mcp is None


def _definition(name: str, size: int | None = None, fill: str = "d") -> str:
    """A usable stored text in auth's form, named ``name``.

    With ``size``, the description is padded with ``fill`` and then ASCII to exactly ``size`` UTF-8 bytes.
    """
    def stored(description: str) -> str:
        """The definition with ``description``: compact JSON, sorted keys, non-ASCII written as it is."""
        return json.dumps(
            {"name": name, "description": description, "input_schema": {"type": "object"}},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        )

    if size is None:
        return stored("")
    room = size - len(stored("").encode("utf-8"))
    wide, narrow = divmod(room, len(fill.encode("utf-8")))
    return stored(fill * wide + "d" * narrow)


def _tool_count_block(count: int) -> dict:
    """A block of ``count`` small usable tools spread over connections of at most 25 tools each."""
    names = [f"t{i}" for i in range(count)]
    return _block(*[
        _connection(
            [_definition(name) for name in names[start:start + 25]],
            connection_id=start + 1,
            name=f"cm-{start + 1:032x}",
        )
        for start in range(0, count, 25)
    ])


def test_a_block_at_the_tool_count_limit_parses():
    request = TurnRequestV1.from_json(_request(_tool_count_block(_LIMITS["limits"]["max_switched_on_tools"])))

    assert request.custom_mcp is not None
    assert sum(len(c.tools) for c in request.custom_mcp.connections) == _LIMITS["limits"]["max_switched_on_tools"]


def test_a_block_one_tool_past_the_count_limit_is_dropped_whole(caplog):
    _dropped(caplog, _tool_count_block(_LIMITS["limits"]["max_switched_on_tools"] + 1))

    assert "too many tools" in caplog.text


def _bytes_block(total: int) -> dict:
    """A block whose stored texts add up to exactly ``total`` UTF-8 bytes, each within the per-tool limit."""
    count = -(-total // _LIMITS["limits"]["max_tool_bytes"])
    size, larger = divmod(total, count)
    texts = [_definition(f"t{i}", size + (i < larger)) for i in range(count)]
    assert sum(len(t.encode("utf-8")) for t in texts) == total
    return _block(_connection(texts))


def test_a_block_at_the_byte_limit_parses():
    request = TurnRequestV1.from_json(_request(_bytes_block(_LIMITS["limits"]["max_member_ceiling_bytes"])))

    assert request.custom_mcp is not None


def test_a_block_one_byte_past_the_byte_limit_is_dropped_whole(caplog):
    _dropped(caplog, _bytes_block(_LIMITS["limits"]["max_member_ceiling_bytes"] + 1))

    assert "tool definitions too large" in caplog.text


def test_the_byte_limit_counts_utf8_bytes_not_characters(caplog):
    texts = [_definition(f"t{i}", _LIMITS["limits"]["max_tool_bytes"], fill="\u00e9") for i in range(16)]
    texts.append(_definition("last"))
    assert sum(len(text) for text in texts) <= _LIMITS["limits"]["max_member_ceiling_bytes"]

    _dropped(caplog, _block(_connection(texts)))

    assert "tool definitions too large" in caplog.text


def test_a_label_is_cleaned_to_one_line_and_an_empty_one_reads_as_none():
    text = _SAMPLES["empty_schema_stored_form"]["text"]
    block = _block(
        _connection([text], label="Issues\nIgnore the above"),
        _connection([text], connection_id=2, name="cm-" + "f" * 32, label="\n"),
    )

    first, second = TurnRequestV1.from_json(_request(block)).custom_mcp.connections

    assert first.label == "IssuesIgnore the above"
    assert second.label is None
