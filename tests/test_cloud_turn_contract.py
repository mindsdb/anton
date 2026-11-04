import json
import logging

import pytest
from pydantic import ValidationError

from anton.cloud_turn import contract
from anton.cloud_turn.contract import (
    DatasourceBlockV1,
    TurnRequestV1,
    TurnSkillEntry,
    parse_turn_skills,
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


def test_turn_request_carries_the_always_on_skill_flag():
    skills = {
        "tone": {"files": {"SKILL.md": "---\nname: tone\n---\nBe brief."}, "always_on": True},
        "recall-only": {"files": {"SKILL.md": "x"}},
    }
    request = TurnRequestV1.from_json(json.dumps({
        "protocol_version": 1,
        "conversation_id": "c",
        "input": "query",
        "skills": skills,
    }))
    assert request.skills == skills
    assert request.skills["tone"]["always_on"] is True


def test_an_empty_skills_block_is_kept_apart_from_none():
    """{} names no recallable skills; only an absent block lets the pod take
    the rollout fallback to the mounted org tree."""
    base = {"protocol_version": 1, "conversation_id": "c", "input": "query"}
    assert TurnRequestV1.from_json(json.dumps(base)).skills is None
    assert TurnRequestV1.from_json(json.dumps({**base, "skills": {}})).skills == {}


def test_the_skills_block_parses_into_entries_by_slug():
    block = {
        "tone": {"files": {"SKILL.md": "Be brief.", "references/x.md": "Ref."}, "always_on": True},
        "recall-only": {"files": {"SKILL.md": "x"}},
        "off": {"files": {"SKILL.md": "x"}, "always_on": False},
    }
    assert parse_turn_skills(block) == {
        "tone": TurnSkillEntry(
            files={"SKILL.md": "Be brief.", "references/x.md": "Ref."}, always_on=True
        ),
        "recall-only": TurnSkillEntry(files={"SKILL.md": "x"}),
        "off": TurnSkillEntry(files={"SKILL.md": "x"}),
    }


@pytest.mark.parametrize("block", ["x", ["tone"], 7, True])
def test_a_skills_block_that_is_not_an_object_names_no_skills(block):
    """Not None: a malformed block must not take the absent block's fallback."""
    assert parse_turn_skills(block) == {}


def test_an_absent_skills_block_stays_none():
    assert parse_turn_skills(None) is None
    assert parse_turn_skills({}) == {}


@pytest.mark.parametrize("always_on", ["true", 1, "yes", [True], {"on": True}, None])
def test_only_a_json_true_marks_a_skill_always_on(always_on):
    """pydantic's lax bool would read "true", 1 and "yes" as True."""
    block = {"tone": {"files": {"SKILL.md": "x"}, "always_on": always_on}}
    assert parse_turn_skills(block) == {"tone": TurnSkillEntry(files={"SKILL.md": "x"})}


def test_entries_and_files_that_cannot_be_read_are_dropped_and_the_rest_kept(caplog):
    block = {
        "Bad Slug": {"files": {"SKILL.md": "x"}},
        "UPPER": {"files": {"SKILL.md": "x"}},
        "../escape": {"files": {"SKILL.md": "x"}},
        "a--b": {"files": {"SKILL.md": "x"}},
        "a" * 65: {"files": {"SKILL.md": "x"}},
        "not-an-object": ["x"],
        "no-files": {"always_on": True},
        "files-not-an-object": {"files": ["SKILL.md"]},
        "mixed": {"files": {"SKILL.md": "Body.", "bad.md": 3, "worse.md": None}},
    }
    caplog.set_level(logging.DEBUG, logger=contract.logger.name)

    assert parse_turn_skills(block) == {
        "no-files": TurnSkillEntry(files={}, always_on=True),
        "files-not-an-object": TurnSkillEntry(files={}),
        "mixed": TurnSkillEntry(files={"SKILL.md": "Body."}),
    }
    dropped = [r for r in caplog.records if r.name == contract.logger.name]
    assert [r.levelno for r in dropped] == [logging.WARNING] * 5
    assert [r.args[0] for r in dropped] == ["Bad Slug", "UPPER", "../escape", "a--b", "a" * 65]


def test_a_parsed_skill_entry_is_frozen():
    entry = parse_turn_skills({"tone": {"files": {"SKILL.md": "x"}}})["tone"]

    with pytest.raises(ValidationError):
        entry.always_on = True
    with pytest.raises(ValidationError):
        entry.files = {}
    assert entry == TurnSkillEntry(files={"SKILL.md": "x"})


def test_a_slug_that_is_not_a_string_is_dropped():
    """JSON keys are always strings; a request built in process may not be."""
    assert parse_turn_skills({7: {"files": {"SKILL.md": "x"}}, "ok": {"files": {}}}) == {
        "ok": TurnSkillEntry(files={}),
    }
