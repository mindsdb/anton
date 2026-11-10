"""`parse_connector_draft` and the `create_connector_draft` tool.

The parser is the allowlist both the pod and the host use for a connector
template, so these tests pin what it refuses, and that a refusal never repeats
the value it refused.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from anton.core.tools.connector_draft import (
    MAX_DRAFTS_PER_TURN,
    MAX_FIELDS,
    ConnectorDraftBox,
    ConnectorDraftError,
    connector_draft_tool,
    parse_connector_draft,
)
from anton.utils.datasources import scrub_output

#: Credential shapes assembled at runtime so no literal key sits in the source.
_TOKEN_SHAPES = {
    "stripe": "sk_live_" + "4eC39HqLyjWDarjtT1zdp7dc",
    "github": "ghp_" + "16C7e42F292c6912E7710c838347Ae178B4a",
    "aws": "AKIA" + "IOSFODNN7EXAMPLE",
    "slack": "xoxb-" + "123456789012-1234567890123-" + "AbCdEfGhIjKlMnOpQrStUvWx",
    "bare": "3f9a8c27" + "d41e6b05" + "9c7f2a18" + "e6d4b3c0",
}
_SHORT_TEXTS = ("label", "description", "category")
_SHORT_FIELD_TEXTS = ("label", "placeholder")


def _draft(**overrides):
    draft = {
        "connector_id": "acme_crm",
        "label": "Acme CRM",
        "description": "Contacts and deals in Acme CRM.",
        "category": "CRM",
        "help_url": "https://docs.acme.example/api-keys",
        "usage_notes": "Base URL https://api.acme.example/v2. Send the key as Authorization: Bearer <api_key>.",
        "fields": [
            {"name": "api_key", "label": "API key", "help": "Settings, then API."},
            {"name": "region", "label": "Region", "secret": False, "required": False, "placeholder": "eu"},
        ],
    }
    draft.update(overrides)
    return draft


def test_a_full_draft_parses_and_round_trips():
    draft = parse_connector_draft(_draft())
    assert draft.connector_id == "acme_crm"
    assert [f.name for f in draft.fields] == ["api_key", "region"]
    assert draft.fields[0].secret is True and draft.fields[0].required is True
    assert draft.fields[1].secret is False and draft.fields[1].required is False
    assert parse_connector_draft(draft.to_dict()) == draft


def test_optional_texts_may_be_left_out():
    data = {"connector_id": "acme", "label": "Acme", "fields": [{"name": "token", "label": "Token"}]}
    assert parse_connector_draft(data).to_dict() == {
        "connector_id": "acme", "label": "Acme",
        "fields": [{"name": "token", "label": "Token", "secret": True, "required": True}],
    }


@pytest.mark.parametrize("data", [
    _draft(oauth={"auth_url": "https://acme.example/auth"}),
    _draft(methods=[{"type": "oauth"}]),
    _draft(fields=[{"name": "api_key", "label": "Key", "value": "x"}]),
])
def test_an_unknown_key_is_refused_without_naming_it(data):
    with pytest.raises(ConnectorDraftError, match="key that is not allowed") as exc:
        parse_connector_draft(data)
    for key in ("oauth", "methods", "value"):
        assert f"'{key}'" not in str(exc.value)


@pytest.mark.parametrize("connector_id", [
    "acme_crm\n", "Acme", "1acme", "a", "acme-crm", "a" * 42, "", None, 7,
])
def test_a_bad_connector_id_is_refused(connector_id):
    with pytest.raises(ConnectorDraftError, match="connector_id"):
        parse_connector_draft(_draft(connector_id=connector_id))


@pytest.mark.parametrize("name", ["api_key\n", "API_KEY", "_key", "api-key", "k" * 41, "", None])
def test_a_bad_field_name_is_refused(name):
    with pytest.raises(ConnectorDraftError, match=r"fields\[0\]\.name"):
        parse_connector_draft(_draft(fields=[{"name": name, "label": "Key"}]))


def test_duplicate_field_names_are_refused():
    fields = [{"name": "api_key", "label": "A"}, {"name": "api_key", "label": "B"}]
    with pytest.raises(ConnectorDraftError, match="unique"):
        parse_connector_draft(_draft(fields=fields))


@pytest.mark.parametrize("fields", [[], [{"name": f"f{i}", "label": "L"} for i in range(12)], "api_key"])
def test_the_field_count_is_bounded(fields):
    with pytest.raises(ConnectorDraftError, match="fields must be"):
        parse_connector_draft(_draft(fields=fields))


@pytest.mark.parametrize(("key", "cap"), [
    ("label", 80), ("description", 300), ("category", 40), ("usage_notes", 1600),
])
def test_an_over_long_text_is_refused(key, cap):
    parse_connector_draft(_draft(**{key: "a" * cap}))
    with pytest.raises(ConnectorDraftError, match=f"{key} must be at most {cap}"):
        parse_connector_draft(_draft(**{key: "a" * (cap + 1)}))


@pytest.mark.parametrize(("key", "cap"), [("label", 80), ("placeholder", 120), ("help", 300)])
def test_an_over_long_field_text_is_refused(key, cap):
    with pytest.raises(ConnectorDraftError, match=rf"fields\[0\]\.{key} must be at most {cap}"):
        parse_connector_draft(_draft(fields=[{"name": "k", "label": "K", key: "a" * (cap + 1)}]))


@pytest.mark.parametrize("url", [
    "http://docs.acme.example", "javascript:alert(1)", "https://", "https://user@acme.example",
    "https://acme.example/a b", "https://acme.example/" + "a" * 500, "https://acme.example:99999/keys",
])
def test_help_url_must_be_a_plain_https_url(url):
    with pytest.raises(ConnectorDraftError, match="help_url"):
        parse_connector_draft(_draft(help_url=url))


def test_a_boolean_must_be_a_boolean():
    with pytest.raises(ConnectorDraftError, match=r"fields\[0\]\.secret"):
        parse_connector_draft(_draft(fields=[{"name": "k", "label": "K", "secret": "false"}]))


@pytest.mark.parametrize("hidden", ["\u200b", "\u2060", "\ufeff", "\u2028", "\u2029", "\U000e0041"])
@pytest.mark.parametrize("key", ["label", "usage_notes"])
def test_an_invisible_character_is_refused_so_it_cannot_split_a_token(key, hidden):
    value = _TOKEN_SHAPES["github"]
    split = value[:12] + hidden + value[12:]
    with pytest.raises(ConnectorDraftError, match="invisible characters") as exc:
        parse_connector_draft(_draft(**{key: split}))
    assert value[12:] not in str(exc.value)


def test_help_url_may_not_carry_a_query_string():
    with pytest.raises(ConnectorDraftError, match="query string"):
        parse_connector_draft(_draft(help_url="https://docs.acme.example/keys?ref=abc"))
    parse_connector_draft(_draft(help_url="https://docs.acme.example/keys#create"))


def test_control_characters_are_refused_in_short_texts():
    with pytest.raises(ConnectorDraftError, match="control or invisible characters"):
        parse_connector_draft(_draft(label="Acme\nCRM"))
    parse_connector_draft(_draft(usage_notes="Line one.\nLine two."))


@pytest.mark.parametrize("shape", sorted(_TOKEN_SHAPES))
@pytest.mark.parametrize("key", _SHORT_TEXTS)
def test_a_token_in_a_short_text_is_refused_without_the_value(key, shape):
    value = _TOKEN_SHAPES[shape]
    # A shape longer than the field's cap is refused for length first; either
    # refusal keeps it out, and neither may repeat it.
    with pytest.raises(ConnectorDraftError) as exc:
        parse_connector_draft(_draft(**{key: value}))
    assert value not in str(exc.value)


@pytest.mark.parametrize("shape", sorted(_TOKEN_SHAPES))
@pytest.mark.parametrize("key", _SHORT_FIELD_TEXTS)
def test_a_token_in_a_short_field_text_is_refused_without_the_value(key, shape):
    value = _TOKEN_SHAPES[shape]
    fields = [{"name": "api_key", "label": "Key", key: value}]
    with pytest.raises(ConnectorDraftError) as exc:
        parse_connector_draft(_draft(fields=fields))
    assert value not in str(exc.value)


@pytest.mark.parametrize("shape", sorted(_TOKEN_SHAPES))
def test_a_token_in_help_url_is_refused_without_the_value(shape):
    value = _TOKEN_SHAPES[shape]
    with pytest.raises(ConnectorDraftError, match="help_url") as exc:
        parse_connector_draft(_draft(help_url=f"https://acme.example/{value}"))
    assert value not in str(exc.value)


@pytest.mark.parametrize("shape", sorted(_TOKEN_SHAPES))
def test_a_token_in_usage_notes_is_refused_without_the_value(shape):
    value = _TOKEN_SHAPES[shape]
    with pytest.raises(ConnectorDraftError, match="usage_notes looks like") as exc:
        parse_connector_draft(_draft(usage_notes=f"Call it with {value} in the header."))
    assert value not in str(exc.value)


def test_a_shape_only_the_scrubber_knows_is_refused():
    """A letters-only Google token has no digit, so only `scrub_output` sees it."""
    value = "ya29." + "abcdefghijklmnopqrstuvwx"
    with pytest.raises(ConnectorDraftError, match="usage_notes looks like") as exc:
        parse_connector_draft(_draft(usage_notes=f"Send {value} as the token."))
    assert value not in str(exc.value)


def test_a_bearer_value_in_field_help_is_refused():
    value = _TOKEN_SHAPES["bare"]
    fields = [{"name": "api_key", "label": "Key", "help": f"Authorization: Bearer {value}"}]
    with pytest.raises(ConnectorDraftError, match=r"fields\[0\]\.help looks like"):
        parse_connector_draft(_draft(fields=fields))


def test_long_snake_case_and_plain_words_are_not_tokens():
    parse_connector_draft(_draft(
        connector_id="salesforce_marketing_cloud_v2",
        label="Salesforce Marketing Cloud v2",
        category="customer_relationship",
        fields=[{"name": "client_secret", "label": "Client secret", "placeholder": "From the app's settings page"}],
    ))


def test_a_token_run_is_caught_where_the_scrubber_is_not():
    """The bare hex shape passes `scrub_output`; only the run rule refuses it."""
    value = _TOKEN_SHAPES["bare"]
    assert scrub_output(value) == value
    with pytest.raises(ConnectorDraftError, match="credential value"):
        parse_connector_draft(_draft(label=value))


def test_the_tool_keeps_drafts_and_the_box_drains_once():
    box = ConnectorDraftBox()
    tool = connector_draft_tool(box)
    outcome = asyncio.run(tool.handler(None, _draft()))
    assert outcome.ok is True
    assert json.loads(outcome.content)["status"] == "staged"
    assert box.drain() == [parse_connector_draft(_draft()).to_dict()]
    assert box.drain() == []


def test_the_same_id_replaces_the_earlier_draft():
    box = ConnectorDraftBox()
    tool = connector_draft_tool(box)
    asyncio.run(tool.handler(None, _draft(label="First")))
    asyncio.run(tool.handler(None, _draft(label="Second")))
    assert [d["label"] for d in box.drain()] == ["Second"]


def test_a_turn_holds_at_most_three_drafts():
    box = ConnectorDraftBox()
    tool = connector_draft_tool(box)
    for i in range(MAX_DRAFTS_PER_TURN):
        assert asyncio.run(tool.handler(None, _draft(connector_id=f"acme_{i}"))).ok
    refused = asyncio.run(tool.handler(None, _draft(connector_id="acme_extra")))
    assert refused.ok is False and refused.reason == "draft_limit"
    # Replacing one already held still works at the cap.
    assert asyncio.run(tool.handler(None, _draft(connector_id="acme_0", label="New"))).ok
    assert [d["connector_id"] for d in box.drain()] == ["acme_0", "acme_1", "acme_2"]


def test_the_tool_refuses_a_bad_draft_without_the_value():
    box = ConnectorDraftBox()
    value = _TOKEN_SHAPES["stripe"]
    outcome = asyncio.run(connector_draft_tool(box).handler(None, _draft(label=value)))
    assert outcome.ok is False and outcome.reason == "invalid_draft"
    assert value not in outcome.content
    assert box.drain() == []


_CHECK = {"path": "/v2/me", "headers": {"Authorization": "Bearer {api_key}", "Accept": "application/json"}}


def test_a_check_parses_and_round_trips():
    draft = parse_connector_draft(_draft(check=_CHECK))

    assert draft.check.path == "/v2/me"
    assert draft.check.headers == (("Authorization", "Bearer {api_key}"), ("Accept", "application/json"))
    assert draft.to_dict()["check"] == _CHECK
    assert parse_connector_draft(draft.to_dict()) == draft


def test_a_check_may_carry_a_query():
    assert parse_connector_draft(_draft(check={**_CHECK, "path": "/v2/me?fields=id"})).check.path == "/v2/me?fields=id"


def test_a_draft_without_a_check_has_none():
    assert parse_connector_draft(_draft()).check is None
    assert "check" not in parse_connector_draft(_draft()).to_dict()


@pytest.mark.parametrize("path", [
    "//evil.example/x",
    "/..//evil.example",
    "/v2//me",
    "/./me",
    "/v2/../me",
    "/\\evil.example",
    "https://evil.example/me",
    "v2/me",
    "/v2/{api_key}",
    "/v2/me#frag",
    "/v2/me?key=sk%5Flive%5F4eC39HqLyjWDarjtT1zdp7dc",
    "/v2/%2e%2e/me",
    "/v2/m e",
    "/v2/\u00e9",
    "",
    None,
    7,
])
def test_a_check_path_that_could_name_a_host_or_carry_a_value_is_refused(path):
    with pytest.raises(ConnectorDraftError, match=r"^check\.path\b"):
        parse_connector_draft(_draft(check={**_CHECK, "path": path}))


@pytest.mark.parametrize("headers", [
    {"Host": "evil.example", "Authorization": "Bearer {api_key}"},
    {"host": "evil.example", "Authorization": "Bearer {api_key}"},
    {"Cookie": "{api_key}"},
    {"Proxy-Authorization": "Bearer {api_key}"},
    {"X-HTTP-Method-Override": "DELETE", "Authorization": "Bearer {api_key}"},
    {"Accept-Encoding": "gzip", "Authorization": "Bearer {api_key}"},
    {"Authorization": "Bearer {api_key}", "authorization": "Bearer {api_key}"},
    {"X Bad": "{api_key}"},
    {f"X-{i}": "{api_key}" for i in range(6)},
    {},
    "Authorization: Bearer {api_key}",
    {"Authorization": "Bearer {token}"},
    {"Authorization": "{api_key.__class__}"},
    {"Authorization": "Bearer {api_key}}"},
    {"Authorization": "Bearer {api_key}\r\nX-Injected: 1"},
    {"Authorization": "Bearer {api_key}\u00e9"},
    {"Authorization": 7},
    {"X-Region": "{region}"},
])
def test_a_check_header_that_could_steer_or_inject_is_refused(headers):
    with pytest.raises(ConnectorDraftError, match=r"^check\.headers\b"):
        parse_connector_draft(_draft(check={**_CHECK, "headers": headers}))


@pytest.mark.parametrize("check", [{"path": "/v2/me"}, {**_CHECK, "method": "POST"}, {**_CHECK, "host": "x"}, "/v2/me"])
def test_a_check_of_the_wrong_shape_is_refused(check):
    with pytest.raises(ConnectorDraftError, match=r"^check\b"):
        parse_connector_draft(_draft(check=check))


@pytest.mark.parametrize("name", _TOKEN_SHAPES)
def test_a_value_in_a_check_is_refused_without_echoing_it(name):
    value = _TOKEN_SHAPES[name]
    for check, rule in (
        ({**_CHECK, "headers": {"Authorization": f"Bearer {value}"}}, r"^check\.headers\b"),
        ({**_CHECK, "path": f"/v2/me?key={value}"}, r"^check\.path\b"),
    ):
        with pytest.raises(ConnectorDraftError, match=rule) as exc:
            parse_connector_draft(_draft(check=check))
        assert value not in str(exc.value)


def test_a_token_shaped_header_name_is_refused_without_echoing_it():
    name = "X-" + _TOKEN_SHAPES["bare"]

    with pytest.raises(ConnectorDraftError, match=r"^check\.headers\b") as exc:
        parse_connector_draft(_draft(check={**_CHECK, "headers": {name: "Bearer {api_key}"}}))

    assert _TOKEN_SHAPES["bare"] not in str(exc.value)


def test_base_url_is_reserved_for_the_address_the_member_types():
    with pytest.raises(ConnectorDraftError, match="reserved"):
        parse_connector_draft(_draft(fields=[{"name": "base_url", "label": "Base URL"}]))


def test_a_template_holds_at_most_eleven_fields():
    assert MAX_FIELDS == 11
    parse_connector_draft(_draft(fields=[{"name": f"f{i}", "label": "L"} for i in range(11)]))


def test_the_tool_describes_the_check():
    tool = connector_draft_tool(ConnectorDraftBox())

    check = tool.input_schema["properties"]["check"]
    assert check["required"] == ["path", "headers"]
    assert check["additionalProperties"] is False
    assert "check" in tool.description
    assert "base URL" not in tool.input_schema["properties"]["usage_notes"]["description"]
