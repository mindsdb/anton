"""`parse_connector_draft` and the `create_connector_draft` tool.

The parser is the allowlist both the pod and the host use for a connector
template, so these tests pin what it refuses, and that a refusal never repeats
the value it refused.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from anton.core.tools.connector_draft import (
    MAX_DRAFTS_PER_TURN,
    MAX_FIELDS,
    ConnectorDraftBox,
    ConnectorDraftError,
    check_draft_text,
    check_header_name,
    connector_draft_tool,
    parse_connector_draft,
)
from anton.utils.datasources import (
    _reset_registered_ds_vars,
    scrub_output,
    set_ds_env_values,
)

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
#: A lowercase key shape both identifier patterns accept.
_ID_SHAPED_KEY = "ka3f9c8b" + "7e6d5f4a" + "3b2c1d0e" + "9f8a7b6c5"
#: Letters only, so the run rule passes it and only the held-value scrub can refuse it.
_HELD_VALUE = "heldvalue" + "qwertyzx"
_AUTH = {"type": "header", "headers": {"Authorization": "Bearer {api_key}", "X-Region": "{region}"}}
#: Parses a draft whose usage_notes is argv[1] and prints the refusal.
_PARSE_NOTES = (
    "import sys\n"
    "from anton.core.tools.connector_draft import ConnectorDraftError, parse_connector_draft\n"
    "draft = {'connector_id': 'acme', 'label': 'Acme', 'usage_notes': sys.argv[1],\n"
    "         'fields': [{'name': 'api_key', 'label': 'Key'}],\n"
    "         'auth': {'type': 'header', 'headers': {'Authorization': 'Bearer {api_key}'}}}\n"
    "try:\n"
    "    parse_connector_draft(draft)\n"
    "except ConnectorDraftError as e:\n"
    "    print(e)\n"
)


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
            {"name": "region", "label": "Region", "secret": False, "placeholder": "eu"},
        ],
        "auth": _AUTH,
    }
    draft.update(overrides)
    return draft


def _one_field_auth(name="api_key"):
    """A header auth section that places only the field ``name`` as a Bearer token."""
    return {"type": "header", "headers": {"Authorization": f"Bearer {{{name}}}"}}


def test_a_full_draft_parses_and_round_trips():
    draft = parse_connector_draft(_draft())
    assert draft.connector_id == "acme_crm"
    assert [f.name for f in draft.fields] == ["api_key", "region"]
    assert draft.fields[0].secret is True and draft.fields[0].required is True
    assert draft.fields[1].secret is False and draft.fields[1].required is True
    assert parse_connector_draft(draft.to_dict()) == draft


def test_optional_texts_may_be_left_out():
    data = {
        "connector_id": "acme", "label": "Acme", "fields": [{"name": "token", "label": "Token"}],
        "auth": _one_field_auth("token"),
    }
    assert parse_connector_draft(data).to_dict() == {
        "connector_id": "acme", "label": "Acme",
        "fields": [{"name": "token", "label": "Token", "secret": True, "required": True}],
        "auth": {"type": "header", "headers": {"Authorization": "Bearer {token}"}},
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
    with pytest.raises(ConnectorDraftError, match=r"fields\[0\]\.required must be true or false"):
        parse_connector_draft(_draft(fields=[{"name": "k", "label": "K", "required": "false"}]))


def test_an_optional_field_is_refused_since_auth_sends_every_field():
    fields = [{"name": "api_key", "label": "Key"}, {"name": "region", "label": "Region", "secret": False, "required": False}]

    with pytest.raises(ConnectorDraftError, match=r"^fields\[1\]\.required must be true"):
        parse_connector_draft(_draft(fields=fields))


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


@pytest.mark.parametrize("key", _SHORT_TEXTS)
def test_one_text_is_checked_with_the_rule_a_whole_draft_gets(key):
    assert check_draft_text(key, "  Acme  ") == "Acme"
    assert check_draft_text(key, "   ") is None
    with pytest.raises(ConnectorDraftError) as caught:
        check_draft_text(key, "Key " + _TOKEN_SHAPES["github"])
    assert _TOKEN_SHAPES["github"] not in str(caught.value)
    with pytest.raises(ConnectorDraftError):
        check_draft_text(key, "x" * 301)


def test_only_a_draft_text_can_be_checked_on_its_own():
    with pytest.raises(ConnectorDraftError):
        check_draft_text("help_url", "https://example.com")


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
        auth=_one_field_auth("client_secret"),
    ))


def _as_connector_id(value):
    return _draft(connector_id=value)


def _as_field_name(value):
    return _draft(fields=[{"name": value, "label": "Key"}])


@pytest.mark.parametrize("build,where", [
    (_as_connector_id, r"^connector_id looks like"),
    (_as_field_name, r"^fields\[0\]\.name looks like"),
])
def test_a_key_shaped_identifier_is_refused_without_the_value(build, where):
    with pytest.raises(ConnectorDraftError, match=where) as exc:
        parse_connector_draft(build(_ID_SHAPED_KEY))
    assert _ID_SHAPED_KEY not in str(exc.value)


@pytest.fixture
def held_value():
    set_ds_env_values({"DS_CUSTOM__TOKEN": _HELD_VALUE})
    yield _HELD_VALUE
    _reset_registered_ds_vars()


@pytest.mark.parametrize("build,where", [
    (_as_connector_id, r"^connector_id looks like"),
    (_as_field_name, r"^fields\[0\]\.name looks like"),
])
def test_a_held_value_as_an_identifier_is_refused_without_the_value(held_value, build, where):
    with pytest.raises(ConnectorDraftError, match=where) as exc:
        parse_connector_draft(build(f"acme_{held_value}"))
    assert held_value not in str(exc.value)


def test_long_snake_case_field_names_are_not_tokens():
    # Placed in a slot too: the token rule reads auth templates with slots removed.
    parse_connector_draft(_draft(
        fields=[{"name": "salesforce_marketing_cloud_api_key_v2", "label": "Key"}],
        auth=_one_field_auth("salesforce_marketing_cloud_api_key_v2"),
    ))


@pytest.mark.parametrize("notes", [
    "Use model gpt4turbopreview20240125 for it.",
    "Fill in salesforce_marketing_cloud_api_key_v2 first.",
    "Workspace ids look like 123e4567-e89b-12d3-a456-426614174000.",
])
def test_a_run_refusal_names_the_rule_so_the_model_can_reword(notes):
    with pytest.raises(ConnectorDraftError, match="run of 20 or more letters, digits, _ or -"):
        parse_connector_draft(_draft(usage_notes=notes))


def test_a_mixed_refusal_is_the_same_in_every_process():
    """The as-typed form fails the run rule, the folded form only the scrubber;
    the as-typed form is checked first whatever the hash seed."""
    fullwidth = "".join(chr(ord(c) + 0xFEE0) for c in "ya29." + "abcdefghijklmnopqrstuvwx")
    notes = f"Use run abcdefghij1234567890xyz and {fullwidth}."
    for seed in ("1", "2", "3", "4", "5", "6", "7"):
        out = subprocess.run(
            [sys.executable, "-c", _PARSE_NOTES, notes],
            capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": seed}, check=True,
        )
        assert "run of 20 or more" in out.stdout, seed


def test_a_fullwidth_key_is_refused_without_the_value():
    value = "".join(chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else c for c in _TOKEN_SHAPES["github"])
    assert scrub_output(value) == value
    with pytest.raises(ConnectorDraftError, match="usage_notes looks like") as exc:
        parse_connector_draft(_draft(usage_notes=f"Send {value} as the key."))
    assert value not in str(exc.value)
    assert _TOKEN_SHAPES["github"] not in str(exc.value)


def test_a_token_run_is_caught_where_the_scrubber_is_not():
    """The bare hex shape passes `scrub_output`; only the run rule refuses it."""
    value = _TOKEN_SHAPES["bare"]
    assert scrub_output(value) == value
    with pytest.raises(ConnectorDraftError, match="credential value"):
        parse_connector_draft(_draft(label=value))


_TOOL_CHECK = {"path": "/v1/me"}


def _tool_draft(**overrides):
    """A draft as the tool takes it, which requires a check."""
    return _draft(**{"check": _TOOL_CHECK, **overrides})


def test_the_tool_keeps_drafts_and_the_box_drains_once():
    box = ConnectorDraftBox()
    tool = connector_draft_tool(box)
    outcome = asyncio.run(tool.handler(None, _tool_draft()))
    assert outcome.ok is True
    assert json.loads(outcome.content)["status"] == "staged"
    assert box.drain() == [parse_connector_draft(_tool_draft()).to_dict()]
    assert box.drain() == []


def test_the_same_id_replaces_the_earlier_tool_draft():
    box = ConnectorDraftBox()
    tool = connector_draft_tool(box)
    asyncio.run(tool.handler(None, _tool_draft(label="First")))
    asyncio.run(tool.handler(None, _tool_draft(label="Second")))
    assert [d["label"] for d in box.drain()] == ["Second"]


def test_a_turn_holds_at_most_three_drafts():
    box = ConnectorDraftBox()
    tool = connector_draft_tool(box)
    for i in range(MAX_DRAFTS_PER_TURN):
        assert asyncio.run(tool.handler(None, _tool_draft(connector_id=f"acme_{i}"))).ok
    refused = asyncio.run(tool.handler(None, _tool_draft(connector_id="acme_extra")))
    assert refused.ok is False and refused.reason == "draft_limit"
    # Replacing one already held still works at the cap.
    assert asyncio.run(tool.handler(None, _tool_draft(connector_id="acme_0", label="New"))).ok
    assert [d["connector_id"] for d in box.drain()] == ["acme_0", "acme_1", "acme_2"]


@pytest.mark.parametrize("check", ["missing", None], ids=["missing", "null"])
def test_the_tool_refuses_a_template_without_a_check_and_says_what_to_do(check):
    box = ConnectorDraftBox()
    draft = _draft() if check == "missing" else _draft(check=None)

    outcome = asyncio.run(connector_draft_tool(box).handler(None, draft))

    assert outcome.ok is False and outcome.reason == "invalid_draft"
    error = json.loads(outcome.content)["error"]
    assert "call it directly from code" in error and "cannot be a custom connector yet" in error
    assert box.drain() == []


@pytest.mark.parametrize("check", [
    {"path": "https://api.acme.example/v1/me"},
    {"path": ""},
], ids=["path-with-host", "empty-path"])
def test_the_tool_says_what_to_do_when_a_check_is_unusable(check):
    """A template whose check cannot be sent is told when to skip the connector."""
    box = ConnectorDraftBox()

    outcome = asyncio.run(connector_draft_tool(box).handler(None, _tool_draft(check=check)))

    assert outcome.ok is False and outcome.reason == "invalid_draft"
    error = json.loads(outcome.content)["error"]
    assert "call it directly from code" in error and "cannot be a custom connector yet" in error
    assert box.drain() == []


def test_the_tool_refuses_a_draft_outside_the_check_without_the_check_guidance():
    outcome = asyncio.run(connector_draft_tool(ConnectorDraftBox()).handler(None, _tool_draft(label="")))

    assert outcome.ok is False
    assert "call it directly from code" not in json.loads(outcome.content)["error"]


def test_the_tool_schema_requires_the_check():
    assert "check" in connector_draft_tool(ConnectorDraftBox()).input_schema["required"]


def test_the_tool_refuses_a_bad_draft_without_the_value():
    box = ConnectorDraftBox()
    value = _TOKEN_SHAPES["stripe"]
    outcome = asyncio.run(connector_draft_tool(box).handler(None, _draft(label=value)))
    assert outcome.ok is False and outcome.reason == "invalid_draft"
    assert value not in outcome.content
    assert box.drain() == []


_CHECK = {"path": "/v2/me"}


def test_a_check_parses_and_round_trips():
    draft = parse_connector_draft(_draft(check=_CHECK))

    assert draft.check.path == "/v2/me"
    assert draft.to_dict()["check"] == _CHECK
    assert parse_connector_draft(draft.to_dict()) == draft


def test_a_check_may_carry_a_query():
    assert parse_connector_draft(_draft(check={"path": "/v2/me?fields=id"})).check.path == "/v2/me?fields=id"


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
    "/v2/é",
    "",
    None,
    7,
])
def test_a_check_path_that_could_name_a_host_or_carry_a_value_is_refused(path):
    with pytest.raises(ConnectorDraftError, match=r"^check\.path\b"):
        parse_connector_draft(_draft(check={"path": path}))


@pytest.mark.parametrize("check", [
    {"path": "/v2/me", "headers": {"Authorization": "Bearer {api_key}"}},
    {"path": "/v2/me", "method": "POST"},
    {"path": "/v2/me", "host": "x"},
    "/v2/me",
])
def test_a_check_of_the_wrong_shape_is_refused(check):
    with pytest.raises(ConnectorDraftError, match=r"^check\b"):
        parse_connector_draft(_draft(check=check))


def test_header_auth_parses_and_round_trips_with_a_literal_header():
    auth = {"type": "header", "headers": {
        "Authorization": "Bearer {api_key}", "Notion-Version": "2022-06-28", "X-Region": "{region}",
    }}
    draft = parse_connector_draft(_draft(auth=auth))

    assert draft.auth.type == "header"
    assert draft.auth.headers == tuple(auth["headers"].items())
    assert draft.to_dict()["auth"] == auth
    assert parse_connector_draft(draft.to_dict()) == draft


_BASIC_FIELDS = [
    {"name": "email", "label": "Email", "secret": False},
    {"name": "api_token", "label": "API token"},
]


@pytest.mark.parametrize("fields,auth", [
    (_BASIC_FIELDS, {"type": "basic", "username": "{email}/token", "password": "{api_token}"}),
    ([{"name": "api_key", "label": "Key"}], {"type": "basic", "username": "{api_key}", "password": "X"}),
    ([{"name": "api_key", "label": "Key"}], {"type": "basic", "username": "{api_key}", "password": ""}),
])
def test_basic_auth_parses_and_round_trips(fields, auth):
    draft = parse_connector_draft(_draft(fields=fields, auth=auth))

    assert (draft.auth.type, draft.auth.username, draft.auth.password) == ("basic", auth["username"], auth["password"])
    assert draft.to_dict()["auth"] == auth
    assert parse_connector_draft(draft.to_dict()) == draft


@pytest.mark.parametrize("auth,rule", [
    (None, r"^auth is required"),
    ("Bearer {api_key}", r"^auth must be an object"),
    ({"headers": {"Authorization": "Bearer {api_key}"}}, r"^auth\.type must be one of"),
    ({"type": "oauth"}, r"^auth\.type must be one of"),
    ({**_AUTH, "username": "{api_key}"}, r"^auth has a key that is not allowed"),
    ({"type": "basic", "username": "{api_key}", "password": "X", "headers": {}}, r"^auth has a key that is not allowed"),
    ({"type": "header", "headers": {"X-Region": "{region}"}}, r"^auth must place a secret field"),
    ({"type": "header", "headers": {"Authorization": "Bearer {api_key}"}}, r"^auth must place every field; fields\[1\]"),
    ({"type": "basic", "username": "{region}", "password": "{api_key}"}, None),
    ({"type": "basic", "username": "admin", "password": "{api_key}"}, r"^auth\.username must place a field"),
    ({"type": "basic", "username": "{region}:x", "password": "{api_key}"}, r"^auth\.username must place a field"),
    ({"type": "basic", "username": "{region}"}, r"^auth\.password is required"),
    ({"type": "basic", "username": "", "password": "{api_key}"}, r"^auth\.username is required"),
    ({"type": "basic", "username": "{region}", "password": 7}, r"^auth\.password must be text"),
])
def test_an_auth_section_of_the_wrong_shape_is_refused(auth, rule):
    if rule is None:
        parse_connector_draft(_draft(auth=auth))
        return
    with pytest.raises(ConnectorDraftError, match=rule):
        parse_connector_draft(_draft(auth=auth))


@pytest.mark.parametrize("headers", [
    {"Host": "evil.example", "Authorization": "Bearer {api_key}"},
    {"host": "evil.example", "Authorization": "Bearer {api_key}"},
    {"Cookie": "{api_key}"},
    {"Proxy-Authorization": "Bearer {api_key}"},
    {"X-HTTP-Method-Override": "DELETE", "Authorization": "Bearer {api_key}"},
    {"Accept-Encoding": "gzip", "Authorization": "Bearer {api_key}"},
    {"X-Forwarded-Host": "evil.example", "Authorization": "Bearer {api_key}"},
    {"Accept": "application/json", "Authorization": "Bearer {api_key}"},
    {"Authorization": "Bearer {api_key}", "authorization": "Bearer {api_key}"},
    {"X Bad": "{api_key}"},
    {f"X-{i}": "{api_key}" for i in range(6)},
    {},
    "Authorization: Bearer {api_key}",
    {"Authorization": "Bearer {token}"},
    {"Authorization": "{api_key.__class__}"},
    {"Authorization": "Bearer {api_key}}"},
    {"Authorization": "Bearer {api_key}\r\nX-Injected: 1"},
    {"Authorization": "Bearer {api_key}é"},
    {"Authorization": ""},
    {"Authorization": 7},
])
def test_an_auth_header_that_could_steer_or_inject_is_refused(headers):
    with pytest.raises(ConnectorDraftError, match=r"^auth\.headers\b"):
        parse_connector_draft(_draft(auth={"type": "header", "headers": headers}))


def test_an_over_long_auth_template_is_refused():
    auth = {"type": "header", "headers": {"Authorization": "Bearer {api_key} " + "x " * 100, "X-Region": "{region}"}}

    with pytest.raises(ConnectorDraftError, match=r"^auth\.headers\[0\] must be at most 200"):
        parse_connector_draft(_draft(auth=auth))


@pytest.mark.parametrize("name", _TOKEN_SHAPES)
def test_a_value_in_auth_or_the_check_is_refused_without_echoing_it(name):
    value = _TOKEN_SHAPES[name]
    for overrides, rule in (
        ({"auth": {"type": "header", "headers": {"Authorization": f"Bearer {value}", "X-Region": "{region}"}}},
         r"^auth\.headers\[0\] looks like"),
        ({"fields": [{"name": "api_key", "label": "Key"}],
          "auth": {"type": "basic", "username": "{api_key}", "password": value}}, r"^auth\.password looks like"),
        ({"check": {"path": f"/v2/me?key={value}"}}, r"^check\.path\b"),
    ):
        with pytest.raises(ConnectorDraftError, match=rule) as exc:
            parse_connector_draft(_draft(**overrides))
        assert value not in str(exc.value)


def test_a_held_value_written_into_auth_is_refused_without_the_value(held_value):
    auth = {"type": "header", "headers": {"Authorization": f"Bearer {held_value}", "X-Region": "{region}"}}

    with pytest.raises(ConnectorDraftError, match=r"^auth\.headers\[0\] looks like") as exc:
        parse_connector_draft(_draft(auth=auth))

    assert held_value not in str(exc.value)


def test_a_held_value_as_a_header_name_is_refused_without_the_value(held_value):
    auth = {"type": "header", "headers": {f"X-{held_value}": "", "Authorization": "Bearer {api_key}"}}

    with pytest.raises(ConnectorDraftError, match=r"^auth\.headers has a header name") as exc:
        parse_connector_draft(_draft(auth=auth))

    assert held_value not in str(exc.value)


@pytest.mark.parametrize("name,value", [
    ("Authorization", "Basic {email}:{api_token}"),
    ("authorization", "Basic {email}:{api_token}"),
    ("Authorization", "basic {api_token}"),
    ("Authorization", "BASIC X"),
    ("X-Authorization", "Basic {email}:{api_token}"),
])
def test_a_hand_built_basic_header_is_refused_in_favour_of_type_basic(name, value):
    auth = {"type": "header", "headers": {name: value}}

    with pytest.raises(ConnectorDraftError, match=r"use auth type 'basic'"):
        parse_connector_draft(_draft(fields=_BASIC_FIELDS, auth=auth))


_QUERY_FIELDS = [{"name": "api_key", "label": "Access key"}]
_QUERY_AUTH = {"type": "query", "params": {"access_key": "{api_key}"}}


def test_query_auth_parses_and_round_trips():
    draft = parse_connector_draft(_draft(fields=_QUERY_FIELDS, auth=_QUERY_AUTH, check={"path": "/v1/eod?symbols=AAPL"}))

    assert draft.auth.type == "query"
    assert draft.auth.params == (("access_key", "{api_key}"),)
    assert draft.to_dict()["auth"] == _QUERY_AUTH
    assert parse_connector_draft(draft.to_dict()) == draft


@pytest.mark.parametrize("params,rule", [
    ({}, r"^auth\.params must be an object of 1 to 5"),
    ({f"k{i}": "{api_key}" for i in range(6)}, r"^auth\.params must be an object of 1 to 5"),
    ({"access.key": "{api_key}"}, r"^auth\.params has a parameter name"),
    ({"1key": "{api_key}"}, r"^auth\.params has a parameter name"),
    ({"access key": "{api_key}"}, r"^auth\.params has a parameter name"),
    ({"X-HTTP-Method-Override": "{api_key}"}, r"^auth\.params has a parameter name"),
    ({"x-method-override": "{api_key}"}, r"^auth\.params has a parameter name"),
    ({"access_key": "{api_key}", "ACCESS_KEY": "{api_key}"}, r"^auth\.params has a parameter name"),
    ({"access_key": "{api_key}&admin=1"}, r"^auth\.params\[0\] may not hold &"),
    ({"access_key": "{api_key}#frag"}, r"^auth\.params\[0\] may not hold &"),
    ({"access_key": "a+{api_key}"}, r"^auth\.params\[0\] may not hold &"),
    ({"access_key": "x={api_key}"}, r"^auth\.params\[0\] may not hold &"),
    ({"access_key": "a {api_key}"}, r"^auth\.params\[0\] may not hold &"),
    ({"access_key": "{token}"}, r"^auth\.params\[0\] names a field"),
    ("access_key={api_key}", r"^auth\.params must be an object"),
])
def test_a_query_auth_section_of_the_wrong_shape_is_refused(params, rule):
    with pytest.raises(ConnectorDraftError, match=rule):
        parse_connector_draft(_draft(fields=_QUERY_FIELDS, auth={"type": "query", "params": params}))


def test_a_token_shaped_or_held_parameter_name_is_refused_without_echoing_it(held_value):
    for name in ("x" + _TOKEN_SHAPES["bare"], held_value):
        auth = {"type": "query", "params": {name: "{api_key}"}}
        with pytest.raises(ConnectorDraftError, match=r"^auth\.params has a parameter name") as exc:
            parse_connector_draft(_draft(fields=_QUERY_FIELDS, auth=auth))
        assert name not in str(exc.value)


@pytest.mark.parametrize("path", [
    "/v1/eod?access_key=demo", "/v1/eod?ACCESS_KEY=demo", "/v1/eod?access_key[]=demo", "/v1/eod?symbols=A&access_key",
    "/v1/eod?access.key=demo", "/v1/eod?access_key[0]=demo", "/v1/eod?symbols=A;access_key=demo",
    "/v1/eod?[access_key]=demo", "/v1/eod?]access_key=demo", "/v1/eod?[access_key][x]=demo",
    "/v1/eod?filter[status]=open",
])
def test_a_check_path_carrying_the_placed_parameter_is_refused(path):
    with pytest.raises(ConnectorDraftError, match=r"^check\.path may carry only plain parameter names"):
        parse_connector_draft(_draft(fields=_QUERY_FIELDS, auth=_QUERY_AUTH, check={"path": path}))


def test_a_header_template_may_carry_a_check_path_with_any_query():
    draft = parse_connector_draft(_draft(check={"path": "/v1/me?access_key=demo"}))
    assert draft.check.path == "/v1/me?access_key=demo"


def test_a_token_shaped_header_name_is_refused_without_echoing_it():
    name = "X-" + _TOKEN_SHAPES["bare"]

    with pytest.raises(ConnectorDraftError, match=r"^auth\.headers\b") as exc:
        parse_connector_draft(_draft(auth={"type": "header", "headers": {name: "Bearer {api_key}"}}))

    assert _TOKEN_SHAPES["bare"] not in str(exc.value)


_HEADER_CASES = Path(__file__).parent / "fixtures" / "header_name_cases.json"
#: Other services carry this file byte for byte; a change here must reach their copies.
_HEADER_CASES_SHA256 = "787974a77b274c7cbb08caca5c76c7d1881a15b824ca1eb2e8b458a5e84e057e"


def test_the_shared_header_name_cases_are_pinned_and_canonical():
    raw = _HEADER_CASES.read_bytes()
    cases = json.loads(raw)

    assert hashlib.sha256(raw).hexdigest() == _HEADER_CASES_SHA256
    assert raw.decode("ascii") == json.dumps(cases, sort_keys=True, indent=2, ensure_ascii=True) + "\n"


@pytest.mark.parametrize("case", json.loads(_HEADER_CASES.read_text(encoding="ascii")), ids=repr)
def test_the_header_name_rule_matches_the_shared_cases(case):
    if case["allowed"]:
        check_header_name(case["name"])
    else:
        with pytest.raises(ConnectorDraftError, match="header name that is not allowed") as exc:
            check_header_name(case["name"])
        assert not case["name"] or case["name"] not in str(exc.value)


def test_the_header_name_rule_refuses_a_held_value_on_its_own(held_value):
    with pytest.raises(ConnectorDraftError, match="header name that is not allowed") as exc:
        check_header_name(f"X-{held_value}")

    assert held_value not in str(exc.value)


@pytest.mark.parametrize("name", [None, 7, b"Authorization"])
def test_a_header_name_that_is_not_text_is_refused(name):
    with pytest.raises(ConnectorDraftError, match=r"^auth\.headers has a header name"):
        check_header_name(name, "auth.headers")


@pytest.mark.parametrize("name", ["base_url", "label", "user_label"])
def test_names_the_host_stores_or_shows_are_reserved(name):
    with pytest.raises(ConnectorDraftError, match="reserved"):
        parse_connector_draft(_draft(fields=[{"name": name, "label": "Taken"}]))


def test_a_template_holds_at_most_eleven_fields():
    assert MAX_FIELDS == 11
    auth = {"type": "header", "headers": {
        "X-Keys": ".".join(f"{{f{i}}}" for i in range(6)),
        "X-More-Keys": ".".join(f"{{f{i}}}" for i in range(6, 11)),
    }}
    parse_connector_draft(_draft(fields=[{"name": f"f{i}", "label": "L"} for i in range(11)], auth=auth))


def test_the_tool_describes_the_check():
    tool = connector_draft_tool(ConnectorDraftBox())

    check = tool.input_schema["properties"]["check"]
    assert check["required"] == ["path"]
    assert set(check["properties"]) == {"path"}
    assert check["additionalProperties"] is False
    assert "check" in tool.description
    assert "call it directly from code" in tool.description
    assert "base URL" not in tool.input_schema["properties"]["usage_notes"]["description"]


def test_the_tool_says_every_path_starts_at_the_host_with_its_version():
    tool = connector_draft_tool(ConnectorDraftBox())
    texts = {
        "description": tool.description,
        "check.path": tool.input_schema["properties"]["check"]["properties"]["path"]["description"],
        "usage_notes": tool.input_schema["properties"]["usage_notes"]["description"],
    }

    for where, text in texts.items():
        assert "only" in text and "host" in text, where
        assert "/v2/" in text, where
        assert "includ" in text and "version" in text, where


def _union_keywords(schema):
    """Every oneOf/anyOf/allOf key anywhere in ``schema``."""
    if isinstance(schema, dict):
        return {k for k in schema if k in ("oneOf", "anyOf", "allOf")} | {
            found for v in schema.values() for found in _union_keywords(v)
        }
    if isinstance(schema, list):
        return {found for v in schema for found in _union_keywords(v)}
    return set()


def test_the_tool_describes_auth_as_one_flat_object():
    tool = connector_draft_tool(ConnectorDraftBox())
    schema = tool.input_schema

    auth = schema["properties"]["auth"]
    assert "auth" in schema["required"]
    assert auth["properties"]["type"]["enum"] == ["basic", "header", "query"]
    assert set(auth["properties"]) == {"type", "headers", "username", "password", "params"}
    assert "type 'query'" in tool.description
    assert auth["properties"]["params"]["maxProperties"] == 5
    field = schema["properties"]["fields"]["items"]["properties"]["required"]
    assert field["enum"] == [True]
    assert auth["additionalProperties"] is False
    assert _union_keywords(schema) == set()
    assert "call_minds_api" in schema["properties"]["usage_notes"]["description"]
    assert "type 'header'" in tool.description and "type 'basic'" in tool.description


def test_the_examples_the_tool_gives_parse():
    header = _draft(fields=[{"name": "api_key", "label": "Key"}], auth=_one_field_auth())
    basic = _draft(fields=_BASIC_FIELDS, auth={"type": "basic", "username": "{email}/token", "password": "{api_token}"})
    notion = _draft(
        fields=[{"name": "api_key", "label": "Key"}],
        auth={"type": "header", "headers": {"Authorization": "Bearer {api_key}", "Notion-Version": "2022-06-28"}},
    )
    for data in (header, basic, notion):
        parse_connector_draft(data)
