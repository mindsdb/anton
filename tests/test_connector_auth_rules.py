"""The shared auth rules module that other services carry byte for byte.

auth and the call gateway keep exact copies of
``anton/core/tools/connector_auth_rules.py`` and pin its SHA-256, so these
tests pin what makes a copy safe to carry: its hash, and that it loads with
the standard library alone.
"""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from anton.core.tools.connector_auth_rules import (
    AuthRuleError,
    parse_auth,
    query_collides,
    query_names,
    render,
    without_missing,
)

_MODULE = Path(__file__).resolve().parents[1] / "anton" / "core" / "tools" / "connector_auth_rules.py"
_MODULE_SHA256 = "e6d4db34127c0ca1565ec4a3f21f3ae57075d0c27a122e378aa10c4eb248b75d"
#: Copies to update, and the hash their tests pin, whenever the module changes.
_COPIES = (
    "auth: auth/datavault/connector_auth_rules.py",
    "mindshub_inference: the custom API gateway's copy",
)
#: Loads the module from its file in a fresh interpreter and prints what it imported.
_LOAD_ALONE = (
    "import importlib.util, json, sys\n"
    "before = set(sys.modules)\n"
    "spec = importlib.util.spec_from_file_location('connector_auth_rules', sys.argv[1])\n"
    "module = importlib.util.module_from_spec(spec)\n"
    "spec.loader.exec_module(module)\n"
    "print(json.dumps(sorted(set(sys.modules) - before)))\n"
)


def test_the_module_hash_is_pinned_so_its_copies_are_resynced():
    digest = hashlib.sha256(_MODULE.read_bytes()).hexdigest()

    assert digest == _MODULE_SHA256, (
        "connector_auth_rules.py changed. Copy it byte for byte to: "
        + "; ".join(_COPIES)
        + f". Then set the pinned SHA-256 here and in each copy's test to {digest}."
    )


_CASES = Path(__file__).parent / "fixtures" / "auth_section_cases.json"
_CASES_SHA256 = "c541a8e0b516371a71c683b910a5146e0cc4c8de5e4456ec5088ea5d1336ed76"
#: Rewrites the case list after a rule change; copy the result to every service that carries the module.
_REGENERATE = (
    "uv run python -c \"from tests.test_connector_auth_rules import _CASES, _cases_text; "
    "_CASES.write_text(_cases_text(), encoding='ascii')\""
)
_KEY = [["api_key", True]]
_HARVEST = [["token", True], ["account_id", False]]
_BASIC = [["email", False], ["api_token", True]]
_LONG = "{api_key} " + "a " * 95
_NAME_41 = "a" * 41
_QUERY = {"type": "query", "params": {"access_key": "{api_key}"}}
#: (fields as [name, secret], auth) pairs for parse_auth. Literals avoid real key
#: shapes; the one token-run literal is made up and only proves that rule.
_PARSE_INPUTS = [
    (_KEY, {"type": "header", "headers": {"Authorization": "Bearer {api_key}"}}),
    (_HARVEST, {"type": "header", "headers": {
        "Authorization": "Bearer {token}", "Harvest-Account-Id": "{account_id}", "Notion-Version": "2022-06-28",
    }}),
    (_KEY, {"type": "header", "headers": {"X-Api-Key": "  {api_key}  "}}),
    (_KEY, {"type": "header", "headers": {f"X-H{i}": "{api_key}" for i in range(5)}}),
    (_KEY, {"type": "header", "headers": {f"X-H{i}": "{api_key}" for i in range(6)}}),
    (_KEY, {"type": "header", "headers": {}}),
    (_KEY, {"type": "header", "headers": {"X-Api-Key": _LONG.strip()}}),
    (_KEY, {"type": "header", "headers": {"X-Api-Key": _LONG.strip() + "ab"}}),
    (_KEY, {"type": "header", "headers": {"Host": "{api_key}"}}),
    (_KEY, {"type": "header", "headers": {"X-Forwarded-For": "{api_key}"}}),
    (_KEY, {"type": "header", "headers": {"Accept": "{api_key}"}}),
    (_KEY, {"type": "header", "headers": {"X-Api-Key": "{api_key}", "x-api-key": "{api_key}"}}),
    (_KEY, {"type": "header", "headers": {"Authorization": "Basic {api_key}"}}),
    (_KEY, {"type": "header", "headers": {"X-Authorization": "basic {api_key}"}}),
    (_KEY, {"type": "header", "headers": {"X-Api-Key": "{api_key}}"}}),
    (_KEY, {"type": "header", "headers": {"X-Api-Key": "{token}"}}),
    (_KEY, {"type": "header", "headers": {"X-Api-Key": "{api_key}\r\nX-Injected: 1"}}),
    (_KEY, {"type": "header", "headers": {"X-Api-Key": "{api_key}é"}}),
    (_KEY, {"type": "header", "headers": {"X-Api-Key": ""}}),
    (_KEY, {"type": "header", "headers": {"X-Api-Key": 7}}),
    (_KEY, {"type": "header", "headers": {"X-Api-Key": "{api_key} abc123def456ghi789jkl0"}}),
    (_KEY, {"type": "header", "headers": {"X-Api-Key": "{api_key}"}, "note": "x"}),
    (_BASIC, {"type": "basic", "username": "{email}/token", "password": "{api_token}"}),
    ([["api_key", True]], {"type": "basic", "username": "{api_key}", "password": "X"}),
    ([["api_key", True]], {"type": "basic", "username": "{api_key}", "password": ""}),
    (_BASIC, {"type": "basic", "username": "admin", "password": "{api_token}"}),
    (_BASIC, {"type": "basic", "username": "{email}:x", "password": "{api_token}"}),
    (_BASIC, {"type": "basic", "username": "{email}"}),
    (_BASIC, {"type": "basic", "username": "{email}", "password": 7}),
    (_BASIC, {"type": "basic", "username": "{email}", "password": "{api_token}", "headers": {}}),
    (_KEY, _QUERY),
    ([["api_key", True], ["client_id", False]], {"type": "query", "params": {"access_key": "{api_key}", "client": "{client_id}"}}),
    (_KEY, {"type": "query", "params": {_NAME_41: "{api_key}"}}),
    (_KEY, {"type": "query", "params": {_NAME_41 + "a": "{api_key}"}}),
    (_KEY, {"type": "query", "params": {}}),
    (_KEY, {"type": "query", "params": {f"k{i}": "{api_key}" for i in range(6)}}),
    (_KEY, {"type": "query", "params": {"access.key": "{api_key}"}}),
    (_KEY, {"type": "query", "params": {"X-HTTP-Method-Override": "{api_key}"}}),
    (_KEY, {"type": "query", "params": {"access_key": "{api_key}", "ACCESS_KEY": "{api_key}"}}),
    (_KEY, {"type": "query", "params": {"access_key": "{api_key}&admin=1"}}),
    (_KEY, {"type": "query", "params": {"access_key": "a+{api_key}"}}),
    (_KEY, {"type": "query", "params": {"access_key": "x={api_key}"}}),
    (_KEY, {"type": "query", "params": {"access_key": "{api_key}#frag"}}),
    (_KEY, {"type": "query", "params": {"access_key": "a {api_key}"}}),
    (_HARVEST, {"type": "header", "headers": {"X-Account": "{account_id}"}}),
    (_HARVEST, {"type": "header", "headers": {"Authorization": "Bearer {token}"}}),
    (_KEY, None),
    (_KEY, "Bearer {api_key}"),
    (_KEY, {"headers": {"X-Api-Key": "{api_key}"}}),
    (_KEY, {"type": "oauth"}),
]
_OPT_KEY = [["api_key", True]]
_OPT_ACCOUNT = [["api_key", True], ["account", False]]
_OPT_BASIC = [["email", True], ["password", True]]
_BEARER = {"type": "header", "headers": {"Authorization": "Bearer {api_key}"}}
_LITERAL = {"type": "header", "headers": {"Notion-Version": "2022-06-28"}}
#: (fields, auth, optional names, stored) for parse_auth with the optional rules; a case
#: records ``optional`` and ``stored`` only when set, so the cases above keep their shape.
_PARSE_OPTIONAL_INPUTS = [
    (_OPT_KEY, _BEARER, ["api_key"], False),
    (_OPT_KEY, _QUERY, ["api_key"], False),
    (_OPT_ACCOUNT, {"type": "header", "headers": {"Authorization": "Bearer {api_key}", "X-Account": "{account}"}}, ["api_key"], False),
    (_OPT_ACCOUNT, {"type": "header", "headers": {"Authorization": "Bearer {api_key}.{account}"}}, ["api_key"], False),
    ([["api_key", True], ["token", True]], {"type": "query", "params": {"k": "{api_key}{token}"}}, ["api_key"], False),
    (_OPT_ACCOUNT, _BEARER, ["account"], False),
    (_OPT_KEY, _BEARER, ["missing"], False),
    ([["email", True], ["password", True]], {"type": "basic", "username": "{email}", "password": "{password}"}, ["password"], False),
    ([["email", True], ["password", True]], {"type": "basic", "username": "{email}", "password": "{password}"}, ["email"], False),
    ([["email", False], ["password", True]], {"type": "basic", "username": "{email}", "password": "{password}"}, ["password"], False),
    ([["user", True], ["password", True], ["tail", True]], {"type": "basic", "username": "{user}", "password": "{password}{tail}"}, ["password"], False),
    (_OPT_KEY, {"type": "none"}, [], False),
    ([], {"type": "none"}, [], True),
    (_OPT_KEY, {"type": "none"}, [], True),
    ([["base", False]], {"type": "none", "headers": {}}, [], True),
    ([], _LITERAL, [], True),
    ([], {"type": "query", "params": {"format": "json"}}, [], True),
    (_OPT_KEY, _LITERAL, [], True),
    ([["account", False]], {"type": "header", "headers": {"X-Account": "{account}"}}, [], True),
    ([], {"type": "basic", "username": "{email}", "password": ""}, [], True),
    ([["email", False]], {"type": "basic", "username": "{email}", "password": ""}, [], True),
    ([], {"type": "oauth"}, [], True),
    (_KEY, _LITERAL, [], False),
    (_OPT_KEY, _BEARER, [], True),
    ([["email", True]], {"type": "basic", "username": "{email}", "password": ""}, [], True),
    (_OPT_ACCOUNT, {"type": "header", "headers": {"Authorization": "Bearer {api_key}", "X-Account": "{account}"}}, ["api_key"], True),
]
#: (auth, held names, optional names) for without_missing.
_WITHOUT_MISSING_INPUTS = [
    (_BEARER, ["api_key"], ["api_key"]),
    (_BEARER, [], ["api_key"]),
    (_BEARER, [], []),
    (_QUERY, [], ["api_key"]),
    ({"type": "header", "headers": {"Authorization": "Bearer {api_key}", "Notion-Version": "2022-06-28"}}, [], ["api_key"]),
    ({"type": "query", "params": {"access_key": "{api_key}", "format": "json"}}, [], ["api_key"]),
    ({"type": "header", "headers": {"Authorization": "Bearer {api_key}", "X-Account": "{account}"}}, ["account"], ["api_key"]),
    ({"type": "header", "headers": {"Authorization": "Bearer {api_key}", "X-Account": "{account}"}}, ["api_key"], ["api_key"]),
    ({"type": "header", "headers": {"Authorization": "Bearer {api_key}", "X-Account": "{account}"}}, ["api_key"], ["api_key", "account"]),
    ({"type": "basic", "username": "{email}", "password": "{password}"}, ["email"], ["password"]),
    ({"type": "basic", "username": "{email}", "password": "{password}"}, ["email", "password"], ["password"]),
    ({"type": "basic", "username": "{email}", "password": "{password}"}, ["password"], ["password"]),
    ({"type": "basic", "username": "{email}", "password": "{password}"}, ["email"], []),
    ({"type": "none"}, [], []),
    ({"type": "oauth"}, [], []),
    ({"type": "basic", "username": "{email}", "password": "{password}"}, ["password"], ["email", "password"]),
    ({"type": "header", "headers": {"A": "{acct}-{api_key}"}}, ["acct"], ["api_key"]),
]
#: (auth, values) pairs for render.
_RENDER_INPUTS = [
    ({"type": "header", "headers": {"Authorization": "Bearer {token}", "Notion-Version": "2022-06-28"}}, {"token": "t0"}),
    ({"type": "basic", "username": "{email}/token", "password": "{api_token}"}, {"email": "me@acme.example", "api_token": "t0"}),
    ({"type": "basic", "username": "{api_key}", "password": ""}, {"api_key": "k0"}),
    ({"type": "query", "params": {"access_key": "{api_key}", "client": "{client_id}"}}, {"api_key": "k0", "client_id": "c0"}),
    ({"type": "header", "headers": {"X-Api-Key": "{api_key}"}}, {}),
    ({"type": "header", "headers": {"X-Api-Key": "{api_key}"}}, {"api_key": "k0\r\nX-Injected: 1"}),
    ({"type": "query", "params": {"access_key": "{api_key}"}}, {"api_key": "k0\x7f"}),
    ({"type": "query", "params": {"access_key": "{api_key}"}}, {"api_key": "k0\x85"}),
    ({"type": "header", "headers": {"X-Api-Key": "{api_key}"}}, {"api_key": "k0 "}),
    ({"type": "oauth"}, {}),
    ({"type": "none"}, {}),
    ({"type": "none"}, {"api_key": "k0"}),
]
#: Raw query strings for query_names.
_QUERY_NAME_INPUTS = ["a=1&access_key=2", "a=1;access_key=2", "access%5Fkey=2&b", "access+key=2", ""]
_MIXED_CASE_QUERY = {"type": "query", "params": {"Access-Key": "{api_key}"}}
#: (auth, request query names) pairs for query_collides.
_COLLIDE_INPUTS = [
    *((_QUERY, names) for names in [
        ["symbols"], [], ["symbols", "date_from"], ["access_key_x"], ["accesskey"],
        ["access_key"], ["ACCESS_KEY"], ["access_key[]"], ["access_key[0]"], ["[access_key]"], ["]access_key"],
        ["[access_key][x]"], ["access.key"], ["access key"], [" access_key"], ["access+key"], ["access[key"],
        ["symbols;access_key"], ["filter[status]"], [_NAME_41], [_NAME_41 + "a"],
    ]),
    (_MIXED_CASE_QUERY, ["access-key"]), (_MIXED_CASE_QUERY, ["ACCESS-KEY"]), (_MIXED_CASE_QUERY, ["access_key"]),
]


def _outcome(call) -> dict:
    """Run ``call`` and record its result or its refusal message."""
    try:
        return {"accepted": True, "result": call()}
    except AuthRuleError as e:
        return {"accepted": False, "error": str(e)}


def _parse_optional_case(fields: list, auth: object, optional: list[str], stored: bool) -> dict:
    """One parse_auth case run with the optional and stored rules, recording only what is set."""
    kwargs: dict = {}
    case: dict = {"fields": fields, "auth": auth}
    if optional:
        kwargs["optional"] = frozenset(optional)
        case["optional"] = optional
    if stored:
        kwargs["stored"] = True
        case["stored"] = True
    return {**case, **_outcome(lambda: parse_auth(auth, [tuple(x) for x in fields], **kwargs))}


def _build_cases() -> dict[str, list[dict]]:
    """Run every input through the module with no secret detector, as each copy will."""
    return {
        "parse_auth": [
            {"fields": fields, "auth": auth, **_outcome(lambda f=fields, a=auth: parse_auth(a, [tuple(x) for x in f]))}
            for fields, auth in _PARSE_INPUTS
        ]
        + [_parse_optional_case(fields, auth, optional, stored) for fields, auth, optional, stored in _PARSE_OPTIONAL_INPUTS],
        "without_missing": [
            {
                "auth": auth, "held": held, "optional": optional,
                **_outcome(lambda a=auth, h=held, o=optional: without_missing(a, h, o)),
            }
            for auth, held, optional in _WITHOUT_MISSING_INPUTS
        ],
        "render": [
            {"auth": auth, "values": values, **_outcome(lambda a=auth, v=values: list(render(a, v)))}
            for auth, values in _RENDER_INPUTS
        ],
        "query_names": [{"query": q, "names": query_names(q)} for q in _QUERY_NAME_INPUTS],
        "query_collides": [
            {"auth": auth, "names": names, "collides": query_collides(auth, names)} for auth, names in _COLLIDE_INPUTS
        ],
    }


def _cases_text() -> str:
    """The case list in its canonical form: sorted keys, 2-space indent, ASCII, final newline."""
    return json.dumps(_build_cases(), sort_keys=True, indent=2, ensure_ascii=True) + "\n"


def test_the_auth_section_cases_are_what_the_module_gives():
    assert _CASES.read_text(encoding="ascii") == _cases_text(), (
        f"Regenerate the case list with: {_REGENERATE}. Then copy it, and the module, to the services that carry them."
    )
    assert hashlib.sha256(_CASES.read_bytes()).hexdigest() == _CASES_SHA256


def test_the_auth_section_cases_cover_both_outcomes_for_each_function():
    cases = _build_cases()
    for kind in ("header", "basic", "query"):
        outcomes = {c["accepted"] for c in cases["parse_auth"] if isinstance(c["auth"], dict) and c["auth"].get("type") == kind}
        assert outcomes == {True, False}
    assert {c["accepted"] for c in cases["render"]} == {True, False}
    assert {c["accepted"] for c in cases["without_missing"]} == {True, False}
    stored = [c for c in cases["parse_auth"] if c.get("stored")]
    assert {c["accepted"] for c in stored} == {True, False}
    optional = [c for c in cases["parse_auth"] if c.get("optional")]
    assert {c["accepted"] for c in optional} == {True, False}
    assert {c["collides"] for c in cases["query_collides"]} == {True, False}


def test_render_fills_header_slots_and_keeps_literals():
    auth = {"type": "header", "headers": {"Authorization": "Bearer {token}", "Notion-Version": "2022-06-28"}}

    assert render(auth, {"token": "t0"}) == ({"Authorization": "Bearer t0", "Notion-Version": "2022-06-28"}, [])


@pytest.mark.parametrize("password,pair", [
    ("{api_token}", "me@acme.example/token:t0"), ("X", "me@acme.example/token:X"), ("", "me@acme.example/token:"),
])
def test_render_builds_one_basic_header(password, pair):
    auth = {"type": "basic", "username": "{email}/token", "password": password}

    headers, params = render(auth, {"email": "me@acme.example", "api_token": "t0"})

    assert headers == {"Authorization": "Basic " + base64.b64encode(pair.encode()).decode("ascii")}
    assert params == []


def test_render_gives_query_parameters_in_template_order():
    auth = {"type": "query", "params": {"access_key": "{api_key}", "client": "{client_id}"}}

    assert render(auth, {"api_key": "k0", "client_id": "c0"}) == ({}, [("access_key", "k0"), ("client", "c0")])


def test_render_refuses_a_missing_value_without_naming_it():
    with pytest.raises(AuthRuleError, match="has no value") as exc:
        render({"type": "header", "headers": {"X-Api-Key": "{secret_field_name}"}}, {})

    assert "secret_field_name" not in str(exc.value)


@pytest.mark.parametrize("value", ["k0\r\nX-Injected: 1", "k0\n", "k0\x00", "k0\x7f", "k0\x85", "k0\x9b", "k0 "])
def test_render_refuses_a_value_with_a_control_character_without_quoting_it(value):
    with pytest.raises(AuthRuleError, match="control character") as exc:
        render({"type": "header", "headers": {"X-Api-Key": "{api_key}"}}, {"api_key": value})

    assert "k0" not in str(exc.value)


def test_render_refuses_a_type_it_does_not_know():
    with pytest.raises(AuthRuleError, match="auth.type is not one"):
        render({"type": "oauth"}, {})


@pytest.mark.parametrize("names,collides", [
    (["symbols"], False), ([], False), (["symbols", "date_from"], False), (["access_key_x"], False),
    (["accesskey"], False), ([_NAME_41], False),
    (["access_key"], True), (["ACCESS_KEY"], True), (["access_key[]"], True), (["access_key[0]"], True),
    (["[access_key]"], True), (["]access_key"], True), (["[access_key][x]"], True), (["access.key"], True),
    (["access key"], True), ([" access_key"], True), (["access+key"], True), (["access[key"], True),
    (["symbols;access_key"], True), (["filter[status]"], True), ([_NAME_41 + "a"], True),
])
def test_query_collides_allows_only_plain_names_other_than_the_placed_one(names, collides):
    assert query_collides(_QUERY, names) is collides


@pytest.mark.parametrize("query,names", [
    ("a=1&access_key=2", ["a", "access_key"]),
    ("a=1;access_key=2", ["a", "access_key"]),
    ("access%5Fkey=2&b", ["access_key", "b"]),
    ("access+key=2", ["access key"]),
    ("", []),
])
def test_query_names_split_on_both_separators_and_decode(query, names):
    assert query_names(query) == names


def test_query_collides_is_false_for_other_types():
    assert query_collides({"type": "header", "headers": {"access_key": "{api_key}"}}, ["[odd]", "access_key"]) is False


def test_the_module_loads_with_the_standard_library_alone():
    result = subprocess.run(
        [sys.executable, "-I", "-c", _LOAD_ALONE, str(_MODULE)], capture_output=True, text=True, check=True
    )

    loaded = json.loads(result.stdout)
    top_level = {name.split(".")[0] for name in loaded}
    assert top_level <= set(sys.stdlib_module_names) | {"connector_auth_rules"}


def _stored(auth, fields, **kwargs):
    return parse_auth(auth, [tuple(x) for x in fields], stored=True, **kwargs)


def test_an_optional_key_placed_alone_is_accepted_in_a_template():
    assert parse_auth(_BEARER, [("api_key", True)], optional=frozenset({"api_key"})) == _BEARER


def test_a_template_with_no_optional_field_behaves_as_before():
    assert parse_auth(_BEARER, [("api_key", True)], optional=frozenset()) == _BEARER
    with pytest.raises(AuthRuleError, match="must place a secret field"):
        parse_auth(_LITERAL, [])


def test_an_optional_field_must_be_a_declared_secret_field():
    with pytest.raises(AuthRuleError, match="optional field must be a secret"):
        parse_auth(_BEARER, [("api_key", True), ("account", False)], optional=frozenset({"account"}))
    with pytest.raises(AuthRuleError, match="optional field must be a secret"):
        parse_auth(_BEARER, [("api_key", True)], optional=frozenset({"missing"}))


def test_an_optional_field_sharing_a_header_or_parameter_is_refused():
    shared = {"type": "header", "headers": {"Authorization": "Bearer {api_key}.{account}"}}
    with pytest.raises(AuthRuleError, match=r"auth\.headers\[0\] places an optional field with another"):
        parse_auth(shared, [("api_key", True), ("account", False)], optional=frozenset({"api_key"}))
    query = {"type": "query", "params": {"k": "{api_key}{token}"}}
    with pytest.raises(AuthRuleError, match=r"auth\.params\[0\] places an optional field with another"):
        parse_auth(query, [("api_key", True), ("token", True)], optional=frozenset({"api_key"}))


def test_a_basic_username_may_not_be_optional():
    auth = {"type": "basic", "username": "{email}", "password": "{password}"}
    with pytest.raises(AuthRuleError, match="only the password may be optional"):
        parse_auth(auth, [("email", True), ("password", True)], optional=frozenset({"email"}))


def test_a_basic_password_is_optional_only_as_the_whole_password_beside_a_required_secret_username():
    auth = {"type": "basic", "username": "{email}", "password": "{password}"}
    fields = [("email", True), ("password", True)]
    assert parse_auth(auth, fields, optional=frozenset({"password"})) == auth
    wrapped = {"type": "basic", "username": "{user}", "password": "{password}{tail}"}
    with pytest.raises(AuthRuleError, match="only the optional field"):
        parse_auth(wrapped, [("user", True), ("password", True), ("tail", True)], optional=frozenset({"password"}))
    with pytest.raises(AuthRuleError, match="needs a username that places a required secret"):
        parse_auth(auth, [("email", False), ("password", True)], optional=frozenset({"password"}))


def test_a_template_may_not_be_none_and_the_type_list_never_offers_it():
    with pytest.raises(AuthRuleError, match=r"must be one of: basic, header, query\.$"):
        parse_auth({"type": "none"}, [])


def test_a_stored_section_may_be_none_when_the_connection_holds_no_field_it_places():
    assert _stored({"type": "none"}, []) == {"type": "none"}


def test_a_stored_none_that_leaves_out_a_held_field_is_refused():
    with pytest.raises(AuthRuleError, match="must place every field"):
        _stored({"type": "none"}, [["api_key", True]])


def test_a_stored_none_holds_no_other_key():
    with pytest.raises(AuthRuleError, match="key that is not allowed"):
        _stored({"type": "none", "headers": {}}, [])


def test_a_stored_section_need_not_place_a_secret_or_any_field():
    assert _stored(_LITERAL, []) == _LITERAL
    account = {"type": "header", "headers": {"X-Account": "{account}"}}
    assert _stored(account, [["account", False]]) == account


def test_a_stored_section_must_still_place_every_held_field():
    with pytest.raises(AuthRuleError, match="must place every field"):
        _stored(_LITERAL, [["api_key", True]])


def test_stored_mode_still_refuses_an_unknown_type():
    with pytest.raises(AuthRuleError, match="auth.type must be one of"):
        _stored({"type": "oauth"}, [])


def test_render_of_none_sends_nothing_even_when_values_exist():
    assert render({"type": "none"}, {"api_key": "k0"}) == ({}, [])


def test_without_missing_keeps_a_section_whose_fields_are_all_held():
    assert without_missing(_BEARER, ["api_key"], ["api_key"]) == _BEARER


def test_without_missing_drops_the_header_of_a_skipped_field():
    auth = {"type": "header", "headers": {"Authorization": "Bearer {api_key}", "X-Account": "{account}"}}

    assert without_missing(auth, ["account"], ["api_key"]) == {"type": "header", "headers": {"X-Account": "{account}"}}


def test_without_missing_drops_the_parameter_of_a_skipped_field_and_keeps_a_literal_one():
    auth = {"type": "query", "params": {"access_key": "{api_key}", "format": "json"}}

    assert without_missing(auth, [], ["api_key"]) == {"type": "query", "params": {"format": "json"}}


def test_without_missing_keeps_a_literal_header_beside_a_dropped_one():
    auth = {"type": "header", "headers": {"Authorization": "Bearer {api_key}", "Notion-Version": "2022-06-28"}}

    assert without_missing(auth, [], ["api_key"]) == _LITERAL


def test_without_missing_gives_none_when_nothing_is_left_to_send():
    assert without_missing(_BEARER, [], ["api_key"]) == {"type": "none"}
    assert without_missing(_QUERY, [], ["api_key"]) == {"type": "none"}


def test_without_missing_empties_a_skipped_basic_password():
    auth = {"type": "basic", "username": "{email}", "password": "{password}"}

    assert without_missing(auth, ["email"], ["password"]) == {"type": "basic", "username": "{email}", "password": ""}


def test_without_missing_refuses_a_missing_field_that_is_not_optional():
    with pytest.raises(AuthRuleError, match="not optional"):
        without_missing(_BEARER, [], [])
    basic = {"type": "basic", "username": "{email}", "password": "{password}"}
    with pytest.raises(AuthRuleError, match="not optional"):
        without_missing(basic, ["password"], ["password"])


def test_without_missing_does_not_change_its_input():
    auth = {"type": "header", "headers": {"Authorization": "Bearer {api_key}", "Notion-Version": "2022-06-28"}}

    without_missing(auth, [], ["api_key"])

    assert auth["headers"]["Authorization"] == "Bearer {api_key}"


def test_what_without_missing_gives_is_a_valid_stored_section():
    auth = {"type": "header", "headers": {"Authorization": "Bearer {api_key}", "X-Account": "{account}"}}
    fields = [("api_key", True), ("account", False)]
    for held in (["account"], ["api_key", "account"]):
        reduced = without_missing(auth, held, ["api_key"])
        assert parse_auth(reduced, [f for f in fields if f[0] in held], stored=True) == reduced


def test_a_stored_basic_section_must_place_a_secret_field():
    auth = {"type": "basic", "username": "{email}", "password": ""}
    with pytest.raises(AuthRuleError, match="must place a secret field"):
        _stored(auth, [["email", False]])
    assert _stored(auth, [["email", True]]) == auth


def test_a_stored_type_error_lists_none_and_a_template_one_does_not():
    with pytest.raises(AuthRuleError, match="basic, header, none, query"):
        _stored({"type": "oauth"}, [])
    with pytest.raises(AuthRuleError) as exc:
        parse_auth({"type": "oauth"}, [])
    assert "none" not in str(exc.value)


def test_a_stored_keyed_section_still_parses():
    assert _stored(_BEARER, [["api_key", True]]) == _BEARER


def test_without_missing_refuses_a_missing_username_even_when_the_password_is_held():
    auth = {"type": "basic", "username": "{email}", "password": "{password}"}
    with pytest.raises(AuthRuleError, match="not optional"):
        without_missing(auth, ["password"], ["email", "password"])


@pytest.mark.parametrize("auth,held,optional", [
    ({"type": "header", "headers": {"A": "{acct}-{api_key}"}}, ["acct"], ["api_key"]),
    ({"type": "query", "params": {"a": "{acct}{api_key}"}}, ["acct"], ["api_key"]),
])
def test_without_missing_refuses_to_drop_a_placement_that_also_places_a_held_field(auth, held, optional):
    with pytest.raises(AuthRuleError, match="also places a field the connection holds"):
        without_missing(auth, held, optional)
