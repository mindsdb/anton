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

from anton.core.tools.connector_auth_rules import AuthRuleError, parse_auth, query_collides, query_names, render

_MODULE = Path(__file__).resolve().parents[1] / "anton" / "core" / "tools" / "connector_auth_rules.py"
_MODULE_SHA256 = "ec4ba9e75d5b06e608dc91085fa9617ef267ebe7d2ee65f4f932b045f89d4e32"
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
_CASES_SHA256 = "a8c1f41e7342d004836999b2b80fcaa048bf187d5cd400838ab03b72caf5d4e5"
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
#: (auth, values) pairs for render.
_RENDER_INPUTS = [
    ({"type": "header", "headers": {"Authorization": "Bearer {token}", "Notion-Version": "2022-06-28"}}, {"token": "t0"}),
    ({"type": "basic", "username": "{email}/token", "password": "{api_token}"}, {"email": "me@acme.example", "api_token": "t0"}),
    ({"type": "basic", "username": "{api_key}", "password": ""}, {"api_key": "k0"}),
    ({"type": "query", "params": {"access_key": "{api_key}", "client": "{client_id}"}}, {"api_key": "k0", "client_id": "c0"}),
    ({"type": "header", "headers": {"X-Api-Key": "{api_key}"}}, {}),
    ({"type": "header", "headers": {"X-Api-Key": "{api_key}"}}, {"api_key": "k0\r\nX-Injected: 1"}),
    ({"type": "query", "params": {"access_key": "{api_key}"}}, {"api_key": "k0\x7f"}),
    ({"type": "oauth"}, {}),
]
#: Raw query strings for query_names.
_QUERY_NAME_INPUTS = ["a=1&access_key=2", "a=1;access_key=2", "access%5Fkey=2&b", "access+key=2", ""]
#: Request query names checked against _QUERY for query_collides.
_COLLIDE_INPUTS = [
    ["symbols"], [], ["symbols", "date_from"], ["access_key_x"], ["accesskey"],
    ["access_key"], ["ACCESS_KEY"], ["access_key[]"], ["access_key[0]"], ["[access_key]"], ["]access_key"],
    ["[access_key][x]"], ["access.key"], ["access key"], [" access_key"], ["access+key"], ["access[key"],
    ["symbols;access_key"], ["filter[status]"], [_NAME_41], [_NAME_41 + "a"],
]


def _outcome(call) -> dict:
    """Run ``call`` and record its result or its refusal message."""
    try:
        return {"accepted": True, "result": call()}
    except AuthRuleError as e:
        return {"accepted": False, "error": str(e)}


def _build_cases() -> dict[str, list[dict]]:
    """Run every input through the module with no secret detector, as each copy will."""
    return {
        "parse_auth": [
            {"fields": fields, "auth": auth, **_outcome(lambda f=fields, a=auth: parse_auth(a, [tuple(x) for x in f]))}
            for fields, auth in _PARSE_INPUTS
        ],
        "render": [
            {"auth": auth, "values": values, **_outcome(lambda a=auth, v=values: list(render(a, v)))}
            for auth, values in _RENDER_INPUTS
        ],
        "query_names": [{"query": q, "names": query_names(q)} for q in _QUERY_NAME_INPUTS],
        "query_collides": [
            {"auth": _QUERY, "names": names, "collides": query_collides(_QUERY, names)} for names in _COLLIDE_INPUTS
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


@pytest.mark.parametrize("value", ["k0\r\nX-Injected: 1", "k0\n", "k0\x00", "k0\x7f"])
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
