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
_MODULE_SHA256 = "b37142d9c2b17961980b803796cc4fab197d122e3fdd57bb867367d5170f052f"
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
_CASES_SHA256 = "8db88436c1936b053529697ac81d5e2a301bbc8bd59dbfb7bb794c8689ec7b7d"
_KEY = [["api_key", True]]
_HARVEST = [["token", True], ["account_id", False]]
_BASIC = [["email", False], ["api_token", True]]
_LONG = "{api_key} " + "a " * 95
#: (fields as [name, secret], auth) pairs. Literals are never key-shaped, so the
#: file is safe to copy; key shapes stay in Python tests that build them at runtime.
_CASE_INPUTS = [
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
    (_KEY, {"type": "query", "params": {"access_key": "{api_key}"}}),
    ([["api_key", True], ["client_id", False]], {"type": "query", "params": {"access_key": "{api_key}", "client": "{client_id}"}}),
    (_KEY, {"type": "query", "params": {}}),
    (_KEY, {"type": "query", "params": {f"k{i}": "{api_key}" for i in range(6)}}),
    (_KEY, {"type": "query", "params": {"access.key": "{api_key}"}}),
    (_KEY, {"type": "query", "params": {"X-HTTP-Method-Override": "{api_key}"}}),
    (_KEY, {"type": "query", "params": {"access_key": "{api_key}", "ACCESS_KEY": "{api_key}"}}),
    (_KEY, {"type": "query", "params": {"access_key": "{api_key}&admin=1"}}),
    (_KEY, {"type": "query", "params": {"access_key": "a {api_key}"}}),
    (_HARVEST, {"type": "header", "headers": {"X-Account": "{account_id}"}}),
    (_HARVEST, {"type": "header", "headers": {"Authorization": "Bearer {token}"}}),
    (_KEY, None),
    (_KEY, "Bearer {api_key}"),
    (_KEY, {"headers": {"X-Api-Key": "{api_key}"}}),
    (_KEY, {"type": "oauth"}),
]


def _build_cases() -> list[dict]:
    """Run every input through the module with no secret detector, as each copy will."""
    cases = []
    for fields, auth in _CASE_INPUTS:
        case = {"fields": fields, "auth": auth}
        try:
            case |= {"accepted": True, "canonical": parse_auth(auth, [tuple(f) for f in fields])}
        except AuthRuleError as e:
            case |= {"accepted": False, "error": str(e)}
        cases.append(case)
    return cases


def _cases_text() -> str:
    """The case list in its canonical form: sorted keys, 2-space indent, ASCII, final newline."""
    return json.dumps(_build_cases(), sort_keys=True, indent=2, ensure_ascii=True) + "\n"


def test_the_auth_section_cases_are_what_the_module_gives():
    assert _CASES.read_text(encoding="ascii") == _cases_text(), (
        "Regenerate tests/fixtures/auth_section_cases.json from _cases_text() and copy it to the services "
        "that carry the module."
    )
    assert hashlib.sha256(_CASES.read_bytes()).hexdigest() == _CASES_SHA256


def test_the_auth_section_cases_cover_both_outcomes_for_each_type():
    cases = _build_cases()
    for kind in ("header", "basic", "query"):
        outcomes = {c["accepted"] for c in cases if isinstance(c["auth"], dict) and c["auth"].get("type") == kind}
        assert outcomes == {True, False}


def test_render_fills_header_slots_and_keeps_literals():
    auth = {"type": "header", "headers": {"Authorization": "Bearer {token}", "Notion-Version": "2022-06-28"}}

    assert render(auth, {"token": "t0"}) == ({"Authorization": "Bearer t0", "Notion-Version": "2022-06-28"}, [])


@pytest.mark.parametrize("password,pair", [("{api_token}", "me@acme.example/token:t0"), ("X", "me@acme.example/token:X"), ("", "me@acme.example/token:")])
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


@pytest.mark.parametrize("names,collides", [
    (["access_key"], True), (["ACCESS_KEY"], True), (["access_key[]"], True),
    (["access_key[0]"], True), (["access.key"], True), (["ACCESS.KEY"], True), (["access key"], True),
    (["symbols;access_key"], True),
    (["symbols"], False), ([], False), (["access_key_x"], False), (["accesskey"], False),
])
def test_query_collides_reads_names_as_servers_do(names, collides):
    assert query_collides({"type": "query", "params": {"access_key": "{api_key}"}}, names) is collides


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
    assert query_collides({"type": "header", "headers": {"access_key": "{api_key}"}}, ["access_key"]) is False


def test_the_module_loads_with_the_standard_library_alone():
    result = subprocess.run(
        [sys.executable, "-I", "-c", _LOAD_ALONE, str(_MODULE)], capture_output=True, text=True, check=True
    )

    loaded = json.loads(result.stdout)
    top_level = {name.split(".")[0] for name in loaded}
    assert top_level <= set(sys.stdlib_module_names) | {"connector_auth_rules"}
