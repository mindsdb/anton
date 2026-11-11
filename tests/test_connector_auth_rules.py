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

from anton.core.tools.connector_auth_rules import AuthRuleError, query_collides, render

_MODULE = Path(__file__).resolve().parents[1] / "anton" / "core" / "tools" / "connector_auth_rules.py"
_MODULE_SHA256 = "6d990ad784d591db7c9b66b66f35f6b560c0715cbbb7f861fbe1fd97655a9b41"
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
    (["symbols"], False), ([], False), (["access_key_x"], False),
])
def test_query_collides_ignores_case_and_brackets(names, collides):
    assert query_collides({"type": "query", "params": {"access_key": "{api_key}"}}, names) is collides


def test_query_collides_is_false_for_other_types():
    assert query_collides({"type": "header", "headers": {"access_key": "{api_key}"}}, ["access_key"]) is False


def test_the_module_loads_with_the_standard_library_alone():
    result = subprocess.run(
        [sys.executable, "-I", "-c", _LOAD_ALONE, str(_MODULE)], capture_output=True, text=True, check=True
    )

    loaded = json.loads(result.stdout)
    top_level = {name.split(".")[0] for name in loaded}
    assert top_level <= set(sys.stdlib_module_names) | {"connector_auth_rules"}
