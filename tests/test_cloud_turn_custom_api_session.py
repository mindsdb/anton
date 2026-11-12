"""A cloud turn's custom API connections in the pod: never fetched, passed to the scratchpad by id only."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

import anton.core.llm.client as llm_client_mod
import anton.core.session as session_mod
from anton.cloud_turn.contract import TurnRequestV1
from anton.cloud_turn.session import _WORKSPACE_PATH_ENV, build_cloud_chat_session
from anton.core.backends import local
from anton.core.backends.sandbox import CellSandbox, overlay_key_allowed
from anton.core.backends.wire import CELL_DELIM, RESULT_END, RESULT_START

REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = Path(__file__).parent / "fixtures" / "turn_custom_api_block.json"
_PRODUCED = json.loads(_FIXTURE.read_text(encoding="utf-8"))["custom_api"]
_CORRELATION_ID = "3f1c9a2e-8b7d-4e6f-a5c4-2d1b0e9f8a7c"
_BASE_URL = "https://minds.internal/v1"
_CUSTOM_ENV = {
    "ANTON_CLOUD_CUSTOM_API_CORRELATION_ID": _CORRELATION_ID,
    "ANTON_CLOUD_CUSTOM_API_CONNECTIONS": json.dumps(
        [
            {"connection_id": 3, "credential_version": 1},
            {"connection_id": 5, "credential_version": 2},
        ],
        separators=(",", ":"),
    ),
}


def _request(**fields) -> TurnRequestV1:
    """A turn request parsed from the wire, carrying the producer's recorded custom API block."""
    body = {
        "protocol_version": 1,
        "conversation_id": "conv_1",
        "correlation_id": _CORRELATION_ID,
        "input": "hello",
        "llm": {
            "provider": "minds-cloud",
            "api_key": _PRODUCED["turn_key"],
            "base_url": _BASE_URL,
        },
        "custom_api": _PRODUCED,
    }
    body.update(fields)
    return TurnRequestV1.from_json(json.dumps(body))


def _captured_config(tmp_path, monkeypatch, request: TurnRequestV1):
    """The ChatSessionConfig the cloud builder hands to ChatSession for ``request``."""
    captured: dict = {}
    monkeypatch.setattr(
        session_mod, "ChatSession", lambda config: captured.setdefault("config", config)
    )
    monkeypatch.setattr(
        llm_client_mod.LLMClient,
        "from_settings",
        classmethod(lambda cls, settings: object()),
    )
    monkeypatch.setenv(_WORKSPACE_PATH_ENV, str(tmp_path))
    for name in ("ANTON_OPENAI_API_KEY", "ANTON_OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    build_cloud_chat_session(request)
    return captured["config"]


def test_a_custom_api_turn_opens_no_connection_and_builds_no_vault(
    tmp_path, monkeypatch
):
    """The pod never asks auth, or anyone, for a custom connection's values."""
    attempts: list = []

    def refuse_connect(sock, address):
        attempts.append(address)
        raise OSError("no network in this test")

    monkeypatch.setattr(socket.socket, "connect", refuse_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse_connect)

    config = _captured_config(tmp_path, monkeypatch, _request())

    assert attempts == []
    assert config.data_vault is None


def test_the_scratchpad_gets_the_connections_by_id_and_version_only(
    tmp_path, monkeypatch
):
    config = _captured_config(tmp_path, monkeypatch, _request())

    assert config.workspace_env_overlay == {"ANTON_CLOUD_TURN": "1", **_CUSTOM_ENV}


def test_a_turn_with_both_blocks_carries_both_sets(tmp_path, monkeypatch):
    datasource = {
        "protocol_version": 1,
        "connections": [{"connection_id": 7, "credential_version": 3}],
    }

    config = _captured_config(tmp_path, monkeypatch, _request(datasource=datasource))

    overlay = config.workspace_env_overlay
    assert {k: overlay[k] for k in _CUSTOM_ENV} == _CUSTOM_ENV
    assert (
        overlay["ANTON_CLOUD_DATASOURCE_CONNECTIONS"]
        == '[{"connection_id":7,"credential_version":3}]'
    )
    assert overlay["ANTON_CLOUD_DATASOURCE_CORRELATION_ID"] == _CORRELATION_ID


def test_a_turn_without_custom_connections_carries_none(tmp_path, monkeypatch):
    config = _captured_config(tmp_path, monkeypatch, _request(custom_api=None))

    assert config.workspace_env_overlay == {"ANTON_CLOUD_TURN": "1"}


@pytest.mark.parametrize(
    "pod_env",
    [
        {"ANTON_OPENAI_API_KEY": "shared-key"},
        {"ANTON_OPENAI_BASE_URL": "https://shared.invalid/v1"},
    ],
    ids=["shared-key", "shared-base"],
)
def test_a_custom_api_turn_fails_when_a_pod_setting_replaces_the_turn_connection(
    tmp_path, monkeypatch, pod_env
):
    """The helper would send the child's key to the child's base, so both must be this turn's."""
    monkeypatch.setenv(_WORKSPACE_PATH_ENV, str(tmp_path))
    for name, value in pod_env.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match="cannot carry custom API calls"):
        build_cloud_chat_session(_request())


@pytest.mark.parametrize(
    "llm",
    [
        {
            "provider": "minds-cloud",
            "api_key": "mdb_other.secret",
            "base_url": _BASE_URL,
        },
        {
            "provider": "minds-cloud",
            "api_key": _PRODUCED["turn_key"],
            "base_url": "http://minds.internal/v1",
        },
        {"provider": "openai", "api_key": _PRODUCED["turn_key"], "base_url": _BASE_URL},
    ],
    ids=["block-key-is-not-the-turn-key", "plaintext-base", "not-minds-cloud"],
)
def test_a_custom_api_turn_fails_when_its_key_or_base_is_not_this_turns(
    tmp_path, monkeypatch, llm
):
    with pytest.raises(ValueError, match="cannot carry custom API calls"):
        _captured_config(tmp_path, monkeypatch, _request(llm=llm))


@pytest.mark.parametrize(
    "correlation_id",
    [None, "", "corr_1", _CORRELATION_ID.upper(), _CORRELATION_ID.replace("-", "")],
    ids=["absent", "empty", "not-a-uuid", "uppercase", "no-hyphens"],
)
def test_a_custom_api_turn_needs_the_correlation_id_the_producer_sends(
    tmp_path, monkeypatch, correlation_id
):
    """The gateway refuses a call whose correlation id is not a UUID, with no part the model could fix."""
    with pytest.raises(ValueError, match="custom API turn credentials are unavailable"):
        _captured_config(tmp_path, monkeypatch, _request(correlation_id=correlation_id))


def test_a_second_turns_overlay_replaces_stale_custom_api_state():
    env = {
        "PATH": "/usr/bin",
        "ANTON_CLOUD_TURN": "1",
        "ANTON_CLOUD_CUSTOM_API_CONNECTIONS": '[{"connection_id":1,"credential_version":1}]',
        "ANTON_CLOUD_CUSTOM_API_CORRELATION_ID": "old",
    }

    local._apply_workspace_overlay(env, {"ANTON_CLOUD_TURN": "1"})

    assert env == {"PATH": "/usr/bin", "ANTON_CLOUD_TURN": "1"}


def test_under_a_sandbox_policy_both_variables_still_reach_the_child(tmp_path):
    policy = CellSandbox(
        project_dir=tmp_path / "project",
        home_dir=tmp_path / "home",
        tmp_dir=tmp_path / "tmp",
        user_name="bob",
    )

    assert all(overlay_key_allowed(key) for key in _CUSTOM_ENV)
    child = policy.child_env(dict(_CUSTOM_ENV), overlay_keys=_CUSTOM_ENV)
    assert {k: child[k] for k in _CUSTOM_ENV} == _CUSTOM_ENV


_PROBE_CELL = """
import json, os
print(json.dumps({"env": dict(os.environ), "names": sorted(globals())}))
"""


def _child_report(tmp_path: Path, overlay: dict[str, str]) -> dict:
    """The environment and namespace names of a real scratchpad child started with ``overlay``."""
    env = local._utf8_env(os.environ)
    local._apply_workspace_overlay(env, overlay)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["ANTON_SCRATCHPAD_HEARTBEAT_INTERVAL"] = "0"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "anton" / "core" / "backends" / "scratchpad_boot.py"),
        ],
        input=_PROBE_CELL + f"\n{CELL_DELIM}\n",
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env=env,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(
        completed.stdout.split(RESULT_START, 1)[1].split(RESULT_END, 1)[0]
    )
    assert result["error"] is None, result["error"]
    return json.loads(result["stdout"])


def test_a_scratchpad_child_holds_nothing_of_a_custom_connection_but_ids_and_versions(
    tmp_path, monkeypatch
):
    """Done-when 1 at the process level: compared with a plain cloud turn's child, a custom API
    turn's child gains only the two id-and-version variables and the helper, and no name, connector
    id or key."""
    overlay = _captured_config(tmp_path, monkeypatch, _request()).workspace_env_overlay
    plain = _child_report(tmp_path, {"ANTON_CLOUD_TURN": "1"})
    custom = _child_report(tmp_path, overlay)

    gained = {k: v for k, v in custom["env"].items() if plain["env"].get(k) != v}
    assert gained == _CUSTOM_ENV
    assert not [k for k in custom["env"] if k.startswith("DS_")]
    request_text = [
        ref[key] for ref in _PRODUCED["connections"] for key in ("connector_id", "name")
    ]
    for value in [*request_text, _PRODUCED["turn_key"]]:
        assert not [k for k, v in gained.items() if value in v], value
    assert set(custom["names"]) - set(plain["names"]) == {"call_minds_api"}
    assert set(plain["names"]) <= set(custom["names"])
