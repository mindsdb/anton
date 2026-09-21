"""The scratchpad child's environment: what a project overlay may and may not
change, and how per-turn cloud state is kept from crossing runtimes."""

from __future__ import annotations

import anton.core.backends.local as local


def test_inherited_cloud_turn_state_is_dropped_and_the_overlay_wins():
    """A previous runtime's bearer or connection set in the parent environment
    must never reach the next pad; the overlay's own values replace them."""
    env = {
        "PATH": "/usr/bin",
        "ANTON_CLOUD_TURN": "1",
        "ANTON_CLOUD_DATASOURCE_TURN_KEY": "mdb_previous.secret",
        "ANTON_CLOUD_DATASOURCE_CONNECTIONS": "[{\"connection_id\": 1, \"credential_version\": 1}]",
        "ANTON_CLOUD_DATASOURCE_CORRELATION_ID": "old",
    }
    local._apply_workspace_overlay(env, {
        "ANTON_CLOUD_TURN": "1",
        "ANTON_CLOUD_DATASOURCE_TURN_KEY": "mdb_current.secret",
        "ANTON_CLOUD_DATASOURCE_CORRELATION_ID": "new",
    })
    assert env["ANTON_CLOUD_DATASOURCE_TURN_KEY"] == "mdb_current.secret"
    assert env["ANTON_CLOUD_DATASOURCE_CORRELATION_ID"] == "new"
    assert "ANTON_CLOUD_DATASOURCE_CONNECTIONS" not in env
    assert env["PATH"] == "/usr/bin"


def test_a_runtime_without_a_block_inherits_no_cloud_turn_state():
    env = {"ANTON_CLOUD_TURN": "1", "ANTON_CLOUD_DATASOURCE_TURN_KEY": "mdb_previous.secret"}
    local._apply_workspace_overlay(env, None)
    assert env == {}


def test_ordinary_overlay_keys_never_override_the_process_environment():
    env = {"PATH": "/usr/bin", "OPENAI_API_KEY": "process"}
    local._apply_workspace_overlay(env, {"PATH": "/hijacked", "OPENAI_API_KEY": "project", "PROJECT_ONLY": "yes"})
    assert env == {"PATH": "/usr/bin", "OPENAI_API_KEY": "process", "PROJECT_ONLY": "yes"}
