"""Unit coverage for TurnKeyDataVault — the cloud-turn DataVault backed by
auth's turn-key token endpoint (see anton/cloud_turn/session.py).
"""
from __future__ import annotations

import urllib.error

import pytest

from anton.core.datasources.data_vault import (
    ANTON_CLOUD_AUTH_BASE_URL_ENV,
    DataVault,
    TurnKeyDataVault,
)


def _vault(**oauth_overrides) -> TurnKeyDataVault:
    oauth = {"turn_key": "tk_abc", "connections": [{"engine": "google_drive", "name": "primary"}]}
    oauth.update(oauth_overrides)
    return TurnKeyDataVault(oauth)


def test_satisfies_the_data_vault_protocol():
    assert isinstance(_vault(), DataVault)


def test_list_connections_reflects_the_oauth_block_without_a_network_call(monkeypatch):
    def fail(*a, **kw):
        raise AssertionError("list_connections must not make a network call")

    monkeypatch.setattr("anton.minds_client.minds_request", fail)
    vault = _vault()
    assert vault.list_connections() == [
        {"engine": "google_drive", "name": "primary", "created_at": ""}
    ]


def test_malformed_connection_entries_are_dropped():
    vault = _vault(connections=[{"engine": "google_drive"}, {"name": "primary"}, "not-a-dict", {}])
    assert vault.list_connections() == []


def test_fetch_refuses_a_connection_not_in_this_turns_list(monkeypatch):
    """Defense in depth: this vault must never fetch a connection
    cowork-server didn't list for this turn (e.g. one the caller's own
    disabled_connections filter deliberately excluded), even though auth's
    own org+user-scoped query would already stop it from crossing an org.

    Tracks whether minds_request was ever called, rather than raising from
    the fake — _fetch_uncached's own `except Exception` would otherwise
    swallow a raised assertion and return None for the wrong reason, letting
    this test pass even without the allowlist check.
    """
    calls = []

    def track(*a, **kw):
        calls.append((a, kw))
        return b'{"access_token": "tok"}'

    monkeypatch.setattr("anton.minds_client.minds_request", track)
    vault = _vault(connections=[{"engine": "google_drive", "name": "primary"}])
    assert vault.load("google_drive", "other-connection") is None
    assert vault.load("gmail", "primary") is None
    assert calls == []


def test_load_fetches_and_shapes_the_token_response(monkeypatch):
    def fake(url, api_key, *, method="GET", payload=None, verify=True, timeout=30):
        assert url.endswith("/v1/oauth/google_drive/token")
        assert method == "POST"
        assert api_key == "tk_abc"
        return b'{"access_token": "tok", "account_email": "a@b.com", "refresh_token": "must-not-leak"}'

    monkeypatch.setattr("anton.minds_client.minds_request", fake)
    fields = _vault().load("google_drive", "primary")
    assert fields == {"access_token": "tok", "account_email": "a@b.com", "auth_type": "oauth"}
    assert "refresh_token" not in fields


def test_load_sends_the_connection_name_so_auth_can_disambiguate(monkeypatch):
    """A name-less request auto-resolves only when an org has exactly one
    connection for the engine — auth 400s on more than one. Sending name
    always avoids that instead of depending on org-specific connection counts."""
    import json as _json

    seen = {}

    def fake(url, api_key, *, method="GET", payload=None, verify=True, timeout=30):
        seen["payload"] = _json.loads(payload.decode()) if payload else None
        return b'{"access_token": "tok"}'

    monkeypatch.setattr("anton.minds_client.minds_request", fake)
    _vault().load("google_drive", "primary")
    assert seen["payload"] == {"name": "primary"}


def test_second_fetch_for_the_same_connection_is_cached(monkeypatch):
    calls = []

    def fake(url, api_key, *, method="GET", payload=None, verify=True, timeout=30):
        calls.append(url)
        return b'{"access_token": "tok"}'

    monkeypatch.setattr("anton.minds_client.minds_request", fake)
    vault = _vault()
    assert vault.load("google_drive", "primary") is not None
    assert vault.env_for("google_drive", "primary") is not None
    assert len(calls) == 1


def test_403_reads_as_needs_reconnect_not_a_crash(monkeypatch):
    def fake(url, api_key, *, method="GET", payload=None, verify=True, timeout=30):
        raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)

    monkeypatch.setattr("anton.minds_client.minds_request", fake)
    assert _vault().load("google_drive", "primary") is None


def test_missing_access_token_in_response_is_treated_as_a_miss(monkeypatch):
    def fake(url, api_key, *, method="GET", payload=None, verify=True, timeout=30):
        return b'{"account_email": "a@b.com"}'

    monkeypatch.setattr("anton.minds_client.minds_request", fake)
    assert _vault().load("google_drive", "primary") is None


def test_no_turn_key_short_circuits_without_a_network_call(monkeypatch):
    def fail(*a, **kw):
        raise AssertionError("must not call out with no turn key")

    monkeypatch.setattr("anton.minds_client.minds_request", fail)
    vault = TurnKeyDataVault({"connections": [{"engine": "gmail", "name": "primary"}]})
    assert vault.load("gmail", "primary") is None


def test_env_for_namespaces_like_local_data_vault(monkeypatch):
    def fake(url, api_key, *, method="GET", payload=None, verify=True, timeout=30):
        return b'{"access_token": "tok", "account_email": "a@b.com"}'

    monkeypatch.setattr("anton.minds_client.minds_request", fake)
    env = _vault().env_for("google_drive", "primary")
    assert env == {
        "DS_GOOGLE_DRIVE_PRIMARY__ACCESS_TOKEN": "tok",
        "DS_GOOGLE_DRIVE_PRIMARY__ACCOUNT_EMAIL": "a@b.com",
        "DS_GOOGLE_DRIVE_PRIMARY__AUTH_TYPE": "oauth",
    }


def test_env_for_flat_mode(monkeypatch):
    def fake(url, api_key, *, method="GET", payload=None, verify=True, timeout=30):
        return b'{"access_token": "tok"}'

    monkeypatch.setattr("anton.minds_client.minds_request", fake)
    env = _vault().env_for("google_drive", "primary", flat=True)
    assert env == {"DS_ACCESS_TOKEN": "tok", "DS_AUTH_TYPE": "oauth"}


def test_read_record_marks_only_access_token_as_secret(monkeypatch):
    def fake(url, api_key, *, method="GET", payload=None, verify=True, timeout=30):
        return b'{"access_token": "tok", "token_type": "Bearer"}'

    monkeypatch.setattr("anton.minds_client.minds_request", fake)
    record = _vault().read_record("google_drive", "primary")
    assert record["secure_keys"] == ["access_token"]
    assert record["fields"]["token_type"] == "Bearer"


def test_auth_type_is_always_synthesized_as_oauth(monkeypatch):
    """Every credential this vault serves is OAuth-backed by construction —
    auth's response doesn't need to say so for build_datasource_context()
    (anton/utils/datasources.py) to recognize a connected account."""
    def fake(url, api_key, *, method="GET", payload=None, verify=True, timeout=30):
        return b'{"access_token": "tok"}'

    monkeypatch.setattr("anton.minds_client.minds_request", fake)
    fields = _vault().load("google_drive", "primary")
    assert fields["auth_type"] == "oauth"


def test_picked_files_passes_through_as_a_json_string(monkeypatch):
    """Forward-compat: auth doesn't send `_picked_files` yet, but once it
    does, this must land in the same JSON-string-in-a-field shape
    _parse_picked_files() (anton/utils/datasources.py) reads from
    LocalDataVault, not as a native list."""
    def fake(url, api_key, *, method="GET", payload=None, verify=True, timeout=30):
        return b'{"access_token": "tok", "_picked_files": [{"id": "f1", "name": "doc.pdf"}]}'

    monkeypatch.setattr("anton.minds_client.minds_request", fake)
    fields = _vault().load("google_drive", "primary")
    assert fields["_picked_files"] == '[{"id": "f1", "name": "doc.pdf"}]'


def test_save_and_delete_are_never_valid_mid_turn():
    vault = _vault()
    with pytest.raises(NotImplementedError):
        vault.save("google_drive", "primary", {"access_token": "x"})
    assert vault.delete("google_drive", "primary") is False


def test_base_url_env_override(monkeypatch):
    monkeypatch.setenv(ANTON_CLOUD_AUTH_BASE_URL_ENV, "https://auth.pr-123.dev.mindshub.ai")
    seen = {}

    def fake(url, api_key, *, method="GET", payload=None, verify=True, timeout=30):
        seen["url"] = url
        return b'{"access_token": "tok"}'

    monkeypatch.setattr("anton.minds_client.minds_request", fake)
    _vault().load("google_drive", "primary")
    assert seen["url"] == "https://auth.pr-123.dev.mindshub.ai/v1/oauth/google_drive/token"


def test_no_turn_key_diagnostic_wins_over_the_allowlist_one(monkeypatch, caplog):
    """A turn with neither a turn key nor a matching connection should log
    the more actionable "no turn key" message, not the allowlist refusal —
    the turn-key check must run first."""
    def fail(*a, **kw):
        raise AssertionError("must not call out with no turn key")

    monkeypatch.setattr("anton.minds_client.minds_request", fail)
    vault = TurnKeyDataVault({"turn_key": "", "connections": []})
    with caplog.at_level("WARNING"):
        assert vault.load("google_drive", "primary") is None
    assert any("no turn key available" in r.message for r in caplog.records)
    assert not any("not in this turn's connection list" in r.message for r in caplog.records)


def test_clear_ds_env_drops_the_per_connection_cache(monkeypatch):
    """A later fetch after clear_ds_env() must not return a cached
    pre-clear value."""
    responses = iter([b'{"access_token": "first"}', b'{"access_token": "second"}'])

    def fake(url, api_key, *, method="GET", payload=None, verify=True, timeout=30):
        return next(responses)

    monkeypatch.setattr("anton.minds_client.minds_request", fake)
    vault = _vault()
    assert vault.load("google_drive", "primary")["access_token"] == "first"
    vault.clear_ds_env()
    assert vault.load("google_drive", "primary")["access_token"] == "second"
