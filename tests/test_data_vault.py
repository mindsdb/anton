"""Unit coverage for the data-vault schema extensions added for the
modify-connection flow: `secure_keys` persistence, `created_at`
preservation across updates, the `is_secret_key` heuristic, and
`read_record` returning the raw record.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from anton.core.datasources.data_vault import (
    ANTON_VAULT_KEEP,
    LocalDataVault,
    is_secret_key,
    resolve_modify_merge,
)


@pytest.fixture
def vault(tmp_path: Path) -> LocalDataVault:
    return LocalDataVault(vault_dir=tmp_path / "vault")


# ─── Sentinel constant ───────────────────────────────────────────────────────


def test_sentinel_constant_is_stable() -> None:
    """The constant is part of the modify-flow contract — server +
    renderer reference it. A drift here is a wire-protocol break,
    so pin the value as a regression check."""
    assert ANTON_VAULT_KEEP == "__anton_vault_keep__"


# ─── Heuristic ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", [
    "password", "PASSWORD", "user_password", "password_hash",
    "secret", "client_secret", "api_secret",
    "api_key", "private_key", "ssh_key", "access_key",
    "token", "access_token", "refresh_token",
    "credentials", "auth_token",
    "passphrase",
])
def test_is_secret_key_heuristic_matches(name: str) -> None:
    """Names containing any of the secret tokens should classify
    as secret when no explicit `secure_keys` list is provided."""
    assert is_secret_key(name) is True


@pytest.mark.parametrize("name", [
    "host", "port", "database", "user", "username", "schema",
    "region", "account", "warehouse", "ssl_mode", "role",
])
def test_is_secret_key_heuristic_misses_config(name: str) -> None:
    """Plain config fields shouldn't be classified as secret by the heuristic."""
    assert is_secret_key(name) is False


def test_is_secret_key_explicit_list_is_authoritative() -> None:
    """When `secure_keys` is supplied the heuristic is skipped — the
    list is the source of truth, even when it disagrees with the
    name shape."""
    # 'host' wouldn't match the heuristic, but the explicit list
    # marks it secret → respected.
    assert is_secret_key("host", secure_keys=["host"]) is True
    # 'password' would match the heuristic, but it's not on the
    # explicit list → respected.
    assert is_secret_key("password", secure_keys=["host"]) is False
    # Empty explicit list = "no secrets". Heuristic must NOT fire.
    assert is_secret_key("password", secure_keys=[]) is False


# ─── Save with secure_keys + read_record ─────────────────────────────────────


def test_save_persists_secure_keys(vault: LocalDataVault) -> None:
    vault.save(
        "postgres", "prod",
        {"host": "db.x", "password": "p"},
        secure_keys=["password"],
    )
    record = vault.read_record("postgres", "prod")
    assert record is not None
    assert record["secure_keys"] == ["password"]
    # `fields` is the credential blob (secrets included). Sentinel
    # substitution is the responsibility of the modify endpoint, not
    # of the vault — the vault returns truth.
    assert record["fields"] == {"host": "db.x", "password": "p"}


def test_save_dedupes_and_sorts_secure_keys(vault: LocalDataVault) -> None:
    """Secure-key persistence is canonicalized so on-disk JSON diffs
    cleanly across updates that don't change set membership."""
    vault.save(
        "postgres", "prod",
        {"a": "1", "b": "2"},
        secure_keys=["b", "a", "b", "a"],
    )
    record = vault.read_record("postgres", "prod")
    assert record["secure_keys"] == ["a", "b"]


def test_save_without_secure_keys_omits_field(vault: LocalDataVault) -> None:
    """Backward compatibility: callers that don't pass `secure_keys`
    don't get the field added — preserves the legacy on-disk shape."""
    vault.save("postgres", "prod", {"host": "db.x", "password": "p"})
    record = vault.read_record("postgres", "prod")
    assert "secure_keys" not in record


def test_read_record_returns_none_for_missing(vault: LocalDataVault) -> None:
    assert vault.read_record("postgres", "missing") is None


def test_read_record_includes_timestamps(vault: LocalDataVault) -> None:
    vault.save("postgres", "prod", {"host": "db.x"})
    record = vault.read_record("postgres", "prod")
    assert record is not None
    # New records get both timestamps stamped at save-time.
    assert "created_at" in record
    assert "updated_at" in record


# ─── created_at preservation across updates ─────────────────────────────────


def test_save_preserves_created_at_on_update(vault: LocalDataVault) -> None:
    """Modify flow saves a second time — `created_at` must keep its
    original value so the timestamp keeps its semantic meaning.
    Only `updated_at` advances."""
    vault.save("postgres", "prod", {"host": "db.x"})
    first = vault.read_record("postgres", "prod")
    original_created = first["created_at"]
    original_updated = first["updated_at"]

    # Re-save (simulating a modify). created_at should be preserved.
    vault.save("postgres", "prod", {"host": "db.y"}, secure_keys=["password"])
    second = vault.read_record("postgres", "prod")
    assert second["created_at"] == original_created
    # updated_at MAY equal original if the same instant — assert at
    # least monotonic non-decrease.
    assert second["updated_at"] >= original_updated


# ─── load() still works on records with secure_keys ─────────────────────────


def test_load_unchanged_when_secure_keys_present(vault: LocalDataVault) -> None:
    """`load` returns the credential dict regardless of the new
    metadata — every existing caller (env injection, agent-side
    reads) must keep working."""
    vault.save(
        "postgres", "prod",
        {"host": "db.x", "password": "p"},
        secure_keys=["password"],
    )
    fields = vault.load("postgres", "prod")
    assert fields == {"host": "db.x", "password": "p"}


# ─── resolve_modify_merge ───────────────────────────────────────────────────


def test_resolve_modify_merge_creates_when_no_prior(vault: LocalDataVault) -> None:
    """No prior record → no sentinel resolution required, but spec +
    heuristic still classify the secret-key set."""
    merged, secure_keys = resolve_modify_merge(
        vault, "postgres", "prod",
        {"host": "db.x", "password": "p"},
        spec_secret_keys=["password"],
    )
    assert merged == {"host": "db.x", "password": "p"}
    assert secure_keys == ["password"]


def test_resolve_modify_merge_replaces_sentinel_with_existing(vault: LocalDataVault) -> None:
    """The sentinel means 'keep existing'. Other fields update normally."""
    vault.save(
        "postgres", "prod",
        {"host": "db.x", "password": "saved"},
        secure_keys=["password"],
    )
    merged, secure_keys = resolve_modify_merge(
        vault, "postgres", "prod",
        {"host": "db.y", "password": ANTON_VAULT_KEEP},
        spec_secret_keys=["password"],
    )
    # host changed; password kept from prior record.
    assert merged == {"host": "db.y", "password": "saved"}
    assert "password" in secure_keys


def test_resolve_modify_merge_drops_orphan_sentinels(vault: LocalDataVault) -> None:
    """Sentinel for a field with no prior value → field is dropped.
    Belt-and-suspenders: prevents the literal sentinel string from
    ever landing on disk if a renderer regression sends one through."""
    merged, _ = resolve_modify_merge(
        vault, "postgres", "fresh",
        {"host": "db.x", "ghost_field": ANTON_VAULT_KEEP},
        spec_secret_keys=[],
    )
    assert merged == {"host": "db.x"}
    assert "ghost_field" not in merged


def test_resolve_modify_merge_distinguishes_clear_from_keep(vault: LocalDataVault) -> None:
    """Empty string ≠ sentinel. Empty = explicit clear."""
    vault.save(
        "postgres", "prod",
        {"host": "db.x", "password": "saved"},
        secure_keys=["password"],
    )
    merged, _ = resolve_modify_merge(
        vault, "postgres", "prod",
        {"host": "db.x", "password": ""},  # explicit clear
        spec_secret_keys=["password"],
    )
    assert merged == {"host": "db.x", "password": ""}


def test_resolve_modify_merge_unions_secure_keys(vault: LocalDataVault) -> None:
    """secure_keys = prior ∪ spec ∪ heuristic. Once secret, always secret."""
    vault.save(
        "postgres", "prod",
        {"host": "db.x", "old_secret": "s"},
        secure_keys=["old_secret"],
    )
    merged, secure_keys = resolve_modify_merge(
        vault, "postgres", "prod",
        {"host": "db.x", "old_secret": ANTON_VAULT_KEEP, "new_token": "t"},
        spec_secret_keys=["new_token"],
    )
    # Prior record marked old_secret → keep it. Spec marks new_token →
    # add it. Heuristic also fires on `new_token` (token substring) and
    # would have caught it anyway. Net union: both names present.
    assert "old_secret" in secure_keys
    assert "new_token" in secure_keys
    assert merged["old_secret"] == "s"
    assert merged["new_token"] == "t"


def test_resolve_modify_merge_heuristic_picks_up_unknown_secrets(vault: LocalDataVault) -> None:
    """Custom engines / agent-driven flows can invent fields not in
    any spec. The heuristic must classify those without help."""
    merged, secure_keys = resolve_modify_merge(
        vault, "custom", "x",
        {"host": "db.x", "weird_password": "p"},
        spec_secret_keys=None,
    )
    assert merged == {"host": "db.x", "weird_password": "p"}
    assert "weird_password" in secure_keys
    assert "host" not in secure_keys
