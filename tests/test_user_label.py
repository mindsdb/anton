from __future__ import annotations

from anton.core.datasources.data_vault import LocalDataVault
from anton.utils.datasources import default_user_label, ensure_unique_user_label


def _save(vault, engine, name, user_label=None, label=None):
    fields = {"host": "x"}
    if user_label is not None:
        fields["_user_label"] = user_label
    if label is not None:
        fields["_label"] = label
    vault.save(engine, name, fields)


class TestEnsureUniqueUserLabel:
    def test_returns_candidate_when_no_collision(self, tmp_path):
        vault = LocalDataVault(vault_dir=tmp_path / "vault")
        assert ensure_unique_user_label(vault, "postgres") == "postgres"

    def test_suffixes_on_collision_with_user_label(self, tmp_path):
        vault = LocalDataVault(vault_dir=tmp_path / "vault")
        _save(vault, "postgres", "a1b2c3", user_label="postgres")
        assert ensure_unique_user_label(vault, "postgres") == "postgres 2"

    def test_suffixes_on_collision_with_legacy_label(self, tmp_path):
        vault = LocalDataVault(vault_dir=tmp_path / "vault")
        _save(vault, "gmail", "abc123", label="Support")
        assert ensure_unique_user_label(vault, "Support") == "Support 2"

    def test_skips_taken_suffix_too(self, tmp_path):
        vault = LocalDataVault(vault_dir=tmp_path / "vault")
        _save(vault, "postgres", "a1", user_label="postgres")
        _save(vault, "postgres", "a2", user_label="postgres 2")
        assert ensure_unique_user_label(vault, "postgres") == "postgres 3"

    def test_uniqueness_is_global_not_per_engine(self, tmp_path):
        vault = LocalDataVault(vault_dir=tmp_path / "vault")
        _save(vault, "postgres", "a1", user_label="prod-db")
        assert ensure_unique_user_label(vault, "prod-db") == "prod-db 2"

    def test_empty_labels_excluded_from_collision_set(self, tmp_path):
        vault = LocalDataVault(vault_dir=tmp_path / "vault")
        vault.save("postgres", "a1", {"host": "x"})  # no _user_label, no _label
        assert ensure_unique_user_label(vault, "postgres") == "postgres"

    def test_exclude_lets_a_connection_keep_its_own_label(self, tmp_path):
        vault = LocalDataVault(vault_dir=tmp_path / "vault")
        _save(vault, "postgres", "a1", user_label="prod-db")
        result = ensure_unique_user_label(
            vault, "prod-db", exclude=("postgres", "a1")
        )
        assert result == "prod-db"

    def test_exclude_only_removes_that_connection(self, tmp_path):
        vault = LocalDataVault(vault_dir=tmp_path / "vault")
        _save(vault, "postgres", "a1", user_label="prod-db")
        _save(vault, "postgres", "a2", user_label="prod-db")  # pre-existing dup
        result = ensure_unique_user_label(
            vault, "prod-db", exclude=("postgres", "a1")
        )
        # a2 still holds "prod-db", so the excluded a1's rename still collides with it
        assert result == "prod-db 2"


class TestDefaultUserLabel:
    def test_returns_engine_id_when_unused(self, tmp_path):
        vault = LocalDataVault(vault_dir=tmp_path / "vault")
        assert default_user_label(vault, "postgres") == "postgres"

    def test_returns_deduplicated_suggestion(self, tmp_path):
        vault = LocalDataVault(vault_dir=tmp_path / "vault")
        _save(vault, "postgres", "a1", user_label="postgres")
        assert default_user_label(vault, "postgres") == "postgres 2"
