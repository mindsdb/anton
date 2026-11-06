"""The credentials a published fullstack artifact's backend receives."""
from __future__ import annotations

from types import SimpleNamespace

from anton.core.artifacts.models import DatasourceRef
from anton.core.datasources.data_vault import LocalDataVault
from anton.publisher import _collect_datasource_secrets


def _artifact(*refs: tuple[str, str]) -> SimpleNamespace:
    return SimpleNamespace(datasources=[DatasourceRef(engine=e, name=n) for e, n in refs])


def test_an_mcp_connections_token_is_not_published(tmp_path):
    """Its access mode only limits the MCP tools; a backend holding the token
    could call the provider's API directly and get past it."""
    vault = LocalDataVault(vault_dir=tmp_path)
    vault.save(
        "notion", "acme",
        {"access_token": "tok", "account_email": "a@b.com", "_method": "mcp"},
        secure_keys=["access_token"],
    )
    secrets, missing = _collect_datasource_secrets(_artifact(("notion", "acme")), vault)
    assert missing == []
    assert "DS_NOTION_ACME__ACCESS_TOKEN" not in secrets
    assert secrets["DS_NOTION_ACME__ACCOUNT_EMAIL"] == "a@b.com"


def test_a_non_mcp_connections_token_is_published(tmp_path):
    vault = LocalDataVault(vault_dir=tmp_path)
    vault.save("postgres", "prod", {"host": "db", "password": "pw"}, secure_keys=["password"])
    secrets, missing = _collect_datasource_secrets(_artifact(("postgres", "prod")), vault)
    assert missing == []
    assert secrets == {"DS_POSTGRES_PROD__HOST": "db", "DS_POSTGRES_PROD__PASSWORD": "pw"}


def test_a_missing_connection_is_reported(tmp_path):
    secrets, missing = _collect_datasource_secrets(_artifact(("postgres", "gone")), LocalDataVault(vault_dir=tmp_path))
    assert (secrets, missing) == ({}, ["postgres-gone"])
