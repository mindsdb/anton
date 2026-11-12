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


def test_bookkeeping_fields_are_not_published(tmp_path):
    """Publish resolves through env_for, the same rule a local launch uses."""
    vault = LocalDataVault(vault_dir=tmp_path)
    vault.save(
        "postgres", "prod",
        {"host": "db", "_method": "form", "_access_mode": "read", "_created_by": "u1"},
        secure_keys=[],
    )
    secrets, _ = _collect_datasource_secrets(_artifact(("postgres", "prod")), vault)
    assert secrets == {"DS_POSTGRES_PROD__HOST": "db"}


def test_publish_names_what_the_app_does_not_get(tmp_path):
    from anton.publisher import _withheld_datasource_secrets

    vault = LocalDataVault(vault_dir=tmp_path)
    vault.save("notion", "acme", {"access_token": "tok", "_method": "mcp"}, secure_keys=["access_token"])
    vault.save("postgres", "prod", {"password": "pw"}, secure_keys=["password"])
    artifact = _artifact(("notion", "acme"), ("postgres", "prod"))
    assert _withheld_datasource_secrets(artifact, vault) == ["DS_NOTION_ACME__ACCESS_TOKEN"]


def test_a_withheld_tokens_marker_does_not_unscrub(tmp_path):
    """unscrub_marker must not turn a withheld MCP token's marker back into the
    token, while a normal field from the same vault still resolves."""
    from anton.utils.datasources import unscrub_marker

    vault = LocalDataVault(vault_dir=tmp_path)
    vault.save("notion", "acme", {"access_token": "tok", "_method": "mcp"}, secure_keys=["access_token"])
    vault.save("postgres", "prod", {"password": "pw"}, secure_keys=["password"])
    assert unscrub_marker(vault, "[DS_NOTION_ACME__ACCESS_TOKEN]") is None
    assert unscrub_marker(vault, "[DS_POSTGRES_PROD__PASSWORD]") == "pw"


def test_the_launcher_names_a_withheld_token(tmp_path, caplog):
    from anton.core.artifacts.backend_launcher import build_datasource_env
    from anton.core.artifacts.models import DatasourceRef

    vault = LocalDataVault(vault_dir=tmp_path)
    vault.save("hubspot", "crm", {"access_token": "tok", "_method": "mcp"}, secure_keys=["access_token"])
    with caplog.at_level("WARNING"):
        env = build_datasource_env(vault, [DatasourceRef(engine="hubspot", name="crm")], slug="app")
    assert "DS_HUBSPOT_CRM__ACCESS_TOKEN" not in env
    assert "access_token" in caplog.text and "hubspot/crm" in caplog.text
