"""The artifact backend generator sees the same usage notes as chat."""
from types import SimpleNamespace

from anton.core.datasources.data_vault import LocalDataVault
from anton.core.tools.generate_artifact.orchestrator import _datasource_context


def test_session_usage_notes_reach_the_artifact_datasource_context(tmp_path):
    vault = LocalDataVault(tmp_path)
    vault.save("langfuse", "prod", {"public_key": "pk", "secret_key": "sk"})
    session = SimpleNamespace(
        _data_vault=vault, _connector_usage_notes={"langfuse": "LF-NOTE"}
    )

    assert "LF-NOTE" in _datasource_context(session)


def test_a_non_mapping_attribute_is_treated_as_no_host_notes(tmp_path):
    vault = LocalDataVault(tmp_path)
    vault.save("langfuse", "prod", {"public_key": "pk", "secret_key": "sk"})
    session = SimpleNamespace(_data_vault=vault, _connector_usage_notes=object())

    ctx = _datasource_context(session)

    assert "### Slug:" in ctx
