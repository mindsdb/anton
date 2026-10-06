"""The artifact backend generator sees the same usage notes as chat."""
from types import SimpleNamespace

import pytest

import anton.utils.datasources as ds
from anton.core.datasources.data_vault import LocalDataVault
from anton.core.tools.generate_artifact.orchestrator import _datasource_catalog


@pytest.fixture
def vault(tmp_path):
    v = LocalDataVault(tmp_path / "vault")
    v.save("langfuse", "prod", {"public_key": "pk", "secret_key": "sk"})
    return v


def test_session_usage_notes_reach_the_artifact_datasource_catalog(vault):
    session = SimpleNamespace(
        _data_vault=vault, _connector_usage_notes={"langfuse": "LF-NOTE"}
    )

    assert "LF-NOTE" in _datasource_catalog(session).render()


def test_a_non_mapping_attribute_is_treated_as_no_host_notes(vault, monkeypatch):
    class _RegistryWithNotes:
        def get(self, engine):
            return SimpleNamespace(display_name="Langfuse", usage_notes="REGISTRY-NOTE")

    monkeypatch.setattr(ds, "DatasourceRegistry", _RegistryWithNotes)
    session = SimpleNamespace(_data_vault=vault, _connector_usage_notes=object())

    ctx = _datasource_catalog(session).render()

    assert "### Slug: `langfuse-prod`" in ctx
    assert "REGISTRY-NOTE" not in ctx


async def test_generate_gives_the_run_the_session_catalog_and_notes(vault, tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from anton.core.tools.generate_artifact import engine, orchestrator

    seen = {}

    async def fake_run(state, *, entry):
        seen["state"] = state
        return {"status": "generated", "files_written": [], "internal_files": [], "trace": []}

    monkeypatch.setattr(orchestrator, "run", fake_run)
    monkeypatch.setattr(engine, "_scratchpads_context", lambda session: "")
    session = AsyncMock()
    session._data_vault = vault
    session._connector_usage_notes = {"langfuse": "LF-NOTE"}
    art = tmp_path / "art"
    art.mkdir()

    await engine.generate(
        session=session, slug="a", artifact_path=art, artifact_type="html-app",
        user_request="r", agent_understanding="u",
    )

    assert "LF-NOTE" in seen["state"].datasource_context
