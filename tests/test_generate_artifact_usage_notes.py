"""The artifact backend generator sees the same usage notes as chat."""
from types import SimpleNamespace

import pytest

import anton.utils.datasources as ds
from anton.core.datasources.data_vault import LocalDataVault
from anton.core.datasources.datasource_registry import DatasourceRegistry
from anton.core.tools.generate_artifact.orchestrator import _datasource_catalog


@pytest.fixture(autouse=True)
def _no_user_registry(tmp_path, monkeypatch):
    """Keep the developer's ~/.anton/datasources.md out of the real registry."""
    monkeypatch.setattr(DatasourceRegistry, "_USER_PATH", tmp_path / "no-user-registry.md")


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


async def test_generate_keeps_the_rendered_section_and_the_catalog_in_step(vault, tmp_path, monkeypatch):
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

    state = seen["state"]
    assert "LF-NOTE" in state.datasource_context
    assert state.datasource_context == state.datasource_catalog.render()


async def test_backend_prompt_gets_only_the_declared_connection_and_its_notes(vault, tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from anton.core.tools.generate_artifact import orchestrator
    from anton.core.tools.generate_artifact.state import GenState, VerifyResult

    vault.save("postgres", "db", {"host": "h", "password": "p"})
    session = SimpleNamespace(
        _data_vault=vault,
        _connector_usage_notes={"langfuse": "LF-NOTE", "postgres": "PG-NOTE"},
    )
    st = GenState(
        session=AsyncMock(), artifact_type="fullstack-stateless-app",
        artifact_path=tmp_path, slug="a", is_fullstack=True,
        datasource_catalog=_datasource_catalog(session),
        declared_sources=["daily metrics from langfuse"],
    )
    st.api_spec = "{}"
    captured = {}

    async def fake_loop(**kw):
        captured["system"] = kw["system"]
        (tmp_path / "backend.py").write_text("x")
        return {"files_written": ["backend.py"], "rounds_used": 1, "summary": "s"}

    async def fake_verify(**kw):
        return VerifyResult(errors=[]), []

    async def fake_declare(state, refs):
        pass

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)
    monkeypatch.setattr(orchestrator.verifiers, "verify_backend", fake_verify)
    monkeypatch.setattr(orchestrator, "_map_datasources", lambda s, k: ([], []))
    monkeypatch.setattr(orchestrator, "_declare_datasources", fake_declare)

    assert await orchestrator._gen_verify_backend(st) is None
    assert "DS_LANGFUSE_PROD__PUBLIC_KEY" in captured["system"]
    assert "LF-NOTE" in captured["system"]
    assert "DS_POSTGRES_DB__HOST" not in captured["system"]
    assert "PG-NOTE" not in captured["system"]
