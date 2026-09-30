"""build_datasource_context renders each connected engine's usage notes once,
from the host's map or — only when the host sent none — from anton's own
registry."""
from types import SimpleNamespace

import pytest

import anton.utils.datasources as ds
from anton.core.datasources.data_vault import LocalDataVault
from anton.utils.datasources import build_datasource_context


@pytest.fixture
def vault(tmp_path):
    v = LocalDataVault(tmp_path)
    v.save("langfuse", "prod", {"public_key": "pk", "secret_key": "sk"})
    return v


class _FakeRegistry:
    """Stands in for DatasourceRegistry: postgres has registry notes."""

    def get(self, engine):
        if engine == "postgres":
            return SimpleNamespace(display_name="PostgreSQL", usage_notes="REGISTRY-NOTE")
        return None


def test_connected_engine_note_is_rendered(vault):
    ctx = build_datasource_context(vault, usage_notes={"langfuse": "Use /metrics/daily."})

    assert "### Usage notes: langfuse (engine `langfuse`)" in ctx
    assert "Use /metrics/daily." in ctx


def test_unconnected_engine_note_is_not_rendered(vault):
    ctx = build_datasource_context(
        vault, usage_notes={"langfuse": "LF", "roam_research": "ROAM-NOTE"}
    )

    assert "ROAM-NOTE" not in ctx


def test_two_connections_of_one_engine_render_the_note_once(vault):
    vault.save("langfuse", "staging", {"public_key": "pk2", "secret_key": "sk2"})

    ctx = build_datasource_context(vault, usage_notes={"langfuse": "ONCE-NOTE"})

    assert ctx.count("ONCE-NOTE") == 1


def test_notes_follow_every_connection_block(vault):
    vault.save("postgres", "db", {"host": "h", "password": "p"})

    ctx = build_datasource_context(vault, usage_notes={"langfuse": "LF-NOTE"})

    assert ctx.index("### Usage notes:") > ctx.rindex("### Slug:")


def test_active_only_limits_notes_to_the_active_engine(vault):
    vault.save("postgres", "db", {"host": "h", "password": "p"})

    ctx = build_datasource_context(
        vault,
        active_only="langfuse-prod",
        usage_notes={"langfuse": "LF-NOTE", "postgres": "PG-NOTE"},
    )

    assert "LF-NOTE" in ctx
    assert "PG-NOTE" not in ctx


def test_none_falls_back_to_the_anton_registry(vault, monkeypatch):
    monkeypatch.setattr(ds, "DatasourceRegistry", _FakeRegistry)
    vault.save("postgres", "db", {"host": "h", "password": "p"})

    ctx = build_datasource_context(vault, usage_notes=None)

    assert "### Usage notes: PostgreSQL (engine `postgres`)" in ctx
    assert "REGISTRY-NOTE" in ctx


def test_a_host_map_even_empty_disables_the_registry(vault, monkeypatch):
    monkeypatch.setattr(ds, "DatasourceRegistry", _FakeRegistry)
    vault.save("postgres", "db", {"host": "h", "password": "p"})

    ctx = build_datasource_context(vault, usage_notes={})

    assert "REGISTRY-NOTE" not in ctx


@pytest.mark.parametrize("value", ["", "   \n", 123, None])
def test_blank_or_non_string_notes_are_skipped(vault, value):
    ctx = build_datasource_context(vault, usage_notes={"langfuse": value})

    assert "Usage notes" not in ctx
