"""build_datasource_context renders each connected engine's usage notes once,
from the host's map or — only when the host sent none — from anton's own
registry."""
from types import SimpleNamespace

import pytest

import anton.utils.datasources as ds
from anton.core.datasources.data_vault import LocalDataVault
from anton.utils.datasources import build_datasource_context, collect_datasource_catalog


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


def test_each_engine_is_looked_up_in_the_registry_once(vault, monkeypatch):
    calls = []

    class _CountingRegistry(_FakeRegistry):
        def get(self, engine):
            calls.append(engine)
            return super().get(engine)

    monkeypatch.setattr(ds, "DatasourceRegistry", _CountingRegistry)
    vault.save("postgres", "db", {"host": "h", "password": "p"})
    vault.save("postgres", "replica", {"host": "h2", "password": "p"})

    ctx = build_datasource_context(vault, usage_notes=None)

    assert "### Usage notes: PostgreSQL (engine `postgres`)" in ctx
    assert sorted(calls) == ["langfuse", "postgres"]


def test_a_non_mapping_host_value_renders_no_notes_and_skips_the_registry(
    vault, monkeypatch, caplog
):
    monkeypatch.setattr(ds, "DatasourceRegistry", _FakeRegistry)
    vault.save("postgres", "db", {"host": "h", "password": "p"})

    with caplog.at_level("WARNING", logger=ds.logger.name):
        ctx = build_datasource_context(vault, usage_notes=["langfuse"])

    assert "### Slug: `langfuse-prod`" in ctx
    assert "### Slug: `postgres-db`" in ctx
    assert "Usage notes" not in ctx
    assert "REGISTRY-NOTE" not in ctx
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


def test_a_host_map_even_empty_disables_the_registry(vault, monkeypatch):
    monkeypatch.setattr(ds, "DatasourceRegistry", _FakeRegistry)
    vault.save("postgres", "db", {"host": "h", "password": "p"})

    ctx = build_datasource_context(vault, usage_notes={})

    assert "REGISTRY-NOTE" not in ctx


@pytest.mark.parametrize("value", ["", "   \n", 123, None])
def test_blank_or_non_string_notes_are_skipped(vault, value):
    ctx = build_datasource_context(vault, usage_notes={"langfuse": value})

    assert "Usage notes" not in ctx


def test_notes_precede_the_google_drive_availability_paragraph(tmp_path):
    v = LocalDataVault(tmp_path)
    v.save("google_drive", "work", {"auth_type": "oauth", "access_token": "t"})

    ctx = build_datasource_context(v, usage_notes={"google_drive": "DRIVE-NOTE"})

    assert ctx.index("DRIVE-NOTE") < ctx.index("Connected Google Drive accounts are available")


def test_two_engines_render_two_headings_in_first_connection_order(tmp_path):
    v = LocalDataVault(tmp_path)
    v.save("postgres", "db", {"host": "h", "password": "p"})
    v.save("langfuse", "prod", {"public_key": "pk", "secret_key": "sk"})
    order = [c["engine"] for c in v.list_connections()]
    assert set(order) == {"postgres", "langfuse"}

    ctx = build_datasource_context(v, usage_notes={"postgres": "PG-NOTE", "langfuse": "LF-NOTE"})

    assert ctx.count("### Usage notes:") == 2
    positions = {e: ctx.index(f"(engine `{e}`)") for e in order}
    assert sorted(positions, key=positions.get) == order


def test_active_only_reads_no_record_of_an_excluded_connection(tmp_path):
    """A remote vault fetches each record over the network, so a connection
    that is not rendered must not be read."""
    reads = []

    class _CountingVault(LocalDataVault):
        def read_record(self, engine, name):
            reads.append((engine, name))
            return super().read_record(engine, name)

        def load(self, engine, name):
            reads.append((engine, name))
            return super().load(engine, name)

    v = _CountingVault(tmp_path / "vault")
    v.save("langfuse", "prod", {"public_key": "pk", "secret_key": "sk"})
    v.save("postgres", "db", {"host": "h", "password": "p"})
    reads.clear()

    ctx = build_datasource_context(v, active_only="langfuse-prod", usage_notes={})

    assert "DS_LANGFUSE_PROD__PUBLIC_KEY" in ctx
    assert reads == [("langfuse", "prod")]


def test_a_bare_string_for_slugs_is_rejected_not_matched_by_substring(vault):
    with pytest.raises(TypeError):
        collect_datasource_catalog(vault, slugs="langfuse-prod")


def test_rendering_with_a_bare_string_for_slugs_is_rejected(vault):
    catalog = collect_datasource_catalog(vault, usage_notes={})

    with pytest.raises(TypeError):
        catalog.render("langfuse-prod")


def test_usage_notes_cannot_be_passed_positionally(vault):
    with pytest.raises(TypeError):
        collect_datasource_catalog(vault, {"langfuse": "LF-NOTE"})
    with pytest.raises(TypeError):
        build_datasource_context(vault, None, {"langfuse": "LF-NOTE"})


def test_a_catalog_is_hashable(vault):
    catalog = collect_datasource_catalog(vault, usage_notes={"langfuse": "LF-NOTE"})

    assert isinstance(hash(catalog), int)
