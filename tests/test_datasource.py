from __future__ import annotations

import io
import json
import os
from textwrap import dedent
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from anton.chat import (
    ChatSession,
    _DS_KNOWN_VARS,
    _DS_SECRET_VARS,
    _build_datasource_context,
    _handle_add_custom_datasource,
    _handle_connect_datasource,
    _handle_list_data_sources,
    _handle_remove_data_source,
    _handle_test_datasource,
    _register_secret_vars,
    _restore_namespaced_env,
    _scrub_credentials,
    parse_connection_slug,
)
from anton.cli import app as _cli_app
from anton.data_vault import DataVault, _slug_env_prefix
from anton.datasource_registry import (
    DatasourceEngine,
    DatasourceRegistry,
    _parse_file,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def vault_dir(tmp_path):
    return tmp_path / "data_vault"


@pytest.fixture()
def vault(vault_dir):
    return DataVault(vault_dir=vault_dir)


@pytest.fixture()
def datasources_md(tmp_path):
    """Write a minimal datasources.md and return its path."""
    path = tmp_path / "datasources.md"
    path.write_text(dedent("""\
        ## PostgreSQL

        ```yaml
        engine: postgresql
        display_name: PostgreSQL
        pip: psycopg2-binary
        name_from: database
        fields:
          - name: host
            required: true
            description: hostname or IP
          - name: port
            required: true
            default: "5432"
            description: port number
          - name: database
            required: true
            description: database name
          - name: user
            required: true
            description: username
          - name: password
            required: true
            secret: true
            description: password
          - name: schema
            required: false
            description: defaults to public
        test_snippet: |
          import psycopg2
          conn = psycopg2.connect(
              host=os.environ["DS_HOST"],
              port=os.environ["DS_PORT"],
              dbname=os.environ["DS_DATABASE"],
              user=os.environ["DS_USER"],
              password=os.environ["DS_PASSWORD"],
          )
          conn.close()
          print("ok")
        ```

        ## HubSpot

        ```yaml
        engine: hubspot
        display_name: HubSpot
        pip: hubspot-api-client
        name_from: access_token
        auth_method: choice
        auth_methods:
          - name: private_app
            display: Private App token (recommended)
            fields:
              - name: access_token
                required: true
                secret: true
                description: pat-na1-xxx token
          - name: oauth
            display: OAuth 2.0
            fields:
              - name: client_id
                required: true
                description: OAuth client ID
              - name: client_secret
                required: true
                secret: true
                description: OAuth client secret
        test_snippet: |
          print("ok")
        ```
    """))
    return path


@pytest.fixture()
def registry(datasources_md):
    """Registry pointing at our temp datasources.md, no user overrides."""
    reg = DatasourceRegistry.__new__(DatasourceRegistry)
    reg._engines = _parse_file(datasources_md)
    return reg


@pytest.fixture()
def make_session():
    """Factory that creates a fresh ChatSession with mocked scratchpads."""
    def _factory():
        mock_llm = AsyncMock()
        session = ChatSession(mock_llm)
        session._scratchpads = AsyncMock()
        return session
    return _factory


@pytest.fixture()
def make_cell():
    """Factory that creates a MagicMock scratchpad execution cell."""
    def _factory(stdout="ok", stderr="", error=None):
        cell = MagicMock()
        cell.stdout = stdout
        cell.stderr = stderr
        cell.error = error
        return cell
    return _factory


@pytest.fixture(autouse=True)
def clean_ds_state():
    """Clear _DS_SECRET_VARS, _DS_KNOWN_VARS, and all DS_* env vars around each test."""
    def _clean():
        _DS_SECRET_VARS.clear()
        _DS_KNOWN_VARS.clear()
        for k in list(os.environ):
            if k.startswith("DS_"):
                del os.environ[k]

    _clean()
    yield
    _clean()


# ─────────────────────────────────────────────────────────────────────────────
# DataVault — save / load / delete
# ─────────────────────────────────────────────────────────────────────────────


class TestDataVaultSaveLoad:
    def test_save_creates_file(self, vault, vault_dir):
        vault.save("postgresql", "prod_db", {"host": "db.example.com", "port": "5432"})
        assert (vault_dir / "postgresql-prod_db").is_file()

    def test_save_file_permissions(self, vault, vault_dir):
        vault.save("postgresql", "prod_db", {"host": "db.example.com"})
        path = vault_dir / "postgresql-prod_db"
        mode = oct(path.stat().st_mode)[-3:]
        assert mode == "600"

    def test_vault_dir_permissions(self, vault, vault_dir):
        vault.save("postgresql", "prod_db", {"host": "db.example.com"})
        mode = oct(vault_dir.stat().st_mode)[-3:]
        assert mode == "700"

    def test_load_returns_fields(self, vault):
        creds = {"host": "db.example.com", "port": "5432", "password": "secret"}
        vault.save("postgresql", "prod_db", creds)
        assert vault.load("postgresql", "prod_db") == creds

    def test_load_missing_returns_none(self, vault):
        assert vault.load("postgresql", "nonexistent") is None

    def test_load_corrupt_file_returns_none(self, vault, vault_dir):
        vault._ensure_dir()
        (vault_dir / "postgresql-bad").write_text("not json")
        assert vault.load("postgresql", "bad") is None

    def test_save_overwrites_existing(self, vault):
        vault.save("postgresql", "prod_db", {"host": "old.host"})
        vault.save("postgresql", "prod_db", {"host": "new.host"})
        assert vault.load("postgresql", "prod_db") == {"host": "new.host"}

    def test_delete_existing(self, vault, vault_dir):
        vault.save("postgresql", "prod_db", {"host": "x"})
        result = vault.delete("postgresql", "prod_db")
        assert result is True
        assert not (vault_dir / "postgresql-prod_db").is_file()

    def test_delete_missing_returns_false(self, vault):
        assert vault.delete("postgresql", "ghost") is False

    def test_special_chars_sanitized_in_filename(self, vault, vault_dir):
        vault.save("postgresql", "my db/prod", {"host": "x"})
        files = list(vault_dir.iterdir())
        assert len(files) == 1
        assert "/" not in files[0].name

    def test_json_contains_metadata(self, vault, vault_dir):
        vault.save("postgresql", "prod_db", {"host": "x"})
        raw = json.loads((vault_dir / "postgresql-prod_db").read_text())
        assert raw["engine"] == "postgresql"
        assert raw["name"] == "prod_db"
        assert "created_at" in raw
        assert raw["fields"] == {"host": "x"}


# ─────────────────────────────────────────────────────────────────────────────
# DataVault — list_connections
# ─────────────────────────────────────────────────────────────────────────────


class TestDataVaultListConnections:
    def test_empty_vault(self, vault):
        assert vault.list_connections() == []

    def test_lists_all_connections(self, vault):
        vault.save("postgresql", "prod_db", {"host": "a"})
        vault.save("hubspot", "main", {"access_token": "pat-xxx"})
        conns = vault.list_connections()
        engines = {c["engine"] for c in conns}
        assert engines == {"postgresql", "hubspot"}

    def test_skips_corrupt_files(self, vault, vault_dir):
        vault._ensure_dir()
        vault.save("postgresql", "good", {"host": "x"})
        (vault_dir / "postgresql-bad").write_text("{{not json")
        conns = vault.list_connections()
        assert len(conns) == 1
        assert conns[0]["name"] == "good"

    def test_vault_dir_missing_returns_empty(self, vault):
        assert vault.list_connections() == []


# ─────────────────────────────────────────────────────────────────────────────
# DataVault — inject_env / clear_ds_env
# ─────────────────────────────────────────────────────────────────────────────


class TestDataVaultEnvInjection:
    def test_inject_sets_ds_vars(self, vault):
        vault.save("postgresql", "prod_db", {"host": "db.example.com", "password": "s3cr3t"})
        var_names = vault.inject_env("postgresql", "prod_db")
        assert os.environ.get("DS_POSTGRESQL_PROD_DB__HOST") == "db.example.com"
        assert os.environ.get("DS_POSTGRESQL_PROD_DB__PASSWORD") == "s3cr3t"
        assert set(var_names) == {"DS_POSTGRESQL_PROD_DB__HOST", "DS_POSTGRESQL_PROD_DB__PASSWORD"}

    def test_inject_missing_returns_none(self, vault):
        assert vault.inject_env("postgresql", "ghost") is None

    def test_clear_removes_ds_vars(self, vault):
        vault.save("postgresql", "prod_db", {"host": "x"})
        vault.inject_env("postgresql", "prod_db")
        vault.clear_ds_env()
        assert "DS_POSTGRESQL_PROD_DB__HOST" not in os.environ

    def test_clear_leaves_non_ds_vars(self, vault, monkeypatch):
        monkeypatch.setenv("MY_VAR", "untouched")
        vault.clear_ds_env()
        assert os.environ.get("MY_VAR") == "untouched"

    def test_inject_uppercases_field_names(self, vault):
        vault.save("postgresql", "prod_db", {"access_token": "tok123"})
        vault.inject_env("postgresql", "prod_db")
        assert os.environ.get("DS_POSTGRESQL_PROD_DB__ACCESS_TOKEN") == "tok123"

    def test_inject_flat_mode_sets_flat_vars(self, vault):
        """flat=True injects legacy DS_FIELD vars, not namespaced ones."""
        vault.save("postgresql", "prod_db", {"host": "db.example.com"})
        var_names = vault.inject_env("postgresql", "prod_db", flat=True)
        assert os.environ.get("DS_HOST") == "db.example.com"
        assert "DS_POSTGRESQL_PROD_DB__HOST" not in os.environ
        assert set(var_names) == {"DS_HOST"}

    def test_two_same_type_connections_no_collision(self, vault):
        """Two connections of the same engine type coexist without overwriting each other."""
        vault.save("postgres", "prod_db", {"host": "prod.example.com"})
        vault.save("postgres", "analytics", {"host": "analytics.example.com"})
        vault.inject_env("postgres", "prod_db")
        vault.inject_env("postgres", "analytics")
        assert os.environ.get("DS_POSTGRES_PROD_DB__HOST") == "prod.example.com"
        assert os.environ.get("DS_POSTGRES_ANALYTICS__HOST") == "analytics.example.com"
        assert os.environ.get("DS_POSTGRES_PROD_DB__HOST") != os.environ.get("DS_POSTGRES_ANALYTICS__HOST")

    def test_different_engines_no_collision(self, vault):
        """Connections from different engines coexist simultaneously."""
        vault.save("postgres", "prod_db", {"host": "pg.example.com"})
        vault.save("hubspot", "main", {"access_token": "pat-abc"})
        vault.inject_env("postgres", "prod_db")
        vault.inject_env("hubspot", "main")
        assert os.environ.get("DS_POSTGRES_PROD_DB__HOST") == "pg.example.com"
        assert os.environ.get("DS_HUBSPOT_MAIN__ACCESS_TOKEN") == "pat-abc"

    def test_slug_env_prefix_sanitizes_special_chars(self, vault):
        """Special characters in names produce correct namespaced vars."""
        assert _slug_env_prefix("postgres", "prod-db.eu") == "DS_POSTGRES_PROD_DB_EU"
        vault.save("postgres", "prod-db.eu", {"host": "eu.pg.com"})
        vault.inject_env("postgres", "prod-db.eu")
        assert os.environ.get("DS_POSTGRES_PROD_DB_EU__HOST") == "eu.pg.com"


# ─────────────────────────────────────────────────────────────────────────────
# DataVault — next_connection_number
# ─────────────────────────────────────────────────────────────────────────────


class TestDataVaultNextConnectionNumber:
    def test_returns_one_when_empty(self, vault):
        assert vault.next_connection_number("postgresql") == 1

    def test_increments_past_existing(self, vault):
        vault.save("postgresql", "1", {"host": "a"})
        vault.save("postgresql", "2", {"host": "b"})
        assert vault.next_connection_number("postgresql") == 3

    def test_ignores_named_connections(self, vault):
        # "prod_db" is not a digit — should not affect numbering
        vault.save("postgresql", "prod_db", {"host": "a"})
        assert vault.next_connection_number("postgresql") == 1

    def test_does_not_confuse_engines(self, vault):
        vault.save("hubspot", "1", {"access_token": "x"})
        vault.save("hubspot", "2", {"access_token": "y"})
        assert vault.next_connection_number("postgresql") == 1


# ─────────────────────────────────────────────────────────────────────────────
# DatasourceRegistry — lookup
# ─────────────────────────────────────────────────────────────────────────────


class TestDatasourceRegistry:
    def test_get_by_slug(self, registry):
        engine = registry.get("postgresql")
        assert engine is not None
        assert engine.display_name == "PostgreSQL"

    def test_get_missing_returns_none(self, registry):
        assert registry.get("mysql") is None

    @pytest.mark.parametrize("query", ["PostgreSQL", "postgresql", "POSTGRESQL"])
    def test_find_by_name_variants(self, registry, query):
        assert registry.find_by_name(query) is not None

    def test_find_unknown_returns_none(self, registry):
        assert registry.find_by_name("MySQL") is None

    def test_all_engines_sorted(self, registry):
        engines = registry.all_engines()
        names = [e.display_name for e in engines]
        assert names == sorted(names)

    def test_fields_parsed_correctly(self, registry):
        engine = registry.get("postgresql")
        field_names = [f.name for f in engine.fields]
        assert "host" in field_names
        assert "password" in field_names

    def test_secret_flag_on_password(self, registry):
        engine = registry.get("postgresql")
        pw = next(f for f in engine.fields if f.name == "password")
        assert pw.secret is True

    def test_required_flag(self, registry):
        engine = registry.get("postgresql")
        schema = next(f for f in engine.fields if f.name == "schema")
        assert schema.required is False

    def test_default_value_on_port(self, registry):
        engine = registry.get("postgresql")
        port = next(f for f in engine.fields if f.name == "port")
        assert port.default == "5432"

    def test_pip_field(self, registry):
        engine = registry.get("postgresql")
        assert engine.pip == "psycopg2-binary"

    def test_test_snippet_present(self, registry):
        engine = registry.get("postgresql")
        assert 'print("ok")' in engine.test_snippet

    def test_auth_method_choice_parsed(self, registry):
        engine = registry.get("hubspot")
        assert engine.auth_method == "choice"
        assert len(engine.auth_methods) == 2
        method_names = [m.name for m in engine.auth_methods]
        assert "private_app" in method_names
        assert "oauth" in method_names

    def test_auth_method_fields_parsed(self, registry):
        engine = registry.get("hubspot")
        private = next(m for m in engine.auth_methods if m.name == "private_app")
        assert len(private.fields) == 1
        assert private.fields[0].name == "access_token"
        assert private.fields[0].secret is True

    def test_validate_file_returns_engines(self, registry, datasources_md):
        result = registry.validate_file(datasources_md)
        assert "postgresql" in result
        assert "hubspot" in result
        assert result["postgresql"].display_name == "PostgreSQL"

    def test_validate_file_missing_returns_empty(self, registry, tmp_path):
        result = registry.validate_file(tmp_path / "nonexistent.md")
        assert result == {}

    def test_reload_picks_up_new_engine(self, tmp_path):
        md = tmp_path / "datasources.md"
        md.write_text(dedent("""\
            ## MySQL

            ```yaml
            engine: mysql
            display_name: MySQL
            pip: pymysql
            name_from: database
            fields:
              - name: host
                required: true
                description: hostname
            test_snippet: |
              print("ok")
            ```
        """))
        reg = DatasourceRegistry.__new__(DatasourceRegistry)
        reg._engines = {}
        reg._BUILTIN_PATH = md
        reg._USER_PATH = tmp_path / "user.md"
        reg.reload()
        assert reg.get("mysql") is not None
        assert reg.get("mysql").display_name == "MySQL"


# ─────────────────────────────────────────────────────────────────────────────
# DatasourceRegistry — derive_name
# ─────────────────────────────────────────────────────────────────────────────


class TestDeriveConnectionName:
    def test_single_field_name_from(self, registry):
        engine = registry.get("postgresql")  # name_from: database
        name = registry.derive_name(engine, {"database": "prod_db", "host": "x"})
        assert name == "prod_db"

    def test_missing_name_from_field_returns_empty(self, registry):
        engine = registry.get("postgresql")
        assert registry.derive_name(engine, {"host": "x"}) == ""

    def test_no_name_from_returns_empty(self):
        engine = DatasourceEngine(engine="test", display_name="Test", name_from="")
        reg = DatasourceRegistry.__new__(DatasourceRegistry)
        reg._engines = {}
        assert reg.derive_name(engine, {"host": "x"}) == ""

    def test_list_name_from(self):
        engine = DatasourceEngine(
            engine="test",
            display_name="Test",
            name_from=["host", "database"],
        )
        reg = DatasourceRegistry.__new__(DatasourceRegistry)
        reg._engines = {}
        name = reg.derive_name(engine, {"host": "db.example.com", "database": "prod"})
        assert name == "db.example.com_prod"

    def test_list_name_from_skips_missing(self):
        engine = DatasourceEngine(
            engine="test",
            display_name="Test",
            name_from=["host", "database"],
        )
        reg = DatasourceRegistry.__new__(DatasourceRegistry)
        reg._engines = {}
        assert reg.derive_name(engine, {"host": "db.example.com"}) == "db.example.com"


# ─────────────────────────────────────────────────────────────────────────────
# DatasourceRegistry — user overrides
# ─────────────────────────────────────────────────────────────────────────────


class TestDatasourceRegistryUserOverrides:
    def test_user_override_wins(self, tmp_path, datasources_md):
        """A user-defined engine with same slug overrides the builtin."""
        user_md = tmp_path / "user_datasources.md"
        user_md.write_text(dedent("""\
            ## PostgreSQL

            ```yaml
            engine: postgresql
            display_name: PostgreSQL (custom)
            pip: psycopg2
            fields:
              - name: host
                required: true
                description: custom host field
            test_snippet: print("ok")
            ```
        """))

        builtin = _parse_file(datasources_md)
        user = _parse_file(user_md)
        merged = {**builtin, **user}

        assert merged["postgresql"].display_name == "PostgreSQL (custom)"
        assert merged["postgresql"].pip == "psycopg2"

    def test_missing_user_file_falls_back_to_builtin(self, tmp_path):
        assert _parse_file(tmp_path / "nonexistent.md") == {}


# ─────────────────────────────────────────────────────────────────────────────
# _handle_connect_datasource — integration-style (mocked I/O)
# ─────────────────────────────────────────────────────────────────────────────


class TestHandleConnectDatasource:
    """Test the slash-command handler with mocked prompts and scratchpad."""

    @pytest.mark.asyncio
    async def test_unknown_engine_returns_early(self, registry, vault_dir, make_session):
        """Typing an unknown engine name aborts without saving anything."""
        session = make_session()
        console = MagicMock()

        with (
            patch("anton.chat.DataVault", return_value=DataVault(vault_dir=vault_dir)),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
            patch("rich.prompt.Prompt.ask", return_value="MySQL"),
        ):
            result = await _handle_connect_datasource(console, session._scratchpads, session)

        assert result is session
        assert DataVault(vault_dir=vault_dir).list_connections() == []

    @pytest.mark.asyncio
    async def test_partial_save_on_n_answer(self, registry, vault_dir, make_session):
        """Answering 'n' saves partial credentials and returns without testing."""
        session = make_session()
        console = MagicMock()
        vault = DataVault(vault_dir=vault_dir)
        prompt_responses = iter(["PostgreSQL", "n", "db.example.com", "", "", "", "", ""])

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
            patch("rich.prompt.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_responses)),
        ):
            await _handle_connect_datasource(console, session._scratchpads, session)

        conns = vault.list_connections()
        assert len(conns) == 1
        assert conns[0]["engine"] == "postgresql"
        assert conns[0]["name"].isdigit()
        session._scratchpads.get_or_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_successful_connection_saves_and_injects_history(
        self, registry, vault_dir, make_session, make_cell
    ):
        """Happy path: test passes, credentials saved, history entry added."""
        session = make_session()
        console = MagicMock()
        vault = DataVault(vault_dir=vault_dir)

        pad = AsyncMock()
        pad.execute = AsyncMock(return_value=make_cell(stdout="ok"))
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        prompt_responses = iter([
            "PostgreSQL", "y",
            "db.example.com", "5432", "prod_db", "alice", "s3cr3t", "",
        ])

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
            patch("rich.prompt.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_responses)),
        ):
            result = await _handle_connect_datasource(console, session._scratchpads, session)

        conns = vault.list_connections()
        assert len(conns) == 1
        saved = vault.load("postgresql", conns[0]["name"])
        assert saved is not None
        assert saved["host"] == "db.example.com"
        assert saved["password"] == "s3cr3t"
        assert result._history
        last = result._history[-1]
        assert last["role"] == "assistant"
        assert "postgresql" in last["content"].lower()

    @pytest.mark.asyncio
    async def test_failed_test_offers_retry(self, registry, vault_dir, make_session, make_cell):
        """Connection test failure prompts for retry; success on second attempt saves."""
        session = make_session()
        console = MagicMock()
        vault = DataVault(vault_dir=vault_dir)

        pad = AsyncMock()
        pad.execute = AsyncMock(side_effect=[
            make_cell(stdout="", stderr="password authentication failed"),
            make_cell(stdout="ok"),
        ])
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        prompt_responses = iter([
            "PostgreSQL", "y",
            "db.example.com", "5432", "prod_db", "alice", "wrongpassword", "",
            "y",              # retry?
            "correctpassword",
        ])

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
            patch("rich.prompt.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_responses)),
        ):
            await _handle_connect_datasource(console, session._scratchpads, session)

        conns = vault.list_connections()
        assert len(conns) == 1
        saved = vault.load("postgresql", conns[0]["name"])
        assert saved is not None
        assert saved["password"] == "correctpassword"

    @pytest.mark.asyncio
    async def test_failed_test_no_retry_returns_without_saving(
        self, registry, vault_dir, make_session, make_cell
    ):
        """Declining retry on failed test leaves vault empty."""
        session = make_session()
        console = MagicMock()
        vault = DataVault(vault_dir=vault_dir)

        pad = AsyncMock()
        pad.execute = AsyncMock(return_value=make_cell(stdout="", error="connection refused"))
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        prompt_responses = iter([
            "PostgreSQL", "y",
            "db.example.com", "5432", "prod_db", "alice", "badpass", "",
            "n",
        ])

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
            patch("rich.prompt.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_responses)),
        ):
            result = await _handle_connect_datasource(console, session._scratchpads, session)

        assert vault.list_connections() == []
        assert not result._history

    @pytest.mark.asyncio
    async def test_ds_env_injected_after_successful_connect(
        self, registry, vault_dir, make_session, make_cell
    ):
        """After a successful connect, namespaced DS_* vars are injected."""
        session = make_session()
        console = MagicMock()
        vault = DataVault(vault_dir=vault_dir)

        pad = AsyncMock()
        pad.execute = AsyncMock(return_value=make_cell(stdout="ok"))
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        prompt_responses = iter([
            "PostgreSQL", "y",
            "db.example.com", "5432", "prod_db", "alice", "s3cr3t", "",
        ])

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
            patch("rich.prompt.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_responses)),
        ):
            await _handle_connect_datasource(console, session._scratchpads, session)

        # name_from=database → name="prod_db" → prefix DS_POSTGRESQL_PROD_DB
        assert os.environ.get("DS_POSTGRESQL_PROD_DB__HOST") == "db.example.com"

    @pytest.mark.asyncio
    async def test_auth_method_choice_selects_fields(
        self, registry, vault_dir, make_session, make_cell
    ):
        """Selecting an auth method filters to that method's fields only."""
        session = make_session()
        console = MagicMock()
        vault = DataVault(vault_dir=vault_dir)

        pad = AsyncMock()
        pad.execute = AsyncMock(return_value=make_cell(stdout="ok"))
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        prompt_responses = iter(["HubSpot", "1", "y", "pat-na1-abc123"])

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
            patch("rich.prompt.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_responses)),
        ):
            await _handle_connect_datasource(console, session._scratchpads, session)

        conns = vault.list_connections()
        assert len(conns) == 1
        saved = vault.load("hubspot", conns[0]["name"])
        assert saved is not None
        # Only private_app fields collected — no client_id or client_secret
        assert "access_token" in saved
        assert "client_id" not in saved
        assert "client_secret" not in saved

    @pytest.mark.asyncio
    async def test_selective_field_collection(
        self, registry, vault_dir, make_session, make_cell
    ):
        """Typing 'host,user,password' collects only those three fields."""
        session = make_session()
        console = MagicMock()
        vault = DataVault(vault_dir=vault_dir)

        pad = AsyncMock()
        pad.execute = AsyncMock(return_value=make_cell(stdout="ok"))
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        prompt_responses = iter([
            "PostgreSQL", "host,user,password",
            "db.example.com", "alice", "s3cr3t",
        ])

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
            patch("rich.prompt.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_responses)),
        ):
            await _handle_connect_datasource(console, session._scratchpads, session)

        conns = vault.list_connections()
        assert len(conns) == 1
        saved = vault.load("postgresql", conns[0]["name"])
        assert saved is not None
        assert set(saved.keys()) == {"host", "user", "password"}


# ─────────────────────────────────────────────────────────────────────────────
# Credential scrubbing
# ─────────────────────────────────────────────────────────────────────────────


class TestCredentialScrubbing:
    """_scrub_credentials and _register_secret_vars — flat and namespaced modes."""

    def test_register_secret_vars_adds_secret_fields(self, registry):
        """Secret fields are added to _DS_SECRET_VARS; non-secret fields are not."""
        pg = registry.get("postgresql")
        assert pg is not None
        _register_secret_vars(pg)
        assert "DS_PASSWORD" in _DS_SECRET_VARS
        assert "DS_HOST" not in _DS_SECRET_VARS
        assert "DS_PORT" not in _DS_SECRET_VARS

    def test_scrub_replaces_registered_secret_value(self, monkeypatch):
        """A registered secret value is replaced with its placeholder."""
        _DS_SECRET_VARS.add("DS_ACCESS_TOKEN")
        monkeypatch.setenv("DS_ACCESS_TOKEN", "supersecrettoken123")
        result = _scrub_credentials("token is supersecrettoken123 here")
        assert "supersecrettoken123" not in result
        assert "[DS_ACCESS_TOKEN]" in result

    def test_scrub_leaves_non_secret_field_readable(self, registry, monkeypatch):
        """Non-secret DS_* values (host, port) are left untouched."""
        pg = registry.get("postgresql")
        assert pg is not None
        _register_secret_vars(pg)
        monkeypatch.setenv("DS_HOST", "mydbhostname")
        monkeypatch.setenv("DS_PASSWORD", "s3cr3tpassword99")
        result = _scrub_credentials("host=mydbhostname pass=s3cr3tpassword99")
        assert "mydbhostname" in result
        assert "s3cr3tpassword99" not in result
        assert "[DS_PASSWORD]" in result

    def test_scrub_skips_short_values(self, monkeypatch):
        """Registered secrets are always scrubbed regardless of length."""
        _DS_SECRET_VARS.add("DS_PASSWORD")
        monkeypatch.setenv("DS_PASSWORD", "short")
        result = _scrub_credentials("password=short")
        assert "short" not in result
        assert "[DS_PASSWORD]" in result

    def test_scrub_fallback_redacts_unknown_long_ds_vars(self, monkeypatch):
        """Long DS_* vars not in _DS_SECRET_VARS are scrubbed as a safety fallback."""
        monkeypatch.setenv("DS_WEBHOOK_SECRET", "wh_sec_abcdefgh1234")
        result = _scrub_credentials("secret=wh_sec_abcdefgh1234 here")
        assert "wh_sec_abcdefgh1234" not in result
        assert "[DS_WEBHOOK_SECRET]" in result

    @pytest.mark.asyncio
    async def test_register_and_scrub_on_connect(self, registry, vault_dir, monkeypatch):
        """After _handle_connect_datasource, the new secret var is immediately scrubbed."""
        vault = DataVault(vault_dir=vault_dir)
        session = MagicMock()
        session._history = []
        session._cortex = None

        pad = AsyncMock()
        pad.execute = AsyncMock(
            return_value=MagicMock(stdout="ok", stderr="", error=None)
        )
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        secret_pw = "supersecretpassword999"
        prompt_responses = iter([
            "PostgreSQL", "y",
            "db.host.com", "5432", "mydb", "alice", secret_pw, "public",
        ])

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
            patch("rich.prompt.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_responses)),
        ):
            await _handle_connect_datasource(MagicMock(), session._scratchpads, session)

        # name_from=database → name="mydb" → DS_POSTGRESQL_MYDB__PASSWORD
        namespaced_pw_var = "DS_POSTGRESQL_MYDB__PASSWORD"
        assert namespaced_pw_var in _DS_SECRET_VARS
        monkeypatch.setenv(namespaced_pw_var, secret_pw)
        result = _scrub_credentials(f"error: auth failed with {secret_pw}")
        assert secret_pw not in result
        assert f"[{namespaced_pw_var}]" in result

    # ── Namespaced mode ──────────────────────────────────────────────────────

    def test_register_with_slug_uses_namespaced_keys(self, registry):
        pg = registry.get("postgresql")
        _register_secret_vars(pg, engine="postgresql", name="prod_db")
        assert "DS_POSTGRESQL_PROD_DB__PASSWORD" in _DS_SECRET_VARS
        assert "DS_POSTGRESQL_PROD_DB__HOST" not in _DS_SECRET_VARS
        assert "DS_POSTGRESQL_PROD_DB__HOST" in _DS_KNOWN_VARS

    def test_register_without_slug_uses_flat_keys(self, registry):
        pg = registry.get("postgresql")
        _register_secret_vars(pg)  # no engine/name → flat mode
        assert "DS_PASSWORD" in _DS_SECRET_VARS
        assert "DS_HOST" not in _DS_SECRET_VARS

    def test_scrub_replaces_namespaced_secret_value(self, registry, monkeypatch):
        pg = registry.get("postgresql")
        _register_secret_vars(pg, engine="postgresql", name="prod_db")
        secret = "namespacedpassword123"
        monkeypatch.setenv("DS_POSTGRESQL_PROD_DB__PASSWORD", secret)
        result = _scrub_credentials(f"error: {secret}")
        assert secret not in result
        assert "[DS_POSTGRESQL_PROD_DB__PASSWORD]" in result

    def test_scrub_leaves_namespaced_non_secret_readable(self, registry, monkeypatch):
        pg = registry.get("postgresql")
        _register_secret_vars(pg, engine="postgresql", name="prod_db")
        monkeypatch.setenv("DS_POSTGRESQL_PROD_DB__HOST", "mydbhostname")
        result = _scrub_credentials("host=mydbhostname")
        assert "mydbhostname" in result


# ─────────────────────────────────────────────────────────────────────────────
# Active datasource scoping
# ─────────────────────────────────────────────────────────────────────────────


class TestActiveDatasourceScoping:
    """Tests for active datasource routing and multi-source context building."""

    def test_active_datasource_defaults_to_none(self, make_session):
        session = make_session()
        assert session._active_datasource is None

    @pytest.mark.asyncio
    async def test_reconnect_sets_active_datasource(self, vault_dir, make_session):
        """Reconnecting to a slug via prefill sets session._active_datasource."""
        vault = DataVault(vault_dir=vault_dir)
        vault.save("hubspot", "2", {"access_token": "pat-xxx"})

        session = make_session()
        console = MagicMock()

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry"),
        ):
            result = await _handle_connect_datasource(
                console, session._scratchpads, session, prefill="hubspot-2"
            )

        assert result._active_datasource == "hubspot-2"

    @pytest.mark.asyncio
    async def test_reconnect_all_namespaced_vars_available(self, vault_dir, make_session):
        """After reconnect, ALL saved connections remain available as namespaced vars."""
        vault = DataVault(vault_dir=vault_dir)
        vault.save("oracle", "1", {"host": "oracle.host", "user": "admin", "password": "orapass"})
        vault.save("hubspot", "2", {"access_token": "pat-xxx"})

        vault.inject_env("oracle", "1")
        vault.inject_env("hubspot", "2")
        assert os.environ.get("DS_ORACLE_1__HOST") == "oracle.host"
        assert os.environ.get("DS_HUBSPOT_2__ACCESS_TOKEN") == "pat-xxx"

        session = make_session()
        console = MagicMock()

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry"),
        ):
            result = await _handle_connect_datasource(
                console, session._scratchpads, session, prefill="hubspot-2"
            )

        assert "DS_HOST" not in os.environ
        assert "DS_ACCESS_TOKEN" not in os.environ
        assert os.environ.get("DS_ORACLE_1__HOST") == "oracle.host"
        assert os.environ.get("DS_HUBSPOT_2__ACCESS_TOKEN") == "pat-xxx"
        assert result._active_datasource == "hubspot-2"

    def test_build_datasource_context_no_filter(self, vault_dir):
        """Without active_only, all vault entries appear in the context."""
        vault = DataVault(vault_dir=vault_dir)
        vault.save("oracle", "1", {"host": "oracle.host"})
        vault.save("hubspot", "2", {"access_token": "pat-xxx"})

        with patch("anton.chat.DataVault", return_value=vault):
            ctx = _build_datasource_context()

        assert "oracle-1" in ctx
        assert "hubspot-2" in ctx

    def test_build_datasource_context_active_only_filters(self, vault_dir):
        """With active_only set, only the matching slug appears."""
        vault = DataVault(vault_dir=vault_dir)
        vault.save("oracle", "1", {"host": "oracle.host"})
        vault.save("hubspot", "2", {"access_token": "pat-xxx"})

        with patch("anton.chat.DataVault", return_value=vault):
            ctx = _build_datasource_context(active_only="hubspot-2")

        assert "hubspot-2" in ctx
        assert "oracle-1" not in ctx

    def test_build_datasource_context_active_only_empty_when_no_match(self, vault_dir):
        """If active_only doesn't match any slug, the section has no entries."""
        vault = DataVault(vault_dir=vault_dir)
        vault.save("oracle", "1", {"host": "oracle.host"})

        with patch("anton.chat.DataVault", return_value=vault):
            ctx = _build_datasource_context(active_only="hubspot-99")

        assert "oracle-1" not in ctx

    def test_build_datasource_context_shows_namespaced_vars(self, vault_dir):
        vault = DataVault(vault_dir=vault_dir)
        vault.save("postgres", "prod_db", {"host": "pg.example.com", "password": "s3cr3t"})

        with patch("anton.chat.DataVault", return_value=vault):
            ctx = _build_datasource_context()

        assert "DS_POSTGRES_PROD_DB__HOST" in ctx
        assert "DS_POSTGRES_PROD_DB__PASSWORD" in ctx
        assert "DS_HOST" not in ctx

    def test_build_datasource_context_shows_slug_and_engine_label(self, vault_dir):
        vault = DataVault(vault_dir=vault_dir)
        vault.save("postgres", "prod_db", {"host": "pg.example.com"})

        with patch("anton.chat.DataVault", return_value=vault):
            ctx = _build_datasource_context()

        assert "postgres-prod_db" in ctx
        assert "(postgres)" in ctx

    def test_multi_source_context_shows_both_connections(self, vault_dir):
        """Both connections are visible in the context with their namespaced vars."""
        vault = DataVault(vault_dir=vault_dir)
        vault.save("postgres", "prod_db", {"host": "pg.example.com"})
        vault.save("hubspot", "main", {"access_token": "pat-abc"})

        with patch("anton.chat.DataVault", return_value=vault):
            ctx = _build_datasource_context()

        assert "postgres-prod_db" in ctx
        assert "DS_POSTGRES_PROD_DB__HOST" in ctx
        assert "hubspot-main" in ctx
        assert "DS_HUBSPOT_MAIN__ACCESS_TOKEN" in ctx


# ─────────────────────────────────────────────────────────────────────────────
# CLI command registration
# ─────────────────────────────────────────────────────────────────────────────


class TestCliCommandRegistration:
    @pytest.mark.parametrize("cmd_name", [
        "connect",
        "list",
        "edit",
        "remove",
        "test",
    ])
    def test_command_registered(self, cmd_name):
        names = [cmd.name for cmd in _cli_app.registered_commands]
        assert cmd_name in names


# ─────────────────────────────────────────────────────────────────────────────
# _handle_list_data_sources
# ─────────────────────────────────────────────────────────────────────────────


class TestHandleListDataSources:
    def test_empty_vault_shows_message(self, vault_dir):
        console = MagicMock()
        with patch("anton.chat.DataVault", return_value=DataVault(vault_dir=vault_dir)):
            _handle_list_data_sources(console)
        printed = " ".join(str(c) for c in console.print.call_args_list)
        assert "No data sources" in printed or "connect" in printed

    def test_complete_connection_shows_saved_with_engine_name(self, vault_dir, registry):
        vault = DataVault(vault_dir=vault_dir)
        vault.save("postgresql", "prod_db", {
            "host": "db.example.com", "port": "5432",
            "database": "prod", "user": "alice", "password": "s3cr3t",
        })

        buf = io.StringIO()
        rich_console = Console(file=buf, highlight=False, markup=False)

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
        ):
            _handle_list_data_sources(rich_console)

        output = buf.getvalue()
        assert "postgresql-prod_db" in output
        assert "saved" in output.lower()
        assert "PostgreSQL" in output  # engine display_name shown

    def test_incomplete_connection_shows_incomplete(self, vault_dir, registry):
        vault = DataVault(vault_dir=vault_dir)
        # Missing required fields: database, user, password
        vault.save("postgresql", "partial", {"host": "db.example.com"})

        buf = io.StringIO()
        rich_console = Console(file=buf, highlight=False, markup=False)

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
        ):
            _handle_list_data_sources(rich_console)

        output = buf.getvalue()
        assert "incomplete" in output.lower()


# ─────────────────────────────────────────────────────────────────────────────
# _handle_test_datasource
# ─────────────────────────────────────────────────────────────────────────────


class TestHandleTestDatasource:
    @pytest.mark.asyncio
    async def test_success_path(self, vault_dir, registry, make_cell):
        vault = DataVault(vault_dir=vault_dir)
        vault.save("postgresql", "prod_db", {
            "host": "db.example.com", "port": "5432",
            "database": "prod", "user": "alice", "password": "s3cr3t",
        })
        console = MagicMock()
        pad = AsyncMock()
        pad.execute = AsyncMock(return_value=make_cell(stdout="ok"))
        scratchpads = AsyncMock()
        scratchpads.get_or_create = AsyncMock(return_value=pad)

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
        ):
            await _handle_test_datasource(console, scratchpads, "postgresql-prod_db")

        printed = " ".join(str(c) for c in console.print.call_args_list)
        assert "✓" in printed or "passed" in printed.lower()

    @pytest.mark.asyncio
    async def test_failure_path(self, vault_dir, registry, make_cell):
        vault = DataVault(vault_dir=vault_dir)
        vault.save("postgresql", "prod_db", {
            "host": "db.example.com", "port": "5432",
            "database": "prod", "user": "alice", "password": "wrongpass",
        })
        console = MagicMock()
        pad = AsyncMock()
        pad.execute = AsyncMock(
            return_value=make_cell(stdout="", stderr="password authentication failed")
        )
        scratchpads = AsyncMock()
        scratchpads.get_or_create = AsyncMock(return_value=pad)

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
        ):
            await _handle_test_datasource(console, scratchpads, "postgresql-prod_db")

        printed = " ".join(str(c) for c in console.print.call_args_list)
        assert "✗" in printed or "failed" in printed.lower()

    @pytest.mark.asyncio
    async def test_unknown_connection(self, vault_dir, registry):
        vault = DataVault(vault_dir=vault_dir)
        console = MagicMock()
        scratchpads = AsyncMock()

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
        ):
            await _handle_test_datasource(console, scratchpads, "postgresql-ghost")

        printed = " ".join(str(c) for c in console.print.call_args_list)
        assert "not found" in printed.lower() or "No connection" in printed

    @pytest.mark.asyncio
    async def test_empty_slug_shows_usage(self, vault_dir, registry):
        console = MagicMock()
        scratchpads = AsyncMock()

        with (
            patch("anton.chat.DataVault", return_value=DataVault(vault_dir=vault_dir)),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
        ):
            await _handle_test_datasource(console, scratchpads, "")

        printed = " ".join(str(c) for c in console.print.call_args_list)
        assert "Usage" in printed or "test" in printed


# ─────────────────────────────────────────────────────────────────────────────
# Edit flow
# ─────────────────────────────────────────────────────────────────────────────


class TestEditDatasourceFlow:
    @pytest.mark.asyncio
    async def test_existing_values_loaded(self, registry, vault_dir, make_session, make_cell):
        """Edit shows existing non-secret values as defaults."""
        vault = DataVault(vault_dir=vault_dir)
        vault.save("postgresql", "prod_db", {
            "host": "old.host", "port": "5432",
            "database": "prod", "user": "alice", "password": "oldpass",
        })

        session = make_session()
        console = MagicMock()
        pad = AsyncMock()
        pad.execute = AsyncMock(return_value=make_cell(stdout="ok"))
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        prompt_values = iter([
            "old.host", "5432", "prod", "alice", "newpass", "",
        ])

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
            patch("rich.prompt.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_values)),
        ):
            await _handle_connect_datasource(
                console, session._scratchpads, session, datasource_name="postgresql-prod_db"
            )

        saved = vault.load("postgresql", "prod_db")
        assert saved is not None
        assert saved["host"] == "old.host"
        assert saved["password"] == "newpass"

    @pytest.mark.asyncio
    async def test_enter_preserves_secret_value(self, registry, vault_dir, make_session, make_cell):
        """Pressing Enter on a secret field keeps the existing value."""
        vault = DataVault(vault_dir=vault_dir)
        original_pass = "original_secret_pass"
        vault.save("postgresql", "prod_db", {
            "host": "db.host", "port": "5432",
            "database": "prod", "user": "alice", "password": original_pass,
        })

        session = make_session()
        console = MagicMock()
        pad = AsyncMock()
        pad.execute = AsyncMock(return_value=make_cell(stdout="ok"))
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        prompt_values = iter([
            "db.host", "5432", "prod", "alice",
            "",   # password — Enter = keep original
            "",   # schema
        ])

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
            patch("rich.prompt.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_values)),
        ):
            await _handle_connect_datasource(
                console, session._scratchpads, session, datasource_name="postgresql-prod_db"
            )

        saved = vault.load("postgresql", "prod_db")
        assert saved is not None
        assert saved["password"] == original_pass

    @pytest.mark.asyncio
    async def test_unknown_slug_returns_session(self, registry, vault_dir, make_session):
        """Editing a non-existent slug returns the session unchanged."""
        vault = DataVault(vault_dir=vault_dir)
        session = make_session()
        console = MagicMock()

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
        ):
            result = await _handle_connect_datasource(
                console, session._scratchpads, session, datasource_name="postgresql-ghost"
            )

        assert result is session
        printed = " ".join(str(c) for c in console.print.call_args_list)
        assert "not found" in printed.lower() or "No connection" in printed


# ─────────────────────────────────────────────────────────────────────────────
# Remove flow
# ─────────────────────────────────────────────────────────────────────────────


class TestRemoveDatasourceFlow:
    def test_confirmation_yes_deletes(self, vault, registry):
        vault.save("postgresql", "prod_db", {"host": "x"})
        console = Console(quiet=True)

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
            patch("rich.prompt.Confirm.ask", return_value=True),
        ):
            _handle_remove_data_source(console, "postgresql-prod_db")

        assert vault.load("postgresql", "prod_db") is None

    def test_confirmation_no_preserves(self, vault, registry):
        vault.save("postgresql", "prod_db", {"host": "x"})
        console = Console(quiet=True)

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
            patch("rich.prompt.Confirm.ask", return_value=False),
        ):
            _handle_remove_data_source(console, "postgresql-prod_db")

        assert vault.load("postgresql", "prod_db") is not None

    def test_unknown_name_shows_message(self, vault_dir, registry):
        vault = DataVault(vault_dir=vault_dir)
        console = MagicMock()

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
        ):
            _handle_remove_data_source(console, "postgresql-ghost")

        printed = " ".join(str(c) for c in console.print.call_args_list)
        assert "not found" in printed.lower() or "No connection" in printed

    def test_invalid_format_shows_warning(self, vault_dir):
        vault = DataVault(vault_dir=vault_dir)
        console = MagicMock()

        with patch("anton.chat.DataVault", return_value=vault):
            _handle_remove_data_source(console, "nohyphen")

        printed = " ".join(str(c) for c in console.print.call_args_list)
        assert "Invalid" in printed or "engine-name" in printed


# ─────────────────────────────────────────────────────────────────────────────
# Environment activation — collision-free behavior
# ─────────────────────────────────────────────────────────────────────────────


class TestEnvActivationCollisionFree:
    @pytest.mark.asyncio
    async def test_connect_clears_previous_ds_vars(
        self, registry, vault_dir, make_session, make_cell, monkeypatch
    ):
        """After a successful new connect, stale DS_* vars are cleared."""
        monkeypatch.setenv("DS_ACCESS_TOKEN", "old-token")
        vault = DataVault(vault_dir=vault_dir)
        session = make_session()
        console = MagicMock()

        pad = AsyncMock()
        pad.execute = AsyncMock(return_value=make_cell(stdout="ok"))
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        prompt_responses = iter([
            "PostgreSQL", "y",
            "db.example.com", "5432", "prod_db", "alice", "s3cr3t", "",
        ])

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
            patch("rich.prompt.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_responses)),
        ):
            await _handle_connect_datasource(console, session._scratchpads, session)

        assert "DS_ACCESS_TOKEN" not in os.environ
        assert os.environ.get("DS_POSTGRESQL_PROD_DB__HOST") == "db.example.com"

    @pytest.mark.asyncio
    async def test_two_same_type_connections_no_collision(
        self, registry, vault_dir, make_session
    ):
        """Both same-type connections remain available as distinct namespaced vars."""
        vault = DataVault(vault_dir=vault_dir)
        vault.save("postgresql", "db1", {
            "host": "host1.example.com", "port": "5432",
            "database": "db1", "user": "u1", "password": "p1",
        })
        vault.save("postgresql", "db2", {
            "host": "host2.example.com", "port": "5432",
            "database": "db2", "user": "u2", "password": "p2",
        })

        session = make_session()
        console = MagicMock()

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
        ):
            await _handle_connect_datasource(
                console, session._scratchpads, session,
                prefill="postgresql-db2",
            )

        assert os.environ.get("DS_POSTGRESQL_DB1__HOST") == "host1.example.com"
        assert os.environ.get("DS_POSTGRESQL_DB2__HOST") == "host2.example.com"
        assert "DS_HOST" not in os.environ
        assert "DS_DATABASE" not in os.environ


# ─────────────────────────────────────────────────────────────────────────────
# Datasource slash-command behavior
# ─────────────────────────────────────────────────────────────────────────────


class TestDatasourceSlashCommandBehavior:
    @pytest.mark.asyncio
    async def test_test_data_source_no_arg_shows_usage(self, vault_dir, registry):
        console = MagicMock()
        scratchpads = AsyncMock()

        with (
            patch("anton.chat.DataVault", return_value=DataVault(vault_dir=vault_dir)),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
        ):
            await _handle_test_datasource(console, scratchpads, "")

        printed = " ".join(str(c) for c in console.print.call_args_list)
        assert "Usage" in printed or "test" in printed

    @pytest.mark.asyncio
    async def test_edit_data_source_no_arg_safe(self, vault_dir, registry, make_session):
        """datasource_name=None triggers new-connect flow without crash."""
        session = make_session()
        console = MagicMock()

        with (
            patch("anton.chat.DataVault", return_value=DataVault(vault_dir=vault_dir)),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
            patch("rich.prompt.Prompt.ask", return_value="UnknownEngine"),
        ):
            updated = await _handle_connect_datasource(
                console, session._scratchpads, session,
                datasource_name=None,
            )

        assert updated is not None


# ─────────────────────────────────────────────────────────────────────────────
# _slug_env_prefix
# ─────────────────────────────────────────────────────────────────────────────


class TestParseConnectionSlug:
    ENGINES = ["postgresql", "sql-server", "google-big-query"]

    def test_simple_engine(self):
        assert parse_connection_slug("postgresql-prod_db", self.ENGINES) == ("postgresql", "prod_db")

    def test_hyphenated_engine(self):
        assert parse_connection_slug("sql-server-prod-db", self.ENGINES) == ("sql-server", "prod-db")

    def test_longest_prefix_wins(self):
        engines = ["google", "google-big-query"]
        assert parse_connection_slug("google-big-query-main", engines) == ("google-big-query", "main")

    def test_ambiguous_resolves_to_longest(self):
        engines = ["sql", "sql-server"]
        assert parse_connection_slug("sql-server-1", engines) == ("sql-server", "1")

    def test_invalid_slug_no_match(self):
        assert parse_connection_slug("unknown-engine-name", self.ENGINES) is None

    def test_slug_with_empty_name_part(self):
        assert parse_connection_slug("postgresql-", self.ENGINES) is None

    def test_fallback_to_vault_for_custom_engine(self, tmp_path):
        """Custom engine not in registry is resolved via vault fallback."""
        vault = DataVault(vault_dir=tmp_path / "vault")
        vault.save("my_custom_db", "prod", {"host": "localhost"})
        result = parse_connection_slug(
            "my_custom_db-prod",
            known_engines=["postgresql"],
            vault=vault,
        )
        assert result == ("my_custom_db", "prod")

    def test_registry_match_takes_priority_over_vault(self, tmp_path):
        """Registry prefix match wins even when vault also has the slug."""
        vault = DataVault(vault_dir=tmp_path / "vault")
        vault.save("postgresql", "prod", {"host": "localhost"})
        result = parse_connection_slug(
            "postgresql-prod",
            known_engines=["postgresql"],
            vault=vault,
        )
        assert result == ("postgresql", "prod")

    def test_no_match_returns_none_with_vault(self, tmp_path):
        """Truly unknown slug returns None even with vault supplied."""
        vault = DataVault(vault_dir=tmp_path / "vault")
        result = parse_connection_slug(
            "ghost-engine-1",
            known_engines=["postgresql"],
            vault=vault,
        )
        assert result is None

    def test_no_vault_still_returns_none_for_unknown(self):
        """Backward compat: no vault arg, unknown engine still returns None."""
        assert parse_connection_slug("custom-1", known_engines=["postgresql"]) is None


class TestSlugEnvPrefix:
    def test_basic_engine_and_name(self):
        assert _slug_env_prefix("postgres", "prod_db") == "DS_POSTGRES_PROD_DB"

    def test_hubspot_main(self):
        assert _slug_env_prefix("hubspot", "main") == "DS_HUBSPOT_MAIN"

    def test_sanitizes_hyphen_and_dot(self):
        assert _slug_env_prefix("postgres", "prod-db.eu") == "DS_POSTGRES_PROD_DB_EU"

    def test_numeric_name(self):
        assert _slug_env_prefix("postgresql", "1") == "DS_POSTGRESQL_1"


# ─────────────────────────────────────────────────────────────────────────────
# Temporary flat activation and restoration
# ─────────────────────────────────────────────────────────────────────────────


class TestTemporaryFlatExecution:
    """Tests that flat vars are used only during test_snippet, then restored."""

    def test_restore_namespaced_env_clears_flat_and_reinjects(self, vault_dir):
        """_restore_namespaced_env replaces flat vars with namespaced vars."""
        vault = DataVault(vault_dir=vault_dir)
        vault.save("postgres", "analytics", {"host": "analytics.example.com"})

        vault.inject_env("postgres", "analytics", flat=True)
        assert os.environ.get("DS_HOST") == "analytics.example.com"
        assert "DS_POSTGRES_ANALYTICS__HOST" not in os.environ

        with patch("anton.chat.DataVault", return_value=vault):
            _restore_namespaced_env(vault)

        assert "DS_HOST" not in os.environ
        assert os.environ.get("DS_POSTGRES_ANALYTICS__HOST") == "analytics.example.com"

    def test_restore_namespaced_env_reinjects_all_connections(self, vault_dir):
        """_restore_namespaced_env restores ALL saved connections, not just one."""
        vault = DataVault(vault_dir=vault_dir)
        vault.save("postgres", "prod_db", {"host": "prod.example.com"})
        vault.save("hubspot", "main", {"access_token": "pat-abc"})

        vault.inject_env("postgres", "prod_db", flat=True)

        with patch("anton.chat.DataVault", return_value=vault):
            _restore_namespaced_env(vault)

        assert "DS_HOST" not in os.environ
        assert os.environ.get("DS_POSTGRES_PROD_DB__HOST") == "prod.example.com"
        assert os.environ.get("DS_HUBSPOT_MAIN__ACCESS_TOKEN") == "pat-abc"

    @pytest.mark.asyncio
    async def test_test_datasource_injects_flat_then_restores_namespaced(
        self, vault_dir, registry
    ):
        """_handle_test_datasource uses flat vars during snippet, then restores namespaced."""
        vault = DataVault(vault_dir=vault_dir)
        vault.save("postgresql", "prod_db", {
            "host": "pg.example.com", "port": "5432",
            "database": "prod_db", "user": "alice", "password": "s3cr3t",
        })
        vault.save("hubspot", "main", {"access_token": "pat-abc"})
        vault.inject_env("postgresql", "prod_db")
        vault.inject_env("hubspot", "main")

        env_during_test: dict = {}

        async def capture_execute(snippet):
            env_during_test["DS_HOST"] = os.environ.get("DS_HOST")
            env_during_test["DS_POSTGRESQL_PROD_DB__HOST"] = os.environ.get(
                "DS_POSTGRESQL_PROD_DB__HOST"
            )
            return MagicMock(stdout="ok", stderr="", error=None)

        pad = AsyncMock()
        pad.execute = capture_execute
        pad.reset = AsyncMock()
        pad.install_packages = AsyncMock()

        scratchpads = AsyncMock()
        scratchpads.get_or_create = AsyncMock(return_value=pad)

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
        ):
            await _handle_test_datasource(MagicMock(), scratchpads, "postgresql-prod_db")

        # During execution: flat var was set, namespaced was absent
        assert env_during_test["DS_HOST"] == "pg.example.com"
        assert env_during_test["DS_POSTGRESQL_PROD_DB__HOST"] is None

        # After execution: flat vars gone, namespaced restored
        assert "DS_HOST" not in os.environ
        assert os.environ.get("DS_POSTGRESQL_PROD_DB__HOST") == "pg.example.com"
        assert os.environ.get("DS_HUBSPOT_MAIN__ACCESS_TOKEN") == "pat-abc"


# ─────────────────────────────────────────────────────────────────────────────
# Stale registration state regression tests
# ─────────────────────────────────────────────────────────────────────────────


class TestStaleDsRegistrationState:
    """Regression tests: _DS_SECRET_VARS/_DS_KNOWN_VARS must mirror vault contents."""

    def test_remove_clears_stale_secret_vars(self, vault_dir, registry):
        """After removing a connection, its secret var names leave _DS_SECRET_VARS."""
        vault = DataVault(vault_dir=vault_dir)
        vault.save("postgresql", "prod_db", {
            "host": "pg.example.com", "port": "5432",
            "database": "prod_db", "user": "alice", "password": "s3cr3t",
        })

        with patch("anton.datasource_registry.DatasourceRegistry", return_value=registry):
            _restore_namespaced_env(vault)

        assert "DS_POSTGRESQL_PROD_DB__PASSWORD" in _DS_SECRET_VARS
        assert "DS_POSTGRESQL_PROD_DB__PASSWORD" in _DS_KNOWN_VARS

        vault.delete("postgresql", "prod_db")

        with patch("anton.datasource_registry.DatasourceRegistry", return_value=registry):
            _restore_namespaced_env(vault)

        assert "DS_POSTGRESQL_PROD_DB__PASSWORD" not in _DS_SECRET_VARS
        assert "DS_POSTGRESQL_PROD_DB__PASSWORD" not in _DS_KNOWN_VARS

    def test_edit_connection_refreshes_secret_vars(self, vault_dir, registry):
        """Overwriting a connection via vault.save rebuilds registration without duplication."""
        vault = DataVault(vault_dir=vault_dir)
        vault.save("postgresql", "prod_db", {
            "host": "pg.example.com", "port": "5432",
            "database": "prod_db", "user": "alice", "password": "old-pass",
        })

        with patch("anton.datasource_registry.DatasourceRegistry", return_value=registry):
            _restore_namespaced_env(vault)

        secret_key = "DS_POSTGRESQL_PROD_DB__PASSWORD"
        assert secret_key in _DS_SECRET_VARS
        count_before = len(_DS_SECRET_VARS)

        # Simulate edit: overwrite with new credentials
        vault.save("postgresql", "prod_db", {
            "host": "pg.example.com", "port": "5432",
            "database": "prod_db", "user": "alice", "password": "new-pass",
        })

        with patch("anton.datasource_registry.DatasourceRegistry", return_value=registry):
            _restore_namespaced_env(vault)

        assert secret_key in _DS_SECRET_VARS
        assert len(_DS_SECRET_VARS) == count_before
        assert os.environ.get(secret_key) == "new-pass"

    def test_reconnect_no_duplicate_secret_vars(self, vault_dir, registry):
        """Calling _restore_namespaced_env multiple times does not grow _DS_SECRET_VARS."""
        vault = DataVault(vault_dir=vault_dir)
        vault.save("postgresql", "prod_db", {
            "host": "pg.example.com", "port": "5432",
            "database": "prod_db", "user": "alice", "password": "s3cr3t",
        })

        with patch("anton.datasource_registry.DatasourceRegistry", return_value=registry):
            _restore_namespaced_env(vault)

        count_after_first = len(_DS_SECRET_VARS)
        known_after_first = len(_DS_KNOWN_VARS)

        with patch("anton.datasource_registry.DatasourceRegistry", return_value=registry):
            _restore_namespaced_env(vault)

        assert len(_DS_SECRET_VARS) == count_after_first
        assert len(_DS_KNOWN_VARS) == known_after_first


# ─────────────────────────────────────────────────────────────────────────────
# TestAddCustomDatasourceFlow
# ─────────────────────────────────────────────────────────────────────────────


class TestAddCustomDatasourceFlow:
    """Tests for _handle_add_custom_datasource field-collection logic."""

    def _make_llm_response(
        self, fields: list[dict], display_name: str = "MyDB"
    ) -> str:
        """Return a JSON string mimicking the LLM's plan() response."""
        import json as _json
        return _json.dumps(
            {"display_name": display_name, "pip": "", "fields": fields}
        )

    def _make_registry(self, tmp_path):
        """Return a minimal registry mock that accepts any slug."""
        reg = MagicMock()
        reg.validate_file.return_value = {"mydb": MagicMock()}
        reg.reload.return_value = None
        reg.get.return_value = None  # triggers inline fallback
        return reg

    def _make_llm(self, json_text: str):
        """Return an AsyncMock LLM whose plan() returns json_text."""
        llm = AsyncMock()
        response = MagicMock()
        response.content = json_text
        llm.plan = AsyncMock(return_value=response)
        return llm

    def _mock_ds_path(self, mock_path_cls, tmp_path):
        """Wire Path mock so datasources.md writes go to tmp_path."""
        mock_path_cls.return_value.expanduser.return_value = tmp_path / "datasources.md"

    @pytest.mark.asyncio
    async def test_missing_required_non_secret_field_prompts_user(
        self, tmp_path, make_session
    ):
        """Required non-secret field without inline value triggers Prompt.ask."""
        session = make_session()
        session._llm = self._make_llm(self._make_llm_response([
            {
                "name": "host", "value": "", "secret": False,
                "required": True, "description": "hostname",
            },
        ]))
        console = MagicMock()
        registry = self._make_registry(tmp_path)

        # first response: initial auth question; second: the host field prompt
        prompt_responses = iter(["I want to connect to mydb", "localhost"])

        with (
            patch(
                "rich.prompt.Prompt.ask",
                side_effect=lambda *a, **kw: next(prompt_responses),
            ),
            patch("anton.chat.Path") as mock_path_cls,
        ):
            self._mock_ds_path(mock_path_cls, tmp_path)
            result = await _handle_add_custom_datasource(
                console, "mydb", registry, session
            )

        assert result is not None
        _, credentials = result
        assert credentials["host"] == "localhost"

    @pytest.mark.asyncio
    async def test_missing_required_secret_field_prompts_user(
        self, tmp_path, make_session
    ):
        """Required secret field without inline value triggers password prompt."""
        session = make_session()
        session._llm = self._make_llm(self._make_llm_response([
            {
                "name": "api_key", "value": "", "secret": True,
                "required": True, "description": "API key",
            },
        ]))
        console = MagicMock()
        registry = self._make_registry(tmp_path)

        password_calls = []

        def fake_prompt(*args, **kwargs):
            if kwargs.get("password"):
                password_calls.append(kwargs)
                return "mysecret"
            return "I want to connect"

        with (
            patch("rich.prompt.Prompt.ask", side_effect=fake_prompt),
            patch("anton.chat.Path") as mock_path_cls,
        ):
            self._mock_ds_path(mock_path_cls, tmp_path)
            result = await _handle_add_custom_datasource(
                console, "mydb", registry, session
            )

        assert result is not None
        _, credentials = result
        assert credentials["api_key"] == "mysecret"
        assert len(password_calls) == 1, "expected exactly one password prompt"

    @pytest.mark.asyncio
    async def test_incomplete_custom_datasource_not_saved(
        self, tmp_path, make_session
    ):
        """Empty responses for all required fields causes a hard stop (None)."""
        session = make_session()
        session._llm = self._make_llm(self._make_llm_response([
            {
                "name": "host", "value": "", "secret": False,
                "required": True, "description": "hostname",
            },
            {
                "name": "api_key", "value": "", "secret": True,
                "required": True, "description": "API key",
            },
        ]))
        console = MagicMock()
        registry = self._make_registry(tmp_path)

        # User presses Enter (empty) for every prompt
        prompt_responses = iter(["I want to connect", "", ""])

        with (
            patch(
                "rich.prompt.Prompt.ask",
                side_effect=lambda *a, **kw: next(prompt_responses),
            ),
            patch("anton.chat.Path") as mock_path_cls,
        ):
            self._mock_ds_path(mock_path_cls, tmp_path)
            result = await _handle_add_custom_datasource(
                console, "mydb", registry, session
            )

        # Must return None — caller must not call vault.save()
        assert result is None


class TestCustomDatasourceConnectFlow:
    """Tests for the custom_source path in _handle_connect_datasource:
    test_snippet is run before saving, and failures prevent saving."""

    # ── helpers (mirrors TestAddCustomDatasourceFlow) ────────────────────

    def _make_llm_response(
        self,
        fields: list[dict],
        display_name: str = "My API Service",
        test_snippet: str = "",
    ) -> str:
        import json as _json
        return _json.dumps({
            "display_name": display_name,
            "pip": "",
            "test_snippet": test_snippet,
            "fields": fields,
        })

    def _make_registry(self, tmp_path):
        reg = MagicMock()
        reg.all_engines.return_value = []
        reg.find_by_name.return_value = None
        reg.fuzzy_find.return_value = []
        reg.validate_file.return_value = {"my_api_service": MagicMock()}
        reg.reload.return_value = None
        reg.get.return_value = None  # triggers inline fallback engine_def
        return reg

    def _make_llm(self, json_text: str):
        llm = AsyncMock()
        response = MagicMock()
        response.content = json_text
        llm.plan = AsyncMock(return_value=response)
        return llm

    def _mock_ds_path(self, mock_path_cls, tmp_path):
        mock_path_cls.return_value.expanduser.return_value = tmp_path / "datasources.md"

    # ── tests ─────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_custom_with_test_snippet_success(
        self, vault_dir, make_session, make_cell, tmp_path
    ):
        """Custom datasource with test_snippet: test passes → connection saved."""
        session = make_session()
        console = MagicMock()
        vault = DataVault(vault_dir=vault_dir)

        pad = AsyncMock()
        pad.execute = AsyncMock(return_value=make_cell(stdout="ok"))
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)
        session._llm = self._make_llm(self._make_llm_response(
            [{"name": "api_key", "value": "", "secret": True, "required": True, "description": "API key"}],
            test_snippet="print('ok')",
        ))

        prompt_responses = iter([
            "0",                  # choose custom
            "My API Service",     # tool name
            "I have an API key",  # auth description
            "my_secret_key",      # api_key (secret prompt)
        ])

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=self._make_registry(tmp_path)),
            patch("rich.prompt.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_responses)),
            patch("anton.chat.Path") as mock_path_cls,
        ):
            self._mock_ds_path(mock_path_cls, tmp_path)
            result = await _handle_connect_datasource(console, session._scratchpads, session)

        conns = vault.list_connections()
        assert len(conns) == 1
        saved = vault.load(conns[0]["engine"], conns[0]["name"])
        assert saved is not None
        assert saved.get("api_key") == "my_secret_key"
        assert result._history
        assert result._history[-1]["role"] == "assistant"
        pad.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_custom_with_test_snippet_fail_no_retry(
        self, vault_dir, make_session, make_cell, tmp_path
    ):
        """Custom datasource: test fails and user declines retry → not saved."""
        session = make_session()
        console = MagicMock()
        vault = DataVault(vault_dir=vault_dir)

        pad = AsyncMock()
        pad.execute = AsyncMock(return_value=make_cell(stdout="", stderr="connection refused"))
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)
        session._llm = self._make_llm(self._make_llm_response(
            [{"name": "api_key", "value": "", "secret": True, "required": True, "description": "API key"}],
            test_snippet="print('ok')",
        ))

        prompt_responses = iter([
            "0",
            "My API Service",
            "I have an API key",
            "bad_key",  # api_key
            "n",        # retry?
        ])

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=self._make_registry(tmp_path)),
            patch("rich.prompt.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_responses)),
            patch("anton.chat.Path") as mock_path_cls,
        ):
            self._mock_ds_path(mock_path_cls, tmp_path)
            result = await _handle_connect_datasource(console, session._scratchpads, session)

        assert vault.list_connections() == []
        assert not result._history

    @pytest.mark.asyncio
    async def test_custom_with_test_snippet_fail_retry_success(
        self, vault_dir, make_session, make_cell, tmp_path
    ):
        """Custom datasource: test fails, user retries with corrected creds → saved."""
        session = make_session()
        console = MagicMock()
        vault = DataVault(vault_dir=vault_dir)

        pad = AsyncMock()
        pad.execute = AsyncMock(side_effect=[
            make_cell(stdout="", stderr="invalid key"),
            make_cell(stdout="ok"),
        ])
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)
        session._llm = self._make_llm(self._make_llm_response(
            [{"name": "api_key", "value": "", "secret": True, "required": True, "description": "API key"}],
            test_snippet="print('ok')",
        ))

        prompt_responses = iter([
            "0",
            "My API Service",
            "I have an API key",
            "bad_key",      # api_key first attempt
            "y",            # retry?
            "good_key",     # api_key retry
        ])

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=self._make_registry(tmp_path)),
            patch("rich.prompt.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_responses)),
            patch("anton.chat.Path") as mock_path_cls,
        ):
            self._mock_ds_path(mock_path_cls, tmp_path)
            result = await _handle_connect_datasource(console, session._scratchpads, session)

        conns = vault.list_connections()
        assert len(conns) == 1
        saved = vault.load(conns[0]["engine"], conns[0]["name"])
        assert saved is not None
        assert saved.get("api_key") == "good_key"
        assert result._history

    @pytest.mark.asyncio
    async def test_custom_without_test_snippet_saves(
        self, vault_dir, make_session, make_cell, tmp_path
    ):
        """Custom datasource without test_snippet: saves directly, no scratchpad call."""
        session = make_session()
        console = MagicMock()
        vault = DataVault(vault_dir=vault_dir)

        pad = AsyncMock()
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)
        session._llm = self._make_llm(self._make_llm_response(
            [{"name": "api_key", "value": "", "secret": True, "required": True, "description": "API key"}],
            test_snippet="",
        ))

        prompt_responses = iter([
            "0",
            "My API Service",
            "I have an API key",
            "my_key",  # api_key
        ])

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=self._make_registry(tmp_path)),
            patch("rich.prompt.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_responses)),
            patch("anton.chat.Path") as mock_path_cls,
        ):
            self._mock_ds_path(mock_path_cls, tmp_path)
            await _handle_connect_datasource(console, session._scratchpads, session)

        conns = vault.list_connections()
        assert len(conns) == 1
        pad.execute.assert_not_called()
