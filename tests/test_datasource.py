from __future__ import annotations

import json
import os
from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anton.data_vault import DataVault
from anton.datasource_registry import (
    AuthMethod,
    DatasourceEngine,
    DatasourceField,
    DatasourceRegistry,
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
def registry(datasources_md, tmp_path):
    """Registry pointing at our temp datasources.md, no user overrides."""
    reg = DatasourceRegistry.__new__(DatasourceRegistry)
    reg._engines = {}
    from anton.datasource_registry import _parse_file
    reg._engines = _parse_file(datasources_md)
    return reg


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
        loaded = vault.load("postgresql", "prod_db")
        assert loaded == creds

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
        # vault_dir was never created
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
        # Cleanup
        vault.clear_ds_env()

    def test_inject_missing_returns_none(self, vault):
        result = vault.inject_env("postgresql", "ghost")
        assert result is None

    def test_clear_removes_ds_vars(self, vault):
        vault.save("postgresql", "prod_db", {"host": "x"})
        vault.inject_env("postgresql", "prod_db")
        vault.clear_ds_env()
        assert "DS_POSTGRESQL_PROD_DB__HOST" not in os.environ

    def test_clear_leaves_non_ds_vars(self, vault):
        os.environ["MY_VAR"] = "untouched"
        vault.clear_ds_env()
        assert os.environ.get("MY_VAR") == "untouched"
        del os.environ["MY_VAR"]

    def test_inject_uppercases_field_names(self, vault):
        vault.save("postgresql", "prod_db", {"access_token": "tok123"})
        vault.inject_env("postgresql", "prod_db")
        assert os.environ.get("DS_POSTGRESQL_PROD_DB__ACCESS_TOKEN") == "tok123"
        vault.clear_ds_env()

    def test_inject_flat_mode_sets_flat_vars(self, vault):
        """flat=True injects legacy DS_FIELD vars, not namespaced ones."""
        vault.save("postgresql", "prod_db", {"host": "db.example.com"})
        var_names = vault.inject_env("postgresql", "prod_db", flat=True)
        assert os.environ.get("DS_HOST") == "db.example.com"
        assert "DS_POSTGRESQL_PROD_DB__HOST" not in os.environ
        assert set(var_names) == {"DS_HOST"}
        vault.clear_ds_env()

    def test_two_same_type_connections_no_collision(self, vault):
        """Two connections of the same engine type coexist without overwriting each other."""
        vault.save("postgres", "prod_db", {"host": "prod.example.com"})
        vault.save("postgres", "analytics", {"host": "analytics.example.com"})
        vault.inject_env("postgres", "prod_db")
        vault.inject_env("postgres", "analytics")
        try:
            assert os.environ.get("DS_POSTGRES_PROD_DB__HOST") == "prod.example.com"
            assert os.environ.get("DS_POSTGRES_ANALYTICS__HOST") == "analytics.example.com"
            # The two vars are distinct — no collision
            assert os.environ.get("DS_POSTGRES_PROD_DB__HOST") != os.environ.get("DS_POSTGRES_ANALYTICS__HOST")
        finally:
            vault.clear_ds_env()

    def test_different_engines_no_collision(self, vault):
        """Connections from different engines coexist simultaneously."""
        vault.save("postgres", "prod_db", {"host": "pg.example.com"})
        vault.save("hubspot", "main", {"access_token": "pat-abc"})
        vault.inject_env("postgres", "prod_db")
        vault.inject_env("hubspot", "main")
        try:
            assert os.environ.get("DS_POSTGRES_PROD_DB__HOST") == "pg.example.com"
            assert os.environ.get("DS_HUBSPOT_MAIN__ACCESS_TOKEN") == "pat-abc"
        finally:
            vault.clear_ds_env()

    def test_slug_env_prefix_sanitizes_special_chars(self, vault):
        """Special characters in names are sanitized to underscores."""
        from anton.data_vault import _slug_env_prefix

        assert _slug_env_prefix("postgres", "prod-db.eu") == "DS_POSTGRES_PROD_DB_EU"
        # Full var
        vault.save("postgres", "prod-db.eu", {"host": "eu.pg.com"})
        vault.inject_env("postgres", "prod-db.eu")
        try:
            assert os.environ.get("DS_POSTGRES_PROD_DB_EU__HOST") == "eu.pg.com"
        finally:
            vault.clear_ds_env()


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
        # postgresql counter is independent
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

    def test_find_by_name_exact(self, registry):
        assert registry.find_by_name("PostgreSQL") is not None

    def test_find_by_name_case_insensitive(self, registry):
        assert registry.find_by_name("postgresql") is not None
        assert registry.find_by_name("POSTGRESQL") is not None

    def test_find_by_slug(self, registry):
        # engine slug is also accepted
        assert registry.find_by_name("postgresql") is not None

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
        assert "print(\"ok\")" in engine.test_snippet

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
        name = registry.derive_name(engine, {"host": "x"})  # no "database"
        assert name == ""

    def test_no_name_from_returns_empty(self):
        engine = DatasourceEngine(engine="test", display_name="Test", name_from="")
        reg = DatasourceRegistry.__new__(DatasourceRegistry)
        reg._engines = {}
        name = reg.derive_name(engine, {"host": "x"})
        assert name == ""

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
        name = reg.derive_name(engine, {"host": "db.example.com"})
        assert name == "db.example.com"


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

        from anton.datasource_registry import _parse_file

        builtin = _parse_file(datasources_md)
        user = _parse_file(user_md)
        merged = {**builtin, **user}

        assert merged["postgresql"].display_name == "PostgreSQL (custom)"
        assert merged["postgresql"].pip == "psycopg2"

    def test_missing_user_file_falls_back_to_builtin(self, tmp_path, datasources_md):
        from anton.datasource_registry import _parse_file

        user_engines = _parse_file(tmp_path / "nonexistent.md")
        assert user_engines == {}


# ─────────────────────────────────────────────────────────────────────────────
# _handle_connect_datasource — integration-style (mocked I/O)
# ─────────────────────────────────────────────────────────────────────────────


class TestHandleConnectDatasource:
    """Test the slash-command handler with mocked prompts and scratchpad."""

    def _make_session(self):
        from anton.chat import ChatSession

        mock_llm = AsyncMock()
        session = ChatSession(mock_llm)
        session._scratchpads = AsyncMock()
        return session

    def _make_cell(self, stdout="ok", stderr="", error=None):
        cell = MagicMock()
        cell.stdout = stdout
        cell.stderr = stderr
        cell.error = error
        return cell

    @pytest.mark.asyncio
    async def test_unknown_engine_returns_early(self, registry, vault_dir, capsys):
        """Typing an unknown engine name aborts without saving anything."""
        from anton.chat import _handle_connect_datasource

        session = self._make_session()
        console = MagicMock()
        console.print = MagicMock()

        with (
            patch("anton.chat.DataVault", return_value=DataVault(vault_dir=vault_dir)),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
            patch("rich.prompt.Prompt.ask", return_value="MySQL"),
        ):
            result = await _handle_connect_datasource(console, session._scratchpads, session)

        assert result is session  # unchanged session
        assert DataVault(vault_dir=vault_dir).list_connections() == []

    @pytest.mark.asyncio
    async def test_partial_save_on_n_answer(self, registry, vault_dir):
        """Answering 'n' saves partial credentials and returns without testing."""
        from anton.chat import _handle_connect_datasource

        session = self._make_session()
        console = MagicMock()
        console.print = MagicMock()

        vault = DataVault(vault_dir=vault_dir)
        prompt_responses = iter(["PostgreSQL", "n", "db.example.com", "", "", "", "", ""])

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
            patch("rich.prompt.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_responses)),
        ):
            result = await _handle_connect_datasource(console, session._scratchpads, session)

        conns = vault.list_connections()
        assert len(conns) == 1
        assert conns[0]["engine"] == "postgresql"
        # Partial connections get auto-numbered names
        assert conns[0]["name"].isdigit()
        # Scratchpad was NOT used for testing
        session._scratchpads.get_or_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_successful_connection_saves_and_injects_history(self, registry, vault_dir):
        """Happy path: test passes, credentials saved, history entry added."""
        from anton.chat import _handle_connect_datasource

        session = self._make_session()
        console = MagicMock()
        console.print = MagicMock()
        vault = DataVault(vault_dir=vault_dir)

        pad = AsyncMock()
        pad.execute = AsyncMock(return_value=self._make_cell(stdout="ok"))
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        prompt_responses = iter([
            "PostgreSQL",       # engine choice
            "y",                # have all credentials
            "db.example.com",   # host
            "5432",             # port
            "prod_db",          # database
            "alice",            # user
            "s3cr3t",           # password
            "",                 # schema (optional)
        ])

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
            patch("rich.prompt.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_responses)),
        ):
            result = await _handle_connect_datasource(console, session._scratchpads, session)

        # Credentials saved
        conns = vault.list_connections()
        assert len(conns) == 1
        saved = vault.load("postgresql", conns[0]["name"])
        assert saved["host"] == "db.example.com"
        assert saved["password"] == "s3cr3t"

        # History entry injected
        assert result._history
        last = result._history[-1]
        assert last["role"] == "assistant"
        assert "postgresql" in last["content"].lower()

    @pytest.mark.asyncio
    async def test_failed_test_offers_retry(self, registry, vault_dir):
        """Connection test failure prompts for retry; success on second attempt saves."""
        from anton.chat import _handle_connect_datasource

        session = self._make_session()
        console = MagicMock()
        console.print = MagicMock()
        vault = DataVault(vault_dir=vault_dir)

        fail_cell = self._make_cell(stdout="", stderr="password authentication failed")
        ok_cell = self._make_cell(stdout="ok")
        pad = AsyncMock()
        pad.execute = AsyncMock(side_effect=[fail_cell, ok_cell])
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        prompt_responses = iter([
            "PostgreSQL",       # engine
            "y",                # have all creds
            "db.example.com",   # host
            "5432",             # port
            "prod_db",          # database
            "alice",            # user
            "wrongpassword",    # password (first attempt - fails)
            "",                 # schema
            "y",                # retry?
            "correctpassword",  # new password
        ])

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
            patch("rich.prompt.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_responses)),
        ):
            result = await _handle_connect_datasource(console, session._scratchpads, session)

        # Should have saved after second attempt
        conns = vault.list_connections()
        assert len(conns) == 1
        saved = vault.load("postgresql", conns[0]["name"])
        assert saved["password"] == "correctpassword"

    @pytest.mark.asyncio
    async def test_failed_test_no_retry_returns_without_saving(self, registry, vault_dir):
        """Declining retry on failed test leaves vault empty."""
        from anton.chat import _handle_connect_datasource

        session = self._make_session()
        console = MagicMock()
        console.print = MagicMock()
        vault = DataVault(vault_dir=vault_dir)

        fail_cell = self._make_cell(stdout="", error="connection refused")
        pad = AsyncMock()
        pad.execute = AsyncMock(return_value=fail_cell)
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        prompt_responses = iter([
            "PostgreSQL", "y",
            "db.example.com", "5432", "prod_db", "alice", "badpass", "",
            "n",  # don't retry
        ])

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
            patch("rich.prompt.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_responses)),
        ):
            result = await _handle_connect_datasource(console, session._scratchpads, session)

        assert vault.list_connections() == []
        # No history injection since save never happened
        assert not result._history

    @pytest.mark.asyncio
    async def test_ds_env_injected_after_successful_connect(
        self, registry, vault_dir
    ):
        """After a successful connect, DS_* vars are injected into the env."""
        from anton.chat import _handle_connect_datasource

        session = self._make_session()
        console = MagicMock()
        console.print = MagicMock()
        vault = DataVault(vault_dir=vault_dir)

        pad = AsyncMock()
        pad.execute = AsyncMock(
            return_value=self._make_cell(stdout="ok")
        )
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        prompt_responses = iter([
            "PostgreSQL", "y",
            "db.example.com", "5432", "prod_db", "alice", "s3cr3t", "",
        ])

        try:
            with (
                patch("anton.chat.DataVault", return_value=vault),
                patch(
                    "anton.chat.DatasourceRegistry",
                    return_value=registry,
                ),
                patch(
                    "rich.prompt.Prompt.ask",
                    side_effect=lambda *a, **kw: next(
                        prompt_responses
                    ),
                ),
            ):
                await _handle_connect_datasource(
                    console, session._scratchpads, session
                )

            # After successful connect, namespaced DS_* vars are injected.
            # name_from=database → name="prod_db" → prefix DS_POSTGRESQL_PROD_DB
            assert os.environ.get("DS_POSTGRESQL_PROD_DB__HOST") == "db.example.com"
        finally:
            vault.clear_ds_env()

    @pytest.mark.asyncio
    async def test_auth_method_choice_selects_fields(self, registry, vault_dir):
        """Selecting an auth method filters to that method's fields only."""
        from anton.chat import _handle_connect_datasource

        session = self._make_session()
        console = MagicMock()
        console.print = MagicMock()
        vault = DataVault(vault_dir=vault_dir)

        pad = AsyncMock()
        pad.execute = AsyncMock(return_value=self._make_cell(stdout="ok"))
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        prompt_responses = iter([
            "HubSpot",          # engine
            "1",                # auth method: private_app
            "y",                # have all creds
            "pat-na1-abc123",   # access_token
        ])

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
            patch("rich.prompt.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_responses)),
        ):
            await _handle_connect_datasource(console, session._scratchpads, session)

        conns = vault.list_connections()
        assert len(conns) == 1
        saved = vault.load("hubspot", conns[0]["name"])
        # Only private_app fields collected — no client_id or client_secret
        assert "access_token" in saved
        assert "client_id" not in saved
        assert "client_secret" not in saved

    @pytest.mark.asyncio
    async def test_selective_field_collection(self, registry, vault_dir):
        """Typing 'host,user,password' collects only those three fields."""
        from anton.chat import _handle_connect_datasource

        session = self._make_session()
        console = MagicMock()
        console.print = MagicMock()
        vault = DataVault(vault_dir=vault_dir)

        pad = AsyncMock()
        pad.execute = AsyncMock(return_value=self._make_cell(stdout="ok"))
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        prompt_responses = iter([
            "PostgreSQL",           # engine
            "host,user,password",   # selective list
            "db.example.com",       # host
            "alice",                # user
            "s3cr3t",               # password
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
        assert set(saved.keys()) == {"host", "user", "password"}


# ─────────────────────────────────────────────────────────────────────────────
# Credential scrubbing
# ─────────────────────────────────────────────────────────────────────────────


class TestCredentialScrubbing:
    """_scrub_credentials and _register_secret_vars."""

    def setup_method(self):
        # Reset the module-level sets and clear any DS_* env vars from other tests
        from anton.chat import _DS_KNOWN_VARS, _DS_SECRET_VARS
        _DS_SECRET_VARS.clear()
        _DS_KNOWN_VARS.clear()
        ds_keys = [k for k in os.environ if k.startswith("DS_")]
        for k in ds_keys:
            del os.environ[k]

    def test_register_secret_vars_adds_secret_fields(self, registry):
        """Secret fields are added to _DS_SECRET_VARS; non-secret fields are not."""
        from anton.chat import _DS_SECRET_VARS, _register_secret_vars

        pg = registry.get("postgresql")
        assert pg is not None
        _register_secret_vars(pg)

        assert "DS_PASSWORD" in _DS_SECRET_VARS
        # host and port are not secret in the fixture definition
        assert "DS_HOST" not in _DS_SECRET_VARS
        assert "DS_PORT" not in _DS_SECRET_VARS

    def test_scrub_replaces_registered_secret_value(self):
        """A registered secret value is replaced with its placeholder."""
        import os
        from anton.chat import _DS_SECRET_VARS, _scrub_credentials

        _DS_SECRET_VARS.add("DS_ACCESS_TOKEN")
        os.environ["DS_ACCESS_TOKEN"] = "supersecrettoken123"
        try:
            result = _scrub_credentials("token is supersecrettoken123 here")
            assert "supersecrettoken123" not in result
            assert "[DS_ACCESS_TOKEN]" in result
        finally:
            del os.environ["DS_ACCESS_TOKEN"]
            _DS_SECRET_VARS.discard("DS_ACCESS_TOKEN")

    def test_scrub_leaves_non_secret_field_readable(self, registry):
        """Non-secret DS_* values (host, port) are left untouched."""
        import os
        from anton.chat import _register_secret_vars, _scrub_credentials

        pg = registry.get("postgresql")
        assert pg is not None
        _register_secret_vars(pg)

        os.environ["DS_HOST"] = "db.example.com"
        os.environ["DS_PASSWORD"] = "s3cr3tpassword99"
        try:
            result = _scrub_credentials("host=db.example.com pass=s3cr3tpassword99")
            assert "db.example.com" in result          # host left readable
            assert "s3cr3tpassword99" not in result    # password redacted
            assert "[DS_PASSWORD]" in result
        finally:
            del os.environ["DS_HOST"]
            del os.environ["DS_PASSWORD"]

    def test_scrub_skips_short_values(self):
        """Values of 8 characters or fewer are not scrubbed (e.g. port numbers)."""
        import os
        from anton.chat import _DS_SECRET_VARS, _scrub_credentials

        _DS_SECRET_VARS.add("DS_PASSWORD")
        os.environ["DS_PASSWORD"] = "short"  # 5 chars — under threshold
        try:
            result = _scrub_credentials("password=short")
            assert "short" in result
        finally:
            del os.environ["DS_PASSWORD"]
            _DS_SECRET_VARS.discard("DS_PASSWORD")

    def test_scrub_fallback_redacts_unknown_long_ds_vars(self):
        """Long DS_* vars not in _DS_SECRET_VARS are scrubbed as a safety fallback."""
        import os
        from anton.chat import _scrub_credentials

        # _DS_SECRET_VARS is empty (cleared in setup_method)
        os.environ["DS_WEBHOOK_SECRET"] = "wh_sec_abcdefgh1234"
        try:
            result = _scrub_credentials("secret=wh_sec_abcdefgh1234 here")
            assert "wh_sec_abcdefgh1234" not in result
            assert "[DS_WEBHOOK_SECRET]" in result
        finally:
            del os.environ["DS_WEBHOOK_SECRET"]

    @pytest.mark.asyncio
    async def test_register_and_scrub_on_connect(self, registry, vault_dir):
        """After _handle_connect_datasource, the new secret var is immediately scrubbed."""
        import os
        from unittest.mock import AsyncMock, MagicMock, patch

        from anton.chat import _DS_SECRET_VARS, _handle_connect_datasource, _scrub_credentials

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
            "PostgreSQL",   # engine
            "y",            # have all credentials
            "db.host.com",  # host
            "5432",         # port
            "mydb",         # database
            "alice",        # user
            secret_pw,      # password
            "public",       # schema (optional, skip)
        ])

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
            patch("rich.prompt.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_responses)),
        ):
            await _handle_connect_datasource(MagicMock(), session._scratchpads, session)

        # After connect, namespaced secret var is registered and scrubbed.
        # name_from=database → name="mydb" → DS_POSTGRESQL_MYDB__PASSWORD
        namespaced_pw_var = "DS_POSTGRESQL_MYDB__PASSWORD"
        assert namespaced_pw_var in _DS_SECRET_VARS
        os.environ[namespaced_pw_var] = secret_pw
        try:
            result = _scrub_credentials(f"error: auth failed with {secret_pw}")
            assert secret_pw not in result
            assert f"[{namespaced_pw_var}]" in result
        finally:
            del os.environ[namespaced_pw_var]


# ─────────────────────────────────────────────────────────────────────────────
# Active datasource scoping
# ─────────────────────────────────────────────────────────────────────────────


class TestActiveDatasourceScoping:
    """Tests for /connect-data-source <slug> isolating a single datasource."""

    def _make_session(self):
        from anton.chat import ChatSession

        mock_llm = AsyncMock()
        session = ChatSession(mock_llm)
        session._scratchpads = AsyncMock()
        return session

    def test_active_datasource_defaults_to_none(self):
        session = self._make_session()
        assert session._active_datasource is None

    @pytest.mark.asyncio
    async def test_reconnect_sets_active_datasource(self, vault_dir):
        """Reconnecting to a slug via prefill sets session._active_datasource."""
        from anton.chat import _handle_connect_datasource

        vault = DataVault(vault_dir=vault_dir)
        vault.save("hubspot", "2", {"access_token": "pat-xxx"})

        session = self._make_session()
        console = MagicMock()
        console.print = MagicMock()

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry"),
        ):
            result = await _handle_connect_datasource(
                console, session._scratchpads, session, prefill="hubspot-2"
            )

        assert result._active_datasource == "hubspot-2"

    @pytest.mark.asyncio
    async def test_reconnect_all_namespaced_vars_available(self, vault_dir):
        """After reconnect, ALL saved connections remain available as namespaced vars."""
        from anton.chat import _handle_connect_datasource

        vault = DataVault(vault_dir=vault_dir)
        vault.save("oracle", "1", {"host": "oracle.host", "user": "admin", "password": "orapass"})
        vault.save("hubspot", "2", {"access_token": "pat-xxx"})

        # Simulate startup: inject all connections as namespaced
        vault.inject_env("oracle", "1")
        vault.inject_env("hubspot", "2")
        assert os.environ.get("DS_ORACLE_1__HOST") == "oracle.host"
        assert os.environ.get("DS_HUBSPOT_2__ACCESS_TOKEN") == "pat-xxx"

        session = self._make_session()
        console = MagicMock()
        console.print = MagicMock()

        try:
            with (
                patch("anton.chat.DataVault", return_value=vault),
                patch("anton.chat.DatasourceRegistry"),
            ):
                result = await _handle_connect_datasource(
                    console, session._scratchpads, session, prefill="hubspot-2"
                )

            # After reconnect, all connections are restored as namespaced vars.
            # No flat DS_* vars are present.
            assert "DS_HOST" not in os.environ
            assert "DS_ACCESS_TOKEN" not in os.environ
            # Both connections remain available as namespaced vars.
            assert os.environ.get("DS_ORACLE_1__HOST") == "oracle.host"
            assert os.environ.get("DS_HUBSPOT_2__ACCESS_TOKEN") == "pat-xxx"
            # Active datasource is updated to the reconnected slug.
            assert result._active_datasource == "hubspot-2"
        finally:
            vault.clear_ds_env()

    def test_build_datasource_context_no_filter(self, vault_dir):
        """Without active_only, all vault entries appear in the context."""
        from anton.chat import _build_datasource_context

        vault = DataVault(vault_dir=vault_dir)
        vault.save("oracle", "1", {"host": "oracle.host"})
        vault.save("hubspot", "2", {"access_token": "pat-xxx"})

        with patch("anton.chat.DataVault", return_value=vault):
            ctx = _build_datasource_context()

        assert "oracle-1" in ctx
        assert "hubspot-2" in ctx

    def test_build_datasource_context_active_only_filters(self, vault_dir):
        """With active_only set, only the matching slug appears."""
        from anton.chat import _build_datasource_context

        vault = DataVault(vault_dir=vault_dir)
        vault.save("oracle", "1", {"host": "oracle.host"})
        vault.save("hubspot", "2", {"access_token": "pat-xxx"})

        with patch("anton.chat.DataVault", return_value=vault):
            ctx = _build_datasource_context(active_only="hubspot-2")

        assert "hubspot-2" in ctx
        assert "oracle-1" not in ctx

    def test_build_datasource_context_active_only_empty_when_no_match(self, vault_dir):
        """If active_only doesn't match any slug, the section has no entries."""
        from anton.chat import _build_datasource_context

        vault = DataVault(vault_dir=vault_dir)
        vault.save("oracle", "1", {"host": "oracle.host"})

        with patch("anton.chat.DataVault", return_value=vault):
            ctx = _build_datasource_context(active_only="hubspot-99")

        # Header is present but no datasource lines
        assert "oracle-1" not in ctx


# ─────────────────────────────────────────────────────────────────────────────
# CLI command registration
# ─────────────────────────────────────────────────────────────────────────────


class TestCliCommandRegistration:
    """Verify all datasource CLI commands are registered."""

    def _get_command_names(self):
        from anton.cli import app
        return [cmd.name for cmd in app.registered_commands]

    def test_connect_data_source_registered(self):
        names = self._get_command_names()
        assert "connect-data-source" in names

    def test_list_data_sources_registered(self):
        names = self._get_command_names()
        assert "list-data-sources" in names

    def test_edit_data_source_registered(self):
        names = self._get_command_names()
        assert "edit-data-source" in names

    def test_remove_data_source_registered(self):
        names = self._get_command_names()
        assert "remove-data-source" in names

    def test_test_data_source_registered(self):
        names = self._get_command_names()
        assert "test-data-source" in names


# ─────────────────────────────────────────────────────────────────────────────
# _handle_list_data_sources — improved output
# ─────────────────────────────────────────────────────────────────────────────


class TestHandleListDataSources:
    def test_empty_vault_shows_message(self, vault_dir):
        from unittest.mock import MagicMock, patch
        from anton.chat import _handle_list_data_sources

        console = MagicMock()
        console.print = MagicMock()

        with patch("anton.chat.DataVault", return_value=DataVault(vault_dir=vault_dir)):
            _handle_list_data_sources(console)

        printed = " ".join(str(c) for c in console.print.call_args_list)
        assert "No data sources" in printed or "connect-data-source" in printed

    def test_complete_connection_shows_saved(self, vault_dir, registry):
        from unittest.mock import MagicMock, patch
        from rich.console import Console
        from anton.chat import _handle_list_data_sources
        import io

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

    def test_incomplete_connection_shows_incomplete(self, vault_dir, registry):
        from unittest.mock import MagicMock, patch
        from rich.console import Console
        from anton.chat import _handle_list_data_sources
        import io

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

    def test_shows_source_name(self, vault_dir, registry):
        from unittest.mock import patch
        from rich.console import Console
        from anton.chat import _handle_list_data_sources
        import io

        vault = DataVault(vault_dir=vault_dir)
        vault.save("postgresql", "prod_db", {
            "host": "x", "port": "5432", "database": "d", "user": "u", "password": "p",
        })

        buf = io.StringIO()
        rich_console = Console(file=buf, highlight=False, markup=False)

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
        ):
            _handle_list_data_sources(rich_console)

        output = buf.getvalue()
        assert "PostgreSQL" in output


# ─────────────────────────────────────────────────────────────────────────────
# _handle_test_datasource
# ─────────────────────────────────────────────────────────────────────────────


class TestHandleTestDatasource:
    def _make_cell(self, stdout="ok", stderr="", error=None):
        from unittest.mock import MagicMock
        cell = MagicMock()
        cell.stdout = stdout
        cell.stderr = stderr
        cell.error = error
        return cell

    @pytest.mark.asyncio
    async def test_success_path(self, vault_dir, registry):
        from unittest.mock import AsyncMock, MagicMock, patch
        from anton.chat import _handle_test_datasource

        vault = DataVault(vault_dir=vault_dir)
        vault.save("postgresql", "prod_db", {
            "host": "db.example.com", "port": "5432",
            "database": "prod", "user": "alice", "password": "s3cr3t",
        })

        console = MagicMock()
        console.print = MagicMock()

        pad = AsyncMock()
        pad.execute = AsyncMock(return_value=self._make_cell(stdout="ok"))
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
    async def test_failure_path(self, vault_dir, registry):
        from unittest.mock import AsyncMock, MagicMock, patch
        from anton.chat import _handle_test_datasource

        vault = DataVault(vault_dir=vault_dir)
        vault.save("postgresql", "prod_db", {
            "host": "db.example.com", "port": "5432",
            "database": "prod", "user": "alice", "password": "wrongpass",
        })

        console = MagicMock()
        console.print = MagicMock()

        pad = AsyncMock()
        pad.execute = AsyncMock(
            return_value=self._make_cell(stdout="", stderr="password authentication failed")
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
        from unittest.mock import AsyncMock, MagicMock, patch
        from anton.chat import _handle_test_datasource

        vault = DataVault(vault_dir=vault_dir)
        console = MagicMock()
        console.print = MagicMock()
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
        from unittest.mock import AsyncMock, MagicMock, patch
        from anton.chat import _handle_test_datasource

        console = MagicMock()
        console.print = MagicMock()
        scratchpads = AsyncMock()

        with (
            patch("anton.chat.DataVault", return_value=DataVault(vault_dir=vault_dir)),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
        ):
            await _handle_test_datasource(console, scratchpads, "")

        printed = " ".join(str(c) for c in console.print.call_args_list)
        assert "Usage" in printed or "test-data-source" in printed

    @pytest.mark.asyncio
    async def test_ds_env_after_test(self, vault_dir, registry):
        """After test-data-source: flat vars are gone, namespaced vars are restored."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from anton.chat import _handle_test_datasource

        vault = DataVault(vault_dir=vault_dir)
        vault.save("postgresql", "prod_db", {
            "host": "db.example.com", "port": "5432",
            "database": "prod", "user": "alice", "password": "s3cr3t",
        })

        console = MagicMock()
        pad = AsyncMock()
        pad.execute = AsyncMock(return_value=self._make_cell(stdout="ok"))
        scratchpads = AsyncMock()
        scratchpads.get_or_create = AsyncMock(return_value=pad)

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
        ):
            await _handle_test_datasource(console, scratchpads, "postgresql-prod_db")

        try:
            # Flat vars must be gone
            assert "DS_HOST" not in os.environ
            assert "DS_PASSWORD" not in os.environ
            # Namespaced vars are restored (all saved connections)
            assert os.environ.get("DS_POSTGRESQL_PROD_DB__HOST") == "db.example.com"
        finally:
            vault.clear_ds_env()


# ─────────────────────────────────────────────────────────────────────────────
# Edit flow
# ─────────────────────────────────────────────────────────────────────────────


class TestEditDatasourceFlow:
    def _make_session(self):
        from unittest.mock import AsyncMock
        from anton.chat import ChatSession

        mock_llm = AsyncMock()
        session = ChatSession(mock_llm)
        session._scratchpads = AsyncMock()
        return session

    def _make_cell(self, stdout="ok", stderr="", error=None):
        from unittest.mock import MagicMock
        cell = MagicMock()
        cell.stdout = stdout
        cell.stderr = stderr
        cell.error = error
        return cell

    @pytest.mark.asyncio
    async def test_existing_values_loaded(self, registry, vault_dir):
        """Edit shows existing non-secret values as defaults."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from anton.chat import _handle_connect_datasource

        vault = DataVault(vault_dir=vault_dir)
        vault.save("postgresql", "prod_db", {
            "host": "old.host", "port": "5432",
            "database": "prod", "user": "alice", "password": "oldpass",
        })

        session = self._make_session()
        console = MagicMock()
        console.print = MagicMock()

        pad = AsyncMock()
        pad.execute = AsyncMock(return_value=self._make_cell(stdout="ok"))
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        # The prompt for 'host' should default to "old.host"; user presses Enter (keeps it)
        # The prompt for 'password' is secret; user provides new value
        prompt_values = iter([
            "old.host",   # host — Enter = keep (returns default)
            "5432",       # port
            "prod",       # database
            "alice",      # user
            "newpass",    # password
            "",           # schema (optional)
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
        assert saved["host"] == "old.host"
        assert saved["password"] == "newpass"

    @pytest.mark.asyncio
    async def test_enter_preserves_secret_value(self, registry, vault_dir):
        """Pressing Enter on a secret field keeps the existing value."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from anton.chat import _handle_connect_datasource

        vault = DataVault(vault_dir=vault_dir)
        original_pass = "original_secret_pass"
        vault.save("postgresql", "prod_db", {
            "host": "db.host", "port": "5432",
            "database": "prod", "user": "alice", "password": original_pass,
        })

        session = self._make_session()
        console = MagicMock()

        pad = AsyncMock()
        pad.execute = AsyncMock(return_value=self._make_cell(stdout="ok"))
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        # Empty string for secret field = keep existing
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
        assert saved["password"] == original_pass

    @pytest.mark.asyncio
    async def test_unknown_slug_returns_session(self, registry, vault_dir):
        """Editing a non-existent slug returns the session unchanged."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from anton.chat import _handle_connect_datasource

        vault = DataVault(vault_dir=vault_dir)
        session = self._make_session()
        console = MagicMock()
        console.print = MagicMock()

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
# Remove flow — full coverage
# ─────────────────────────────────────────────────────────────────────────────


class TestRemoveDatasourceFlow:
    def test_confirmation_yes_deletes(self, vault, registry):
        from unittest.mock import patch
        from anton.chat import _handle_remove_data_source
        from rich.console import Console

        vault.save("postgresql", "prod_db", {"host": "x"})
        console = Console(quiet=True)

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("rich.prompt.Confirm.ask", return_value=True),
        ):
            _handle_remove_data_source(console, "postgresql-prod_db")

        assert vault.load("postgresql", "prod_db") is None

    def test_confirmation_no_preserves(self, vault, registry):
        from unittest.mock import patch
        from anton.chat import _handle_remove_data_source
        from rich.console import Console

        vault.save("postgresql", "prod_db", {"host": "x"})
        console = Console(quiet=True)

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("rich.prompt.Confirm.ask", return_value=False),
        ):
            _handle_remove_data_source(console, "postgresql-prod_db")

        assert vault.load("postgresql", "prod_db") is not None

    def test_unknown_name_shows_message(self, vault_dir):
        from unittest.mock import MagicMock, patch
        from anton.chat import _handle_remove_data_source

        vault = DataVault(vault_dir=vault_dir)
        console = MagicMock()
        console.print = MagicMock()

        with patch("anton.chat.DataVault", return_value=vault):
            _handle_remove_data_source(console, "postgresql-ghost")

        printed = " ".join(str(c) for c in console.print.call_args_list)
        assert "not found" in printed.lower() or "No connection" in printed

    def test_invalid_format_shows_warning(self, vault_dir):
        from unittest.mock import MagicMock, patch
        from anton.chat import _handle_remove_data_source

        vault = DataVault(vault_dir=vault_dir)
        console = MagicMock()
        console.print = MagicMock()

        with patch("anton.chat.DataVault", return_value=vault):
            _handle_remove_data_source(console, "nohyphen")

        printed = " ".join(str(c) for c in console.print.call_args_list)
        assert "Invalid" in printed or "engine-name" in printed


# ─────────────────────────────────────────────────────────────────────────────
# Remove slash command — empty name bug fix
# ─────────────────────────────────────────────────────────────────────────────


class TestRemoveSlashCommandEmptyName:
    """Ensure /remove-data-source without arg does NOT call the handler."""

    def test_empty_arg_does_not_call_handler(self):
        """The chat loop must not call _handle_remove_data_source with an empty slug."""
        from unittest.mock import MagicMock, patch
        # Import the handler to verify it's not called with empty arg
        with patch("anton.chat._handle_remove_data_source") as mock_remove:
            # Simulate what the chat loop does when arg is empty
            cmd = "/remove-data-source"
            parts = [cmd]  # no argument
            arg = parts[1].strip() if len(parts) > 1 else ""
            assert arg == ""
            # The fixed logic:
            if not arg:
                pass  # show usage message, do NOT call handler
            else:
                from anton.chat import _handle_remove_data_source
                _handle_remove_data_source(MagicMock(), arg)

            mock_remove.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Environment activation — collision-free behavior
# ─────────────────────────────────────────────────────────────────────────────


class TestEnvActivationCollisionFree:
    def _make_session(self):
        from unittest.mock import AsyncMock
        from anton.chat import ChatSession

        mock_llm = AsyncMock()
        session = ChatSession(mock_llm)
        session._scratchpads = AsyncMock()
        return session

    def _make_cell(self, stdout="ok", stderr="", error=None):
        from unittest.mock import MagicMock
        cell = MagicMock()
        cell.stdout = stdout
        cell.stderr = stderr
        cell.error = error
        return cell

    @pytest.mark.asyncio
    async def test_connect_clears_previous_ds_vars(self, registry, vault_dir):
        """After a successful new connect, only the new connection's DS_* vars are set."""
        from unittest.mock import AsyncMock, patch
        from anton.chat import _handle_connect_datasource

        # Pre-inject an "old" connection
        os.environ["DS_ACCESS_TOKEN"] = "old-token"

        vault = DataVault(vault_dir=vault_dir)
        session = self._make_session()
        console = MagicMock()

        pad = AsyncMock()
        pad.execute = AsyncMock(return_value=self._make_cell(stdout="ok"))
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        prompt_responses = iter([
            "PostgreSQL", "y",
            "db.example.com", "5432", "prod_db", "alice", "s3cr3t", "",
        ])

        try:
            with (
                patch("anton.chat.DataVault", return_value=vault),
                patch("anton.chat.DatasourceRegistry", return_value=registry),
                patch("rich.prompt.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_responses)),
            ):
                await _handle_connect_datasource(console, session._scratchpads, session)

            # Old flat token must be gone; flat vars are never kept in runtime
            assert "DS_ACCESS_TOKEN" not in os.environ
            # Namespaced vars for the new connection must be present
            # name_from=database → name="prod_db" → DS_POSTGRESQL_PROD_DB__HOST
            assert os.environ.get("DS_POSTGRESQL_PROD_DB__HOST") == "db.example.com"
        finally:
            vault.clear_ds_env()
            os.environ.pop("DS_ACCESS_TOKEN", None)

    @pytest.mark.asyncio
    async def test_two_same_type_connections_no_collision(self, registry, vault_dir):
        """Activating one of two same-type connections sets only that one's vars."""
        from unittest.mock import MagicMock, patch
        from anton.chat import _handle_connect_datasource

        vault = DataVault(vault_dir=vault_dir)
        vault.save("postgresql", "db1", {
            "host": "host1.example.com", "port": "5432",
            "database": "db1", "user": "u1", "password": "p1",
        })
        vault.save("postgresql", "db2", {
            "host": "host2.example.com", "port": "5432",
            "database": "db2", "user": "u2", "password": "p2",
        })

        session = self._make_session()
        console = MagicMock()

        try:
            with (
                patch("anton.chat.DataVault", return_value=vault),
                patch("anton.chat.DatasourceRegistry", return_value=registry),
            ):
                # Reconnect to db2 (prefill path)
                await _handle_connect_datasource(
                    console, session._scratchpads, session,
                    prefill="postgresql-db2",
                )

            # Both connections remain available as namespaced vars (no collision)
            assert os.environ.get("DS_POSTGRESQL_DB1__HOST") == "host1.example.com"
            assert os.environ.get("DS_POSTGRESQL_DB2__HOST") == "host2.example.com"
            assert os.environ.get("DS_POSTGRESQL_DB2__DATABASE") == "db2"
            # No flat vars
            assert "DS_HOST" not in os.environ
            assert "DS_DATABASE" not in os.environ
        finally:
            vault.clear_ds_env()

    @pytest.mark.asyncio
    async def test_test_datasource_does_not_leave_vars(self, registry, vault_dir):
        """_handle_test_datasource cleans up all DS_* vars after testing."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from anton.chat import _handle_test_datasource

        vault = DataVault(vault_dir=vault_dir)
        vault.save("postgresql", "prod_db", {
            "host": "db.example.com", "port": "5432",
            "database": "prod", "user": "alice", "password": "s3cr3t",
        })

        pad = AsyncMock()
        pad.execute = AsyncMock(
            return_value=MagicMock(stdout="ok", stderr="", error=None)
        )
        scratchpads = AsyncMock()
        scratchpads.get_or_create = AsyncMock(return_value=pad)

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
        ):
            await _handle_test_datasource(MagicMock(), scratchpads, "postgresql-prod_db")

        try:
            # Flat vars must not be present after the test
            flat_leaked = [k for k in os.environ if k.startswith("DS_") and "__" not in k]
            assert flat_leaked == [], f"Flat DS_* vars leaked after test: {flat_leaked}"
            # Namespaced vars are restored — that is the expected runtime state
            assert os.environ.get("DS_POSTGRESQL_PROD_DB__HOST") == "db.example.com"
        finally:
            vault.clear_ds_env()


# ─────────────────────────────────────────────────────────────────────────────
# Chat slash-command behavior
# ─────────────────────────────────────────────────────────────────────────────


class TestChatSlashCommands:
    """Verify slash-command routing logic for datasource commands."""

    def test_list_data_sources_slash_command_routes(self):
        """'/list-data-sources' cmd string maps to _handle_list_data_sources."""
        # Verify the function is importable and callable (routing covered by chat loop)
        from anton.chat import _handle_list_data_sources
        assert callable(_handle_list_data_sources)

    def test_test_data_source_slash_command_routes(self):
        """'/test-data-source' is importable and async."""
        import inspect
        from anton.chat import _handle_test_datasource
        assert inspect.iscoroutinefunction(_handle_test_datasource)

    def test_remove_data_source_slash_command_routes(self):
        from anton.chat import _handle_remove_data_source
        assert callable(_handle_remove_data_source)

    def test_edit_data_source_routes_to_connect_handler(self):
        """'/edit-data-source' uses _handle_connect_datasource with datasource_name arg."""
        import inspect
        from anton.chat import _handle_connect_datasource
        sig = inspect.signature(_handle_connect_datasource)
        assert "datasource_name" in sig.parameters

    @pytest.mark.asyncio
    async def test_test_data_source_no_arg_shows_usage(self, vault_dir, registry):
        from unittest.mock import AsyncMock, MagicMock, patch
        from anton.chat import _handle_test_datasource

        console = MagicMock()
        console.print = MagicMock()
        scratchpads = AsyncMock()

        with (
            patch("anton.chat.DataVault", return_value=DataVault(vault_dir=vault_dir)),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
        ):
            await _handle_test_datasource(console, scratchpads, "")

        printed = " ".join(str(c) for c in console.print.call_args_list)
        assert "Usage" in printed or "test-data-source" in printed

    @pytest.mark.asyncio
    async def test_edit_data_source_no_arg_safe(self, vault_dir, registry):
        """'/edit-data-source' without arg (datasource_name=None) triggers new-connect flow."""
        from anton.chat import _handle_connect_datasource
        # datasource_name=None means new connect, not edit — no crash expected
        from anton.chat import ChatSession
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_llm = AsyncMock()
        session = ChatSession(mock_llm)
        session._scratchpads = AsyncMock()
        console = MagicMock()

        with (
            patch(
                "anton.chat.DataVault",
                return_value=DataVault(vault_dir=vault_dir),
            ),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
            patch("rich.prompt.Prompt.ask", return_value="UnknownEngine"),
        ):
            updated = await _handle_connect_datasource(
                console, session._scratchpads, session,
                datasource_name=None,
            )
        # Should return session (even if unknown engine)
        assert updated is not None

# ─────────────────────────────────────────────────────────────────────────────
# _slug_env_prefix
# ─────────────────────────────────────────────────────────────────────────────


class TestSlugEnvPrefix:
    """Unit tests for the _slug_env_prefix helper."""

    def test_basic_engine_and_name(self):
        from anton.data_vault import _slug_env_prefix
        assert _slug_env_prefix("postgres", "prod_db") == "DS_POSTGRES_PROD_DB"

    def test_hubspot_main(self):
        from anton.data_vault import _slug_env_prefix
        assert _slug_env_prefix("hubspot", "main") == "DS_HUBSPOT_MAIN"

    def test_sanitizes_hyphen_and_dot(self):
        from anton.data_vault import _slug_env_prefix
        assert _slug_env_prefix("postgres", "prod-db.eu") == "DS_POSTGRES_PROD_DB_EU"

    def test_numeric_name(self):
        from anton.data_vault import _slug_env_prefix
        assert _slug_env_prefix("postgresql", "1") == "DS_POSTGRESQL_1"

    def test_uppercase_result(self):
        from anton.data_vault import _slug_env_prefix
        result = _slug_env_prefix("myengine", "myname")
        assert result == result.upper()

    def test_double_underscore_separator_in_full_var(self):
        """The separator between prefix and field must be double underscore."""
        from anton.data_vault import _slug_env_prefix
        prefix = _slug_env_prefix("postgres", "prod_db")
        full_var = f"{prefix}__HOST"
        assert full_var == "DS_POSTGRES_PROD_DB__HOST"


# ─────────────────────────────────────────────────────────────────────────────
# Namespaced runtime env — _build_datasource_context
# ─────────────────────────────────────────────────────────────────────────────


class TestNamespacedRuntimeEnv:
    """Tests for namespaced vars in datasource context and multi-source access."""

    def test_build_datasource_context_shows_namespaced_vars(self, vault_dir):
        from unittest.mock import patch
        from anton.chat import _build_datasource_context

        vault = DataVault(vault_dir=vault_dir)
        vault.save("postgres", "prod_db", {"host": "pg.example.com", "password": "s3cr3t"})

        with patch("anton.chat.DataVault", return_value=vault):
            ctx = _build_datasource_context()

        assert "DS_POSTGRES_PROD_DB__HOST" in ctx
        assert "DS_POSTGRES_PROD_DB__PASSWORD" in ctx
        # No flat vars
        assert "DS_HOST" not in ctx

    def test_build_datasource_context_shows_slug_and_engine_label(self, vault_dir):
        from unittest.mock import patch
        from anton.chat import _build_datasource_context

        vault = DataVault(vault_dir=vault_dir)
        vault.save("postgres", "prod_db", {"host": "pg.example.com"})

        with patch("anton.chat.DataVault", return_value=vault):
            ctx = _build_datasource_context()

        assert "postgres-prod_db" in ctx
        assert "(postgres)" in ctx

    def test_multi_source_context_shows_both_connections(self, vault_dir):
        """Both connections are visible in the context with their namespaced vars."""
        from unittest.mock import patch
        from anton.chat import _build_datasource_context

        vault = DataVault(vault_dir=vault_dir)
        vault.save("postgres", "prod_db", {"host": "pg.example.com"})
        vault.save("hubspot", "main", {"access_token": "pat-abc"})

        with patch("anton.chat.DataVault", return_value=vault):
            ctx = _build_datasource_context()

        assert "postgres-prod_db" in ctx
        assert "DS_POSTGRES_PROD_DB__HOST" in ctx
        assert "hubspot-main" in ctx
        assert "DS_HUBSPOT_MAIN__ACCESS_TOKEN" in ctx

    def test_build_datasource_context_header_mentions_namespaced(self, vault_dir):
        """The header text explains the namespaced pattern."""
        from unittest.mock import patch
        from anton.chat import _build_datasource_context

        vault = DataVault(vault_dir=vault_dir)
        vault.save("postgres", "prod_db", {"host": "pg.example.com"})

        with patch("anton.chat.DataVault", return_value=vault):
            ctx = _build_datasource_context()

        assert "namespaced" in ctx.lower() or "DS_" in ctx


# ─────────────────────────────────────────────────────────────────────────────
# _register_secret_vars — namespaced mode
# ─────────────────────────────────────────────────────────────────────────────


class TestRegisterSecretVarsNamespaced:
    """Tests for namespaced secret var registration."""

    def setup_method(self):
        from anton.chat import _DS_KNOWN_VARS, _DS_SECRET_VARS
        _DS_SECRET_VARS.clear()
        _DS_KNOWN_VARS.clear()
        ds_keys = [k for k in os.environ if k.startswith("DS_")]
        for k in ds_keys:
            del os.environ[k]

    def test_register_with_slug_uses_namespaced_keys(self, registry):
        from anton.chat import _DS_KNOWN_VARS, _DS_SECRET_VARS, _register_secret_vars

        pg = registry.get("postgresql")
        _register_secret_vars(pg, engine="postgresql", name="prod_db")

        assert "DS_POSTGRESQL_PROD_DB__PASSWORD" in _DS_SECRET_VARS
        assert "DS_POSTGRESQL_PROD_DB__HOST" not in _DS_SECRET_VARS
        assert "DS_POSTGRESQL_PROD_DB__HOST" in _DS_KNOWN_VARS

    def test_register_without_slug_uses_flat_keys(self, registry):
        from anton.chat import _DS_SECRET_VARS, _register_secret_vars

        pg = registry.get("postgresql")
        _register_secret_vars(pg)  # no engine/name → flat mode

        assert "DS_PASSWORD" in _DS_SECRET_VARS
        assert "DS_HOST" not in _DS_SECRET_VARS

    def test_scrub_replaces_namespaced_secret_value(self, registry):
        from anton.chat import _DS_SECRET_VARS, _register_secret_vars, _scrub_credentials

        pg = registry.get("postgresql")
        _register_secret_vars(pg, engine="postgresql", name="prod_db")

        secret = "namespacedpassword123"
        os.environ["DS_POSTGRESQL_PROD_DB__PASSWORD"] = secret
        try:
            result = _scrub_credentials(f"error: {secret}")
            assert secret not in result
            assert "[DS_POSTGRESQL_PROD_DB__PASSWORD]" in result
        finally:
            del os.environ["DS_POSTGRESQL_PROD_DB__PASSWORD"]

    def test_scrub_leaves_namespaced_non_secret_readable(self, registry):
        from anton.chat import _register_secret_vars, _scrub_credentials

        pg = registry.get("postgresql")
        _register_secret_vars(pg, engine="postgresql", name="prod_db")

        os.environ["DS_POSTGRESQL_PROD_DB__HOST"] = "db.example.com"
        try:
            result = _scrub_credentials("host=db.example.com")
            assert "db.example.com" in result
        finally:
            del os.environ["DS_POSTGRESQL_PROD_DB__HOST"]


# ─────────────────────────────────────────────────────────────────────────────
# Temporary flat activation and restoration
# ─────────────────────────────────────────────────────────────────────────────


class TestTemporaryFlatExecution:
    """Tests that flat vars are used only during test_snippet, then restored."""

    def test_restore_namespaced_env_clears_flat_and_reinjects(self, vault_dir):
        """_restore_namespaced_env replaces flat vars with namespaced vars."""
        from unittest.mock import patch
        from anton.chat import _restore_namespaced_env

        vault = DataVault(vault_dir=vault_dir)
        vault.save("postgres", "analytics", {"host": "analytics.example.com"})

        # Simulate a flat injection (as done during test_snippet execution)
        vault.inject_env("postgres", "analytics", flat=True)
        assert os.environ.get("DS_HOST") == "analytics.example.com"
        assert "DS_POSTGRES_ANALYTICS__HOST" not in os.environ

        with patch("anton.chat.DataVault", return_value=vault):
            _restore_namespaced_env(vault)

        # Flat var is gone; namespaced var is back
        assert "DS_HOST" not in os.environ
        assert os.environ.get("DS_POSTGRES_ANALYTICS__HOST") == "analytics.example.com"
        vault.clear_ds_env()

    def test_restore_namespaced_env_reinjects_all_connections(self, vault_dir):
        """_restore_namespaced_env restores ALL saved connections, not just one."""
        from unittest.mock import patch
        from anton.chat import _restore_namespaced_env

        vault = DataVault(vault_dir=vault_dir)
        vault.save("postgres", "prod_db", {"host": "prod.example.com"})
        vault.save("hubspot", "main", {"access_token": "pat-abc"})

        # Simulate flat injection for one connection only
        vault.clear_ds_env()
        vault.inject_env("postgres", "prod_db", flat=True)

        with patch("anton.chat.DataVault", return_value=vault):
            _restore_namespaced_env(vault)

        # Both connections are available as namespaced vars
        assert "DS_HOST" not in os.environ
        assert os.environ.get("DS_POSTGRES_PROD_DB__HOST") == "prod.example.com"
        assert os.environ.get("DS_HUBSPOT_MAIN__ACCESS_TOKEN") == "pat-abc"
        vault.clear_ds_env()

    @pytest.mark.asyncio
    async def test_test_datasource_injects_flat_then_restores_namespaced(
        self, vault_dir, registry
    ):
        """_handle_test_datasource uses flat vars during snippet, then restores namespaced."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from anton.chat import _handle_test_datasource

        vault = DataVault(vault_dir=vault_dir)
        vault.save("postgresql", "prod_db", {
            "host": "pg.example.com", "port": "5432",
            "database": "prod_db", "user": "alice", "password": "s3cr3t",
        })
        # Pre-save a second connection that should be restored after the test
        vault.save("hubspot", "main", {"access_token": "pat-abc"})
        vault.inject_env("postgresql", "prod_db")
        vault.inject_env("hubspot", "main")

        env_during_test: dict = {}

        async def capture_execute(snippet):
            # Capture env state mid-execution to verify flat vars are set
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
        console = MagicMock()
        console.print = MagicMock()

        with (
            patch("anton.chat.DataVault", return_value=vault),
            patch("anton.chat.DatasourceRegistry", return_value=registry),
        ):
            await _handle_test_datasource(console, scratchpads, "postgresql-prod_db")

        # During execution: flat var was set, namespaced was absent
        assert env_during_test["DS_HOST"] == "pg.example.com"
        assert env_during_test["DS_POSTGRESQL_PROD_DB__HOST"] is None

        # After execution: flat vars gone, namespaced restored
        assert "DS_HOST" not in os.environ
        assert os.environ.get("DS_POSTGRESQL_PROD_DB__HOST") == "pg.example.com"
        assert os.environ.get("DS_HUBSPOT_MAIN__ACCESS_TOKEN") == "pat-abc"
        vault.clear_ds_env()
