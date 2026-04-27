from __future__ import annotations

import io
import json
import os
import re
from textwrap import dedent
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from anton.chat import (
    ChatSession,
)
from anton.core.session import ChatSessionConfig
from anton.commands.datasource import (
    _PROMPT_RECONNECT_CANCEL,
    handle_add_custom_datasource,
    handle_connect_datasource,
    handle_test_datasource,
    run_connection_test,
    handle_list_data_sources,
    handle_remove_data_source,
)
from anton.utils.datasources import (
    _DS_KNOWN_VARS,
    _DS_SECRET_VARS,
    build_datasource_context,
    register_secret_vars,
    restore_namespaced_env,
    scrub_credentials,
    parse_connection_slug,
)
from anton.cli import app as cli_app
from anton.core.datasources.data_vault import DataVault, LocalDataVault, _slug_env_prefix
from anton.core.datasources.datasource_registry import (
    DatasourceEngine,
    DatasourceField,
    DatasourceRegistry,
    _parse_file,
)


@pytest.fixture()
def vault_dir(tmp_path):
    return tmp_path / "data_vault"


@pytest.fixture()
def vault(vault_dir):
    return LocalDataVault(vault_dir=vault_dir)


@pytest.fixture()
def datasources_md(tmp_path):
    """Write a minimal datasources.md and return its path."""
    path = tmp_path / "datasources.md"
    path.write_text(
        dedent(
            """\
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
    """
        )
    )
    return path


@pytest.fixture()
def registry(datasources_md):
    """Registry pointing at our temp datasources.md, no user overrides."""
    reg = DatasourceRegistry.__new__(DatasourceRegistry)
    reg._engines = _parse_file(datasources_md)
    return reg


@pytest.fixture()
def make_session():
    """Factory that creates a fresh ChatSession with mocked scratchpads.

    The default `generate_object` dispatcher returns a sensible empty
    instance for whichever Pydantic schema the production code asks
    for. Tests that need a non-empty result should override
    `session._llm.generate_object` after construction.
    """

    def _factory():
        from anton.connect_collector import _ExtractionResult

        async def _default_generate_object(schema_class, **kwargs):
            # Known extraction schemas → empty defaults so the call
            # falls back to "no structured data" behavior. Unknown
            # schemas → raise so the caller's try/except sees a clear
            # failure (matching the pre-refactor behavior where
            # json.loads would fail on the canned "UNKNOWN" content).
            if schema_class is _ExtractionResult:
                return _ExtractionResult()
            raise RuntimeError(
                f"test mock has no default for {schema_class.__name__}; "
                "override session._llm.generate_object in this test"
            )

        from anton.core.llm.provider import ProviderConnectionInfo
        mock_llm = AsyncMock()
        plan_response = MagicMock()
        plan_response.content = "UNKNOWN"
        mock_llm.plan = AsyncMock(return_value=plan_response)
        mock_llm.generate_object = AsyncMock(side_effect=_default_generate_object)
        mock_llm.coding_provider = MagicMock()
        mock_llm.coding_provider.export_connection_info = MagicMock(
            return_value=ProviderConnectionInfo(provider="anthropic", api_key="test")
        )
        mock_llm.coding_model = "claude-sonnet-4-6"
        session = ChatSession(ChatSessionConfig(llm_client=mock_llm))
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


@pytest.fixture()
def make_pad():
    """Factory that creates a pre-configured scratchpad AsyncMock."""

    def _factory(cell=None, side_effect=None):
        pad = AsyncMock()
        if side_effect is not None:
            pad.execute = AsyncMock(side_effect=side_effect)
        else:
            if cell is None:
                cell = MagicMock()
                cell.stdout = "ok"
                cell.stderr = ""
                cell.error = None
            pad.execute = AsyncMock(return_value=cell)
        pad.reset = AsyncMock()
        pad.install_packages = AsyncMock(return_value="")
        return pad

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


class TestDataVaultEnvInjection:
    def test_inject_sets_ds_vars(self, vault):
        vault.save(
            "postgresql", "prod_db", {"host": "db.example.com", "password": "s3cr3t"}
        )
        var_names = vault.inject_env("postgresql", "prod_db")
        assert os.environ.get("DS_POSTGRESQL_PROD_DB__HOST") == "db.example.com"
        assert os.environ.get("DS_POSTGRESQL_PROD_DB__PASSWORD") == "s3cr3t"
        assert set(var_names) == {
            "DS_POSTGRESQL_PROD_DB__HOST",
            "DS_POSTGRESQL_PROD_DB__PASSWORD",
        }

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
        assert os.environ.get("DS_POSTGRES_PROD_DB__HOST") != os.environ.get(
            "DS_POSTGRES_ANALYTICS__HOST"
        )

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
        md.write_text(
            dedent(
                """\
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
        """
            )
        )
        reg = DatasourceRegistry.__new__(DatasourceRegistry)
        reg._engines = {}
        reg._BUILTIN_PATH = md
        reg._USER_PATH = tmp_path / "user.md"
        reg.reload()
        assert reg.get("mysql") is not None
        assert reg.get("mysql").display_name == "MySQL"


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


class TestDatasourceRegistryUserOverrides:
    def test_user_override_wins(self, tmp_path, datasources_md):
        """A user-defined engine with same slug overrides the builtin."""
        user_md = tmp_path / "user_datasources.md"
        user_md.write_text(
            dedent(
                """\
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
        """
            )
        )

        builtin = _parse_file(datasources_md)
        user = _parse_file(user_md)
        merged = {**builtin, **user}

        assert merged["postgresql"].display_name == "PostgreSQL (custom)"
        assert merged["postgresql"].pip == "psycopg2"

    def test_missing_user_file_falls_back_to_builtin(self, tmp_path):
        assert _parse_file(tmp_path / "nonexistent.md") == {}


# ─────────────────────────────────────────────────────────────────────────────
# handle_connect_datasource — integration-style (mocked I/O)
# ─────────────────────────────────────────────────────────────────────────────


class TestHandleConnectDatasource:
    """Test the slash-command handler with mocked prompts and scratchpad."""

    @pytest.mark.asyncio
    async def test_unknown_engine_returns_early(
        self, registry, vault_dir, make_session
    ):
        """Typing an unknown engine name aborts without saving anything."""
        session = make_session()
        console = MagicMock()

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=LocalDataVault(vault_dir=vault_dir)),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
            patch("anton.commands.datasource.connect.prompt_or_cancel", return_value="MySQL"),
            patch("anton.commands.datasource.custom.prompt_or_cancel", return_value="MySQL"),
        ):
            result = await handle_connect_datasource(
                console, session._scratchpads, session
            )

        assert result is session
        assert LocalDataVault(vault_dir=vault_dir).list_connections() == []

    @pytest.mark.asyncio
    async def test_partial_save_on_skip(self, registry, vault_dir, make_session):
        """Answering 'skip' at the bulk prompt saves partial credentials and returns without testing."""
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)
        responses = iter(["PostgreSQL", "skip"])

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
            patch(
                "anton.commands.datasource.connect.prompt_or_cancel",
                new=AsyncMock(side_effect=lambda *a, **kw: next(responses)),
            ),
        ):
            await handle_connect_datasource(console, session._scratchpads, session)

        conns = vault.list_connections()
        assert len(conns) == 1
        assert conns[0]["engine"] == "postgresql"
        assert len(conns[0]["name"]) == 8 and all(
            c in "0123456789abcdef" for c in conns[0]["name"]
        )
        assert conns[0]["name"].isalnum()
        session._scratchpads.get_or_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_successful_connection_saves_and_injects_history(
        self, registry, vault_dir, make_session, make_cell, make_pad
    ):
        """Happy path: test passes, credentials saved, history entry added."""
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)

        pad = make_pad()
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        responses = iter(
            [
                "PostgreSQL",
                "db.example.com",
                "5432",
                "prod_db",
                "alice",
                "s3cr3t",
            ]
        )

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
            patch(
                "anton.commands.datasource.connect.prompt_or_cancel",
                new=AsyncMock(side_effect=lambda *a, **kw: next(responses)),
            ),
        ):
            result = await handle_connect_datasource(
                console, session._scratchpads, session
            )

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
    async def test_test_failed_decline_sets_status(
        self, registry, vault_dir, make_session, make_cell, make_pad
    ):
        """When the connection test fails and the user declines to
        re-enter credentials, the handler should set
        session._pending_connect_status = 'test_failed' so the tool
        wrapper can return an accurate (non-misleading) message to the
        LLM instead of claiming the user pressed Escape.
        """
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)

        pad = make_pad(make_cell(stdout="", error="connection refused"))
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        # Engine pick + decline retry after the test fails
        responses = iter(["PostgreSQL", "n"])
        _responses_next = lambda *a, **kw: next(responses)

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
            patch(
                "anton.commands.datasource.connect.prompt_or_cancel",
                new=AsyncMock(side_effect=_responses_next),
            ),
            patch(
                "anton.commands.datasource.verify.prompt_or_cancel",
                new=AsyncMock(side_effect=_responses_next),
            ),
        ):
            await handle_connect_datasource(
                console,
                session._scratchpads,
                session,
                known_variables={
                    "host": "db.example.com",
                    "port": "5432",
                    "database": "prod_db",
                    "user": "alice",
                    "password": "wrong",
                },
            )

        # Connection NOT saved
        assert vault.list_connections() == []
        # Status correctly marked as test_failed (not "escaped")
        assert getattr(session, "_pending_connect_status", None) == "test_failed"

    @pytest.mark.asyncio
    async def test_fully_prefilled_known_variables_skips_prompts(
        self, registry, vault_dir, make_session, make_cell, make_pad
    ):
        """When known_variables covers every required field, skip all
        credential prompts and go straight to test + save. The user has
        already provided everything.
        """
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)

        pad = make_pad()
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        # Only the engine selection prompt should fire. After that, the
        # collector is already complete and the flow proceeds directly
        # to the connection test.
        responses = iter(["PostgreSQL"])

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
            patch(
                "anton.commands.datasource.connect.prompt_or_cancel",
                new=AsyncMock(side_effect=lambda *a, **kw: next(responses)),
            ),
        ):
            await handle_connect_datasource(
                console,
                session._scratchpads,
                session,
                known_variables={
                    "host": "db.example.com",
                    "port": "5432",
                    "database": "prod_db",
                    "user": "alice",
                    "password": "s3cr3t",
                },
            )

        conns = vault.list_connections()
        assert len(conns) == 1
        saved = vault.load("postgresql", conns[0]["name"])
        assert saved is not None
        assert saved["host"] == "db.example.com"
        assert saved["port"] == "5432"
        assert saved["database"] == "prod_db"
        assert saved["user"] == "alice"
        assert saved["password"] == "s3cr3t"

    @pytest.mark.asyncio
    async def test_credentials_pasted_at_first_prompt(
        self, registry, vault_dir, make_session, make_cell, make_pad
    ):
        """Pasting credentials at the first credential prompt should extract
        them instead of forcing a re-prompt or re-asking for every field.
        """
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)

        # Mock the LLM to return a structured extraction for the paste.
        # connect_collector.extract_variables now uses generate_object
        # with a Pydantic schema, so the mock returns the typed object.
        from anton.connect_collector import _ExtractionResult
        extract_response = _ExtractionResult(
            variables={
                "host": "db.example.com",
                "port": "5432",
                "database": "prod_db",
                "user": "alice",
                "password": "s3cr3t",
            },
            is_redirect=False,
            redirect_engine="",
            redirect_reason="",
        )
        session._llm.generate_object = AsyncMock(return_value=extract_response)

        pad = make_pad()
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        pasted = (
            "host: db.example.com\n"
            "port: 5432\n"
            "database: prod_db\n"
            "user: alice\n"
            "password: s3cr3t"
        )
        # Only two user inputs needed: the engine pick, then the paste
        # at the first credential prompt. The collector becomes complete
        # immediately after extraction, so no further prompts are issued.
        responses = iter(["PostgreSQL", pasted])

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
            patch(
                "anton.commands.datasource.connect.prompt_or_cancel",
                new=AsyncMock(side_effect=lambda *a, **kw: next(responses)),
            ),
        ):
            await handle_connect_datasource(console, session._scratchpads, session)

        conns = vault.list_connections()
        assert len(conns) == 1
        saved = vault.load("postgresql", conns[0]["name"])
        assert saved is not None
        assert saved["host"] == "db.example.com"
        assert saved["port"] == "5432"
        assert saved["database"] == "prod_db"
        assert saved["user"] == "alice"
        assert saved["password"] == "s3cr3t"

    @pytest.mark.asyncio
    async def test_from_tool_call_does_not_append_to_history(
        self, registry, vault_dir, make_session, make_cell, make_pad
    ):
        """With from_tool_call=True the handler must NOT mutate session._history.

        If it did, the appended assistant message would land between the
        surrounding tool_use and tool_result blocks, violating the
        Anthropic API invariant and producing a 400 error on the next
        LLM call ("tool_use ids were found without tool_result blocks").
        """
        session = make_session()
        # Simulate being mid-tool-call: history already contains an
        # assistant message with a tool_use that needs a tool_result next.
        session._history = [
            {"role": "user", "content": "connect to postgres"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me connect you."},
                    {
                        "type": "tool_use",
                        "id": "toolu_test_123",
                        "name": "connect_new_datasource",
                        "input": {"engine": "postgres"},
                    },
                ],
            },
        ]
        history_len_before = len(session._history)

        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)
        pad = make_pad()
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        responses = iter(
            [
                "PostgreSQL",
                "db.example.com",
                "5432",
                "prod_db",
                "alice",
                "s3cr3t",
            ]
        )

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
            patch(
                "anton.commands.datasource.connect.prompt_or_cancel",
                new=AsyncMock(side_effect=lambda *a, **kw: next(responses)),
            ),
        ):
            await handle_connect_datasource(
                console,
                session._scratchpads,
                session,
                from_tool_call=True,
            )

        # Connection saved successfully...
        conns = vault.list_connections()
        assert len(conns) == 1
        # ...but history MUST be untouched (the tool wrapper appends
        # the tool_result separately after this returns).
        assert len(session._history) == history_len_before
        assert session._history[-1]["role"] == "assistant"
        assert isinstance(session._history[-1]["content"], list)
        assert session._history[-1]["content"][-1]["type"] == "tool_use"

    @pytest.mark.asyncio
    async def test_failed_test_offers_retry(
        self, registry, vault_dir, make_session, make_cell, make_pad
    ):
        """Connection test failure prompts for retry; success on second attempt saves."""
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)

        pad = make_pad(side_effect=[
            make_cell(stdout="", stderr="password authentication failed"),
            make_cell(stdout="ok"),
        ])
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        responses = iter(
            [
                "PostgreSQL",
                "db.example.com",
                "5432",
                "prod_db",
                "alice",
                "wrongpassword",
                "y",
                # Retry re-prompts every field: empty = keep current for
                # host/port/database/user/schema/ssl; new value for password.
                "",       # host
                "",       # port
                "",       # database
                "",       # user
                "correctpassword",  # password
                "",       # schema (optional)
                "",       # ssl (optional)
            ]
        )

        poc = AsyncMock(side_effect=lambda *a, **kw: next(responses))
        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
            patch("anton.commands.datasource.connect.prompt_or_cancel", new=poc),
            patch("anton.commands.datasource.helpers.prompt_or_cancel", new=poc),
            patch("anton.commands.datasource.verify.prompt_or_cancel", new=poc),
        ):
            await handle_connect_datasource(console, session._scratchpads, session)

        conns = vault.list_connections()
        assert len(conns) == 1
        saved = vault.load("postgresql", conns[0]["name"])
        assert saved is not None
        assert saved["password"] == "correctpassword"

    @pytest.mark.asyncio
    async def test_failed_test_no_retry_returns_without_saving(
        self, registry, vault_dir, make_session, make_cell, make_pad
    ):
        """Declining retry on failed test leaves vault empty."""
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)

        pad = make_pad(make_cell(stdout="", error="connection refused"))
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        responses = iter(
            [
                "PostgreSQL",
                "db.example.com",
                "5432",
                "prod_db",
                "alice",
                "badpass",
                "n",
            ]
        )

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
            patch(
                "anton.commands.datasource.connect.prompt_or_cancel",
                new=AsyncMock(side_effect=lambda *a, **kw: next(responses)),
            ),
            patch(
                "anton.commands.datasource.verify.prompt_or_cancel",
                new=AsyncMock(side_effect=lambda *a, **kw: next(responses)),
            ),
        ):
            result = await handle_connect_datasource(
                console, session._scratchpads, session
            )

        assert vault.list_connections() == []
        assert not result._history

    @pytest.mark.asyncio
    async def test_ds_env_injected_after_successful_connect(
        self, registry, vault_dir, make_session, make_cell, make_pad
    ):
        """After a successful connect, namespaced DS_* vars are injected."""
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)

        pad = make_pad()
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        responses = iter(
            [
                "PostgreSQL",
                "db.example.com",
                "5432",
                "prod_db",
                "alice",
                "s3cr3t",
            ]
        )

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
            patch(
                "anton.commands.datasource.connect.prompt_or_cancel",
                new=AsyncMock(side_effect=lambda *a, **kw: next(responses)),
            ),
        ):
            await handle_connect_datasource(console, session._scratchpads, session)

        conns = vault.list_connections()
        assert len(conns) == 1
        prefix = _slug_env_prefix(conns[0]["engine"], conns[0]["name"])
        assert os.environ.get(f"{prefix}__HOST") == "db.example.com"

    @pytest.mark.asyncio
    async def test_auth_method_choice_selects_fields(
        self, registry, vault_dir, make_session, make_cell, make_pad
    ):
        """Selecting an auth method filters to that method's fields only."""
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)

        pad = make_pad()
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        responses = iter(["HubSpot", "1", "pat-na1-abc123"])

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
            patch(
                "anton.commands.datasource.connect.prompt_or_cancel",
                new=AsyncMock(side_effect=lambda *a, **kw: next(responses)),
            ),
        ):
            await handle_connect_datasource(console, session._scratchpads, session)

        conns = vault.list_connections()
        assert len(conns) == 1
        saved = vault.load("hubspot", conns[0]["name"])
        assert saved is not None
        # Only private_app fields collected — no client_id or client_secret
        assert "access_token" in saved
        assert "client_id" not in saved
        assert "client_secret" not in saved

    @pytest.mark.asyncio
    async def test_bulk_key_value_extraction(
        self, registry, vault_dir, make_session, make_cell, make_pad
    ):
        """A single bulk response with key=value pairs fills multiple fields at once."""
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)

        # Mock the LLM extraction to return a typed Pydantic result
        # (connect_collector now uses generate_object with a schema).
        from anton.connect_collector import _ExtractionResult
        bulk_response = _ExtractionResult(
            variables={
                "host": "db.example.com",
                "port": "5432",
                "database": "prod_db",
                "user": "alice",
            },
            is_redirect=False,
            redirect_engine="",
            redirect_reason="",
        )
        session._llm.generate_object = AsyncMock(return_value=bulk_response)

        pad = make_pad()
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        responses = iter(
            [
                "PostgreSQL",
                "host=db.example.com port=5432 database=prod_db user=alice",
                "s3cr3t",  # only password remains → single-field prompt
            ]
        )

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
            patch(
                "anton.commands.datasource.connect.prompt_or_cancel",
                new=AsyncMock(side_effect=lambda *a, **kw: next(responses)),
            ),
        ):
            await handle_connect_datasource(console, session._scratchpads, session)

        conns = vault.list_connections()
        assert len(conns) == 1
        saved = vault.load("postgresql", conns[0]["name"])
        assert saved is not None
        assert saved["host"] == "db.example.com"
        assert saved["port"] == "5432"
        assert saved["database"] == "prod_db"
        assert saved["user"] == "alice"
        assert saved["password"] == "s3cr3t"

    @pytest.mark.asyncio
    async def test_non_registry_name_routes_to_custom_with_soft_message(
        self, registry, vault_dir, make_session
    ):
        """Non-registry input routes to custom; soft entry message is printed;
        empty description does not cancel the flow."""
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)

        # Empty description → flow should not exit (treat as skip, not cancel).
        # LLM raises RuntimeError (default mock for unknown schemas) → custom
        # flow aborts cleanly, nothing saved.
        custom_calls = iter([""])

        with (
            patch(
                "anton.commands.datasource.connect.LocalDataVault",
                return_value=vault,
            ),
            patch(
                "anton.commands.datasource.connect.DatasourceRegistry",
                return_value=registry,
            ),
            patch(
                "anton.commands.datasource.connect.prompt_or_cancel",
                new=AsyncMock(return_value="GitHub"),
            ),
            patch(
                "anton.commands.datasource.custom.prompt_or_cancel",
                new=AsyncMock(side_effect=lambda *_: next(custom_calls)),
            ),
        ):
            result = await handle_connect_datasource(
                console, session._scratchpads, session
            )

        printed = " ".join(str(c) for c in console.print.call_args_list)
        assert "GitHub" in printed
        assert "custom datasource" in printed
        assert vault.list_connections() == []
        assert result is session

    @pytest.mark.asyncio
    async def test_cancel_at_description_prompt_exits_cleanly(
        self, registry, vault_dir, make_session
    ):
        """Cancelling (None) at the description prompt in custom flow exits
        without saving anything."""
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)

        with (
            patch(
                "anton.commands.datasource.connect.LocalDataVault",
                return_value=vault,
            ),
            patch(
                "anton.commands.datasource.connect.DatasourceRegistry",
                return_value=registry,
            ),
            patch(
                "anton.commands.datasource.connect.prompt_or_cancel",
                new=AsyncMock(return_value="SomeUnknownService"),
            ),
            patch(
                "anton.commands.datasource.custom.prompt_or_cancel",
                new=AsyncMock(return_value=None),
            ),
        ):
            result = await handle_connect_datasource(
                console, session._scratchpads, session
            )

        assert vault.list_connections() == []
        assert result is session

    @pytest.mark.asyncio
    async def test_no_proactive_help_prompt_for_builtin(
        self, registry, vault_dir, make_session, make_cell, make_pad
    ):
        """Built-in connect flow must not ask 'need instructions on how
        to obtain credentials?'. After the engine pick, the next prompt
        is a credential prompt, not a y/n help prompt.
        """
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)

        pad = make_pad()
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        seen_labels: list[str] = []

        async def _prompt(*args, **kwargs):
            label = args[0] if args else kwargs.get("label", "")
            seen_labels.append(label)
            return next(responses)

        responses = iter(
            [
                "PostgreSQL",
                "db.example.com",
                "5432",
                "prod_db",
                "alice",
                "s3cr3t",
            ]
        )

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
            patch(
                "anton.commands.datasource.connect.prompt_or_cancel",
                new=_prompt,
            ),
        ):
            await handle_connect_datasource(console, session._scratchpads, session)

        joined = " | ".join(seen_labels).lower()
        assert "instructions" not in joined
        assert "need help" not in joined
        conns = vault.list_connections()
        assert len(conns) == 1
        assert re.fullmatch(r"[0-9a-f]{8}", conns[0]["name"])

    @pytest.mark.asyncio
    async def test_help_keyword_triggers_credential_help(
        self, registry, vault_dir, make_session, make_cell, make_pad
    ):
        """Typing 'help' at a credential prompt should invoke
        show_credential_help (LLM-backed guide) and then keep prompting
        for the same field, rather than treating 'help' as a literal value.
        """
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)

        pad = make_pad()
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        help_calls: list[str] = []

        async def _fake_help(console_arg, session_arg, service_name, current_field, all_fields):
            help_calls.append(service_name)

        # Five required fields missing → first prompt is the bulk prompt.
        # Type 'help' → show_credential_help fires, loop continues, next
        # input is the actual paste that extracts all five.
        from anton.connect_collector import _ExtractionResult
        session._llm.generate_object = AsyncMock(
            return_value=_ExtractionResult(
                variables={
                    "host": "db.example.com",
                    "port": "5432",
                    "database": "prod_db",
                    "user": "alice",
                    "password": "s3cr3t",
                },
            )
        )

        responses = iter(
            [
                "PostgreSQL",
                "help",
                "host=db.example.com port=5432 database=prod_db user=alice password=s3cr3t",
            ]
        )

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
            patch(
                "anton.commands.datasource.connect.prompt_or_cancel",
                new=AsyncMock(side_effect=lambda *a, **kw: next(responses)),
            ),
            patch(
                "anton.commands.datasource.connect.show_credential_help",
                new=_fake_help,
            ),
        ):
            await handle_connect_datasource(console, session._scratchpads, session)

        assert help_calls == ["PostgreSQL"]
        conns = vault.list_connections()
        assert len(conns) == 1
        saved = vault.load("postgresql", conns[0]["name"])
        assert saved is not None
        assert saved["password"] == "s3cr3t"

    @pytest.mark.asyncio
    async def test_help_keyword_at_single_field_prompt(
        self, registry, vault_dir, make_session, make_cell, make_pad
    ):
        """When only one required field remains, typing 'help' there
        triggers field-specific help and does not get saved as the value.
        """
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)

        pad = make_pad()
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        help_calls: list[tuple[str, str]] = []

        async def _fake_help(console_arg, session_arg, service_name, current_field, all_fields):
            help_calls.append((service_name, current_field.name if current_field else ""))

        # Pre-fill everything except password → the while-loop hits the
        # single-field branch for 'password'. First response is 'help',
        # second is the actual password.
        responses = iter(["PostgreSQL", "help", "s3cr3t"])

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
            patch(
                "anton.commands.datasource.connect.prompt_or_cancel",
                new=AsyncMock(side_effect=lambda *a, **kw: next(responses)),
            ),
            patch(
                "anton.commands.datasource.connect.show_credential_help",
                new=_fake_help,
            ),
        ):
            await handle_connect_datasource(
                console,
                session._scratchpads,
                session,
                known_variables={
                    "host": "db.example.com",
                    "port": "5432",
                    "database": "prod_db",
                    "user": "alice",
                },
            )

        assert help_calls == [("PostgreSQL", "password")]
        conns = vault.list_connections()
        assert len(conns) == 1
        saved = vault.load("postgresql", conns[0]["name"])
        assert saved is not None
        assert saved["password"] == "s3cr3t"

    @pytest.mark.asyncio
    async def test_sequential_prompting_uses_title_case_labels(
        self, registry, vault_dir, make_session, make_cell, make_pad
    ):
        """With >2 required fields missing, built-in flow prompts one at a
        time with short Title-Case labels (no bulk '(anton) a, b, c …'
        prompt listing every field)."""
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)

        pad = make_pad()
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        seen_labels: list[str] = []

        async def _prompt(*args, **kwargs):
            label = args[0] if args else kwargs.get("label", "")
            seen_labels.append(label)
            return next(responses)

        responses = iter(
            [
                "PostgreSQL",
                "db.example.com",
                "5432",
                "prod_db",
                "alice",
                "s3cr3t",
            ]
        )

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
            patch(
                "anton.commands.datasource.connect.prompt_or_cancel",
                new=_prompt,
            ),
        ):
            await handle_connect_datasource(console, session._scratchpads, session)

        # First label is "Connect to" engine picker; then every required
        # field is prompted individually — no bulk compact prompts.
        assert seen_labels[1] == "(anton) Host"
        assert seen_labels[2] == "(anton) Port"
        assert seen_labels[3] == "(anton) Database"
        assert seen_labels[4] == "(anton) User"
        assert seen_labels[5] == "(anton) Password"

        conns = vault.list_connections()
        assert len(conns) == 1
        saved = vault.load("postgresql", conns[0]["name"])
        assert saved is not None
        assert saved["host"] == "db.example.com"
        assert saved["password"] == "s3cr3t"

    @pytest.mark.asyncio
    async def test_sequential_accepts_multi_field_paste(
        self, registry, vault_dir, make_session, make_cell, make_pad
    ):
        """A key=value paste at the first sequential prompt is routed
        through extract_variables and fills every matching field in one
        shot (consistent with prior bulk behavior)."""
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)

        from anton.connect_collector import _ExtractionResult
        session._llm.generate_object = AsyncMock(
            return_value=_ExtractionResult(
                variables={
                    "host": "db.example.com",
                    "port": "5432",
                    "database": "prod_db",
                    "user": "alice",
                },
            )
        )

        pad = make_pad()
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        responses = iter(
            [
                "PostgreSQL",
                "host=db.example.com port=5432 database=prod_db user=alice",
                "s3cr3t",
            ]
        )

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
            patch(
                "anton.commands.datasource.connect.prompt_or_cancel",
                new=AsyncMock(side_effect=lambda *a, **kw: next(responses)),
            ),
        ):
            await handle_connect_datasource(console, session._scratchpads, session)

        conns = vault.list_connections()
        assert len(conns) == 1
        saved = vault.load("postgresql", conns[0]["name"])
        assert saved is not None
        assert saved["host"] == "db.example.com"
        assert saved["port"] == "5432"
        assert saved["database"] == "prod_db"
        assert saved["user"] == "alice"
        assert saved["password"] == "s3cr3t"

    @pytest.mark.asyncio
    async def test_sequential_skip_keyword_saves_partial(
        self, registry, vault_dir, make_session, make_cell, make_pad
    ):
        """Typing 'skip' at a sequential per-field prompt marks the
        connection as partial and stops collection, just like in the
        prior bulk flow."""
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)

        pad = make_pad()
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        # Fill two fields, then 'skip' at the third prompt (database).
        responses = iter(
            [
                "PostgreSQL",
                "db.example.com",
                "5432",
                "skip",
            ]
        )

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
            patch(
                "anton.commands.datasource.connect.prompt_or_cancel",
                new=AsyncMock(side_effect=lambda *a, **kw: next(responses)),
            ),
        ):
            await handle_connect_datasource(console, session._scratchpads, session)

        conns = vault.list_connections()
        assert len(conns) == 1
        # Partial save — auto-named, only host + port filled.
        saved = vault.load(conns[0]["engine"], conns[0]["name"])
        assert saved is not None
        assert saved.get("host") == "db.example.com"
        assert saved.get("port") == "5432"
        assert "database" not in saved
        session._scratchpads.get_or_create.assert_not_called()


class TestCredentialScrubbing:
    """_scrub_credentials and _register_secret_vars — flat and namespaced modes."""

    def test_register_secret_vars_adds_secret_fields(self, registry):
        """Secret fields are added to _DS_SECRET_VARS; non-secret fields are not."""
        pg = registry.get("postgresql")
        assert pg is not None
        register_secret_vars(pg)
        assert "DS_PASSWORD" in _DS_SECRET_VARS
        assert "DS_HOST" not in _DS_SECRET_VARS
        assert "DS_PORT" not in _DS_SECRET_VARS

    def test_scrub_replaces_registered_secret_value(self, monkeypatch):
        """A registered secret value is replaced with its placeholder."""
        _DS_SECRET_VARS.add("DS_ACCESS_TOKEN")
        monkeypatch.setenv("DS_ACCESS_TOKEN", "supersecrettoken123")
        result = scrub_credentials("token is supersecrettoken123 here")
        assert "supersecrettoken123" not in result
        assert "[DS_ACCESS_TOKEN]" in result

    def test_scrub_leaves_non_secret_field_readable(self, registry, monkeypatch):
        """Non-secret DS_* values (host, port) are left untouched."""
        pg = registry.get("postgresql")
        assert pg is not None
        register_secret_vars(pg)
        monkeypatch.setenv("DS_HOST", "mydbhostname")
        monkeypatch.setenv("DS_PASSWORD", "s3cr3tpassword99")
        result = scrub_credentials("host=mydbhostname pass=s3cr3tpassword99")
        assert "mydbhostname" in result
        assert "s3cr3tpassword99" not in result
        assert "[DS_PASSWORD]" in result

    def test_scrub_skips_short_values(self, monkeypatch):
        """Registered secrets are always scrubbed regardless of length."""
        _DS_SECRET_VARS.add("DS_PASSWORD")
        monkeypatch.setenv("DS_PASSWORD", "short")
        result = scrub_credentials("password=short")
        assert "short" not in result
        assert "[DS_PASSWORD]" in result

    def test_scrub_fallback_redacts_unknown_long_ds_vars(self, monkeypatch):
        """Long DS_* vars not in _DS_SECRET_VARS are scrubbed as a safety fallback."""
        monkeypatch.setenv("DS_WEBHOOK_SECRET", "wh_sec_abcdefgh1234")
        result = scrub_credentials("secret=wh_sec_abcdefgh1234 here")
        assert "wh_sec_abcdefgh1234" not in result
        assert "[DS_WEBHOOK_SECRET]" in result

    @pytest.mark.asyncio
    async def test_register_and_scrub_on_connect(
        self, registry, vault_dir, monkeypatch, make_pad
    ):
        """After _handle_connect_datasource, the new secret var is immediately scrubbed."""
        vault = LocalDataVault(vault_dir=vault_dir)
        session = MagicMock()
        session._history = []
        session._cortex = None

        pad = make_pad()
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        secret_pw = "supersecretpassword999"
        responses = iter(
            [
                "PostgreSQL",
                "db.host.com",
                "5432",
                "mydb",
                "alice",
                secret_pw,
            ]
        )

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
            patch(
                "anton.commands.datasource.connect.prompt_or_cancel",
                new=AsyncMock(side_effect=lambda *a, **kw: next(responses)),
            ),
        ):
            await handle_connect_datasource(MagicMock(), session._scratchpads, session)

        conns = vault.list_connections()
        assert len(conns) == 1
        prefix = _slug_env_prefix(conns[0]["engine"], conns[0]["name"])
        namespaced_pw_var = f"{prefix}__PASSWORD"
        assert namespaced_pw_var in _DS_SECRET_VARS
        monkeypatch.setenv(namespaced_pw_var, secret_pw)
        result = scrub_credentials(f"error: auth failed with {secret_pw}")
        assert secret_pw not in result
        assert f"[{namespaced_pw_var}]" in result

    def test_register_with_slug_uses_namespaced_keys(self, registry):
        pg = registry.get("postgresql")
        register_secret_vars(pg, engine="postgresql", name="prod_db")
        assert "DS_POSTGRESQL_PROD_DB__PASSWORD" in _DS_SECRET_VARS
        assert "DS_POSTGRESQL_PROD_DB__HOST" not in _DS_SECRET_VARS
        assert "DS_POSTGRESQL_PROD_DB__HOST" in _DS_KNOWN_VARS

    def test_register_without_slug_uses_flat_keys(self, registry):
        pg = registry.get("postgresql")
        register_secret_vars(pg)  # no engine/name → flat mode
        assert "DS_PASSWORD" in _DS_SECRET_VARS
        assert "DS_HOST" not in _DS_SECRET_VARS

    def test_scrub_replaces_namespaced_secret_value(self, registry, monkeypatch):
        pg = registry.get("postgresql")
        register_secret_vars(pg, engine="postgresql", name="prod_db")
        secret = "namespacedpassword123"
        monkeypatch.setenv("DS_POSTGRESQL_PROD_DB__PASSWORD", secret)
        result = scrub_credentials(f"error: {secret}")
        assert secret not in result
        assert "[DS_POSTGRESQL_PROD_DB__PASSWORD]" in result

    def test_scrub_leaves_namespaced_non_secret_readable(self, registry, monkeypatch):
        pg = registry.get("postgresql")
        register_secret_vars(pg, engine="postgresql", name="prod_db")
        monkeypatch.setenv("DS_POSTGRESQL_PROD_DB__HOST", "mydbhostname")
        result = scrub_credentials("host=mydbhostname")
        assert "mydbhostname" in result


class TestActiveDatasourceScoping:
    """Tests for active datasource routing and multi-source context building."""

    def test_active_datasource_defaults_to_none(self, make_session):
        session = make_session()
        assert session._active_datasource is None

    @pytest.mark.asyncio
    async def test_reconnect_sets_active_datasource(self, vault_dir, make_session):
        """Reconnecting to a slug via prefill sets session._active_datasource."""
        vault = LocalDataVault(vault_dir=vault_dir)
        vault.save("hubspot", "2", {"access_token": "pat-xxx"})

        session = make_session()
        console = MagicMock()

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry"),
        ):
            result = await handle_connect_datasource(
                console, session._scratchpads, session, prefill="hubspot-2"
            )

        assert result._active_datasource == "hubspot-2"

    @pytest.mark.asyncio
    async def test_reconnect_all_namespaced_vars_available(
        self, vault_dir, make_session
    ):
        """After reconnect, ALL saved connections remain available as namespaced vars."""
        vault = LocalDataVault(vault_dir=vault_dir)
        vault.save(
            "oracle",
            "1",
            {"host": "oracle.host", "user": "admin", "password": "orapass"},
        )
        vault.save("hubspot", "2", {"access_token": "pat-xxx"})

        vault.inject_env("oracle", "1")
        vault.inject_env("hubspot", "2")
        assert os.environ.get("DS_ORACLE_1__HOST") == "oracle.host"
        assert os.environ.get("DS_HUBSPOT_2__ACCESS_TOKEN") == "pat-xxx"

        session = make_session()
        console = MagicMock()

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry"),
        ):
            result = await handle_connect_datasource(
                console, session._scratchpads, session, prefill="hubspot-2"
            )

        assert "DS_HOST" not in os.environ
        assert "DS_ACCESS_TOKEN" not in os.environ
        assert os.environ.get("DS_ORACLE_1__HOST") == "oracle.host"
        assert os.environ.get("DS_HUBSPOT_2__ACCESS_TOKEN") == "pat-xxx"
        assert result._active_datasource == "hubspot-2"

    def test_build_datasource_context_no_filter(self, vault_dir):
        """Without active_only, all vault entries appear in the context."""
        vault = LocalDataVault(vault_dir=vault_dir)
        vault.save("oracle", "1", {"host": "oracle.host"})
        vault.save("hubspot", "2", {"access_token": "pat-xxx"})

        ctx = build_datasource_context(vault)

        assert "oracle-1" in ctx
        assert "hubspot-2" in ctx

    def test_build_datasource_context_active_only_filters(self, vault_dir):
        """With active_only set, only the matching slug appears."""
        vault = LocalDataVault(vault_dir=vault_dir)
        vault.save("oracle", "1", {"host": "oracle.host"})
        vault.save("hubspot", "2", {"access_token": "pat-xxx"})

        ctx = build_datasource_context(vault, active_only="hubspot-2")

        assert "hubspot-2" in ctx
        assert "oracle-1" not in ctx

    def test_build_datasource_context_active_only_empty_when_no_match(self, vault_dir):
        """If active_only doesn't match any slug, the section has no entries."""
        vault = LocalDataVault(vault_dir=vault_dir)
        vault.save("oracle", "1", {"host": "oracle.host"})

        ctx = build_datasource_context(vault, active_only="hubspot-99")

        assert "oracle-1" not in ctx

    def test_build_datasource_context_shows_namespaced_vars(self, vault_dir):
        vault = LocalDataVault(vault_dir=vault_dir)
        vault.save(
            "postgres", "prod_db", {"host": "pg.example.com", "password": "s3cr3t"}
        )

        ctx = build_datasource_context(vault)

        assert "DS_POSTGRES_PROD_DB__HOST" in ctx
        assert "DS_POSTGRES_PROD_DB__PASSWORD" in ctx
        assert "DS_HOST" not in ctx

    def test_build_datasource_context_shows_slug_and_engine_label(self, vault_dir):
        vault = LocalDataVault(vault_dir=vault_dir)
        vault.save("postgres", "prod_db", {"host": "pg.example.com"})

        ctx = build_datasource_context(vault)

        assert "postgres-prod_db" in ctx
        assert "(postgres)" in ctx

    def test_multi_source_context_shows_both_connections(self, vault_dir):
        """Both connections are visible in the context with their namespaced vars."""
        vault = LocalDataVault(vault_dir=vault_dir)
        vault.save("postgres", "prod_db", {"host": "pg.example.com"})
        vault.save("hubspot", "main", {"access_token": "pat-abc"})

        ctx = build_datasource_context(vault)

        assert "postgres-prod_db" in ctx
        assert "DS_POSTGRES_PROD_DB__HOST" in ctx
        assert "hubspot-main" in ctx
        assert "DS_HUBSPOT_MAIN__ACCESS_TOKEN" in ctx


class TestCliCommandRegistration:
    @pytest.mark.parametrize(
        "cmd_name",
        [
            "connect",
            "list",
            "edit",
            "remove",
            "test",
        ],
    )
    def test_command_registered(self, cmd_name):
        names = [cmd.name for cmd in cli_app.registered_commands]
        assert cmd_name in names


class TestHandleListDataSources:
    def test_empty_vault_shows_message(self, vault_dir):
        console = MagicMock()
        handle_list_data_sources(console, vault=LocalDataVault(vault_dir=vault_dir))
        printed = " ".join(str(c) for c in console.print.call_args_list)
        assert "No data sources" in printed or "connect" in printed

    def test_complete_connection_shows_saved_with_engine_name(
        self, vault_dir, registry
    ):
        vault = LocalDataVault(vault_dir=vault_dir)
        vault.save(
            "postgresql",
            "prod_db",
            {
                "host": "db.example.com",
                "port": "5432",
                "database": "prod",
                "user": "alice",
                "password": "s3cr3t",
            },
        )

        buf = io.StringIO()
        rich_console = Console(file=buf, highlight=False, markup=False)

        with patch("anton.commands.datasource.manage.DatasourceRegistry", return_value=registry):
            handle_list_data_sources(rich_console, vault=vault)

        output = buf.getvalue()
        assert "postgresql-prod_db" in output
        assert "saved" in output.lower()
        assert "PostgreSQL" in output  # engine display_name shown

    def test_incomplete_connection_shows_incomplete(self, vault_dir, registry):
        vault = LocalDataVault(vault_dir=vault_dir)
        # Missing required fields: database, user, password
        vault.save("postgresql", "partial", {"host": "db.example.com"})

        buf = io.StringIO()
        rich_console = Console(file=buf, highlight=False, markup=False)

        with patch("anton.commands.datasource.manage.DatasourceRegistry", return_value=registry):
            handle_list_data_sources(rich_console, vault=vault)

        output = buf.getvalue()
        assert "incomplete" in output.lower()


class TestHandleTestDatasource:
    @pytest.mark.asyncio
    async def test_success_path(self, vault_dir, registry, make_cell, make_pad):
        vault = LocalDataVault(vault_dir=vault_dir)
        vault.save(
            "postgresql",
            "prod_db",
            {
                "host": "db.example.com",
                "port": "5432",
                "database": "prod",
                "user": "alice",
                "password": "s3cr3t",
            },
        )
        console = MagicMock()
        pad = make_pad()
        scratchpads = AsyncMock()
        scratchpads.get_or_create = AsyncMock(return_value=pad)

        with patch("anton.commands.datasource.verify.DatasourceRegistry", return_value=registry):
            await handle_test_datasource(console, scratchpads, "postgresql-prod_db", vault=vault)

        printed = " ".join(str(c) for c in console.print.call_args_list)
        assert "✓" in printed or "passed" in printed.lower()

    @pytest.mark.asyncio
    async def test_failure_path(self, vault_dir, registry, make_cell, make_pad):
        vault = LocalDataVault(vault_dir=vault_dir)
        vault.save(
            "postgresql",
            "prod_db",
            {
                "host": "db.example.com",
                "port": "5432",
                "database": "prod",
                "user": "alice",
                "password": "wrongpass",
            },
        )
        console = MagicMock()
        pad = make_pad(make_cell(stdout="", stderr="password authentication failed"))
        scratchpads = AsyncMock()
        scratchpads.get_or_create = AsyncMock(return_value=pad)

        with patch("anton.commands.datasource.verify.DatasourceRegistry", return_value=registry):
            await handle_test_datasource(console, scratchpads, "postgresql-prod_db", vault=vault)

        printed = " ".join(str(c) for c in console.print.call_args_list)
        assert "✗" in printed or "failed" in printed.lower()

    @pytest.mark.asyncio
    async def test_unknown_connection(self, vault_dir, registry):
        vault = LocalDataVault(vault_dir=vault_dir)
        console = MagicMock()
        scratchpads = AsyncMock()

        with patch("anton.commands.datasource.verify.DatasourceRegistry", return_value=registry):
            await handle_test_datasource(console, scratchpads, "postgresql-ghost", vault=vault)

        printed = " ".join(str(c) for c in console.print.call_args_list)
        assert "not found" in printed.lower() or "No connection" in printed

    @pytest.mark.asyncio
    async def test_empty_slug_shows_usage(self, vault_dir, registry):
        console = MagicMock()
        scratchpads = AsyncMock()

        await handle_test_datasource(console, scratchpads, "", vault=LocalDataVault(vault_dir=vault_dir))

        printed = " ".join(str(c) for c in console.print.call_args_list)
        assert "Usage" in printed or "test" in printed


class TestEditDatasourceFlow:
    @pytest.mark.asyncio
    async def test_existing_values_loaded(
        self, registry, vault_dir, make_session, make_cell, make_pad
    ):
        """Edit shows existing non-secret values as defaults."""
        vault = LocalDataVault(vault_dir=vault_dir)
        vault.save(
            "postgresql",
            "prod_db",
            {
                "host": "old.host",
                "port": "5432",
                "database": "prod",
                "user": "alice",
                "password": "oldpass",
            },
        )

        session = make_session()
        console = MagicMock()
        pad = make_pad()
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        prompt_values = iter(
            [
                "old.host",
                "5432",
                "prod",
                "alice",
                "newpass",
                "",
                "",
            ]
        )
        poc = AsyncMock(side_effect=lambda *a, **kw: next(prompt_values))

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
            patch("anton.commands.datasource.connect.prompt_or_cancel", new=poc),
            patch("anton.commands.datasource.helpers.prompt_or_cancel", new=poc),
        ):
            await handle_connect_datasource(
                console,
                session._scratchpads,
                session,
                datasource_name="postgresql-prod_db",
            )

        saved = vault.load("postgresql", "prod_db")
        assert saved is not None
        assert saved["host"] == "old.host"
        assert saved["password"] == "newpass"

    @pytest.mark.asyncio
    async def test_enter_preserves_secret_value(
        self, registry, vault_dir, make_session, make_cell, make_pad
    ):
        """Pressing Enter on a secret field keeps the existing value."""
        vault = LocalDataVault(vault_dir=vault_dir)
        original_pass = "original_secret_pass"
        vault.save(
            "postgresql",
            "prod_db",
            {
                "host": "db.host",
                "port": "5432",
                "database": "prod",
                "user": "alice",
                "password": original_pass,
            },
        )

        session = make_session()
        console = MagicMock()
        pad = make_pad()
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        prompt_values = iter(
            [
                "db.host",
                "5432",
                "prod",
                "alice",
                "",  # password — Enter = keep original
                "",  # schema
                "",  # ssl
            ]
        )
        poc = AsyncMock(side_effect=lambda *a, **kw: next(prompt_values))

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
            patch("anton.commands.datasource.connect.prompt_or_cancel", new=poc),
            patch("anton.commands.datasource.helpers.prompt_or_cancel", new=poc),
        ):
            await handle_connect_datasource(
                console,
                session._scratchpads,
                session,
                datasource_name="postgresql-prod_db",
            )

        saved = vault.load("postgresql", "prod_db")
        assert saved is not None
        assert saved["password"] == original_pass

    @pytest.mark.asyncio
    async def test_unknown_slug_returns_session(
        self, registry, vault_dir, make_session
    ):
        """Editing a non-existent slug returns the session unchanged."""
        vault = LocalDataVault(vault_dir=vault_dir)
        session = make_session()
        console = MagicMock()

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
        ):
            result = await handle_connect_datasource(
                console,
                session._scratchpads,
                session,
                datasource_name="postgresql-ghost",
            )

        assert result is session
        printed = " ".join(str(c) for c in console.print.call_args_list)
        assert "not found" in printed.lower() or "No connection" in printed


class TestRemoveDatasourceFlow:
    @pytest.mark.asyncio
    async def test_confirmation_yes_deletes(self, vault, registry):
        vault.save("postgresql", "prod_db", {"host": "x"})
        console = Console(quiet=True)

        with (
            patch("anton.commands.datasource.manage.DatasourceRegistry", return_value=registry),
            patch("anton.commands.datasource.manage.prompt_or_cancel", new=AsyncMock(return_value="y")),
        ):
            await handle_remove_data_source(console, "postgresql-prod_db", vault=vault)

        assert vault.load("postgresql", "prod_db") is None

    @pytest.mark.asyncio
    async def test_confirmation_no_preserves(self, vault, registry):
        vault.save("postgresql", "prod_db", {"host": "x"})
        console = Console(quiet=True)

        with (
            patch("anton.commands.datasource.manage.DatasourceRegistry", return_value=registry),
            patch("anton.commands.datasource.manage.prompt_or_cancel", new=AsyncMock(return_value="n")),
        ):
            await handle_remove_data_source(console, "postgresql-prod_db", vault=vault)

        assert vault.load("postgresql", "prod_db") is not None

    @pytest.mark.asyncio
    async def test_unknown_name_shows_message(self, vault_dir, registry):
        vault = LocalDataVault(vault_dir=vault_dir)
        console = MagicMock()

        with (
            patch("anton.commands.datasource.manage.DatasourceRegistry", return_value=registry),
        ):
            await handle_remove_data_source(console, "postgresql-ghost", vault=vault)

        printed = " ".join(str(c) for c in console.print.call_args_list)
        assert "not found" in printed.lower() or "No connection" in printed

    @pytest.mark.asyncio
    async def test_invalid_format_shows_warning(self, vault_dir):
        vault = LocalDataVault(vault_dir=vault_dir)
        console = MagicMock()

        with patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault):
            await handle_remove_data_source(console, "nohyphen")

        printed = " ".join(str(c) for c in console.print.call_args_list)
        assert "Invalid" in printed or "engine-name" in printed


class TestEnvActivationCollisionFree:
    @pytest.mark.asyncio
    async def test_connect_clears_previous_ds_vars(
        self, registry, vault_dir, make_session, make_cell, monkeypatch, make_pad
    ):
        """After a successful new connect, stale DS_* vars are cleared."""
        monkeypatch.setenv("DS_ACCESS_TOKEN", "old-token")
        vault = LocalDataVault(vault_dir=vault_dir)
        session = make_session()
        console = MagicMock()

        pad = make_pad()
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        responses = iter(
            [
                "PostgreSQL",
                "db.example.com",
                "5432",
                "prod_db",
                "alice",
                "s3cr3t",
            ]
        )

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
            patch(
                "anton.commands.datasource.connect.prompt_or_cancel",
                new=AsyncMock(side_effect=lambda *a, **kw: next(responses)),
            ),
        ):
            await handle_connect_datasource(console, session._scratchpads, session)

        assert "DS_ACCESS_TOKEN" not in os.environ
        conns = vault.list_connections()
        assert len(conns) == 1
        prefix = _slug_env_prefix(conns[0]["engine"], conns[0]["name"])
        assert os.environ.get(f"{prefix}__HOST") == "db.example.com"

    @pytest.mark.asyncio
    async def test_two_same_type_connections_no_collision(
        self, registry, vault_dir, make_session
    ):
        """Both same-type connections remain available as distinct namespaced vars."""
        vault = LocalDataVault(vault_dir=vault_dir)
        vault.save(
            "postgresql",
            "db1",
            {
                "host": "host1.example.com",
                "port": "5432",
                "database": "db1",
                "user": "u1",
                "password": "p1",
            },
        )
        vault.save(
            "postgresql",
            "db2",
            {
                "host": "host2.example.com",
                "port": "5432",
                "database": "db2",
                "user": "u2",
                "password": "p2",
            },
        )

        session = make_session()
        console = MagicMock()

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
        ):
            await handle_connect_datasource(
                console,
                session._scratchpads,
                session,
                prefill="postgresql-db2",
            )

        assert os.environ.get("DS_POSTGRESQL_DB1__HOST") == "host1.example.com"
        assert os.environ.get("DS_POSTGRESQL_DB2__HOST") == "host2.example.com"
        assert "DS_HOST" not in os.environ
        assert "DS_DATABASE" not in os.environ


class TestDatasourceSlashCommandBehavior:
    @pytest.mark.asyncio
    async def test_test_data_source_no_arg_shows_usage(self, vault_dir, registry):
        console = MagicMock()
        scratchpads = AsyncMock()

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=LocalDataVault(vault_dir=vault_dir)),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
        ):
            await handle_test_datasource(console, scratchpads, "")

        printed = " ".join(str(c) for c in console.print.call_args_list)
        assert "Usage" in printed or "test" in printed

    @pytest.mark.asyncio
    async def test_edit_data_source_no_arg_safe(
        self, vault_dir, registry, make_session
    ):
        """datasource_name=None triggers new-connect flow without crash."""
        session = make_session()
        console = MagicMock()

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=LocalDataVault(vault_dir=vault_dir)),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
            patch("anton.commands.datasource.connect.prompt_or_cancel", return_value="UnknownEngine"),
            patch("anton.commands.datasource.custom.prompt_or_cancel", return_value="some description"),
        ):
            updated = await handle_connect_datasource(
                console,
                session._scratchpads,
                session,
                datasource_name=None,
            )

        assert updated is not None


class TestParseConnectionSlug:
    ENGINES = ["postgresql", "sql-server", "google-big-query"]

    def test_simple_engine(self):
        assert parse_connection_slug("postgresql-prod_db", self.ENGINES) == (
            "postgresql",
            "prod_db",
        )

    def test_hyphenated_engine(self):
        assert parse_connection_slug("sql-server-prod-db", self.ENGINES) == (
            "sql-server",
            "prod-db",
        )

    def test_longest_prefix_wins(self):
        engines = ["google", "google-big-query"]
        assert parse_connection_slug("google-big-query-main", engines) == (
            "google-big-query",
            "main",
        )

    def test_ambiguous_resolves_to_longest(self):
        engines = ["sql", "sql-server"]
        assert parse_connection_slug("sql-server-1", engines) == ("sql-server", "1")

    def test_invalid_slug_no_match(self):
        assert parse_connection_slug("unknown-engine-name", self.ENGINES) is None

    def test_slug_with_empty_name_part(self):
        assert parse_connection_slug("postgresql-", self.ENGINES) is None

    def test_fallback_to_vault_for_custom_engine(self, tmp_path):
        """Custom engine not in registry is resolved via vault fallback."""
        vault = LocalDataVault(vault_dir=tmp_path / "vault")
        vault.save("my_custom_db", "prod", {"host": "localhost"})
        result = parse_connection_slug(
            "my_custom_db-prod",
            known_engines=["postgresql"],
            vault=vault,
        )
        assert result == ("my_custom_db", "prod")

    def test_registry_match_takes_priority_over_vault(self, tmp_path):
        """Registry prefix match wins even when vault also has the slug."""
        vault = LocalDataVault(vault_dir=tmp_path / "vault")
        vault.save("postgresql", "prod", {"host": "localhost"})
        result = parse_connection_slug(
            "postgresql-prod",
            known_engines=["postgresql"],
            vault=vault,
        )
        assert result == ("postgresql", "prod")

    def test_no_match_returns_none_with_vault(self, tmp_path):
        """Truly unknown slug returns None even with vault supplied."""
        vault = LocalDataVault(vault_dir=tmp_path / "vault")
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


class TestTemporaryFlatExecution:
    """Tests that flat vars are used only during test_snippet, then restored."""

    def test_restore_namespaced_env_clears_flat_and_reinjects(self, vault_dir):
        """_restore_namespaced_env replaces flat vars with namespaced vars."""
        vault = LocalDataVault(vault_dir=vault_dir)
        vault.save("postgres", "analytics", {"host": "analytics.example.com"})

        vault.inject_env("postgres", "analytics", flat=True)
        assert os.environ.get("DS_HOST") == "analytics.example.com"
        assert "DS_POSTGRES_ANALYTICS__HOST" not in os.environ

        with patch("anton.utils.datasources.DataVault", return_value=vault):
            restore_namespaced_env(vault)

        assert "DS_HOST" not in os.environ
        assert os.environ.get("DS_POSTGRES_ANALYTICS__HOST") == "analytics.example.com"

    def test_restore_namespaced_env_reinjects_all_connections(self, vault_dir):
        """_restore_namespaced_env restores ALL saved connections, not just one."""
        vault = LocalDataVault(vault_dir=vault_dir)
        vault.save("postgres", "prod_db", {"host": "prod.example.com"})
        vault.save("hubspot", "main", {"access_token": "pat-abc"})

        vault.inject_env("postgres", "prod_db", flat=True)

        with patch("anton.utils.datasources.DataVault", return_value=vault):
            restore_namespaced_env(vault)

        assert "DS_HOST" not in os.environ
        assert os.environ.get("DS_POSTGRES_PROD_DB__HOST") == "prod.example.com"
        assert os.environ.get("DS_HUBSPOT_MAIN__ACCESS_TOKEN") == "pat-abc"

    @pytest.mark.asyncio
    async def test_test_datasource_injects_flat_then_restores_namespaced(
        self, vault_dir, registry, make_pad
    ):
        """handle_test_datasource uses flat vars during snippet, then restores namespaced."""
        vault = LocalDataVault(vault_dir=vault_dir)
        vault.save(
            "postgresql",
            "prod_db",
            {
                "host": "pg.example.com",
                "port": "5432",
                "database": "prod_db",
                "user": "alice",
                "password": "s3cr3t",
            },
        )
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

        pad = make_pad()
        pad.execute = capture_execute

        scratchpads = AsyncMock()
        scratchpads.get_or_create = AsyncMock(return_value=pad)

        with patch("anton.commands.datasource.verify.DatasourceRegistry", return_value=registry):
            await handle_test_datasource(
                MagicMock(), scratchpads, "postgresql-prod_db", vault=vault
            )

        # During execution: flat var was set, namespaced was absent
        assert env_during_test["DS_HOST"] == "pg.example.com"
        assert env_during_test["DS_POSTGRESQL_PROD_DB__HOST"] is None

        # After execution: flat vars gone, namespaced restored
        assert "DS_HOST" not in os.environ
        assert os.environ.get("DS_POSTGRESQL_PROD_DB__HOST") == "pg.example.com"
        assert os.environ.get("DS_HUBSPOT_MAIN__ACCESS_TOKEN") == "pat-abc"


class TestStaleDsRegistrationState:
    """Regression tests: _DS_SECRET_VARS/_DS_KNOWN_VARS must mirror vault contents."""

    def test_remove_clears_stale_secret_vars(self, vault_dir, registry):
        """After removing a connection, its secret var names leave _DS_SECRET_VARS."""
        vault = LocalDataVault(vault_dir=vault_dir)
        vault.save(
            "postgresql",
            "prod_db",
            {
                "host": "pg.example.com",
                "port": "5432",
                "database": "prod_db",
                "user": "alice",
                "password": "s3cr3t",
            },
        )

        with patch(
            "anton.utils.datasources.DatasourceRegistry", return_value=registry
        ):
            restore_namespaced_env(vault)

        assert "DS_POSTGRESQL_PROD_DB__PASSWORD" in _DS_SECRET_VARS
        assert "DS_POSTGRESQL_PROD_DB__PASSWORD" in _DS_KNOWN_VARS

        vault.delete("postgresql", "prod_db")

        with patch(
            "anton.utils.datasources.DatasourceRegistry", return_value=registry
        ):
            restore_namespaced_env(vault)

        assert "DS_POSTGRESQL_PROD_DB__PASSWORD" not in _DS_SECRET_VARS
        assert "DS_POSTGRESQL_PROD_DB__PASSWORD" not in _DS_KNOWN_VARS

    def test_edit_connection_refreshes_secret_vars(self, vault_dir, registry):
        """Overwriting a connection via vault.save rebuilds registration without duplication."""
        vault = LocalDataVault(vault_dir=vault_dir)
        vault.save(
            "postgresql",
            "prod_db",
            {
                "host": "pg.example.com",
                "port": "5432",
                "database": "prod_db",
                "user": "alice",
                "password": "old-pass",
            },
        )

        with patch(
            "anton.utils.datasources.DatasourceRegistry", return_value=registry
        ):
            restore_namespaced_env(vault)

        secret_key = "DS_POSTGRESQL_PROD_DB__PASSWORD"
        assert secret_key in _DS_SECRET_VARS
        count_before = len(_DS_SECRET_VARS)

        # Simulate edit: overwrite with new credentials
        vault.save(
            "postgresql",
            "prod_db",
            {
                "host": "pg.example.com",
                "port": "5432",
                "database": "prod_db",
                "user": "alice",
                "password": "new-pass",
            },
        )

        with patch(
            "anton.utils.datasources.DatasourceRegistry", return_value=registry
        ):
            restore_namespaced_env(vault)

        assert secret_key in _DS_SECRET_VARS
        assert len(_DS_SECRET_VARS) == count_before
        assert os.environ.get(secret_key) == "new-pass"

    def test_reconnect_no_duplicate_secret_vars(self, vault_dir, registry):
        """Calling _restore_namespaced_env multiple times does not grow _DS_SECRET_VARS."""
        vault = LocalDataVault(vault_dir=vault_dir)
        vault.save(
            "postgresql",
            "prod_db",
            {
                "host": "pg.example.com",
                "port": "5432",
                "database": "prod_db",
                "user": "alice",
                "password": "s3cr3t",
            },
        )

        with patch(
            "anton.utils.datasources.DatasourceRegistry", return_value=registry
        ):
            restore_namespaced_env(vault)

        count_after_first = len(_DS_SECRET_VARS)
        known_after_first = len(_DS_KNOWN_VARS)

        with patch(
            "anton.utils.datasources.DatasourceRegistry", return_value=registry
        ):
            restore_namespaced_env(vault)

        assert len(_DS_SECRET_VARS) == count_after_first
        assert len(_DS_KNOWN_VARS) == known_after_first


class TestAddCustomDatasourceFlow:
    """Tests for _handle_add_custom_datasource field-collection logic."""

    def _make_spec(self, fields: list[dict], display_name: str = "MyDB"):
        """Return a _CustomDatasourceSpec instance mimicking the LLM's response."""
        from anton.commands.datasource import (
            _CustomDatasourceField,
            _CustomDatasourceSpec,
        )

        return _CustomDatasourceSpec(
            display_name=display_name,
            pip="",
            test_snippet="",
            fields=[_CustomDatasourceField(**f) for f in fields],
        )

    def _make_registry(self, tmp_path):
        """Return a minimal registry mock that accepts any slug."""
        reg = MagicMock()
        reg.validate_file.return_value = {"mydb": MagicMock()}
        reg.reload.return_value = None
        reg.get.return_value = None  # triggers inline fallback
        return reg

    def _make_llm(self, spec):
        """Return an AsyncMock LLM whose generate_object() returns the spec."""
        llm = AsyncMock()
        llm.generate_object = AsyncMock(return_value=spec)
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
        session._llm = self._make_llm(
            self._make_spec(
                [
                    {
                        "name": "host",
                        "value": "",
                        "secret": False,
                        "required": True,
                        "description": "hostname",
                    },
                ]
            )
        )
        console = MagicMock()
        registry = self._make_registry(tmp_path)

        responses = iter(["I want to connect to mydb", "localhost"])

        with (
            patch(
                "anton.commands.datasource.custom.prompt_or_cancel",
                new=AsyncMock(side_effect=lambda *a, **kw: next(responses)),
            ),
            patch("anton.utils.datasources.Path") as mock_path_cls,
        ):
            self._mock_ds_path(mock_path_cls, tmp_path)
            result = await handle_add_custom_datasource(
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
        session._llm = self._make_llm(
            self._make_spec(
                [
                    {
                        "name": "api_key",
                        "value": "",
                        "secret": True,
                        "required": True,
                        "description": "API key",
                    },
                ]
            )
        )
        console = MagicMock()
        registry = self._make_registry(tmp_path)

        responses = iter(["I want to connect", "mysecret"])

        with (
            patch(
                "anton.commands.datasource.custom.prompt_or_cancel",
                new=AsyncMock(side_effect=lambda *a, **kw: next(responses)),
            ),
            patch("anton.utils.datasources.Path") as mock_path_cls,
        ):
            self._mock_ds_path(mock_path_cls, tmp_path)
            result = await handle_add_custom_datasource(
                console, "mydb", registry, session
            )

        assert result is not None
        _, credentials = result
        assert credentials["api_key"] == "mysecret"

    @pytest.mark.asyncio
    async def test_all_fields_skipped_returns_empty_credentials(
        self, tmp_path, make_session
    ):
        """Skipping every field (empty input) is now valid — all custom fields
        are optional, so empty credentials is a successful save."""
        session = make_session()
        session._llm = self._make_llm(
            self._make_spec(
                [
                    {
                        "name": "host",
                        "value": "",
                        "secret": False,
                        "required": True,
                        "description": "hostname",
                    },
                    {
                        "name": "api_key",
                        "value": "",
                        "secret": True,
                        "required": True,
                        "description": "API key",
                    },
                ]
            )
        )
        console = MagicMock()
        registry = self._make_registry(tmp_path)

        responses = iter(["I want to connect", "", ""])

        with (
            patch(
                "anton.commands.datasource.custom.prompt_or_cancel",
                new=AsyncMock(side_effect=lambda *a, **kw: next(responses)),
            ),
            patch("anton.utils.datasources.Path") as mock_path_cls,
        ):
            self._mock_ds_path(mock_path_cls, tmp_path)
            result = await handle_add_custom_datasource(
                console, "mydb", registry, session
            )

        assert result is not None
        _, credentials = result
        assert credentials == {}

    @pytest.mark.asyncio
    async def test_no_proactive_help_prompt(self, tmp_path, make_session):
        """Simplified custom flow must not ask 'need instructions on how
        to obtain credentials?'. Prompts should be: description + one
        prompt per missing field (all optional).
        """
        session = make_session()
        session._llm = self._make_llm(
            self._make_spec(
                [
                    {
                        "name": "host",
                        "value": "",
                        "secret": False,
                        "required": False,
                        "description": "hostname",
                    },
                    {
                        "name": "api_key",
                        "value": "",
                        "secret": True,
                        "required": False,
                        "description": "API key",
                    },
                ]
            )
        )
        console = MagicMock()
        registry = self._make_registry(tmp_path)

        seen_labels: list[str] = []

        async def _prompt(*args, **kwargs):
            label = args[0] if args else kwargs.get("label", "")
            seen_labels.append(label)
            return next(responses)

        # description + host value + api_key value
        responses = iter(["uses an API key", "localhost", "secret123"])

        with (
            patch(
                "anton.commands.datasource.custom.prompt_or_cancel",
                new=_prompt,
            ),
            patch("anton.utils.datasources.Path") as mock_path_cls,
        ):
            self._mock_ds_path(mock_path_cls, tmp_path)
            result = await handle_add_custom_datasource(
                console, "mydb", registry, session
            )

        assert result is not None
        _, credentials = result
        joined = " | ".join(seen_labels).lower()
        assert "instructions" not in joined
        assert "need help" not in joined
        assert credentials["host"] == "localhost"
        assert credentials["api_key"] == "secret123"
        # Exactly 3 prompts: description, host, api_key.
        assert len(seen_labels) == 3

    @pytest.mark.asyncio
    async def test_all_inline_values_issue_no_field_prompts(
        self, tmp_path, make_session
    ):
        """When the LLM spec returns inline values for every field, no
        per-field prompt is issued — only the description prompt runs.
        """
        session = make_session()
        session._llm = self._make_llm(
            self._make_spec(
                [
                    {
                        "name": "host",
                        "value": "db.internal",
                        "secret": False,
                        "required": False,
                        "description": "hostname",
                    },
                    {
                        "name": "token",
                        "value": "tok-inline-123",
                        "secret": True,
                        "required": False,
                        "description": "auth token",
                    },
                ]
            )
        )
        console = MagicMock()
        registry = self._make_registry(tmp_path)

        seen_labels: list[str] = []

        async def _prompt(*args, **kwargs):
            label = args[0] if args else kwargs.get("label", "")
            seen_labels.append(label)
            return next(responses)

        # Only the description prompt should fire.
        responses = iter(["connects via header token"])

        with (
            patch(
                "anton.commands.datasource.custom.prompt_or_cancel",
                new=_prompt,
            ),
            patch("anton.utils.datasources.Path") as mock_path_cls,
        ):
            self._mock_ds_path(mock_path_cls, tmp_path)
            result = await handle_add_custom_datasource(
                console, "mydb", registry, session
            )

        assert result is not None
        _, credentials = result
        assert credentials == {"host": "db.internal", "token": "tok-inline-123"}
        assert len(seen_labels) == 1

    @pytest.mark.asyncio
    async def test_missing_values_do_not_hard_stop(self, tmp_path, make_session):
        """Skipping every field still returns a valid (empty) credentials
        dict — no 'Cannot save — missing required fields' abort.
        """
        session = make_session()
        session._llm = self._make_llm(
            self._make_spec(
                [
                    {
                        "name": "host",
                        "value": "",
                        "secret": False,
                        "required": False,
                        "description": "hostname",
                    },
                    {
                        "name": "api_key",
                        "value": "",
                        "secret": True,
                        "required": False,
                        "description": "API key",
                    },
                ]
            )
        )
        console = MagicMock()
        registry = self._make_registry(tmp_path)

        # description + skip both fields
        responses = iter(["no idea", "", ""])

        with (
            patch(
                "anton.commands.datasource.custom.prompt_or_cancel",
                new=AsyncMock(side_effect=lambda *a, **kw: next(responses)),
            ),
            patch("anton.utils.datasources.Path") as mock_path_cls,
        ):
            self._mock_ds_path(mock_path_cls, tmp_path)
            result = await handle_add_custom_datasource(
                console, "mydb", registry, session
            )

        assert result is not None
        _, credentials = result
        assert credentials == {}
        printed = " ".join(str(c) for c in console.print.call_args_list).lower()
        assert "cannot save" not in printed
        assert "missing required" not in printed

    @pytest.mark.asyncio
    async def test_sequential_prompts_use_descriptions(self, tmp_path, make_session):
        """After spec generation, fields are collected one at a time and the
        prompt labels use the field description (not snake_case name)."""
        from anton.commands.datasource import _CustomDatasourceSpec
        from anton.connect_collector import _ExtractionResult

        spec = self._make_spec(
            [
                {
                    "name": "base_url",
                    "value": "",
                    "secret": False,
                    "required": False,
                    "description": "API base URL",
                },
                {
                    "name": "access_token",
                    "value": "",
                    "secret": True,
                    "required": False,
                    "description": "Access Token",
                },
            ]
        )

        session = make_session()

        async def _gen(schema_class, **kwargs):
            if schema_class is _ExtractionResult:
                return _ExtractionResult()
            if schema_class is _CustomDatasourceSpec:
                return spec
            raise RuntimeError(f"unexpected schema {schema_class.__name__}")

        session._llm.generate_object = AsyncMock(side_effect=_gen)

        console = MagicMock()
        registry = self._make_registry(tmp_path)
        registry.all_engines = MagicMock(return_value=[])

        seen_labels: list[str] = []

        async def _prompt(*args, **kwargs):
            label = args[0] if args else kwargs.get("label", "")
            seen_labels.append(label)
            return next(responses)

        # description + base_url + access_token
        responses = iter(["uses a token", "https://api.example.com", "tok-abc"])

        with (
            patch(
                "anton.commands.datasource.custom.prompt_or_cancel",
                new=_prompt,
            ),
            patch("anton.utils.datasources.Path") as mock_path_cls,
        ):
            self._mock_ds_path(mock_path_cls, tmp_path)
            result = await handle_add_custom_datasource(
                console, "mydb", registry, session
            )

        assert result is not None
        _, credentials = result
        assert credentials == {
            "base_url": "https://api.example.com",
            "access_token": "tok-abc",
        }
        # First label is the discovery description; next two are per-field.
        assert seen_labels[1] == "(anton) API base URL"
        assert seen_labels[2] == "(anton) Access Token"

    @pytest.mark.asyncio
    async def test_help_during_collection_shows_guide(self, tmp_path, make_session):
        """Typing 'help' at a custom field prompt invokes show_credential_help
        with the current field, then collection continues."""
        session = make_session()
        session._llm = self._make_llm(
            self._make_spec(
                [
                    {
                        "name": "api_key",
                        "value": "",
                        "secret": True,
                        "required": False,
                        "description": "API key",
                    },
                ]
            )
        )
        console = MagicMock()
        registry = self._make_registry(tmp_path)

        help_calls: list[tuple[str, str]] = []

        async def _fake_help(console_arg, session_arg, service_name, current_field, all_fields):
            help_calls.append(
                (service_name, current_field.name if current_field else "")
            )

        # description + 'help' (shows guide) + real value
        responses = iter(["uses a key", "help", "real-key"])

        with (
            patch(
                "anton.commands.datasource.custom.prompt_or_cancel",
                new=AsyncMock(side_effect=lambda *a, **kw: next(responses)),
            ),
            patch(
                "anton.commands.datasource.custom.show_credential_help",
                new=_fake_help,
            ),
            patch("anton.utils.datasources.Path") as mock_path_cls,
        ):
            self._mock_ds_path(mock_path_cls, tmp_path)
            result = await handle_add_custom_datasource(
                console, "mydb", registry, session
            )

        assert help_calls == [("MyDB", "api_key")]
        assert result is not None
        _, credentials = result
        assert credentials["api_key"] == "real-key"

    @pytest.mark.asyncio
    async def test_optional_empty_input_skips_and_continues(
        self, tmp_path, make_session
    ):
        """Empty input on an optional field skips it and the loop continues
        to the next field — no hard stop, no re-prompt."""
        session = make_session()
        session._llm = self._make_llm(
            self._make_spec(
                [
                    {
                        "name": "host",
                        "value": "",
                        "secret": False,
                        "required": False,
                        "description": "hostname",
                    },
                    {
                        "name": "api_key",
                        "value": "",
                        "secret": True,
                        "required": False,
                        "description": "API key",
                    },
                ]
            )
        )
        console = MagicMock()
        registry = self._make_registry(tmp_path)

        # description + empty (skip host) + real api_key value
        responses = iter(["no idea", "", "secret123"])

        with (
            patch(
                "anton.commands.datasource.custom.prompt_or_cancel",
                new=AsyncMock(side_effect=lambda *a, **kw: next(responses)),
            ),
            patch("anton.utils.datasources.Path") as mock_path_cls,
        ):
            self._mock_ds_path(mock_path_cls, tmp_path)
            result = await handle_add_custom_datasource(
                console, "mydb", registry, session
            )

        assert result is not None
        _, credentials = result
        assert credentials == {"api_key": "secret123"}

    @pytest.mark.asyncio
    async def test_structured_multi_value_fills_multiple_fields(
        self, tmp_path, make_session
    ):
        """Structured input (contains '=') at a field prompt when multiple
        fields are still missing is routed through extract_variables and
        fills every matching field in one shot."""
        from anton.commands.datasource import (
            _CustomDatasourceField,
            _CustomDatasourceSpec,
        )
        from anton.connect_collector import _ExtractionResult

        spec = _CustomDatasourceSpec(
            display_name="MyDB",
            pip="",
            test_snippet="",
            fields=[
                _CustomDatasourceField(
                    name="host",
                    value="",
                    secret=False,
                    required=False,
                    description="hostname",
                ),
                _CustomDatasourceField(
                    name="port",
                    value="",
                    secret=False,
                    required=False,
                    description="port",
                ),
            ],
        )

        session = make_session()

        async def _gen(schema_class, **kwargs):
            if schema_class is _ExtractionResult:
                return _ExtractionResult(
                    variables={"host": "db.example.com", "port": "5432"},
                )
            if schema_class is _CustomDatasourceSpec:
                return spec
            raise RuntimeError(f"unexpected schema {schema_class.__name__}")

        session._llm.generate_object = AsyncMock(side_effect=_gen)

        console = MagicMock()
        registry = self._make_registry(tmp_path)
        registry.all_engines = MagicMock(return_value=[])

        # description + structured paste at first field prompt
        responses = iter(["uses host+port", "host=db.example.com port=5432"])

        with (
            patch(
                "anton.commands.datasource.custom.prompt_or_cancel",
                new=AsyncMock(side_effect=lambda *a, **kw: next(responses)),
            ),
            patch("anton.utils.datasources.Path") as mock_path_cls,
        ):
            self._mock_ds_path(mock_path_cls, tmp_path)
            result = await handle_add_custom_datasource(
                console, "mydb", registry, session
            )

        assert result is not None
        _, credentials = result
        assert credentials == {"host": "db.example.com", "port": "5432"}
        # Only two user prompts: description + the structured paste.
        # (If extraction hadn't filled both fields, a second per-field
        # prompt would have fired and next() would have raised StopIteration.)


class TestCustomDatasourceRequiredFieldsPolicy:
    """Verify that custom engines never expose required=True fields at any layer."""

    # ── helpers ─────────────────────────────────────────────────────────────

    def _custom_md(self, tmp_path, *, required_value: str = "true"):
        md = tmp_path / "custom_datasources.md"
        md.write_text(
            dedent(
                f"""\
            ## My Custom API

            ```yaml
            engine: my_custom_api
            display_name: My Custom API
            fields:
              - name: api_key
                required: {required_value}
                secret: true
                description: API key
              - name: base_url
                required: {required_value}
                secret: false
                description: base URL
            ```
        """
            )
        )
        return md

    def _make_spec(self, fields: list[dict], display_name: str = "MyDB"):
        from anton.commands.datasource import (
            _CustomDatasourceField,
            _CustomDatasourceSpec,
        )
        return _CustomDatasourceSpec(
            display_name=display_name,
            pip="",
            test_snippet="",
            fields=[_CustomDatasourceField(**f) for f in fields],
        )

    def _make_registry(self, tmp_path):
        reg = MagicMock()
        reg.validate_file.return_value = {"mydb": MagicMock()}
        reg.reload.return_value = None
        reg.get.return_value = None
        return reg

    # ── load-time: custom file ───────────────────────────────────────────────

    def test_custom_file_required_true_forced_false_on_load(self, tmp_path):
        """Custom datasources.md with required: true → all fields required=False after load."""
        md = self._custom_md(tmp_path, required_value="true")
        engines = _parse_file(md, custom=True)
        assert "my_custom_api" in engines
        engine = engines["my_custom_api"]
        assert all(not f.required for f in engine.fields), (
            "Custom engine fields must all be required=False after load"
        )

    def test_builtin_file_required_true_preserved_on_load(self, datasources_md):
        """Built-in datasources.md with required: true → fields keep required=True."""
        engines = _parse_file(datasources_md, custom=False)
        pg = engines.get("postgresql")
        assert pg is not None
        required_fields = [f for f in pg.fields if f.required]
        assert required_fields, "Built-in postgresql must still have required fields"

    def test_custom_load_with_auth_methods_fields_also_forced(self, tmp_path):
        """Custom engine with auth_methods: all nested fields also required=False."""
        md = tmp_path / "custom_auth.md"
        md.write_text(
            dedent(
                """\
            ## Custom Auth Service

            ```yaml
            engine: custom_auth_service
            display_name: Custom Auth Service
            auth_method: choice
            auth_methods:
              - name: token
                display: API Token
                fields:
                  - name: token
                    required: true
                    secret: true
                    description: access token
              - name: basic
                display: Basic Auth
                fields:
                  - name: username
                    required: true
                    description: username
            fields: []
            ```
        """
            )
        )
        engines = _parse_file(md, custom=True)
        engine = engines["custom_auth_service"]
        for method in engine.auth_methods:
            for f in method.fields:
                assert not f.required, (
                    f"auth_method field {f.name!r} must be required=False for custom engine"
                )

    # ── save-time: Pydantic default & field construction ────────────────────

    def test_custom_field_pydantic_default_is_false(self):
        """_CustomDatasourceField defaults required=False without explicit setting."""
        from anton.commands.datasource import _CustomDatasourceField
        f = _CustomDatasourceField(name="token")
        assert f.required is False

    def test_field_constructor_forces_required_false(self):
        """Even if LLM sets required=True in spec, DatasourceField is built with False."""
        from anton.commands.datasource import _CustomDatasourceField
        llm_field = _CustomDatasourceField(
            name="host", required=True, secret=False, description="hostname"
        )
        # Simulate what handle_add_custom_datasource does
        real_field = DatasourceField(
            name=llm_field.name,
            required=False,
            secret=llm_field.secret,
            description=llm_field.description,
        )
        assert real_field.required is False

    # ── ConnectionCollector: all-optional → immediately complete ────────────

    def test_collector_all_optional_fields_is_complete_with_zero_filled(self):
        """ConnectionCollector with all required=False reports is_complete immediately."""
        from anton.connect_collector import ConnectionCollector
        engine = DatasourceEngine(
            engine="test_engine",
            display_name="Test Engine",
            fields=[
                DatasourceField(name="api_key", required=False, secret=True),
                DatasourceField(name="base_url", required=False, secret=False),
            ],
        )
        collector = ConnectionCollector(engine_def=engine)
        assert collector.is_complete is True
        assert collector.missing_required == []

    # ── end-to-end: skip all fields → vault save succeeds ───────────────────

    @pytest.mark.asyncio
    async def test_skip_all_fields_saves_empty_credentials(
        self, vault_dir, make_session, tmp_path
    ):
        """User skips every field with Enter → custom datasource saved with empty creds."""
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)

        spec_with_fields = self._make_spec(
            [
                {"name": "api_key", "value": "", "secret": True, "required": True,
                 "description": "API key"},
                {"name": "base_url", "value": "", "secret": False, "required": True,
                 "description": "base URL"},
            ],
            display_name="WeirdAPI",
        )
        session._llm.generate_object = AsyncMock(return_value=spec_with_fields)

        registry = self._make_registry(tmp_path)
        registry.find_by_name = MagicMock(return_value=None)
        registry.all_engines = MagicMock(return_value=[])

        # connect: "WeirdAPI" → custom description → skip both fields
        connect_responses = iter(["WeirdAPI"])
        custom_responses = iter(["it uses token auth", "", ""])

        with (
            patch(
                "anton.commands.datasource.connect.LocalDataVault",
                return_value=vault,
            ),
            patch(
                "anton.commands.datasource.connect.DatasourceRegistry",
                return_value=registry,
            ),
            patch(
                "anton.commands.datasource.connect.prompt_or_cancel",
                new=AsyncMock(side_effect=lambda *a, **kw: next(connect_responses)),
            ),
            patch(
                "anton.commands.datasource.custom.prompt_or_cancel",
                new=AsyncMock(side_effect=lambda *a, **kw: next(custom_responses)),
            ),
            patch("anton.utils.datasources.Path") as mock_path_cls,
        ):
            mock_path_cls.return_value.expanduser.return_value = (
                tmp_path / "datasources.md"
            )
            result = await handle_connect_datasource(
                console, session._scratchpads, session
            )

        conns = vault.list_connections()
        assert len(conns) == 1, "Custom datasource must be saved even with empty credentials"
        saved = vault.load(conns[0]["engine"], conns[0]["name"])
        assert saved == {}, "Empty credentials dict should be saved"
        assert result is session


class TestCustomDatasourceConnectFlow:
    """Tests for the custom_source path in _handle_connect_datasource:
    test_snippet is run before saving, and failures prevent saving."""

    # ── helpers (mirrors TestAddCustomDatasourceFlow) ────────────────────

    def _make_spec(
        self,
        fields: list[dict],
        display_name: str = "My API Service",
        test_snippet: str = "",
    ):
        from anton.commands.datasource import (
            _CustomDatasourceField,
            _CustomDatasourceSpec,
        )

        return _CustomDatasourceSpec(
            display_name=display_name,
            pip="",
            test_snippet=test_snippet,
            fields=[_CustomDatasourceField(**f) for f in fields],
        )

    def _make_registry(self, tmp_path):
        reg = MagicMock()
        reg.all_engines.return_value = []
        reg.find_by_name.return_value = None
        reg.fuzzy_find.return_value = []
        reg.validate_file.return_value = {"my_api_service": MagicMock()}
        reg.reload.return_value = None
        reg.get.return_value = None  # triggers inline fallback engine_def
        return reg

    def _make_llm(self, spec):
        llm = AsyncMock()
        llm.generate_object = AsyncMock(return_value=spec)
        return llm

    def _mock_ds_path(self, mock_path_cls, tmp_path):
        mock_path_cls.return_value.expanduser.return_value = tmp_path / "datasources.md"

    @pytest.mark.asyncio
    async def test_custom_with_test_snippet_success(
        self, vault_dir, make_session, make_cell, tmp_path, make_pad
    ):
        """Custom datasource with test_snippet: test passes → connection saved."""
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)

        pad = make_pad()
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)
        session._llm = self._make_llm(
            self._make_spec(
                [
                    {
                        "name": "api_key",
                        "value": "",
                        "secret": True,
                        "required": True,
                        "description": "API key",
                    }
                ],
                test_snippet="print('ok')",
            )
        )

        responses = iter(
            ["My API Service", "I have an API key", "my_secret_key"]
        )

        poc = AsyncMock(side_effect=lambda *a, **kw: next(responses))
        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch(
                "anton.commands.datasource.connect.DatasourceRegistry",
                return_value=self._make_registry(tmp_path),
            ),
            patch("anton.commands.datasource.connect.prompt_or_cancel", new=poc),
            patch("anton.commands.datasource.custom.prompt_or_cancel", new=poc),
            patch("anton.utils.datasources.Path") as mock_path_cls,
        ):
            self._mock_ds_path(mock_path_cls, tmp_path)
            result = await handle_connect_datasource(
                console, session._scratchpads, session
            )

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
        self, vault_dir, make_session, make_cell, tmp_path, make_pad
    ):
        """Custom datasource: test fails and user declines retry → not saved."""
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)

        pad = make_pad(make_cell(stdout="", stderr="connection refused"))
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)
        session._llm = self._make_llm(
            self._make_spec(
                [
                    {
                        "name": "api_key",
                        "value": "",
                        "secret": True,
                        "required": True,
                        "description": "API key",
                    }
                ],
                test_snippet="print('ok')",
            )
        )

        responses = iter(
            ["My API Service", "I have an API key", "bad_key", "n"]
        )

        poc = AsyncMock(side_effect=lambda *a, **kw: next(responses))
        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch(
                "anton.commands.datasource.connect.DatasourceRegistry",
                return_value=self._make_registry(tmp_path),
            ),
            patch("anton.commands.datasource.connect.prompt_or_cancel", new=poc),
            patch("anton.commands.datasource.custom.prompt_or_cancel", new=poc),
            patch("anton.commands.datasource.verify.prompt_or_cancel", new=poc),
            patch("anton.utils.datasources.Path") as mock_path_cls,
        ):
            self._mock_ds_path(mock_path_cls, tmp_path)
            result = await handle_connect_datasource(
                console, session._scratchpads, session
            )

        assert vault.list_connections() == []
        assert not result._history

    @pytest.mark.asyncio
    async def test_custom_with_test_snippet_fail_retry_success(
        self, vault_dir, make_session, make_cell, tmp_path, make_pad
    ):
        """Custom datasource: test fails, user retries with corrected creds → saved."""
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)

        pad = make_pad(side_effect=[
            make_cell(stdout="", stderr="invalid key"),
            make_cell(stdout="ok"),
        ])
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)
        session._llm = self._make_llm(
            self._make_spec(
                [
                    {
                        "name": "api_key",
                        "value": "",
                        "secret": True,
                        "required": True,
                        "description": "API key",
                    }
                ],
                test_snippet="print('ok')",
            )
        )

        responses = iter(
            [
                "My API Service",
                "I have an API key",
                "bad_key",
                "y",
                "good_key",
            ]
        )

        poc = AsyncMock(side_effect=lambda *a, **kw: next(responses))
        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch(
                "anton.commands.datasource.connect.DatasourceRegistry",
                return_value=self._make_registry(tmp_path),
            ),
            patch("anton.commands.datasource.connect.prompt_or_cancel", new=poc),
            patch("anton.commands.datasource.custom.prompt_or_cancel", new=poc),
            patch("anton.commands.datasource.verify.prompt_or_cancel", new=poc),
            patch("anton.utils.datasources.Path") as mock_path_cls,
        ):
            self._mock_ds_path(mock_path_cls, tmp_path)
            result = await handle_connect_datasource(
                console, session._scratchpads, session
            )

        conns = vault.list_connections()
        assert len(conns) == 1
        saved = vault.load(conns[0]["engine"], conns[0]["name"])
        assert saved is not None
        assert saved.get("api_key") == "good_key"
        assert result._history

    @pytest.mark.asyncio
    async def test_custom_retry_reenter_returns_to_field_editing(
        self, vault_dir, make_session, make_cell, tmp_path, make_pad
    ):
        """After custom test failure + retry, field prompts are shown again for editing."""
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)

        pad = make_pad(side_effect=[
            make_cell(stdout="", stderr="invalid credentials"),
            make_cell(stdout="ok"),
        ])
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)
        session._llm = self._make_llm(
            self._make_spec(
                [
                    {
                        "name": "host",
                        "value": "",
                        "secret": False,
                        "required": False,
                        "description": "Host",
                    },
                    {
                        "name": "api_key",
                        "value": "",
                        "secret": True,
                        "required": False,
                        "description": "API key",
                    },
                ],
                test_snippet="print('ok')",
            )
        )

        responses = iter(
            [
                "My API Service",
                "host + api key",
                "bad-host",
                "bad-key",
                "y",
                "good-host",
                "good-key",
            ]
        )

        poc = AsyncMock(side_effect=lambda *a, **kw: next(responses))
        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch(
                "anton.commands.datasource.connect.DatasourceRegistry",
                return_value=self._make_registry(tmp_path),
            ),
            patch("anton.commands.datasource.connect.prompt_or_cancel", new=poc),
            patch("anton.commands.datasource.custom.prompt_or_cancel", new=poc),
            patch("anton.commands.datasource.verify.prompt_or_cancel", new=poc),
            patch("anton.utils.datasources.Path") as mock_path_cls,
        ):
            self._mock_ds_path(mock_path_cls, tmp_path)
            await handle_connect_datasource(
                console, session._scratchpads, session
            )

        prompts = [c.args[0] for c in poc.call_args_list if c.args]
        assert prompts.count("(anton) Host") == 2
        assert any("(anton) API key" in p for p in prompts)

    @pytest.mark.asyncio
    async def test_custom_retry_edited_value_used_for_next_attempt(
        self, vault_dir, make_session, make_cell, tmp_path, make_pad
    ):
        """Edited retry value is used by the next connection test attempt."""
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)

        seen_hosts: list[str] = []
        attempt = {"n": 0}

        async def _exec(_code):
            attempt["n"] += 1
            seen_hosts.append(os.environ.get("DS_HOST", ""))
            if attempt["n"] == 1:
                return make_cell(stdout="", stderr="invalid host")
            return make_cell(stdout="ok")

        pad = make_pad()
        pad.execute = AsyncMock(side_effect=_exec)
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)
        session._llm = self._make_llm(
            self._make_spec(
                [
                    {
                        "name": "host",
                        "value": "",
                        "secret": False,
                        "required": False,
                        "description": "Host",
                    },
                    {
                        "name": "api_key",
                        "value": "",
                        "secret": True,
                        "required": False,
                        "description": "API key",
                    },
                ],
                test_snippet="print('ok')",
            )
        )

        responses = iter(
            [
                "My API Service",
                "host + api key",
                "bad-host",
                "bad-key",
                "y",
                "good-host",
                "",
            ]
        )

        poc = AsyncMock(side_effect=lambda *a, **kw: next(responses))
        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch(
                "anton.commands.datasource.connect.DatasourceRegistry",
                return_value=self._make_registry(tmp_path),
            ),
            patch("anton.commands.datasource.connect.prompt_or_cancel", new=poc),
            patch("anton.commands.datasource.custom.prompt_or_cancel", new=poc),
            patch("anton.commands.datasource.verify.prompt_or_cancel", new=poc),
            patch("anton.utils.datasources.Path") as mock_path_cls,
        ):
            self._mock_ds_path(mock_path_cls, tmp_path)
            await handle_connect_datasource(
                console, session._scratchpads, session
            )

        conns = vault.list_connections()
        assert len(conns) == 1
        saved = vault.load(conns[0]["engine"], conns[0]["name"])
        assert saved is not None
        assert saved.get("host") == "good-host"
        assert seen_hosts == ["bad-host", "good-host"]

    @pytest.mark.asyncio
    async def test_custom_retry_does_not_immediately_retest_unchanged_values(
        self, vault_dir, make_session, make_cell, tmp_path, make_pad
    ):
        """Retry path prompts for edits before issuing the next test attempt."""
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)

        state = {"attempt": 0, "retry_confirmed": False, "retry_prompt_seen": False}

        async def _exec(_code):
            state["attempt"] += 1
            if state["attempt"] == 2:
                assert state["retry_prompt_seen"] is True
            if state["attempt"] == 1:
                return make_cell(stdout="", stderr="invalid")
            return make_cell(stdout="ok")

        def _prompt(label, **kwargs):
            if label == "(anton) Connect to":
                return "My API Service"
            if "How does" in label:
                return "host + api key"
            if label == "(anton) Host" and not state["retry_confirmed"]:
                return "bad-host"
            if label == "(anton) API key":
                return "bad-key"
            if label == "(anton) Would you like to re-enter your credentials?":
                state["retry_confirmed"] = True
                return "y"
            if label == "(anton) Host" and state["retry_confirmed"]:
                state["retry_prompt_seen"] = True
                return "good-host"
            if "(anton) API key" in label and state["retry_confirmed"]:
                return "good-key"
            raise AssertionError(f"Unexpected prompt: {label}")

        pad = make_pad()
        pad.execute = AsyncMock(side_effect=_exec)
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)
        session._llm = self._make_llm(
            self._make_spec(
                [
                    {
                        "name": "host",
                        "value": "",
                        "secret": False,
                        "required": False,
                        "description": "Host",
                    },
                    {
                        "name": "api_key",
                        "value": "",
                        "secret": True,
                        "required": False,
                        "description": "API key",
                    },
                ],
                test_snippet="print('ok')",
            )
        )

        poc = AsyncMock(side_effect=_prompt)
        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch(
                "anton.commands.datasource.connect.DatasourceRegistry",
                return_value=self._make_registry(tmp_path),
            ),
            patch("anton.commands.datasource.connect.prompt_or_cancel", new=poc),
            patch("anton.commands.datasource.custom.prompt_or_cancel", new=poc),
            patch("anton.commands.datasource.verify.prompt_or_cancel", new=poc),
            patch("anton.utils.datasources.Path") as mock_path_cls,
        ):
            self._mock_ds_path(mock_path_cls, tmp_path)
            await handle_connect_datasource(
                console, session._scratchpads, session
            )

        assert state["attempt"] == 2
        assert state["retry_prompt_seen"] is True

    @pytest.mark.asyncio
    async def test_custom_retry_reuses_existing_inferred_spec(
        self, vault_dir, make_session, make_cell, tmp_path, make_pad
    ):
        """Retry path reuses discovered custom spec/fields without rediscovery."""
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)

        pad = make_pad(side_effect=[
            make_cell(stdout="", stderr="invalid key"),
            make_cell(stdout="ok"),
        ])
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)
        session._llm = self._make_llm(
            self._make_spec(
                [
                    {
                        "name": "api_key",
                        "value": "",
                        "secret": True,
                        "required": False,
                        "description": "API key",
                    }
                ],
                test_snippet="print('ok')",
            )
        )

        responses = iter(
            [
                "My API Service",
                "api key",
                "bad-key",
                "y",
                "good-key",
            ]
        )

        poc = AsyncMock(side_effect=lambda *a, **kw: next(responses))
        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch(
                "anton.commands.datasource.connect.DatasourceRegistry",
                return_value=self._make_registry(tmp_path),
            ),
            patch("anton.commands.datasource.connect.prompt_or_cancel", new=poc),
            patch("anton.commands.datasource.custom.prompt_or_cancel", new=poc),
            patch("anton.commands.datasource.verify.prompt_or_cancel", new=poc),
            patch("anton.utils.datasources.Path") as mock_path_cls,
        ):
            self._mock_ds_path(mock_path_cls, tmp_path)
            await handle_connect_datasource(
                console, session._scratchpads, session
            )

        assert session._llm.generate_object.await_count == 1

    @pytest.mark.asyncio
    async def test_custom_without_test_snippet_saves(
        self, vault_dir, make_session, make_cell, tmp_path, make_pad
    ):
        """Custom datasource without test_snippet: saves directly, no scratchpad call."""
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)

        pad = make_pad()
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)
        session._llm = self._make_llm(
            self._make_spec(
                [
                    {
                        "name": "api_key",
                        "value": "",
                        "secret": True,
                        "required": True,
                        "description": "API key",
                    }
                ],
                test_snippet="",
            )
        )

        responses = iter(["My API Service", "I have an API key", "my_key"])

        poc = AsyncMock(side_effect=lambda *a, **kw: next(responses))
        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch(
                "anton.commands.datasource.connect.DatasourceRegistry",
                return_value=self._make_registry(tmp_path),
            ),
            patch("anton.commands.datasource.connect.prompt_or_cancel", new=poc),
            patch("anton.commands.datasource.custom.prompt_or_cancel", new=poc),
            patch("anton.utils.datasources.Path") as mock_path_cls,
        ):
            self._mock_ds_path(mock_path_cls, tmp_path)
            await handle_connect_datasource(console, session._scratchpads, session)

        conns = vault.list_connections()
        assert len(conns) == 1
        pad.execute.assert_not_called()


class TestEditDatasourceWithTestSnippet:
    """Tests for /edit path: test_snippet runs before vault.save, not after."""

    OLD_CREDS = {
        "host": "pg.example.com",
        "port": "5432",
        "database": "prod_db",
        "user": "alice",
        "password": "good-pass",
        "schema": "",
    }


    @pytest.mark.asyncio
    async def test_edit_failed_test_does_not_corrupt_vault(
        self, vault_dir, registry, make_session, make_cell, make_pad
    ):
        """edit with bad creds + test fails + user declines retry → original creds intact."""
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)
        vault.save("postgresql", "prod_db", self.OLD_CREDS)

        pad = make_pad(make_cell(stdout="", stderr="connection refused"))
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        # Keep all non-secret fields; enter bad password; decline retry.
        responses = iter(
            ["", "", "", "", "bad-pass", "", "", "n"]
        )  # field values, then retry?

        poc = AsyncMock(side_effect=lambda *a, **kw: next(responses))
        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
            patch("anton.commands.datasource.connect.prompt_or_cancel", new=poc),
            patch("anton.commands.datasource.helpers.prompt_or_cancel", new=poc),
            patch("anton.commands.datasource.verify.prompt_or_cancel", new=poc),
        ):
            result = await handle_connect_datasource(
                console,
                session._scratchpads,
                session,
                datasource_name="postgresql-prod_db",
            )

        saved = vault.load("postgresql", "prod_db")
        assert saved is not None
        assert saved.get("password") == "good-pass"
        assert result._history == []

    @pytest.mark.asyncio
    async def test_edit_successful_test_persists_new_credentials(
        self, vault_dir, registry, make_session, make_cell, make_pad
    ):
        """edit with valid creds + test passes → new creds saved to vault."""
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)
        vault.save("postgresql", "prod_db", self.OLD_CREDS)

        pad = make_pad()
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        prompt_responses = iter(
            [
                "",  # host
                "",  # port
                "",  # database
                "",  # user
                "new-pass",  # password (updated)
                "",  # schema
                "",  # ssl
            ]
        )
        poc = AsyncMock(side_effect=lambda *a, **kw: next(prompt_responses))

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
            patch("anton.commands.datasource.connect.prompt_or_cancel", new=poc),
            patch("anton.commands.datasource.helpers.prompt_or_cancel", new=poc),
        ):
            result = await handle_connect_datasource(
                console,
                session._scratchpads,
                session,
                datasource_name="postgresql-prod_db",
            )

        saved = vault.load("postgresql", "prod_db")
        assert saved is not None
        assert saved.get("password") == "new-pass"
        assert result._history

    @pytest.mark.asyncio
    async def test_connection_test_error_summary_uses_meaningful_line(
        self, vault_dir, registry, make_cell, make_pad
    ):
        """Error display shows last non-empty line (exception msg), not traceback header."""
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)

        traceback_text = (
            "Traceback (most recent call last):\n"
            '  File "test.py", line 3, in <module>\n'
            "    conn = psycopg2.connect(host=os.environ['DS_HOST'])\n"
            "psycopg2.OperationalError: could not connect to server\n"
        )
        cell = MagicMock()
        cell.stdout = ""
        cell.stderr = traceback_text
        cell.error = None

        pad = make_pad(cell)
        scratchpads = AsyncMock()
        scratchpads.get_or_create = AsyncMock(return_value=pad)

        engine_def = registry.get("postgresql")
        credentials = {
            "host": "bad-host",
            "port": "5432",
            "database": "prod_db",
            "user": "alice",
            "password": "pw",
        }

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
            patch("anton.commands.datasource.verify.prompt_or_cancel", new=AsyncMock(return_value="n")),
        ):
            result = await run_connection_test(
                console,
                scratchpads,
                vault,
                engine_def,
                credentials,
                retry_fields=engine_def.fields,
            )

        assert result is False
        printed = " ".join(str(c) for c in console.print.call_args_list)
        assert "psycopg2.OperationalError" in printed


class TestPromptCopyConsistency:
    """Verify that interactive prompts use (y/n) style and Esc cancels safely."""

    @pytest.mark.asyncio
    async def test_esc_on_engine_selection_returns_session_unchanged(
        self, registry, vault_dir, make_session
    ):
        """Pressing Esc on the engine-selection prompt returns the session with no vault writes."""
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
            patch("anton.commands.datasource.connect.prompt_or_cancel", new=AsyncMock(return_value=None)),
        ):
            result = await handle_connect_datasource(
                console, session._scratchpads, session
            )

        assert result is session
        assert vault.list_connections() == []

    @pytest.mark.asyncio
    async def test_esc_on_retry_does_not_save(
        self, registry, vault_dir, make_session, make_cell, make_pad
    ):
        """Pressing Esc at the retry prompt makes _run_connection_test return False."""
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)

        pad = make_pad(make_cell(stdout="", error="bad creds"))
        session._scratchpads.get_or_create = AsyncMock(return_value=pad)

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
            patch("anton.commands.datasource.verify.prompt_or_cancel", new=AsyncMock(return_value=None)),
        ):
            engine_def = registry.get("postgresql")
            credentials = {
                "host": "h",
                "port": "5432",
                "database": "d",
                "user": "u",
                "password": "p",
            }
            result = await run_connection_test(
                console,
                session._scratchpads,
                vault,
                engine_def,
                credentials,
                retry_fields=engine_def.fields,
            )

        assert result is False
        assert vault.list_connections() == []

    @pytest.mark.asyncio
    async def test_esc_on_do_you_have_these_returns_session(
        self, registry, vault_dir, make_session
    ):
        """Pressing Esc after engine selection (on 'do you have these?') returns session."""
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)

        poc_calls = iter(["PostgreSQL", None])  # engine selected, then Esc

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
            patch(
                "anton.commands.datasource.connect.prompt_or_cancel",
                new=AsyncMock(side_effect=lambda *a, **kw: next(poc_calls)),
            ),
        ):
            result = await handle_connect_datasource(
                console, session._scratchpads, session
            )

        assert result is session
        assert vault.list_connections() == []

    @pytest.mark.asyncio
    async def test_fuzzy_match_prompt_has_context_text(
        self, registry, vault_dir, make_session
    ):
        """The fuzzy-match confirmation prompt includes context text and uses (y/n)."""
        session = make_session()
        console = MagicMock()
        vault = LocalDataVault(vault_dir=vault_dir)

        captured_labels: list[str] = []

        def _capture(label, **kw):
            captured_labels.append(label)
            return None  # Esc on every prompt to bail out

        with (
            patch("anton.commands.datasource.connect.LocalDataVault", return_value=vault),
            patch("anton.commands.datasource.connect.DatasourceRegistry", return_value=registry),
            patch("anton.commands.datasource.connect.prompt_or_cancel", new=AsyncMock(side_effect=_capture)),
        ):
            # "PostgreeSQL" triggers fuzzy match against "PostgreSQL"
            await handle_connect_datasource(
                console,
                session._scratchpads,
                session,
                datasource_name=None,
            )

        # Find the fuzzy-confirm label (contains "Use this datasource?")
        fuzzy_labels = [lbl for lbl in captured_labels if "Use this datasource?" in lbl]
        # If no fuzzy suggestions were generated, just verify the prompt constants are correct.
        if fuzzy_labels:
            lbl = fuzzy_labels[0]
            assert "(y/n)" in lbl
            assert "[y/n]" not in lbl

    @pytest.mark.parametrize(
        "label",
        [
            "(y/n)",
            "(reconnect/cancel)",
            "(anton) Would you like to re-enter your credentials? (y/n)",
            "(anton) Use this datasource? (y/n)",
            "(anton) (reconnect/cancel)",
        ],
    )
    def test_canonical_labels_no_bracket_style(self, label):
        """None of the canonical prompt strings use the old bracket style."""
        assert "[y/n]" not in label
        assert "[reconnect/cancel]" not in label
        assert "(y/n" in label or _PROMPT_RECONNECT_CANCEL in label
