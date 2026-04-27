"""Tests for handle_connect_datasource in tools.py — non-interactive path."""

from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anton.chat import ChatSession
from anton.core.datasources.data_vault import LocalDataVault
from anton.core.session import ChatSessionConfig
from anton.tools import handle_connect_datasource

_UUID8 = re.compile(r"^[0-9a-f]{8}$")


@pytest.fixture()
def vault_dir(tmp_path):
    return tmp_path / "vault"


def _make_session(vault_dir):
    """Return a minimal ChatSession wired for tool-handler tests."""
    from anton.core.llm.provider import ProviderConnectionInfo

    mock_llm = AsyncMock()
    mock_llm.coding_provider = MagicMock()
    mock_llm.coding_provider.export_connection_info = MagicMock(
        return_value=ProviderConnectionInfo(provider="anthropic", api_key="test")
    )
    mock_llm.coding_model = "claude-sonnet-4-6"
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm))
    session._console = MagicMock()
    session._scratchpads = AsyncMock()
    session._data_vault = LocalDataVault(vault_dir=vault_dir)
    session._settings = None
    return session


def _fake_interactive(session):
    """Return an AsyncMock for the interactive handler that exits cleanly."""

    async def _side_effect(console, scratchpads, sess, **kwargs):
        sess._pending_connect_redirect = "handled by test"
        return sess

    return AsyncMock(side_effect=_side_effect)


def _patch_test_ok():
    """Patch run_connection_test (imported inside tools) to always succeed."""
    return patch(
        "anton.commands.datasource.verify.run_connection_test",
        new=AsyncMock(return_value=True),
    )


def _patch_test_fail():
    """Patch run_connection_test to always fail."""
    return patch(
        "anton.commands.datasource.verify.run_connection_test",
        new=AsyncMock(return_value=False),
    )


class TestNonInteractiveSave:
    @pytest.mark.asyncio
    async def test_saves_with_hash_name(self, vault_dir):
        """Full credential set → saved under <engine>-<8-hex>, no prompts."""
        session = _make_session(vault_dir)
        with _patch_test_ok():
            result = await handle_connect_datasource(
                session,
                {
                    "engine": "postgres",
                    "known_variables": {
                        "host": "db.example.com",
                        "database": "sales",
                        "user": "admin",
                        "password": "x",
                    },
                },
            )
        vault = session._data_vault
        conns = vault.list_connections()
        assert len(conns) == 1
        assert conns[0]["engine"] == "postgres"
        assert _UUID8.match(conns[0]["name"])
        # Name is never derived from credential fields like `database`.
        assert conns[0]["name"] != "sales"
        saved = vault.load("postgres", conns[0]["name"])
        assert saved is not None
        assert saved["host"] == "db.example.com"
        assert saved["database"] == "sales"
        # Required `port` (with default) is auto-filled.
        assert saved["port"] == "5432"
        assert result.startswith("Saved connection")

    @pytest.mark.asyncio
    async def test_missing_required_falls_through_to_interactive(self, vault_dir):
        """When required fields are missing, do NOT save; run interactive."""
        session = _make_session(vault_dir)
        interactive = _fake_interactive(session)
        with patch(
            "anton.commands.datasource.handle_connect_datasource", new=interactive
        ), _patch_test_ok():
            await handle_connect_datasource(
                session,
                {
                    "engine": "postgres",
                    "known_variables": {"host": "db.example.com"},
                },
            )
        interactive.assert_called_once()
        # No partial YOLO save.
        assert session._data_vault.list_connections() == []

    @pytest.mark.asyncio
    async def test_test_failure_prevents_save(self, vault_dir):
        """Connection test failure → nothing saved, structured retry message returned."""
        session = _make_session(vault_dir)
        with _patch_test_fail():
            result = await handle_connect_datasource(
                session,
                {
                    "engine": "postgres",
                    "known_variables": {
                        "host": "db.example.com",
                        "database": "sales",
                        "user": "admin",
                        "password": "x",
                    },
                },
            )
        assert session._data_vault.list_connections() == []
        assert result.startswith("Connection test failed")
        assert "'postgres'" in result

    @pytest.mark.asyncio
    async def test_filtered_out_keys_are_warned(self, vault_dir):
        """Keys not belonging to the engine are filtered out and warned to console."""
        session = _make_session(vault_dir)
        with _patch_test_ok():
            await handle_connect_datasource(
                session,
                {
                    "engine": "postgres",
                    "known_variables": {
                        "host": "db.example.com",
                        "database": "sales",
                        "user": "admin",
                        "password": "x",
                        "bogus_field": "ignore-me",
                    },
                },
            )
        printed = " ".join(
            str(c.args[0]) for c in session._console.print.call_args_list if c.args
        )
        assert "Ignoring keys" in printed
        assert "bogus_field" in printed
        saved = session._data_vault.load(
            "postgres", session._data_vault.list_connections()[0]["name"]
        )
        assert "bogus_field" not in saved

    @pytest.mark.asyncio
    async def test_new_connection_reports_saved(self, vault_dir):
        """Happy path: reports Saved, not Updated."""
        session = _make_session(vault_dir)
        with _patch_test_ok():
            result = await handle_connect_datasource(
                session,
                {
                    "engine": "postgres",
                    "known_variables": {
                        "host": "db.example.com",
                        "database": "analytics",
                        "user": "admin",
                        "password": "x",
                    },
                },
            )
        assert result.startswith("Saved connection")
        assert "Updated connection" not in result

    @pytest.mark.asyncio
    async def test_unknown_engine_with_credentials_saves_silently(
        self, vault_dir, tmp_path
    ):
        """Engine not in registry + known_variables → silent YOLO save, no interactive."""
        session = _make_session(vault_dir)
        interactive = _fake_interactive(session)
        ds_md = tmp_path / "datasources.md"
        mock_path = MagicMock()
        mock_path.return_value.expanduser.return_value = ds_md
        with patch(
            "anton.commands.datasource.handle_connect_datasource", new=interactive
        ), patch("anton.utils.datasources.Path", new=mock_path), patch(
            "anton.core.datasources.datasource_registry."
            "DatasourceRegistry._USER_PATH",
            new=ds_md,
        ):
            result = await handle_connect_datasource(
                session,
                {"engine": "posthog", "known_variables": {"api_key": "phx_xyz"}},
            )
        interactive.assert_not_called()
        conns = session._data_vault.list_connections()
        assert len(conns) == 1
        assert conns[0]["engine"] == "posthog"
        assert _UUID8.match(conns[0]["name"])
        assert result.startswith("Saved connection")
        assert session._active_datasource == f"posthog-{conns[0]['name']}"
        # Engine was persisted to the user datasources file.
        assert ds_md.is_file()
        ds_text = ds_md.read_text()
        assert "engine: posthog" in ds_text
        assert "name: api_key" in ds_text
        assert "secret: true" in ds_text

    @pytest.mark.asyncio
    async def test_unknown_engine_secret_heuristic(self, vault_dir, tmp_path):
        """Only field names matching secret-y tokens are marked secret."""
        session = _make_session(vault_dir)
        interactive = _fake_interactive(session)
        ds_md = tmp_path / "datasources.md"
        mock_path = MagicMock()
        mock_path.return_value.expanduser.return_value = ds_md
        with patch(
            "anton.commands.datasource.handle_connect_datasource", new=interactive
        ), patch("anton.utils.datasources.Path", new=mock_path), patch(
            "anton.core.datasources.datasource_registry."
            "DatasourceRegistry._USER_PATH",
            new=ds_md,
        ):
            await handle_connect_datasource(
                session,
                {
                    "engine": "mixpanel",
                    "known_variables": {
                        "api_secret": "s3cr3t",
                        "workspace_id": "ws-1",
                    },
                },
            )
        ds_text = ds_md.read_text()
        api_line = next(ln for ln in ds_text.splitlines() if "api_secret" in ln)
        ws_line = next(ln for ln in ds_text.splitlines() if "workspace_id" in ln)
        assert "secret: true" in api_line
        assert "secret: false" in ws_line

    @pytest.mark.asyncio
    async def test_unknown_engine_no_known_variables_falls_through(self, vault_dir):
        """Engine not in registry AND empty known_variables → still interactive."""
        session = _make_session(vault_dir)
        interactive = _fake_interactive(session)
        with patch("anton.commands.datasource.handle_connect_datasource", new=interactive):
            await handle_connect_datasource(
                session,
                {"engine": "NotARealEngine"},
            )
        interactive.assert_called_once()
        assert session._data_vault.list_connections() == []

    @pytest.mark.asyncio
    async def test_no_field_overlap_falls_through_to_interactive(self, vault_dir):
        """known_variables with no matching engine fields → interactive called, nothing saved."""
        session = _make_session(vault_dir)
        interactive = _fake_interactive(session)
        with patch("anton.commands.datasource.handle_connect_datasource", new=interactive):
            await handle_connect_datasource(
                session,
                {"engine": "postgres", "known_variables": {"wrongfield": "foo"}},
            )
        interactive.assert_called_once()
        assert session._data_vault.list_connections() == []

    @pytest.mark.asyncio
    async def test_scrubbed_placeholder_values_are_dropped(self, vault_dir):
        """Bracketed DS_* values look like scrubber output → drop before save."""
        session = _make_session(vault_dir)
        interactive = _fake_interactive(session)
        with patch(
            "anton.commands.datasource.handle_connect_datasource", new=interactive
        ):
            await handle_connect_datasource(
                session,
                {
                    "engine": "posthog",
                    "known_variables": {
                        "api_key": "[DS_POSTHOG_ABCD1234__API_KEY]",
                    },
                },
            )

        interactive.assert_called_once()
        assert session._data_vault.list_connections() == []
        printed = " ".join(
            str(c.args[0])
            for c in session._console.print.call_args_list
            if c.args
        )
        assert "scrubbed-placeholder" in printed
        assert "api_key" in printed

    @pytest.mark.asyncio
    async def test_mixed_real_and_scrubbed_values(self, vault_dir, tmp_path):
        """Real values are saved; scrubbed placeholders are dropped with a warning."""
        session = _make_session(vault_dir)
        ds_md = tmp_path / "datasources.md"
        mock_path = MagicMock()
        mock_path.return_value.expanduser.return_value = ds_md
        with patch(
            "anton.utils.datasources.Path", new=mock_path
        ), patch(
            "anton.core.datasources.datasource_registry."
            "DatasourceRegistry._USER_PATH",
            new=ds_md,
        ):
            await handle_connect_datasource(
                session,
                {
                    "engine": "posthog",
                    "known_variables": {
                        "api_key": "phx_real",
                        "host": "[DS_POSTHOG_X__HOST]",
                    },
                },
            )
        conns = session._data_vault.list_connections()
        assert len(conns) == 1
        saved = session._data_vault.load("posthog", conns[0]["name"])
        assert saved["api_key"] == "phx_real"
        assert "host" not in saved
        printed = " ".join(
            str(c.args[0])
            for c in session._console.print.call_args_list
            if c.args
        )
        assert "scrubbed-placeholder" in printed
        assert "host" in printed

    @pytest.mark.asyncio
    async def test_yolo_dedups_matching_connection(self, vault_dir):
        """YOLO save for a DB already in the vault reuses the existing slug."""
        session = _make_session(vault_dir)
        session._data_vault.save(
            "postgres",
            "abc12345",
            {
                "host": "db.example.com",
                "port": "5432",
                "database": "sales",
                "user": "admin",
                "password": "old-pw",
            },
        )
        with _patch_test_ok():
            result = await handle_connect_datasource(
                session,
                {
                    "engine": "postgres",
                    "known_variables": {
                        "host": "db.example.com",
                        "database": "sales",
                        "user": "admin",
                        "password": "new-pw",
                    },
                },
            )
        conns = session._data_vault.list_connections()
        assert len(conns) == 1
        assert conns[0]["name"] == "abc12345"
        saved = session._data_vault.load("postgres", "abc12345")
        assert saved["password"] == "new-pw"
        assert result.startswith("Updated connection")
        assert "password" in result

    @pytest.mark.asyncio
    async def test_yolo_different_db_does_not_dedup(self, vault_dir):
        """Different database name → treated as a new connection."""
        session = _make_session(vault_dir)
        session._data_vault.save(
            "postgres",
            "abc12345",
            {
                "host": "db.example.com",
                "database": "sales",
                "user": "admin",
                "password": "pw",
            },
        )
        with _patch_test_ok():
            await handle_connect_datasource(
                session,
                {
                    "engine": "postgres",
                    "known_variables": {
                        "host": "db.example.com",
                        "database": "analytics",
                        "user": "admin",
                        "password": "pw",
                    },
                },
            )
        conns = session._data_vault.list_connections()
        assert len(conns) == 2

    @pytest.mark.asyncio
    async def test_no_known_variables_runs_interactive(self, vault_dir):
        """No known_variables → interactive flow is invoked, no non-interactive save."""
        session = _make_session(vault_dir)
        interactive = _fake_interactive(session)
        with patch("anton.commands.datasource.handle_connect_datasource", new=interactive):
            await handle_connect_datasource(session, {"engine": "postgres"})
        interactive.assert_called_once()
        assert session._data_vault.list_connections() == []
