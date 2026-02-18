"""Tests for the minds.common.feature_flags module (Statsig-based)."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from minds.common.settings.app_settings import (
    AppSettings,
    FeatureFlagSettings,
    StatsigSettings,
)
from minds.requests.context import Context

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_settings() -> AppSettings:
    """Return an AppSettings instance with Statsig in offline/no-network mode."""
    return AppSettings(
        env="local",
        statsig=StatsigSettings(
            sdk_key="secret-test-key",
            disable_network=True,
            disable_all_logging=True,
            environment="test",
        ),
        feature_flag_enable_langfuse=FeatureFlagSettings(
            name="enable-langfuse",
            default_value=True,
        ),
    )


@pytest.fixture
def mock_settings_network_enabled() -> AppSettings:
    """Return AppSettings with networking enabled."""
    return AppSettings(
        env="staging",
        statsig=StatsigSettings(
            sdk_key="secret-staging-key",
            disable_network=False,
            disable_all_logging=False,
            environment="staging",
        ),
        feature_flag_enable_langfuse=FeatureFlagSettings(
            name="enable-langfuse",
            default_value=True,
        ),
    )


@pytest.fixture
def mock_context() -> Context:
    return Context(
        user_id=UUID("00000000-0000-0000-0000-000000000001"),
        organization_id=UUID("00000000-0000-0000-0000-000000000002"),
        user_email="test@example.com",
    )


# ===================================================================
# Tests for client.py
# ===================================================================


class TestBuildStatsigOptions:
    """Tests for build_statsig_options."""

    def test_builds_options_from_settings(self, mock_settings):
        from minds.common.feature_flags.client import build_statsig_options

        options = build_statsig_options(mock_settings)

        assert options.environment == "test"
        assert options.disable_network is True
        assert options.disable_all_logging is True

    def test_builds_options_with_network_enabled(self, mock_settings_network_enabled):
        from minds.common.feature_flags.client import build_statsig_options

        options = build_statsig_options(mock_settings_network_enabled)

        assert options.environment == "staging"
        assert options.disable_network is False
        assert options.disable_all_logging is False


class TestInitStatsig:
    """Tests for init_statsig."""

    @patch("minds.common.feature_flags.client.atexit")
    @patch("minds.common.feature_flags.client.apply_offline_overrides")
    @patch("minds.common.feature_flags.client.Statsig")
    def test_initializes_client(self, mock_statsig_cls, mock_apply_overrides, mock_atexit, mock_settings):
        import minds.common.feature_flags.client as client_mod

        # Reset module-level _client
        client_mod._client = None
        try:
            mock_instance = MagicMock()
            mock_future = MagicMock()
            mock_instance.initialize.return_value = mock_future
            mock_statsig_cls.return_value = mock_instance

            result = client_mod.init_statsig(settings=mock_settings)

            # Statsig was constructed with the right SDK key + options
            mock_statsig_cls.assert_called_once()
            args, kwargs = mock_statsig_cls.call_args
            assert kwargs["sdk_key"] == "secret-test-key"

            # initialize().wait() was called
            mock_instance.initialize.assert_called_once()
            mock_future.wait.assert_called_once()

            # Offline overrides applied
            mock_apply_overrides.assert_called_once_with(statsig=mock_instance, settings=mock_settings)

            # atexit registered
            mock_atexit.register.assert_called_once()

            assert result is mock_instance
        finally:
            client_mod._client = None

    @patch("minds.common.feature_flags.client.atexit")
    @patch("minds.common.feature_flags.client.apply_offline_overrides")
    @patch("minds.common.feature_flags.client.Statsig")
    def test_returns_existing_client_on_second_call(
        self, mock_statsig_cls, mock_apply_overrides, mock_atexit, mock_settings
    ):
        import minds.common.feature_flags.client as client_mod

        client_mod._client = None
        try:
            mock_instance = MagicMock()
            mock_instance.initialize.return_value = MagicMock()
            mock_statsig_cls.return_value = mock_instance

            first = client_mod.init_statsig(settings=mock_settings)
            second = client_mod.init_statsig(settings=mock_settings)

            assert first is second
            # Constructor only called once
            assert mock_statsig_cls.call_count == 1
        finally:
            client_mod._client = None

    @patch("minds.common.feature_flags.client.atexit")
    @patch("minds.common.feature_flags.client.apply_offline_overrides")
    @patch("minds.common.feature_flags.client.Statsig")
    def test_thread_safety(self, mock_statsig_cls, mock_apply_overrides, mock_atexit, mock_settings):
        """Concurrent init_statsig calls should only create one client."""
        import minds.common.feature_flags.client as client_mod

        client_mod._client = None
        try:
            mock_instance = MagicMock()
            mock_instance.initialize.return_value = MagicMock()
            mock_statsig_cls.return_value = mock_instance

            results: list = []
            errors: list = []

            def _init():
                try:
                    results.append(client_mod.init_statsig(settings=mock_settings))
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=_init) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors
            # All threads got the same instance
            assert all(r is results[0] for r in results)
            # Constructor called at most once
            assert mock_statsig_cls.call_count == 1
        finally:
            client_mod._client = None


class TestGetStatsig:
    """Tests for get_statsig."""

    @patch("minds.common.feature_flags.client.init_statsig")
    def test_returns_existing_client(self, mock_init, mock_settings):
        import minds.common.feature_flags.client as client_mod

        sentinel = MagicMock()
        client_mod._client = sentinel
        try:
            result = client_mod.get_statsig(settings=mock_settings)
            assert result is sentinel
            mock_init.assert_not_called()
        finally:
            client_mod._client = None

    @patch("minds.common.feature_flags.client.init_statsig")
    def test_initializes_when_no_client(self, mock_init, mock_settings):
        import minds.common.feature_flags.client as client_mod

        client_mod._client = None
        try:
            mock_instance = MagicMock()

            def _side_effect(settings):
                client_mod._client = mock_instance

            mock_init.side_effect = _side_effect

            result = client_mod.get_statsig(settings=mock_settings)

            mock_init.assert_called_once_with(settings=mock_settings)
            assert result is mock_instance
        finally:
            client_mod._client = None


class TestShutdownStatsig:
    """Tests for shutdown_statsig."""

    def test_shutdown_calls_wait(self):
        import minds.common.feature_flags.client as client_mod

        mock_instance = MagicMock()
        mock_future = MagicMock()
        mock_instance.shutdown.return_value = mock_future
        client_mod._client = mock_instance
        try:
            client_mod.shutdown_statsig()
            mock_instance.shutdown.assert_called_once()
            mock_future.wait.assert_called_once()
            assert client_mod._client is None
        finally:
            client_mod._client = None

    def test_shutdown_noop_when_no_client(self):
        import minds.common.feature_flags.client as client_mod

        client_mod._client = None
        # Should not raise
        client_mod.shutdown_statsig()
        assert client_mod._client is None

    def test_shutdown_resets_client_on_exception(self):
        import minds.common.feature_flags.client as client_mod

        mock_instance = MagicMock()
        mock_instance.shutdown.side_effect = RuntimeError("shutdown failed")
        client_mod._client = mock_instance
        try:
            with pytest.raises(RuntimeError, match="shutdown failed"):
                client_mod.shutdown_statsig()
            # _client is still reset to None
            assert client_mod._client is None
        finally:
            client_mod._client = None


# ===================================================================
# Tests for offline.py
# ===================================================================


class TestApplyOfflineOverrides:
    """Tests for apply_offline_overrides."""

    def test_overrides_gate_when_network_disabled(self, mock_settings):
        from minds.common.feature_flags.offline import apply_offline_overrides

        mock_statsig = MagicMock()
        apply_offline_overrides(statsig=mock_statsig, settings=mock_settings)

        mock_statsig.override_gate.assert_called_once_with("enable-langfuse", True)

    def test_no_overrides_when_network_enabled(self, mock_settings_network_enabled):
        from minds.common.feature_flags.offline import apply_offline_overrides

        mock_statsig = MagicMock()
        apply_offline_overrides(statsig=mock_statsig, settings=mock_settings_network_enabled)

        mock_statsig.override_gate.assert_not_called()

    def test_uses_feature_flag_name_from_settings(self):
        from minds.common.feature_flags.offline import apply_offline_overrides

        settings = AppSettings(
            statsig=StatsigSettings(disable_network=True),
            feature_flag_enable_langfuse=FeatureFlagSettings(
                name="custom-langfuse-gate",
                default_value=False,
            ),
        )
        mock_statsig = MagicMock()

        apply_offline_overrides(statsig=mock_statsig, settings=settings)

        mock_statsig.override_gate.assert_called_once_with("custom-langfuse-gate", True)


# ===================================================================
# Tests for users.py
# ===================================================================


class TestBuildStatsigUser:
    """Tests for build_statsig_user."""

    def test_builds_user_from_context(self, mock_context, mock_settings):
        from minds.common.feature_flags.users import build_statsig_user

        user = build_statsig_user(context=mock_context, settings=mock_settings)

        assert user.user_id == "00000000-0000-0000-0000-000000000001"
        assert user.email == "test@example.com"
        assert user.custom["organization_id"] == "00000000-0000-0000-0000-000000000002"
        assert user.custom["user_id"] == "00000000-0000-0000-0000-000000000001"
        assert user.custom["env"] == "local"

    def test_builds_user_with_different_context(self, mock_settings):
        from minds.common.feature_flags.users import build_statsig_user

        context = Context(
            user_id=UUID("11111111-1111-1111-1111-111111111111"),
            organization_id=UUID("22222222-2222-2222-2222-222222222222"),
            user_email="john.doe@example.com",
        )
        user = build_statsig_user(context=context, settings=mock_settings)

        assert user.user_id == "11111111-1111-1111-1111-111111111111"
        assert user.email == "john.doe@example.com"
        assert user.custom["organization_id"] == "22222222-2222-2222-2222-222222222222"

    def test_builds_user_with_empty_email(self, mock_settings):
        from minds.common.feature_flags.users import build_statsig_user

        context = Context(
            user_id=UUID("00000000-0000-0000-0000-000000000001"),
            organization_id=UUID("00000000-0000-0000-0000-000000000002"),
            user_email="",
        )
        user = build_statsig_user(context=context, settings=mock_settings)

        assert user.email == ""

    @patch("minds.common.feature_flags.users.get_app_settings")
    def test_uses_default_settings_when_none(self, mock_get_settings, mock_context, mock_settings):
        from minds.common.feature_flags.users import build_statsig_user

        mock_get_settings.return_value = mock_settings

        user = build_statsig_user(context=mock_context, settings=None)

        mock_get_settings.assert_called_once()
        assert user.user_id == "00000000-0000-0000-0000-000000000001"


# ===================================================================
# Tests for flags.py  (is_langfuse_enabled)
# ===================================================================


class TestIsLangfuseEnabled:
    """Tests for is_langfuse_enabled function (replaces old is_langfuse_disabled tests)."""

    @patch("minds.common.feature_flags.flags.get_statsig")
    @patch("minds.common.feature_flags.flags.build_statsig_user")
    def test_returns_true_when_gate_enabled(self, mock_build_user, mock_get_statsig, mock_context, mock_settings):
        from minds.common.feature_flags.flags import is_langfuse_enabled

        mock_user = MagicMock()
        mock_build_user.return_value = mock_user
        mock_statsig = MagicMock()
        mock_statsig.check_gate.return_value = True
        mock_get_statsig.return_value = mock_statsig

        result = is_langfuse_enabled(context=mock_context, settings=mock_settings)

        assert result is True
        mock_get_statsig.assert_called_once_with(settings=mock_settings)
        mock_build_user.assert_called_once_with(context=mock_context, settings=mock_settings)
        mock_statsig.check_gate.assert_called_once_with(user=mock_user, name="enable-langfuse")

    @patch("minds.common.feature_flags.flags.get_statsig")
    @patch("minds.common.feature_flags.flags.build_statsig_user")
    def test_returns_false_when_gate_disabled(self, mock_build_user, mock_get_statsig, mock_context, mock_settings):
        from minds.common.feature_flags.flags import is_langfuse_enabled

        mock_user = MagicMock()
        mock_build_user.return_value = mock_user
        mock_statsig = MagicMock()
        mock_statsig.check_gate.return_value = False
        mock_get_statsig.return_value = mock_statsig

        result = is_langfuse_enabled(context=mock_context, settings=mock_settings)

        assert result is False

    @patch("minds.common.feature_flags.flags.get_statsig")
    @patch("minds.common.feature_flags.flags.build_statsig_user")
    def test_uses_custom_gate_name(self, mock_build_user, mock_get_statsig, mock_context):
        from minds.common.feature_flags.flags import is_langfuse_enabled

        settings = AppSettings(
            feature_flag_enable_langfuse=FeatureFlagSettings(
                name="custom-gate-name",
                default_value=False,
            ),
        )
        mock_user = MagicMock()
        mock_build_user.return_value = mock_user
        mock_statsig = MagicMock()
        mock_statsig.check_gate.return_value = True
        mock_get_statsig.return_value = mock_statsig

        is_langfuse_enabled(context=mock_context, settings=settings)

        mock_statsig.check_gate.assert_called_once_with(user=mock_user, name="custom-gate-name")

    @patch("minds.common.feature_flags.flags.get_statsig")
    @patch("minds.common.feature_flags.flags.build_statsig_user")
    def test_passes_correct_user_to_check_gate(self, mock_build_user, mock_get_statsig, mock_settings):
        from minds.common.feature_flags.flags import is_langfuse_enabled

        context = Context(
            user_id=UUID("11111111-1111-1111-1111-111111111111"),
            organization_id=UUID("22222222-2222-2222-2222-222222222222"),
            user_email="john.doe@example.com",
        )
        mock_user = MagicMock()
        mock_build_user.return_value = mock_user
        mock_statsig = MagicMock()
        mock_statsig.check_gate.return_value = False
        mock_get_statsig.return_value = mock_statsig

        is_langfuse_enabled(context=context, settings=mock_settings)

        mock_build_user.assert_called_once_with(context=context, settings=mock_settings)

    @patch("minds.common.feature_flags.flags.logger")
    @patch("minds.common.feature_flags.flags.get_statsig")
    @patch("minds.common.feature_flags.flags.build_statsig_user")
    def test_logs_debug_information(self, mock_build_user, mock_get_statsig, mock_logger, mock_context, mock_settings):
        from minds.common.feature_flags.flags import is_langfuse_enabled

        mock_build_user.return_value = MagicMock()
        mock_statsig = MagicMock()
        mock_statsig.check_gate.return_value = True
        mock_get_statsig.return_value = mock_statsig

        is_langfuse_enabled(context=mock_context, settings=mock_settings)

        assert mock_logger.debug.call_count >= 3
        mock_logger.debug.assert_any_call("Checking if Langfuse is enabled: test@example.com")
        mock_logger.debug.assert_any_call("Feature flag name: enable-langfuse")
        mock_logger.debug.assert_any_call("Feature flag default value: True")

    @patch("minds.common.feature_flags.flags.get_statsig")
    @patch("minds.common.feature_flags.flags.build_statsig_user")
    def test_with_different_user_emails(self, mock_build_user, mock_get_statsig, mock_settings):
        from minds.common.feature_flags.flags import is_langfuse_enabled

        mock_statsig = MagicMock()
        mock_statsig.check_gate.return_value = True
        mock_get_statsig.return_value = mock_statsig

        emails = ["alice@example.com", "bob@test.org", "charlie@company.net"]
        for email in emails:
            context = Context(
                user_id=UUID("00000000-0000-0000-0000-000000000001"),
                organization_id=UUID("00000000-0000-0000-0000-000000000002"),
                user_email=email,
            )
            mock_user = MagicMock()
            mock_build_user.return_value = mock_user

            result = is_langfuse_enabled(context=context, settings=mock_settings)

            assert result is True
            mock_build_user.assert_called_with(context=context, settings=mock_settings)

    @patch("minds.common.feature_flags.flags.get_statsig")
    @patch("minds.common.feature_flags.flags.build_statsig_user")
    def test_with_empty_user_email(self, mock_build_user, mock_get_statsig, mock_settings):
        from minds.common.feature_flags.flags import is_langfuse_enabled

        context = Context(
            user_id=UUID("00000000-0000-0000-0000-000000000001"),
            organization_id=UUID("00000000-0000-0000-0000-000000000002"),
            user_email="",
        )
        mock_user = MagicMock()
        mock_build_user.return_value = mock_user
        mock_statsig = MagicMock()
        mock_statsig.check_gate.return_value = False
        mock_get_statsig.return_value = mock_statsig

        result = is_langfuse_enabled(context=context, settings=mock_settings)

        assert result is False
        mock_build_user.assert_called_once_with(context=context, settings=mock_settings)

    @patch("minds.common.feature_flags.flags.get_app_settings")
    @patch("minds.common.feature_flags.flags.get_statsig")
    @patch("minds.common.feature_flags.flags.build_statsig_user")
    def test_uses_default_settings_when_none(
        self, mock_build_user, mock_get_statsig, mock_get_settings, mock_context, mock_settings
    ):
        from minds.common.feature_flags.flags import is_langfuse_enabled

        mock_get_settings.return_value = mock_settings
        mock_build_user.return_value = MagicMock()
        mock_statsig = MagicMock()
        mock_statsig.check_gate.return_value = True
        mock_get_statsig.return_value = mock_statsig

        result = is_langfuse_enabled(context=mock_context, settings=None)

        assert result is True
        mock_get_settings.assert_called_once()
