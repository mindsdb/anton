from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from minds.common.launch_darkly.disable_langfuse import is_langfuse_disabled
from minds.requests.context import Context


@pytest.fixture
def mock_context():
    """Fixture for creating a test context with a user email."""
    return Context(
        user_id=UUID("00000000-0000-0000-0000-000000000001"),
        tenant_id=UUID("00000000-0000-0000-0000-000000000002"),
        user_email="test@example.com",
    )


class TestIsLangfuseDisabled:
    """Test cases for is_langfuse_disabled function."""

    @patch("minds.common.launch_darkly.disable_langfuse.settings")
    @patch("minds.common.launch_darkly.disable_langfuse.ldclient")
    def test_is_langfuse_disabled_returns_false_when_feature_flag_disabled(
        self, mock_ldclient_module, mock_settings_module, mock_context
    ):
        """Test that Langfuse is NOT disabled when feature flag is off."""
        # Arrange
        mock_settings_module.feature_flag_disable_langfuse.name = "disable-langfuse"
        mock_settings_module.feature_flag_disable_langfuse.default_value = False

        mock_ld_context = MagicMock()
        mock_context_builder = MagicMock()
        mock_context_builder.kind.return_value = mock_context_builder
        mock_context_builder.name.return_value = mock_context_builder
        mock_context_builder.set.return_value = mock_context_builder
        mock_context_builder.build.return_value = mock_ld_context
        mock_ldclient_module.Context.builder.return_value = mock_context_builder
        mock_ldclient_module.get.return_value.variation.return_value = False

        # Act
        result = is_langfuse_disabled(mock_context)

        # Assert
        assert result is False
        mock_ldclient_module.Context.builder.assert_called_once_with("test@example.com")
        mock_context_builder.kind.assert_called_once_with("user")
        mock_context_builder.name.assert_called_once_with("test@example.com")
        mock_context_builder.set.assert_called_once_with("email", "test@example.com")
        mock_ldclient_module.get.return_value.variation.assert_called_once_with(
            "disable-langfuse", mock_ld_context, False
        )

    @patch("minds.common.launch_darkly.disable_langfuse.settings")
    @patch("minds.common.launch_darkly.disable_langfuse.ldclient")
    def test_is_langfuse_disabled_returns_true_when_feature_flag_enabled(
        self, mock_ldclient_module, mock_settings_module, mock_context
    ):
        """Test that Langfuse IS disabled when feature flag is on."""
        # Arrange
        mock_settings_module.feature_flag_disable_langfuse.name = "disable-langfuse"
        mock_settings_module.feature_flag_disable_langfuse.default_value = False

        mock_ld_context = MagicMock()
        mock_context_builder = MagicMock()
        mock_context_builder.kind.return_value = mock_context_builder
        mock_context_builder.name.return_value = mock_context_builder
        mock_context_builder.set.return_value = mock_context_builder
        mock_context_builder.build.return_value = mock_ld_context
        mock_ldclient_module.Context.builder.return_value = mock_context_builder
        mock_ldclient_module.get.return_value.variation.return_value = True

        # Act
        result = is_langfuse_disabled(mock_context)

        # Assert
        assert result is True
        mock_ldclient_module.get.return_value.variation.assert_called_once_with(
            "disable-langfuse", mock_ld_context, False
        )

    @patch("minds.common.launch_darkly.disable_langfuse.settings")
    @patch("minds.common.launch_darkly.disable_langfuse.ldclient")
    def test_is_langfuse_disabled_uses_custom_feature_flag_name(
        self, mock_ldclient_module, mock_settings_module, mock_context
    ):
        """Test that the function uses the custom feature flag name from settings."""
        # Arrange
        mock_settings_module.feature_flag_disable_langfuse.name = "custom-flag-name"
        mock_settings_module.feature_flag_disable_langfuse.default_value = False

        mock_ld_context = MagicMock()
        mock_context_builder = MagicMock()
        mock_context_builder.kind.return_value = mock_context_builder
        mock_context_builder.name.return_value = mock_context_builder
        mock_context_builder.set.return_value = mock_context_builder
        mock_context_builder.build.return_value = mock_ld_context
        mock_ldclient_module.Context.builder.return_value = mock_context_builder
        mock_ldclient_module.get.return_value.variation.return_value = False

        # Act
        result = is_langfuse_disabled(mock_context)

        # Assert
        assert result is False
        mock_ldclient_module.get.return_value.variation.assert_called_once_with(
            "custom-flag-name", mock_ld_context, False
        )

    @patch("minds.common.launch_darkly.disable_langfuse.settings")
    @patch("minds.common.launch_darkly.disable_langfuse.ldclient")
    def test_is_langfuse_disabled_uses_custom_default_value(
        self, mock_ldclient_module, mock_settings_module, mock_context
    ):
        """Test that the function uses the custom default value from settings."""
        # Arrange
        mock_settings_module.feature_flag_disable_langfuse.name = "disable-langfuse"
        mock_settings_module.feature_flag_disable_langfuse.default_value = True

        mock_ld_context = MagicMock()
        mock_context_builder = MagicMock()
        mock_context_builder.kind.return_value = mock_context_builder
        mock_context_builder.name.return_value = mock_context_builder
        mock_context_builder.set.return_value = mock_context_builder
        mock_context_builder.build.return_value = mock_ld_context
        mock_ldclient_module.Context.builder.return_value = mock_context_builder
        mock_ldclient_module.get.return_value.variation.return_value = True

        # Act
        result = is_langfuse_disabled(mock_context)

        # Assert
        assert result is True
        mock_ldclient_module.get.return_value.variation.assert_called_once_with(
            "disable-langfuse", mock_ld_context, True
        )

    @patch("minds.common.launch_darkly.disable_langfuse.settings")
    @patch("minds.common.launch_darkly.disable_langfuse.ldclient")
    def test_is_langfuse_disabled_builds_context_correctly(self, mock_ldclient_module, mock_settings_module):
        """Test that the LaunchDarkly context is built correctly with user information."""
        # Arrange
        test_context = Context(
            user_id=UUID("11111111-1111-1111-1111-111111111111"),
            tenant_id=UUID("22222222-2222-2222-2222-222222222222"),
            user_email="john.doe@example.com",
        )
        mock_settings_module.feature_flag_disable_langfuse.name = "disable-langfuse"
        mock_settings_module.feature_flag_disable_langfuse.default_value = False

        mock_ld_context = MagicMock()
        mock_context_builder = MagicMock()
        mock_context_builder.kind.return_value = mock_context_builder
        mock_context_builder.name.return_value = mock_context_builder
        mock_context_builder.set.return_value = mock_context_builder
        mock_context_builder.build.return_value = mock_ld_context
        mock_ldclient_module.Context.builder.return_value = mock_context_builder
        mock_ldclient_module.get.return_value.variation.return_value = False

        # Act
        is_langfuse_disabled(test_context)

        # Assert
        mock_ldclient_module.Context.builder.assert_called_once_with("john.doe@example.com")
        mock_context_builder.kind.assert_called_once_with("user")
        mock_context_builder.name.assert_called_once_with("john.doe@example.com")
        mock_context_builder.set.assert_called_once_with("email", "john.doe@example.com")
        mock_context_builder.build.assert_called_once()

    @patch("minds.common.launch_darkly.disable_langfuse.settings")
    @patch("minds.common.launch_darkly.disable_langfuse.ldclient")
    def test_is_langfuse_disabled_with_different_user_emails(self, mock_ldclient_module, mock_settings_module):
        """Test that the function works correctly with different user emails."""
        # Arrange
        mock_settings_module.feature_flag_disable_langfuse.name = "disable-langfuse"
        mock_settings_module.feature_flag_disable_langfuse.default_value = False

        user_emails = ["alice@example.com", "bob@test.org", "charlie@company.net"]

        for email in user_emails:
            test_context = Context(
                user_id=UUID("00000000-0000-0000-0000-000000000001"),
                tenant_id=UUID("00000000-0000-0000-0000-000000000002"),
                user_email=email,
            )

            mock_ld_context = MagicMock()
            mock_context_builder = MagicMock()
            mock_context_builder.kind.return_value = mock_context_builder
            mock_context_builder.name.return_value = mock_context_builder
            mock_context_builder.set.return_value = mock_context_builder
            mock_context_builder.build.return_value = mock_ld_context
            mock_ldclient_module.Context.builder.return_value = mock_context_builder
            mock_ldclient_module.get.return_value.variation.return_value = False

            # Act
            result = is_langfuse_disabled(test_context)

            # Assert
            assert result is False
            mock_ldclient_module.Context.builder.assert_called_with(email)
            mock_context_builder.set.assert_called_with("email", email)

    @patch("minds.common.launch_darkly.disable_langfuse.logger")
    @patch("minds.common.launch_darkly.disable_langfuse.settings")
    @patch("minds.common.launch_darkly.disable_langfuse.ldclient")
    def test_is_langfuse_disabled_logs_debug_information(
        self, mock_ldclient_module, mock_settings_module, mock_logger, mock_context
    ):
        """Test that the function logs appropriate debug information."""
        # Arrange
        mock_settings_module.feature_flag_disable_langfuse.name = "disable-langfuse"
        mock_settings_module.feature_flag_disable_langfuse.default_value = False

        mock_ld_context = MagicMock()
        mock_context_builder = MagicMock()
        mock_context_builder.kind.return_value = mock_context_builder
        mock_context_builder.name.return_value = mock_context_builder
        mock_context_builder.set.return_value = mock_context_builder
        mock_context_builder.build.return_value = mock_ld_context
        mock_ldclient_module.Context.builder.return_value = mock_context_builder
        mock_ldclient_module.get.return_value.variation.return_value = False

        # Act
        is_langfuse_disabled(mock_context)

        # Assert
        assert mock_logger.debug.call_count >= 3
        mock_logger.debug.assert_any_call("Checking if Langfuse is disabled: test@example.com")
        mock_logger.debug.assert_any_call("Feature flag name: disable-langfuse")
        mock_logger.debug.assert_any_call("Feature flag default value: False")

    @patch("minds.common.launch_darkly.disable_langfuse.settings")
    @patch("minds.common.launch_darkly.disable_langfuse.ldclient")
    def test_is_langfuse_disabled_returns_default_on_launchdarkly_error(
        self, mock_ldclient_module, mock_settings_module, mock_context
    ):
        """Test that the function returns the default value when LaunchDarkly fails."""
        # Arrange
        mock_settings_module.feature_flag_disable_langfuse.name = "disable-langfuse"
        mock_settings_module.feature_flag_disable_langfuse.default_value = False

        mock_ld_context = MagicMock()
        mock_context_builder = MagicMock()
        mock_context_builder.kind.return_value = mock_context_builder
        mock_context_builder.name.return_value = mock_context_builder
        mock_context_builder.set.return_value = mock_context_builder
        mock_context_builder.build.return_value = mock_ld_context
        mock_ldclient_module.Context.builder.return_value = mock_context_builder

        # Simulate LaunchDarkly error by having variation return the default value
        mock_ldclient_module.get.return_value.variation.return_value = False

        # Act
        result = is_langfuse_disabled(mock_context)

        # Assert
        assert result is False
        mock_ldclient_module.get.return_value.variation.assert_called_once_with(
            "disable-langfuse", mock_ld_context, False
        )

    @patch("minds.common.launch_darkly.disable_langfuse.settings")
    @patch("minds.common.launch_darkly.disable_langfuse.ldclient")
    def test_is_langfuse_disabled_with_empty_user_email(self, mock_ldclient_module, mock_settings_module):
        """Test that the function handles empty user_email gracefully."""
        # Arrange
        test_context = Context(
            user_id=UUID("00000000-0000-0000-0000-000000000001"),
            tenant_id=UUID("00000000-0000-0000-0000-000000000002"),
            user_email="",
        )
        mock_settings_module.feature_flag_disable_langfuse.name = "disable-langfuse"
        mock_settings_module.feature_flag_disable_langfuse.default_value = False

        mock_ld_context = MagicMock()
        mock_context_builder = MagicMock()
        mock_context_builder.kind.return_value = mock_context_builder
        mock_context_builder.name.return_value = mock_context_builder
        mock_context_builder.set.return_value = mock_context_builder
        mock_context_builder.build.return_value = mock_ld_context
        mock_ldclient_module.Context.builder.return_value = mock_context_builder
        mock_ldclient_module.get.return_value.variation.return_value = False

        # Act
        result = is_langfuse_disabled(test_context)

        # Assert
        assert result is False
        mock_ldclient_module.Context.builder.assert_called_once_with("")
        mock_context_builder.name.assert_called_once_with("")
        mock_context_builder.set.assert_called_once_with("email", "")
