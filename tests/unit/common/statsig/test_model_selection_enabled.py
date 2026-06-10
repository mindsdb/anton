"""
Unit tests for is_model_selection_enabled.

Tests:
- Self-hosted → returns configured default (Statsig never called)
- Cloud → returns Statsig gate result
- Cloud + Statsig failure → falls back to configured default
"""

from unittest.mock import patch
from uuid import UUID

from minds.common.settings.app_settings import AppSettings, DeploymentMode
from minds.common.statsig.feature_flags.model_selection_enabled import is_model_selection_enabled
from minds.requests.context import Context


def _make_context() -> Context:
    return Context(
        user_id=UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
        organization_id=UUID("11111111-2222-3333-4444-555555555555"),
        user_email="alice@example.com",
        user_roles=["pro"],
    )


def _make_settings(
    deployment_mode: DeploymentMode = DeploymentMode.CLOUD,
    model_selection_default: bool = True,
) -> AppSettings:
    return AppSettings(
        deployment_mode=deployment_mode,
        statsig={
            "sdk_key": "test-key",
            "environment": "test",
            "disable_network": True,
            "disable_all_logging": True,
        },
        feature_flag_enable_model_selection={
            "name": "enable-model-selection",
            "default_value": model_selection_default,
        },
    )


class TestSelfHostedMode:
    """Self-hosted deployment returns the configured default without Statsig."""

    @patch("minds.common.statsig.feature_flags.model_selection_enabled.get_statsig")
    def test_returns_default_true(self, mock_get_statsig):
        ctx = _make_context()
        settings = _make_settings(DeploymentMode.SELF_HOSTED, model_selection_default=True)

        result = is_model_selection_enabled(context=ctx, settings=settings)

        assert result is True, f"Self-hosted with default=True should return True, got {result}"
        mock_get_statsig.assert_not_called()

    @patch("minds.common.statsig.feature_flags.model_selection_enabled.get_statsig")
    def test_returns_default_false(self, mock_get_statsig):
        ctx = _make_context()
        settings = _make_settings(DeploymentMode.SELF_HOSTED, model_selection_default=False)

        result = is_model_selection_enabled(context=ctx, settings=settings)

        assert result is False, f"Self-hosted with default=False should return False, got {result}"
        mock_get_statsig.assert_not_called()


class TestCloudMode:
    """Cloud deployment checks the Statsig gate."""

    @patch("minds.common.statsig.feature_flags.model_selection_enabled.build_statsig_user")
    @patch("minds.common.statsig.feature_flags.model_selection_enabled.get_statsig")
    def test_returns_true_when_gate_enabled(self, mock_get_statsig, mock_build_user):
        mock_get_statsig.return_value.check_gate.return_value = True

        ctx = _make_context()
        settings = _make_settings(DeploymentMode.CLOUD)
        result = is_model_selection_enabled(context=ctx, settings=settings)

        assert result is True, f"Cloud with gate enabled should return True, got {result}"

    @patch("minds.common.statsig.feature_flags.model_selection_enabled.build_statsig_user")
    @patch("minds.common.statsig.feature_flags.model_selection_enabled.get_statsig")
    def test_returns_false_when_gate_disabled(self, mock_get_statsig, mock_build_user):
        mock_get_statsig.return_value.check_gate.return_value = False

        ctx = _make_context()
        settings = _make_settings(DeploymentMode.CLOUD)
        result = is_model_selection_enabled(context=ctx, settings=settings)

        assert result is False, f"Cloud with gate disabled should return False, got {result}"

    @patch("minds.common.statsig.feature_flags.model_selection_enabled.build_statsig_user")
    @patch("minds.common.statsig.feature_flags.model_selection_enabled.get_statsig")
    def test_calls_check_gate_with_correct_name(self, mock_get_statsig, mock_build_user):
        mock_statsig = mock_get_statsig.return_value
        mock_statsig.check_gate.return_value = True

        ctx = _make_context()
        settings = _make_settings(DeploymentMode.CLOUD)
        is_model_selection_enabled(context=ctx, settings=settings)

        mock_statsig.check_gate.assert_called_once()
        call_kwargs = mock_statsig.check_gate.call_args
        assert call_kwargs.kwargs["name"] == "enable-model-selection", (
            f"Gate name should be 'enable-model-selection', got '{call_kwargs.kwargs.get('name')}'"
        )


class TestCloudStatsigFailure:
    """Cloud mode falls back to the configured default when Statsig fails."""

    @patch("minds.common.statsig.feature_flags.model_selection_enabled.get_statsig")
    def test_falls_back_to_default_true_on_exception(self, mock_get_statsig):
        mock_get_statsig.side_effect = RuntimeError("Statsig is down")

        ctx = _make_context()
        settings = _make_settings(DeploymentMode.CLOUD, model_selection_default=True)
        result = is_model_selection_enabled(context=ctx, settings=settings)

        assert result is True, f"Fallback with default=True should return True, got {result}"

    @patch("minds.common.statsig.feature_flags.model_selection_enabled.get_statsig")
    def test_falls_back_to_default_false_on_exception(self, mock_get_statsig):
        mock_get_statsig.side_effect = RuntimeError("Statsig is down")

        ctx = _make_context()
        settings = _make_settings(DeploymentMode.CLOUD, model_selection_default=False)
        result = is_model_selection_enabled(context=ctx, settings=settings)

        assert result is False, f"Fallback with default=False should return False, got {result}"

    @patch("minds.common.statsig.feature_flags.model_selection_enabled.build_statsig_user")
    @patch("minds.common.statsig.feature_flags.model_selection_enabled.get_statsig")
    def test_falls_back_when_check_gate_raises(self, mock_get_statsig, mock_build_user):
        mock_get_statsig.return_value.check_gate.side_effect = Exception("gate error")

        ctx = _make_context()
        settings = _make_settings(DeploymentMode.CLOUD, model_selection_default=True)
        result = is_model_selection_enabled(context=ctx, settings=settings)

        assert result is True, f"Fallback on gate error with default=True should return True, got {result}"
