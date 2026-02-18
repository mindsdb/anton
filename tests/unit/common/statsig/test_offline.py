"""
Unit tests for Statsig offline overrides.

Tests that gate overrides are applied when network is disabled,
and skipped when network is enabled.
"""

from unittest.mock import MagicMock

from minds.common.settings.app_settings import AppSettings
from minds.common.statsig.offline import apply_offline_overrides


def _make_settings(disable_network: bool = True) -> AppSettings:
    return AppSettings(
        statsig={
            "sdk_key": "test-key",
            "environment": "test",
            "disable_network": disable_network,
            "disable_all_logging": True,
        },
        feature_flag_enable_langfuse={"name": "enable-langfuse", "default_value": True},
    )


class TestApplyOfflineOverrides:
    """Tests for apply_offline_overrides."""

    def test_overrides_applied_when_network_disabled(self):
        mock_statsig = MagicMock()
        settings = _make_settings(disable_network=True)

        apply_offline_overrides(statsig=mock_statsig, settings=settings)

        mock_statsig.override_gate.assert_called_once_with("enable-langfuse", True)

    def test_overrides_skipped_when_network_enabled(self):
        mock_statsig = MagicMock()
        settings = _make_settings(disable_network=False)

        apply_offline_overrides(statsig=mock_statsig, settings=settings)

        mock_statsig.override_gate.assert_not_called()
