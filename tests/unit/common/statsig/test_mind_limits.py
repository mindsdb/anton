"""
Unit tests for get_mind_limits_config.

Tests:
- Self-hosted → unlimited (Statsig never called)
- Cloud → Statsig values parsed correctly
- Cloud + Statsig failure → fail open (unlimited)
"""

from unittest.mock import MagicMock, patch
from uuid import UUID

from minds.common.settings.app_settings import AppSettings, DeploymentMode
from minds.common.statsig.dynamic_config.mind_limits import get_mind_limits_config
from minds.requests.context import Context
from minds.schemas.limits import UNLIMITED, MindLimitsConfig, ResourceUsageConfig, UsageConfig


def _resource_fields() -> list[str]:
    """Return all ResourceUsageConfig field names on MindLimitsConfig."""
    return [
        name
        for name, field_info in MindLimitsConfig.model_fields.items()
        if field_info.annotation is ResourceUsageConfig
    ]


def _make_context() -> Context:
    return Context(
        user_id=UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
        organization_id=UUID("11111111-2222-3333-4444-555555555555"),
        user_email="alice@example.com",
        user_roles=["pro"],
    )


def _make_settings(deployment_mode: DeploymentMode = DeploymentMode.CLOUD) -> AppSettings:
    return AppSettings(
        deployment_mode=deployment_mode,
        statsig={
            "sdk_key": "test-key",
            "environment": "test",
            "disable_network": True,
            "disable_all_logging": True,
        },
    )


# Realistic Statsig response payload — must cover every resource field.
STATSIG_PAYLOAD = {
    "tokens": {"limit": {"lifetime": "-1", "monthly": "1000000"}},
    "minds": {"limit": {"lifetime": "1", "monthly": "1"}},
    "datasources": {"limit": {"lifetime": "3", "monthly": "3"}},
    "questions": {"limit": {"lifetime": "-1", "monthly": "250"}},
}


class TestStatsigPayloadCoverage:
    """Ensure STATSIG_PAYLOAD covers all resource fields on MindLimitsConfig."""

    def test_payload_covers_all_resource_fields(self):
        actual_fields = set(_resource_fields())
        payload_fields = set(STATSIG_PAYLOAD.keys())
        missing = actual_fields - payload_fields
        assert not missing, (
            f"STATSIG_PAYLOAD is missing keys for resource fields: {missing}. "
            f"Add them so cloud-mode parsing tests cover all resources."
        )


class TestSelfHostedMode:
    """Self-hosted deployment should return unlimited without touching Statsig."""

    @patch("minds.common.statsig.dynamic_config.mind_limits.get_statsig")
    def test_returns_unlimited(self, mock_get_statsig):
        ctx = _make_context()
        settings = _make_settings(DeploymentMode.SELF_HOSTED)

        config = get_mind_limits_config(context=ctx, settings=settings)

        # Every resource field should be unlimited with zero usage
        for resource in _resource_fields():
            section = getattr(config, resource)
            assert section.limit.lifetime == UNLIMITED, f"{resource}.limit.lifetime should be unlimited"
            assert section.limit.monthly == UNLIMITED, f"{resource}.limit.monthly should be unlimited"
            assert section.usage == UsageConfig(), f"{resource}.usage should be default UsageConfig"

        # Statsig should never be called
        mock_get_statsig.assert_not_called()

    @patch("minds.common.statsig.dynamic_config.mind_limits.get_statsig")
    def test_returns_fresh_instance(self, mock_get_statsig):
        """Each call should return a new MindLimitsConfig (not shared mutable)."""
        settings = _make_settings(DeploymentMode.SELF_HOSTED)
        a = get_mind_limits_config(context=_make_context(), settings=settings)
        b = get_mind_limits_config(context=_make_context(), settings=settings)
        assert a is not b, "Self-hosted should return a fresh instance each time, not a shared object"


class TestCloudMode:
    """Cloud deployment should fetch limits from Statsig."""

    @patch("minds.common.statsig.dynamic_config.mind_limits.build_statsig_user")
    @patch("minds.common.statsig.dynamic_config.mind_limits.get_statsig")
    def test_parses_statsig_values(self, mock_get_statsig, mock_build_user):
        mock_dynamic_config = MagicMock()
        mock_dynamic_config.value = STATSIG_PAYLOAD
        mock_get_statsig.return_value.get_dynamic_config.return_value = mock_dynamic_config

        ctx = _make_context()
        settings = _make_settings(DeploymentMode.CLOUD)
        config = get_mind_limits_config(context=ctx, settings=settings)

        assert config.tokens.limit.lifetime == -1, (
            f"tokens.limit.lifetime should be -1, got {config.tokens.limit.lifetime}"
        )
        assert config.tokens.limit.monthly == 1_000_000, (
            f"tokens.limit.monthly should be 1000000, got {config.tokens.limit.monthly}"
        )
        assert config.minds.limit.lifetime == 1, f"minds.limit.lifetime should be 1, got {config.minds.limit.lifetime}"
        assert config.minds.limit.monthly == 1, f"minds.limit.monthly should be 1, got {config.minds.limit.monthly}"
        assert config.datasources.limit.lifetime == 3, (
            f"datasources.limit.lifetime should be 3, got {config.datasources.limit.lifetime}"
        )
        assert config.datasources.limit.monthly == 3, (
            f"datasources.limit.monthly should be 3, got {config.datasources.limit.monthly}"
        )
        assert config.questions.limit.lifetime == -1, (
            f"questions.limit.lifetime should be -1, got {config.questions.limit.lifetime}"
        )
        assert config.questions.limit.monthly == 250, (
            f"questions.limit.monthly should be 250, got {config.questions.limit.monthly}"
        )

    @patch("minds.common.statsig.dynamic_config.mind_limits.build_statsig_user")
    @patch("minds.common.statsig.dynamic_config.mind_limits.get_statsig")
    def test_calls_statsig_with_correct_config_name(self, mock_get_statsig, mock_build_user):
        mock_dynamic_config = MagicMock()
        mock_dynamic_config.value = STATSIG_PAYLOAD
        mock_statsig = mock_get_statsig.return_value
        mock_statsig.get_dynamic_config.return_value = mock_dynamic_config

        ctx = _make_context()
        settings = _make_settings(DeploymentMode.CLOUD)
        get_mind_limits_config(context=ctx, settings=settings)

        mock_statsig.get_dynamic_config.assert_called_once()
        call_kwargs = mock_statsig.get_dynamic_config.call_args
        assert call_kwargs.kwargs["name"] == "mind-usage-limits", (
            f"Dynamic config name should be 'mind-usage-limits', got '{call_kwargs.kwargs.get('name')}'"
        )


class TestCloudStatsigFailure:
    """Cloud mode should fail open when Statsig is unavailable."""

    @patch("minds.common.statsig.dynamic_config.mind_limits.get_statsig")
    def test_falls_back_to_unlimited_on_exception(self, mock_get_statsig):
        mock_get_statsig.side_effect = RuntimeError("Statsig is down")

        ctx = _make_context()
        settings = _make_settings(DeploymentMode.CLOUD)
        config = get_mind_limits_config(context=ctx, settings=settings)

        # Should return unlimited defaults for every resource, not raise
        for resource in _resource_fields():
            section = getattr(config, resource)
            assert section.limit.lifetime == UNLIMITED, f"{resource}.limit.lifetime should be unlimited on fallback"
            assert section.limit.monthly == UNLIMITED, f"{resource}.limit.monthly should be unlimited on fallback"
            assert section.usage == UsageConfig(), f"{resource}.usage should be default UsageConfig on fallback"

    @patch("minds.common.statsig.dynamic_config.mind_limits.build_statsig_user")
    @patch("minds.common.statsig.dynamic_config.mind_limits.get_statsig")
    def test_falls_back_on_dynamic_config_parse_error(self, mock_get_statsig, mock_build_user):
        """If Statsig returns garbage, we still fail open."""
        mock_dynamic_config = MagicMock()
        mock_dynamic_config.value = {"tokens": "not_a_valid_structure"}
        mock_get_statsig.return_value.get_dynamic_config.return_value = mock_dynamic_config

        ctx = _make_context()
        settings = _make_settings(DeploymentMode.CLOUD)
        config = get_mind_limits_config(context=ctx, settings=settings)

        # Should fall back to unlimited
        for resource in _resource_fields():
            section = getattr(config, resource)
            assert section.limit.lifetime == UNLIMITED, f"{resource}.limit.lifetime should be unlimited on parse error"
            assert section.limit.monthly == UNLIMITED, f"{resource}.limit.monthly should be unlimited on parse error"
