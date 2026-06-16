"""
Unit tests for get_passthrough_model_config.

Tests:
- Self-hosted → empty policy (Statsig never called)
- Cloud → Statsig values parsed correctly
- Cloud + Statsig failure / garbage → fail open (empty policy)
- allowed_aliases None vs [] semantics
- unknown search_provider dropped to None
"""

from unittest.mock import MagicMock, patch
from uuid import UUID

from minds.common.settings.app_settings import AppSettings, DeploymentMode
from minds.common.statsig.dynamic_config.model_config import get_passthrough_model_config
from minds.requests.context import Context
from minds.schemas.passthrough import PassthroughModelStatsigConfig


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


STATSIG_PAYLOAD = {
    "alias_overrides": {"opus": "claude-opus-4-8"},
    "allowed_aliases": ["opus", "sonnet"],
    "search_provider": "exa",
    "search_enabled": False,
}


class TestSelfHostedMode:
    @patch("minds.common.statsig.dynamic_config.model_config.get_statsig")
    def test_returns_empty_policy_without_calling_statsig(self, mock_get_statsig):
        config = get_passthrough_model_config(
            context=_make_context(), settings=_make_settings(DeploymentMode.SELF_HOSTED)
        )

        assert config == PassthroughModelStatsigConfig()
        assert config.alias_overrides == {}
        assert config.allowed_aliases is None
        assert config.search_provider is None
        assert config.search_enabled is True
        mock_get_statsig.assert_not_called()


class TestCloudMode:
    @patch("minds.common.statsig.dynamic_config.model_config.build_statsig_user")
    @patch("minds.common.statsig.dynamic_config.model_config.get_statsig")
    def test_parses_statsig_values(self, mock_get_statsig, mock_build_user):
        mock_dynamic_config = MagicMock()
        mock_dynamic_config.value = STATSIG_PAYLOAD
        mock_get_statsig.return_value.get_dynamic_config.return_value = mock_dynamic_config

        config = get_passthrough_model_config(context=_make_context(), settings=_make_settings())

        assert config.alias_overrides == {"opus": "claude-opus-4-8"}
        assert config.allowed_aliases == ["opus", "sonnet"]
        assert config.search_provider == "exa"
        assert config.search_enabled is False

    @patch("minds.common.statsig.dynamic_config.model_config.build_statsig_user")
    @patch("minds.common.statsig.dynamic_config.model_config.get_statsig")
    def test_calls_statsig_with_correct_config_name(self, mock_get_statsig, mock_build_user):
        mock_dynamic_config = MagicMock()
        mock_dynamic_config.value = {}
        mock_statsig = mock_get_statsig.return_value
        mock_statsig.get_dynamic_config.return_value = mock_dynamic_config

        get_passthrough_model_config(context=_make_context(), settings=_make_settings())

        mock_statsig.get_dynamic_config.assert_called_once()
        assert mock_statsig.get_dynamic_config.call_args.kwargs["name"] == "passthrough-model-config"

    @patch("minds.common.statsig.dynamic_config.model_config.build_statsig_user")
    @patch("minds.common.statsig.dynamic_config.model_config.get_statsig")
    def test_empty_payload_means_no_policy(self, mock_get_statsig, mock_build_user):
        mock_dynamic_config = MagicMock()
        mock_dynamic_config.value = {}
        mock_get_statsig.return_value.get_dynamic_config.return_value = mock_dynamic_config

        config = get_passthrough_model_config(context=_make_context(), settings=_make_settings())

        # None means "all allowed" — must not collapse to an empty-list lockout.
        assert config.allowed_aliases is None
        assert config.alias_overrides == {}

    @patch("minds.common.statsig.dynamic_config.model_config.build_statsig_user")
    @patch("minds.common.statsig.dynamic_config.model_config.get_statsig")
    def test_empty_allowed_list_is_preserved_as_block_all(self, mock_get_statsig, mock_build_user):
        # An explicit [] is an intentional total block, distinct from absent/None.
        mock_dynamic_config = MagicMock()
        mock_dynamic_config.value = {"allowed_aliases": []}
        mock_get_statsig.return_value.get_dynamic_config.return_value = mock_dynamic_config

        config = get_passthrough_model_config(context=_make_context(), settings=_make_settings())

        assert config.allowed_aliases == []

    @patch("minds.common.statsig.dynamic_config.model_config.build_statsig_user")
    @patch("minds.common.statsig.dynamic_config.model_config.get_statsig")
    def test_unknown_search_provider_dropped_to_none(self, mock_get_statsig, mock_build_user):
        mock_dynamic_config = MagicMock()
        mock_dynamic_config.value = {"search_provider": "tavily"}  # not implemented yet
        mock_get_statsig.return_value.get_dynamic_config.return_value = mock_dynamic_config

        config = get_passthrough_model_config(context=_make_context(), settings=_make_settings())

        assert config.search_provider is None


class TestCloudStatsigFailure:
    @patch("minds.common.statsig.dynamic_config.model_config.get_statsig")
    def test_falls_back_to_empty_on_exception(self, mock_get_statsig):
        mock_get_statsig.side_effect = RuntimeError("Statsig is down")

        config = get_passthrough_model_config(context=_make_context(), settings=_make_settings())

        assert config == PassthroughModelStatsigConfig()

    @patch("minds.common.statsig.dynamic_config.model_config.build_statsig_user")
    @patch("minds.common.statsig.dynamic_config.model_config.get_statsig")
    def test_falls_back_on_parse_error(self, mock_get_statsig, mock_build_user):
        mock_dynamic_config = MagicMock()
        mock_dynamic_config.value = {"alias_overrides": "not_a_dict"}
        mock_get_statsig.return_value.get_dynamic_config.return_value = mock_dynamic_config

        config = get_passthrough_model_config(context=_make_context(), settings=_make_settings())

        assert config == PassthroughModelStatsigConfig()
