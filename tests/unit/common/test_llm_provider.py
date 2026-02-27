"""
Unit tests for LLM provider utilities.

Tests:
- LLMProvider enum and normalize()
- get_supported_models_by_provider(context, settings)
- validate_provider_and_model_name(provider, model_name, context, settings)
"""

from unittest.mock import patch
from uuid import UUID

import pytest

from minds.common.llm_provider import (
    LLMProvider,
    get_supported_models_by_provider,
    validate_provider_and_model_name,
)
from minds.common.settings.app_settings import AppSettings
from minds.requests.context import Context


def _make_context() -> Context:
    return Context(
        user_id=UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
        organization_id=UUID("11111111-2222-3333-4444-555555555555"),
        user_email="alice@example.com",
        user_roles=["pro"],
    )


def _make_settings(enable_model_selection: bool = False, **overrides) -> AppSettings:
    defaults = {
        "minds": {"enable_model_selection": enable_model_selection},
        "statsig": {
            "sdk_key": "test-key",
            "environment": "test",
            "disable_network": True,
            "disable_all_logging": True,
        },
    }
    defaults.update(overrides)
    return AppSettings(**defaults)


class TestLLMProvider:
    """Test cases for the LLMProvider enum."""

    def test_enum_values(self):
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"

    def test_enum_members_count(self):
        assert len(LLMProvider) == 2

    def test_aliases_returns_dict(self):
        aliases = LLMProvider._aliases()
        assert isinstance(aliases, dict)
        assert aliases == {"gemini": "google"}


class TestLLMProviderNormalize:
    """Test cases for LLMProvider.normalize()."""

    def test_normalize_openai(self):
        assert LLMProvider.normalize("openai") == LLMProvider.OPENAI

    def test_normalize_anthropic(self):
        assert LLMProvider.normalize("anthropic") == LLMProvider.ANTHROPIC

    def test_normalize_case_insensitive_upper(self):
        assert LLMProvider.normalize("OPENAI") == LLMProvider.OPENAI
        assert LLMProvider.normalize("ANTHROPIC") == LLMProvider.ANTHROPIC

    def test_normalize_case_insensitive_mixed(self):
        assert LLMProvider.normalize("OpenAI") == LLMProvider.OPENAI
        assert LLMProvider.normalize("Anthropic") == LLMProvider.ANTHROPIC

    def test_normalize_unsupported_provider_raises(self):
        with pytest.raises(ValueError, match="Provider 'unknown' is not supported"):
            LLMProvider.normalize("unknown")

    def test_normalize_unsupported_provider_lists_supported(self):
        with pytest.raises(ValueError) as exc_info:
            LLMProvider.normalize("unsupported")
        error_msg = str(exc_info.value)
        assert "openai" in error_msg
        assert "anthropic" in error_msg

    def test_normalize_alias_gemini_raises_when_google_not_enabled(self):
        with pytest.raises(ValueError, match="Provider 'gemini' is not supported"):
            LLMProvider.normalize("gemini")

    def test_normalize_empty_string_raises(self):
        with pytest.raises(ValueError, match="is not supported"):
            LLMProvider.normalize("")


class TestGetSupportedModelsByProvider:
    """Test cases for get_supported_models_by_provider(context, settings)."""

    @patch("minds.common.llm_provider.is_model_selection_enabled", return_value=False)
    def test_disabled_returns_only_default(self, mock_flag):
        ctx = _make_context()
        settings = _make_settings()

        is_enabled, default_provider, default_model, providers = get_supported_models_by_provider(
            context=ctx, settings=settings
        )

        assert is_enabled is False
        assert default_provider == "openai"
        assert default_model == "gpt-4o"
        assert providers == {"openai": ["gpt-4o"]}

    @patch("minds.common.llm_provider.is_model_selection_enabled", return_value=True)
    def test_enabled_returns_all_providers(self, mock_flag):
        ctx = _make_context()
        settings = _make_settings()

        is_enabled, default_provider, default_model, providers = get_supported_models_by_provider(
            context=ctx, settings=settings
        )

        assert is_enabled is True
        assert default_provider == "openai"
        assert default_model == "gpt-4o"
        assert set(providers.keys()) == {p.value for p in LLMProvider}
        assert settings.openai.supported_models == providers["openai"]
        assert settings.anthropic.supported_models == providers["anthropic"]

    @patch("minds.common.llm_provider.is_model_selection_enabled", return_value=False)
    def test_disabled_with_anthropic_default(self, mock_flag):
        ctx = _make_context()
        settings = _make_settings(
            default_models={"default_provider": "anthropic", "anthropic_model": "claude-sonnet-4-5"},
        )

        is_enabled, default_provider, default_model, providers = get_supported_models_by_provider(
            context=ctx, settings=settings
        )

        assert is_enabled is False
        assert default_provider == "anthropic"
        assert default_model == "claude-sonnet-4-5"
        assert providers == {"anthropic": ["claude-sonnet-4-5"]}

    @patch("minds.common.llm_provider.is_model_selection_enabled", return_value=True)
    def test_enabled_iterates_all_enum_members(self, mock_flag):
        ctx = _make_context()
        settings = _make_settings()

        _, _, _, providers = get_supported_models_by_provider(context=ctx, settings=settings)

        assert set(providers.keys()) == {p.value for p in LLMProvider}

    @patch("minds.common.llm_provider.is_model_selection_enabled", return_value=False)
    def test_passes_context_and_settings_to_flag(self, mock_flag):
        ctx = _make_context()
        settings = _make_settings()

        get_supported_models_by_provider(context=ctx, settings=settings)

        mock_flag.assert_called_once_with(context=ctx, settings=settings)


class TestValidateWithoutContext:
    """Schema-level validation (no context) only checks provider normalization."""

    def test_none_provider_and_model_is_noop(self):
        validate_provider_and_model_name(provider=None, model_name=None)

    def test_valid_provider_accepted(self):
        validate_provider_and_model_name(provider="openai", model_name=None)

    def test_invalid_provider_rejected(self):
        with pytest.raises(ValueError, match="is not supported"):
            validate_provider_and_model_name(provider="bedrock", model_name=None)

    def test_model_without_context_skips_selection_check(self):
        validate_provider_and_model_name(provider="openai", model_name="any-model")


class TestValidateProviderAndModelName:
    """Test cases for validate_provider_and_model_name(provider, model_name, context, settings)."""

    @patch("minds.common.llm_provider.is_model_selection_enabled", return_value=False)
    def test_none_provider_and_model_returns_early(self, mock_flag):
        ctx = _make_context()
        settings = _make_settings()

        validate_provider_and_model_name(provider=None, model_name=None, context=ctx, settings=settings)
        mock_flag.assert_not_called()

    @patch("minds.common.llm_provider.is_model_selection_enabled", return_value=False)
    def test_disabled_default_provider_accepted(self, mock_flag):
        ctx = _make_context()
        settings = _make_settings()

        validate_provider_and_model_name(provider="openai", model_name=None, context=ctx, settings=settings)

    @patch("minds.common.llm_provider.is_model_selection_enabled", return_value=False)
    def test_disabled_non_default_provider_rejected(self, mock_flag):
        ctx = _make_context()
        settings = _make_settings()

        with pytest.raises(ValueError, match="Model selection is disabled"):
            validate_provider_and_model_name(provider="anthropic", model_name=None, context=ctx, settings=settings)

    @patch("minds.common.llm_provider.is_model_selection_enabled", return_value=False)
    def test_disabled_default_model_accepted(self, mock_flag):
        ctx = _make_context()
        settings = _make_settings()

        validate_provider_and_model_name(provider=None, model_name="gpt-4o", context=ctx, settings=settings)

    @patch("minds.common.llm_provider.is_model_selection_enabled", return_value=False)
    def test_disabled_non_default_model_rejected(self, mock_flag):
        ctx = _make_context()
        settings = _make_settings()

        with pytest.raises(ValueError, match="Model selection is disabled"):
            validate_provider_and_model_name(provider=None, model_name="gpt-5.2", context=ctx, settings=settings)

    @patch("minds.common.llm_provider.is_model_selection_enabled", return_value=True)
    def test_enabled_valid_provider_accepted(self, mock_flag):
        ctx = _make_context()
        settings = _make_settings()

        validate_provider_and_model_name(provider="anthropic", model_name=None, context=ctx, settings=settings)

    @patch("minds.common.llm_provider.is_model_selection_enabled", return_value=True)
    def test_enabled_unsupported_provider_rejected(self, mock_flag):
        ctx = _make_context()
        settings = _make_settings()

        with pytest.raises(ValueError, match="is not supported"):
            validate_provider_and_model_name(provider="bedrock", model_name=None, context=ctx, settings=settings)

    @patch("minds.common.llm_provider.is_model_selection_enabled", return_value=True)
    def test_enabled_valid_model_accepted(self, mock_flag):
        ctx = _make_context()
        settings = _make_settings()

        validate_provider_and_model_name(provider="openai", model_name="gpt-4o", context=ctx, settings=settings)

    @patch("minds.common.llm_provider.is_model_selection_enabled", return_value=True)
    def test_enabled_invalid_model_for_provider_rejected(self, mock_flag):
        ctx = _make_context()
        settings = _make_settings()

        with pytest.raises(ValueError, match="is not supported for provider"):
            validate_provider_and_model_name(
                provider="openai", model_name="claude-sonnet-4-5", context=ctx, settings=settings
            )

    @patch("minds.common.llm_provider.is_model_selection_enabled", return_value=True)
    def test_enabled_model_without_provider_uses_default(self, mock_flag):
        ctx = _make_context()
        settings = _make_settings()

        validate_provider_and_model_name(provider=None, model_name="gpt-4o", context=ctx, settings=settings)

    @patch("minds.common.llm_provider.is_model_selection_enabled", return_value=True)
    def test_enabled_invalid_model_without_provider_rejected(self, mock_flag):
        ctx = _make_context()
        settings = _make_settings()

        with pytest.raises(ValueError, match="is not supported for provider"):
            validate_provider_and_model_name(
                provider=None, model_name="nonexistent-model", context=ctx, settings=settings
            )
