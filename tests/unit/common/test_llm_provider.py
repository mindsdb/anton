from unittest.mock import patch

import pytest

from minds.common.llm_provider import LLMProvider, get_supported_models_by_provider


class TestLLMProvider:
    """Test cases for the LLMProvider enum."""

    def test_enum_values(self):
        """Test that all expected providers are defined."""
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"

    def test_enum_members_count(self):
        """Test the total number of enum members."""
        assert len(LLMProvider) == 2

    def test_aliases_returns_dict(self):
        """Test that _aliases returns the expected alias mapping."""
        aliases = LLMProvider._aliases()
        assert isinstance(aliases, dict)
        assert aliases == {"gemini": "google"}


class TestLLMProviderNormalize:
    """Test cases for LLMProvider.normalize()."""

    def test_normalize_openai(self):
        """Test normalizing 'openai' returns OPENAI."""
        assert LLMProvider.normalize("openai") == LLMProvider.OPENAI

    def test_normalize_anthropic(self):
        """Test normalizing 'anthropic' returns ANTHROPIC."""
        assert LLMProvider.normalize("anthropic") == LLMProvider.ANTHROPIC

    def test_normalize_case_insensitive_upper(self):
        """Test that normalization is case-insensitive (uppercase)."""
        assert LLMProvider.normalize("OPENAI") == LLMProvider.OPENAI
        assert LLMProvider.normalize("ANTHROPIC") == LLMProvider.ANTHROPIC

    def test_normalize_case_insensitive_mixed(self):
        """Test that normalization is case-insensitive (mixed case)."""
        assert LLMProvider.normalize("OpenAI") == LLMProvider.OPENAI
        assert LLMProvider.normalize("Anthropic") == LLMProvider.ANTHROPIC

    def test_normalize_unsupported_provider_raises(self):
        """Test that an unsupported provider raises ValueError with descriptive message."""
        with pytest.raises(ValueError, match="Provider 'unknown' is not supported"):
            LLMProvider.normalize("unknown")

    def test_normalize_unsupported_provider_lists_supported(self):
        """Test that the error message lists all supported providers."""
        with pytest.raises(ValueError) as exc_info:
            LLMProvider.normalize("unsupported")
        error_msg = str(exc_info.value)
        assert "openai" in error_msg
        assert "anthropic" in error_msg

    def test_normalize_alias_gemini_raises_when_google_not_enabled(self):
        """Test that 'gemini' alias resolves to 'google' which is not a current member, so raises."""
        # 'gemini' maps to 'google' via aliases, but GOOGLE is commented out in the enum
        with pytest.raises(ValueError, match="Provider 'gemini' is not supported"):
            LLMProvider.normalize("gemini")

    def test_normalize_empty_string_raises(self):
        """Test that an empty string raises ValueError."""
        with pytest.raises(ValueError, match="is not supported"):
            LLMProvider.normalize("")


class TestSupportedModelsByProvider:
    """Test cases for supported_models_by_provider()."""

    @patch("minds.common.llm_provider.settings")
    def test_model_selection_disabled(self, mock_settings):
        """Test that only the default provider/model is returned when selection is disabled."""
        # Arrange
        mock_settings.minds.enable_model_selection = False
        mock_settings.default_models.default_provider = "openai"
        mock_settings.default_models.openai_model = "gpt-4o"

        # Act
        is_enabled, default_provider, default_model, providers_and_models = get_supported_models_by_provider()

        # Assert
        assert is_enabled is False
        assert default_provider == "openai"
        assert default_model == "gpt-4o"
        assert providers_and_models == {"openai": ["gpt-4o"]}

    @patch("minds.common.llm_provider.settings")
    def test_model_selection_enabled(self, mock_settings):
        """Test that all providers and their models are returned when selection is enabled."""
        # Arrange
        mock_settings.minds.enable_model_selection = True
        mock_settings.default_models.default_provider = "openai"
        mock_settings.default_models.openai_model = "gpt-4o"
        mock_settings.openai.supported_models = ["gpt-4o", "gpt-4o-mini"]
        mock_settings.anthropic.supported_models = ["claude-sonnet-4-5", "claude-haiku-3-5"]

        # Act
        is_enabled, default_provider, default_model, providers_and_models = get_supported_models_by_provider()

        # Assert
        assert is_enabled is True
        assert default_provider == "openai"
        assert default_model == "gpt-4o"
        assert providers_and_models == {
            "openai": ["gpt-4o", "gpt-4o-mini"],
            "anthropic": ["claude-sonnet-4-5", "claude-haiku-3-5"],
        }

    @patch("minds.common.llm_provider.settings")
    def test_model_selection_disabled_with_anthropic_default(self, mock_settings):
        """Test disabled selection with anthropic as the default provider."""
        # Arrange
        mock_settings.minds.enable_model_selection = False
        mock_settings.default_models.default_provider = "anthropic"
        mock_settings.default_models.anthropic_model = "claude-sonnet-4-5"

        # Act
        is_enabled, default_provider, default_model, providers_and_models = get_supported_models_by_provider()

        # Assert
        assert is_enabled is False
        assert default_provider == "anthropic"
        assert default_model == "claude-sonnet-4-5"
        assert providers_and_models == {"anthropic": ["claude-sonnet-4-5"]}

    @patch("minds.common.llm_provider.settings")
    def test_model_selection_enabled_iterates_all_providers(self, mock_settings):
        """Test that enabled selection iterates over all LLMProvider members."""
        # Arrange
        mock_settings.minds.enable_model_selection = True
        mock_settings.default_models.default_provider = "openai"
        mock_settings.default_models.openai_model = "gpt-4o"
        mock_settings.openai.supported_models = ["gpt-4o"]
        mock_settings.anthropic.supported_models = ["claude-sonnet-4-5"]

        # Act
        _, _, _, providers_and_models = get_supported_models_by_provider()

        # Assert – keys must match exactly the enum members
        assert set(providers_and_models.keys()) == {p.value for p in LLMProvider}

    @patch("minds.common.llm_provider.settings")
    def test_enable_model_selection_truthy_values(self, mock_settings):
        """Test that enable_model_selection handles truthy values via bool()."""
        # Arrange
        mock_settings.minds.enable_model_selection = 1  # truthy but not True
        mock_settings.default_models.default_provider = "openai"
        mock_settings.default_models.openai_model = "gpt-4o"
        mock_settings.openai.supported_models = ["gpt-4o"]
        mock_settings.anthropic.supported_models = ["claude-sonnet-4-5"]

        # Act
        is_enabled, _, _, providers_and_models = get_supported_models_by_provider()

        # Assert
        assert is_enabled is True
        assert len(providers_and_models) == len(LLMProvider)

    @patch("minds.common.llm_provider.settings")
    def test_enable_model_selection_falsy_values(self, mock_settings):
        """Test that enable_model_selection handles falsy values via bool()."""
        # Arrange
        mock_settings.minds.enable_model_selection = 0  # falsy but not False
        mock_settings.default_models.default_provider = "openai"
        mock_settings.default_models.openai_model = "gpt-4o"

        # Act
        is_enabled, _, _, providers_and_models = get_supported_models_by_provider()

        # Assert
        assert is_enabled is False
        assert providers_and_models == {"openai": ["gpt-4o"]}
