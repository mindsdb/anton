"""Tests for ModelResolver implementation."""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from minds.inference.model_resolver import ModelResolver
from minds.inference.types import ApiKind


@pytest.fixture
def mock_settings():
    """Create a mock AppSettings with all providers configured."""
    settings = MagicMock()

    # Anthropic
    settings.anthropic.api_key = "test-anthropic-key"
    settings.anthropic.passthrough_sonnet_model = "claude-3-5-sonnet-20241022"
    settings.anthropic.passthrough_opus_model = "claude-3-opus-20240229"
    settings.anthropic.passthrough_haiku_model = "claude-3-haiku-20240307"
    settings.anthropic.passthrough_mythos_model = "claude-mythos-5"
    settings.anthropic.passthrough_fable_model = "claude-fable-5"

    # OpenAI
    settings.openai.api_key = "test-openai-key"
    settings.openai.api_url = None
    settings.openai.passthrough_gpt_model = "gpt-4o"
    settings.openai.passthrough_gpt_codex_model = "gpt-4o"
    settings.openai.passthrough_gpt_mini_model = "gpt-4o-mini"
    settings.openai.passthrough_gpt_nano_model = "gpt-4o-mini"

    # Gemini
    settings.gemini.api_key = "test-gemini-key"
    settings.gemini.passthrough_gemini_model = "gemini-2.0-flash"
    settings.gemini.passthrough_gemini_flash_model = "gemini-2.0-flash"

    # Fireworks
    settings.fireworks.api_key = "test-fireworks-key"
    settings.fireworks.anthropic_base_url = "https://api.fireworks.ai/account/v1/completions"
    settings.fireworks.passthrough_kimi_model = "accounts/fireworks/models/kimi-k2.5"
    settings.fireworks.passthrough_deepseek_model = "accounts/fireworks/models/deepseek-r1"
    settings.fireworks.passthrough_qwen_model = "accounts/fireworks/models/qwen-qwq"

    return settings


class TestModelResolverCanonicalAliases:
    """Test resolution of canonical aliases."""

    def test_resolve_sonnet(self, mock_settings):
        """Resolve latest:sonnet to Anthropic Claude 3.5 Sonnet."""
        resolver = ModelResolver(mock_settings)
        config = resolver.resolve("latest:sonnet")

        assert config.api_kind == ApiKind.ANTHROPIC_MESSAGES
        assert config.model_name == "claude-3-5-sonnet-20241022"
        assert config.api_key == "test-anthropic-key"
        assert config.label == "anthropic"
        assert config.alias == "sonnet"

    def test_resolve_opus(self, mock_settings):
        """Resolve latest:opus to Anthropic Claude 3 Opus."""
        resolver = ModelResolver(mock_settings)
        config = resolver.resolve("latest:opus")

        assert config.api_kind == ApiKind.ANTHROPIC_MESSAGES
        assert config.model_name == "claude-3-opus-20240229"
        assert config.alias == "opus"

    def test_resolve_haiku(self, mock_settings):
        """Resolve latest:haiku to Anthropic Claude 3 Haiku."""
        resolver = ModelResolver(mock_settings)
        config = resolver.resolve("latest:haiku")

        assert config.api_kind == ApiKind.ANTHROPIC_MESSAGES
        assert config.model_name == "claude-3-haiku-20240307"
        assert config.alias == "haiku"

    def test_resolve_gpt(self, mock_settings):
        """Resolve latest:gpt to OpenAI GPT with low reasoning effort."""
        resolver = ModelResolver(mock_settings)
        config = resolver.resolve("latest:gpt")

        assert config.api_kind == ApiKind.OPENAI_RESPONSES
        assert config.model_name == "gpt-4o"
        assert config.label == "openai"
        assert config.alias == "gpt"
        assert config.reasoning_effort == "low"

    def test_resolve_gpt_low(self, mock_settings):
        """Resolve latest:gpt-low to OpenAI GPT with low reasoning effort."""
        resolver = ModelResolver(mock_settings)
        config = resolver.resolve("latest:gpt-low")

        assert config.api_kind == ApiKind.OPENAI_RESPONSES
        assert config.reasoning_effort == "low"
        assert config.alias == "gpt-low"

    def test_resolve_gpt_medium(self, mock_settings):
        """Resolve latest:gpt-medium to OpenAI GPT with medium reasoning effort."""
        resolver = ModelResolver(mock_settings)
        config = resolver.resolve("latest:gpt-medium")

        assert config.api_kind == ApiKind.OPENAI_RESPONSES
        assert config.reasoning_effort == "medium"
        assert config.alias == "gpt-medium"

    def test_resolve_gpt_high(self, mock_settings):
        """Resolve latest:gpt-high to OpenAI GPT with high reasoning effort."""
        resolver = ModelResolver(mock_settings)
        config = resolver.resolve("latest:gpt-high")

        assert config.api_kind == ApiKind.OPENAI_RESPONSES
        assert config.reasoning_effort == "high"
        assert config.alias == "gpt-high"

    def test_resolve_gpt_codex(self, mock_settings):
        """Resolve latest:gpt-codex to OpenAI GPT Codex."""
        resolver = ModelResolver(mock_settings)
        config = resolver.resolve("latest:gpt-codex")

        assert config.api_kind == ApiKind.OPENAI_RESPONSES
        assert config.alias == "gpt-codex"
        assert config.reasoning_effort is None

    def test_resolve_gpt_mini(self, mock_settings):
        """Resolve latest:gpt-mini to OpenAI GPT Mini."""
        resolver = ModelResolver(mock_settings)
        config = resolver.resolve("latest:gpt-mini")

        assert config.api_kind == ApiKind.OPENAI_RESPONSES
        assert config.alias == "gpt-mini"

    def test_resolve_gpt_nano(self, mock_settings):
        """Resolve latest:gpt-nano to OpenAI GPT Nano."""
        resolver = ModelResolver(mock_settings)
        config = resolver.resolve("latest:gpt-nano")

        assert config.api_kind == ApiKind.OPENAI_RESPONSES
        assert config.alias == "gpt-nano"

    def test_resolve_gemini(self, mock_settings):
        """Resolve latest:gemini to Google Gemini."""
        resolver = ModelResolver(mock_settings)
        config = resolver.resolve("latest:gemini")

        assert config.api_kind == ApiKind.GEMINI_NATIVE
        assert config.model_name == "gemini-2.0-flash"
        assert config.label == "gemini"
        assert config.alias == "gemini"

    def test_resolve_gemini_flash(self, mock_settings):
        """Resolve latest:gemini-flash to Google Gemini Flash."""
        resolver = ModelResolver(mock_settings)
        config = resolver.resolve("latest:gemini-flash")

        assert config.api_kind == ApiKind.GEMINI_NATIVE
        assert config.alias == "gemini-flash"

    def test_resolve_kimi(self, mock_settings):
        """Resolve latest:kimi to Fireworks Kimi."""
        resolver = ModelResolver(mock_settings)
        config = resolver.resolve("latest:kimi")

        assert config.api_kind == ApiKind.ANTHROPIC_MESSAGES
        assert config.label == "fireworks"
        assert config.alias == "kimi"
        assert config.base_url == "https://api.fireworks.ai/account/v1/completions"

    def test_resolve_deepseek(self, mock_settings):
        """Resolve latest:deepseek to Fireworks DeepSeek."""
        resolver = ModelResolver(mock_settings)
        config = resolver.resolve("latest:deepseek")

        assert config.api_kind == ApiKind.ANTHROPIC_MESSAGES
        assert config.label == "fireworks"
        assert config.alias == "deepseek"

    def test_resolve_qwen(self, mock_settings):
        """Resolve latest:qwen to Fireworks Qwen."""
        resolver = ModelResolver(mock_settings)
        config = resolver.resolve("latest:qwen")

        assert config.api_kind == ApiKind.ANTHROPIC_MESSAGES
        assert config.label == "fireworks"
        assert config.alias == "qwen"


class TestModelResolverDeprecatedAliases:
    """Test resolution of deprecated aliases."""

    def test_resolve_deprecated_reason(self, mock_settings):
        """Resolve _reason_ to sonnet with deprecated alias preserved."""
        resolver = ModelResolver(mock_settings)
        config = resolver.resolve("_reason_")

        assert config.api_kind == ApiKind.ANTHROPIC_MESSAGES
        assert config.model_name == "claude-3-5-sonnet-20241022"
        assert config.alias == "_reason_"

    def test_resolve_deprecated_code(self, mock_settings):
        """Resolve _code_ to haiku with deprecated alias preserved."""
        resolver = ModelResolver(mock_settings)
        config = resolver.resolve("_code_")

        assert config.api_kind == ApiKind.ANTHROPIC_MESSAGES
        assert config.model_name == "claude-3-haiku-20240307"
        assert config.alias == "_code_"


class TestModelResolverIsPassthroughModel:
    """Test is_passthrough_model predicate."""

    def test_is_passthrough_canonical(self, mock_settings):
        """Test canonical latest:* pattern matching."""
        resolver = ModelResolver(mock_settings)

        assert resolver.is_passthrough_model("latest:sonnet") is True
        assert resolver.is_passthrough_model("latest:gpt-high") is True
        assert resolver.is_passthrough_model("latest:gemini-flash") is True

    def test_is_passthrough_deprecated(self, mock_settings):
        """Test deprecated alias matching."""
        resolver = ModelResolver(mock_settings)

        assert resolver.is_passthrough_model("_reason_") is True
        assert resolver.is_passthrough_model("_code_") is True

    def test_is_not_passthrough_invalid(self, mock_settings):
        """Test non-passthrough model names."""
        resolver = ModelResolver(mock_settings)

        assert resolver.is_passthrough_model("gpt-4") is False
        assert resolver.is_passthrough_model("claude-3-sonnet") is False
        assert resolver.is_passthrough_model("gemini-1.5") is False
        assert resolver.is_passthrough_model("") is False

    def test_is_not_passthrough_malformed(self, mock_settings):
        """Test malformed latest: patterns."""
        resolver = ModelResolver(mock_settings)

        assert resolver.is_passthrough_model("latest:") is False
        assert resolver.is_passthrough_model("latest:") is False
        assert resolver.is_passthrough_model("latest_sonnet") is False


class TestModelResolverErrorCases:
    """Test error handling for invalid or unconfigured aliases."""

    def test_resolve_unknown_alias(self, mock_settings):
        """Raise 400 for unknown alias."""
        resolver = ModelResolver(mock_settings)

        with pytest.raises(HTTPException) as exc_info:
            resolver.resolve("latest:unknown-alias")

        assert exc_info.value.status_code == 400
        assert "Unknown passthrough alias" in exc_info.value.detail

    def test_resolve_invalid_pattern(self, mock_settings):
        """Raise ValueError for malformed alias pattern."""
        resolver = ModelResolver(mock_settings)

        with pytest.raises(ValueError, match="is not a valid passthrough model name"):
            resolver.resolve("gpt-4")

    def test_resolve_no_provider_configured(self, mock_settings):
        """Raise 400 when no provider is configured for alias."""
        # Remove Anthropic API key
        mock_settings.anthropic.api_key = ""

        resolver = ModelResolver(mock_settings)

        with pytest.raises(HTTPException) as exc_info:
            resolver.resolve("latest:sonnet")

        assert exc_info.value.status_code == 400
        assert "No provider configured" in exc_info.value.detail

    def test_resolve_openai_api_key_not_set(self, mock_settings):
        """OpenAI is not available when api_key is 'not set' string."""
        mock_settings.openai.api_key = "not set"

        resolver = ModelResolver(mock_settings)

        with pytest.raises(HTTPException) as exc_info:
            resolver.resolve("latest:gpt")

        assert exc_info.value.status_code == 400


class TestModelResolverListAvailable:
    """Test list_available() returns correctly filtered configs."""

    def test_list_available_all_providers(self, mock_settings):
        """List all available models when all providers configured."""
        resolver = ModelResolver(mock_settings)
        configs = resolver.list_available()

        # Should have one config per alias
        assert len(configs) > 0

        # Verify all expected aliases are present
        aliases = {config.alias for config in configs}
        expected = {
            "sonnet",
            "opus",
            "haiku",
            "mythos",
            "fable",
            "gpt",
            "gpt-low",
            "gpt-medium",
            "gpt-high",
            "gpt-codex",
            "gpt-mini",
            "gpt-nano",
            "gemini",
            "gemini-flash",
            "kimi",
            "deepseek",
            "qwen",
        }
        assert aliases == expected

    def test_list_available_partial_providers(self, mock_settings):
        """List only models from configured providers."""
        # Remove Fireworks API key
        mock_settings.fireworks.api_key = ""

        resolver = ModelResolver(mock_settings)
        configs = resolver.list_available()

        aliases = {config.alias for config in configs}

        # Fireworks models should be absent
        assert "kimi" not in aliases
        assert "deepseek" not in aliases
        assert "qwen" not in aliases

        # Anthropic, OpenAI, Gemini models should be present
        assert "sonnet" in aliases
        assert "gpt" in aliases
        assert "gemini" in aliases

    def test_list_available_excludes_deprecated(self, mock_settings):
        """List does not include deprecated aliases."""
        resolver = ModelResolver(mock_settings)
        configs = resolver.list_available()

        aliases = {config.alias for config in configs}

        # Deprecated aliases should not be in the list
        assert "_reason_" not in aliases
        assert "_code_" not in aliases

    def test_list_available_only_anthropic(self, mock_settings):
        """List only Anthropic models when only Anthropic is configured."""
        mock_settings.openai.api_key = ""
        mock_settings.gemini.api_key = ""
        mock_settings.fireworks.api_key = ""

        resolver = ModelResolver(mock_settings)
        configs = resolver.list_available()

        aliases = {config.alias for config in configs}
        expected = {"sonnet", "opus", "haiku", "mythos", "fable"}
        assert aliases == expected


class TestModelResolverWebSearchMode:
    """Test web search mode configuration per provider."""

    def test_openai_has_native_web_search(self, mock_settings):
        """OpenAI models should have OPENAI_NATIVE web search mode."""
        from minds.inference.types import WebSearchMode

        resolver = ModelResolver(mock_settings)
        config = resolver.resolve("latest:gpt")

        assert config.web_search_mode == WebSearchMode.OPENAI_NATIVE

    def test_anthropic_has_native_web_search(self, mock_settings):
        """Anthropic models should have ANTHROPIC_NATIVE web search mode."""
        from minds.inference.types import WebSearchMode

        resolver = ModelResolver(mock_settings)
        config = resolver.resolve("latest:sonnet")

        assert config.web_search_mode == WebSearchMode.ANTHROPIC_NATIVE

    def test_gemini_has_google_search(self, mock_settings):
        """Gemini models should have GEMINI_GOOGLE_SEARCH web search mode."""
        from minds.inference.types import WebSearchMode

        resolver = ModelResolver(mock_settings)
        config = resolver.resolve("latest:gemini")

        assert config.web_search_mode == WebSearchMode.GEMINI_GOOGLE_SEARCH

    def test_fireworks_drops_web_search(self, mock_settings):
        """Fireworks models should have DROP web search mode."""
        from minds.inference.types import WebSearchMode

        resolver = ModelResolver(mock_settings)
        config = resolver.resolve("latest:kimi")

        assert config.web_search_mode == WebSearchMode.DROP


class TestModelResolverObservabilityMetadata:
    """Test observability metadata generation."""

    def test_metadata_includes_alias(self, mock_settings):
        """Observability metadata should include the alias."""
        resolver = ModelResolver(mock_settings)
        config = resolver.resolve("latest:sonnet")
        metadata = config.to_observability_metadata()

        assert metadata.passthrough_alias == "sonnet"
        assert metadata.provider == "anthropic"

    def test_metadata_preserves_deprecated_alias(self, mock_settings):
        """Observability metadata should preserve deprecated alias."""
        resolver = ModelResolver(mock_settings)
        config = resolver.resolve("_reason_")
        metadata = config.to_observability_metadata()

        assert metadata.passthrough_alias == "_reason_"

    def test_metadata_includes_reasoning_effort(self, mock_settings):
        """Observability metadata should include reasoning_effort when present."""
        resolver = ModelResolver(mock_settings)
        config = resolver.resolve("latest:gpt-high")
        metadata = config.to_observability_metadata()

        assert metadata.reasoning_effort == "high"

    def test_metadata_excludes_none_fields(self, mock_settings):
        """Metadata to_metadata() dict should exclude None fields."""
        resolver = ModelResolver(mock_settings)
        config = resolver.resolve("latest:sonnet")
        metadata = config.to_observability_metadata()
        metadata_dict = metadata.to_metadata()

        # reasoning_effort should not be in dict since it's None
        assert "reasoning_effort" not in metadata_dict
        assert metadata_dict["passthrough_alias"] == "sonnet"
