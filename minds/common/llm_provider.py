from enum import Enum

from minds.common.settings.app_settings import get_app_settings

settings = get_app_settings()


class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    # GOOGLE = "google"
    # BEDROCK = "bedrock"

    @classmethod
    def _aliases(cls) -> dict[str, str]:
        """Return mapping of provider aliases to canonical names."""
        return {
            "gemini": "google",
        }

    @classmethod
    def normalize(cls, provider: str) -> "LLMProvider":
        """Normalize provider name, handling aliases like 'gemini' -> 'google'."""
        normalized = cls._aliases().get(provider.lower(), provider.lower())
        try:
            return cls(normalized)
        except ValueError:
            raise ValueError(
                f"Provider '{provider}' is not supported. Supported providers: {', '.join([p.value for p in cls])}"
            ) from None


def get_supported_models_by_provider() -> tuple[bool, str, str, dict[str, list[str]]]:
    """
    Get the supported models by provider.
    If model selection is disabled, return only the default provider and model.
    If model selection is enabled, return all supported providers and models.

    Returns:
        tuple[bool, str, str, dict[str, list[str]]]: A tuple with a boolean flag indicating if model selection is
        enabled, the default provider, the default model, and a dictionary of supported providers and models
    """
    # If model selection is disabled, return only the default provider and model
    is_model_selection_enabled = bool(settings.minds.enable_model_selection)
    default_provider = settings.default_models.default_provider
    default_model = getattr(settings.default_models, f"{default_provider}_model")
    if not is_model_selection_enabled:
        providers_and_models = {
            default_provider: [default_model],
        }

    else:
        providers_and_models: dict[str, list[str]] = {}
        for provider in LLMProvider:
            provider_str = provider.value
            providers_and_models[provider_str] = getattr(settings, provider_str).supported_models

    return is_model_selection_enabled, default_provider, default_model, providers_and_models


def validate_provider_and_model_name(
    provider: str | None,
    model_name: str | None,
) -> None:
    """Validate provider and model name.
    If model selection is disabled, only the default provider and model can be used.
    If model selection is enabled, validate the provider and model against the supported models for the provider.
    """
    if provider is None and model_name is None:
        return

    is_model_selection_enabled, default_provider, default_model, providers_and_models = (
        get_supported_models_by_provider()
    )

    if provider:
        provider = LLMProvider.normalize(provider).value
        if not is_model_selection_enabled:
            if provider != default_provider:
                raise ValueError(
                    f"Model selection is disabled. Only the default provider '{default_provider}' can be used."
                )
        else:
            if provider not in providers_and_models:
                raise ValueError(
                    f"Provider '{provider}' is not supported. "
                    f"Supported providers: {', '.join(providers_and_models.keys())}"
                )

    if model_name:
        if not is_model_selection_enabled:
            if model_name != default_model:
                raise ValueError(f"Model selection is disabled. Only the default model '{default_model}' can be used.")
        else:
            supported_models = providers_and_models.get(
                provider,
                providers_and_models[default_provider],
            )
            if model_name not in supported_models:
                raise ValueError(
                    f"Model '{model_name}' is not supported for provider '{provider or default_provider}'. "
                    f"Supported models: {', '.join(supported_models)}"
                )
