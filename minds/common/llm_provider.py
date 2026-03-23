from enum import Enum

from minds.common.settings.app_settings import AppSettings, get_app_settings
from minds.common.statsig.feature_flags.model_selection_enabled import is_model_selection_enabled
from minds.requests.context import Context

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


def get_supported_models_by_provider(
    context: Context, settings: AppSettings | None = None
) -> tuple[bool, str, str, str, dict[str, dict[str, list[str]]]]:
    """
    Get the supported models by provider.
    If model selection is disabled, return only the default provider and model.
    If model selection is enabled, return all supported providers and models.

    Returns:
        tuple[bool, str, str, str, dict[str, dict[str, list[str]]]]: A tuple with a boolean flag indicating if model
        selection is enabled, the default provider, the default reasoning model, the default coding model, and a
        dictionary mapping providers -> {reasoning_models: [...], coding_models: [...]}.
    """
    settings = settings or get_app_settings()

    # If model selection is disabled, return only the default provider and model
    model_selection_enabled = is_model_selection_enabled(context=context, settings=settings)

    default_provider = settings.default_models.default_provider
    default_model = getattr(settings.default_models, f"{default_provider}_model")
    default_coding_model = getattr(settings.default_models, f"{default_provider}_coding_model")
    if not model_selection_enabled:
        providers_and_models = {
            default_provider: {
                "reasoning_models": [default_model],
                "coding_models": [default_coding_model],
            }
        }

    else:
        providers_and_models: dict[str, dict[str, list[str]]] = {}
        for provider in LLMProvider:
            provider_str = provider.value
            providers_and_models[provider_str] = {
                "reasoning_models": getattr(settings, provider_str).supported_models,
                "coding_models": getattr(settings, provider_str).supported_coding_models,
            }

    return model_selection_enabled, default_provider, default_model, default_coding_model, providers_and_models


def validate_provider_and_model_name(
    provider: str | None,
    model_name: str | None,
    context: Context | None = None,
    settings: AppSettings | None = None,
) -> None:
    """Validate provider and model name.

    When called without *context* (e.g. from a Pydantic schema validator),
    only basic provider normalization is performed.  Full model-selection
    checks require a request context and are enforced at the endpoint /
    service layer.
    """
    if provider is None and model_name is None:
        return

    # Without context we can only validate that the provider string is recognised.
    if context is None:
        if provider:
            LLMProvider.normalize(provider)
        return

    settings = settings or get_app_settings()

    model_selection_enabled, default_provider, default_model, _default_coding_model, providers_and_models = (
        get_supported_models_by_provider(context=context, settings=settings)
    )

    if provider:
        provider = LLMProvider.normalize(provider).value
        if not model_selection_enabled:
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
        if not model_selection_enabled:
            if model_name != default_model:
                raise ValueError(f"Model selection is disabled. Only the default model '{default_model}' can be used.")
        else:
            provider_key = provider or default_provider
            supported = providers_and_models.get(provider_key, providers_and_models[default_provider])
            supported_reasoning_models = supported.get("reasoning_models", [])
            if model_name not in supported_reasoning_models:
                raise ValueError(
                    f"Model '{model_name}' is not supported for provider '{provider or default_provider}'. "
                    f"Supported models: {', '.join(supported_reasoning_models)}"
                )
