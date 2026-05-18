from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from minds.client.openai_client import create_openai_client
from minds.common.llm_provider import LLMProvider
from minds.common.logger import get_logger
from minds.common.settings.app_settings import get_app_settings

logger = get_logger(__name__)
settings = get_app_settings()


def get_llm_config(provider: str | None = None, model_name: str | None = None) -> OpenAIModel | AnthropicModel:
    """Get LLM configuration based on agent settings.
    Args:
        provider: The provider name (e.g., 'openai', 'anthropic').
        model_name: The model name to use (optional).

    Returns:
        Union[OpenAIModel, AnthropicModel]: Configuration for the specified model

    Raises:
        ValueError: If an unsupported provider is specified
    """
    # Normalize provider name using enum
    provider = provider or settings.default_models.default_provider
    normalized_provider = LLMProvider.normalize(provider)

    if normalized_provider == LLMProvider.OPENAI:
        model_name = model_name or settings.default_models.openai_model
        return OpenAIModel(
            model_name=model_name,
            provider=OpenAIProvider(openai_client=create_openai_client(chat_completions_model=model_name).client),
        )

    elif normalized_provider == LLMProvider.ANTHROPIC:
        model_name = model_name or settings.default_models.anthropic_model
        return AnthropicModel(
            model_name=model_name,
            provider=AnthropicProvider(api_key=settings.anthropic.api_key),
        )

    else:
        raise ValueError(f"Unsupported provider: {normalized_provider.value}")
