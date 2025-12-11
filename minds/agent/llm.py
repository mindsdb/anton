from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from minds.client.openai_client import create_openai_client
from minds.common.logger import setup_logging
from minds.common.settings.app_settings import get_app_settings

logger = setup_logging()
settings = get_app_settings()


def get_llm_config(provider: str, model_name: str | None = None) -> OpenAIModel | GeminiModel:
    """Get LLM configuration based on agent settings.
    Args:
        provider: The provider name (e.g., 'openai', 'google').
        model_name: The model name to use (optional).
    Returns:
        Union[OpenAIModel, GeminiModel]: Configuration for the specified model
    Raises:
        ValueError: If an unsupported provider is specified
    """

    if provider == "openai":
        # Use default if no model specified
        if not model_name:
            logger.debug(f"No model name specified, using default model: {settings.default_models.mind_model}")
            model_name = settings.default_models.mind_model

        openai_client = create_openai_client(chat_completions_model=model_name)
        logger.debug(f"Created OpenAIClient for model: {model_name}")

        return OpenAIModel(model_name=model_name, provider=OpenAIProvider(openai_client=openai_client.client))
    elif provider in ["google", "gemini"]:
        # Use Google default if no model specified
        if not model_name:
            model_name = settings.default_models.google_model
        return GeminiModel(model_name=model_name)
    else:
        raise ValueError(f"Provider '{provider}' is not supported yet. Supported providers: openai, google.")
