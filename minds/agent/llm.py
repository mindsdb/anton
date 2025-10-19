from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from minds.common.vars import DEFAULT_GOOGLE_MODEL, DEFAULT_MIND_MODEL, OPEN_AI_API_KEY


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
            model_name = DEFAULT_MIND_MODEL
        openai_provider = OpenAIProvider(api_key=OPEN_AI_API_KEY)
        return OpenAIModel(model_name=model_name, provider=openai_provider)
    elif provider in ["google", "gemini"]:
        # Use Google default if no model specified
        if not model_name:
            model_name = DEFAULT_GOOGLE_MODEL
        return GeminiModel(model_name=model_name)
    else:
        raise ValueError(f"Provider '{provider}' is not supported yet. Supported providers: openai, google.")
