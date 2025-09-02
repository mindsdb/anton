from typing import Optional, Union

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.gemini import GeminiModel
from minds.common.vars import DEFAULT_MIND_MODEL, DEFAULT_GOOGLE_MODEL


def get_llm_config(provider: str, model_name: Optional[str] = None) -> Union[OpenAIModel, GeminiModel]:
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
        return OpenAIModel(model_name=model_name)
    elif provider in ["google", "gemini"]:
        # Use Google default if no model specified
        if not model_name:
            model_name = DEFAULT_GOOGLE_MODEL
        return GeminiModel(model_name=model_name)
    else:
        raise ValueError(
            f"Provider '{provider}' is not supported yet. Supported providers: openai, google."
        )