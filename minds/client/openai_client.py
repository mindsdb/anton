from functools import lru_cache

from openai import AsyncOpenAI

from minds.common.logger import setup_logging
from minds.common.settings.app_settings import get_app_settings
from minds.schemas.chat import Message

logger = setup_logging()
settings = get_app_settings()


class OpenAIClient:
    def __init__(self, api_url: str, api_key: str, chat_completions_model: str, max_tokens: int):
        self.api_url = api_url
        self.chat_completions_model = chat_completions_model
        self.api_key = api_key
        self.max_tokens = max_tokens

        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_url)

    @observe(name="Chat Completions - OpenAI", as_type="generation")
    async def chat_completions(
        self,
        messages: list[Message],
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """
        Chat completions with performance logging.

        Args:
            messages: The messages to send to the model.
            stream: Whether to stream the response.
            temperature: The temperature to use for the model.
            max_tokens: The maximum number of tokens to generate in the response.
        """
        # Log input statistics
        total_chars = sum(len(msg.content) for msg in messages)
        logger.info(f"📤 Sending {len(messages)} messages, {total_chars} chars")

        try:
            response = await self.client.chat.completions.create(
                model=self.chat_completions_model,
                messages=messages,
                stream=stream,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if stream:
                # Streaming response - yield chunks with delta content
                response_chars = 0
                async for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        chunk_content = chunk.choices[0].delta.content
                        response_chars += len(chunk_content)
                        yield chunk_content

                logger.info(f"📥 Streamed response: {response_chars} chars")
            else:
                # Non-streaming response - yield the complete content
                if response.choices and response.choices[0].message.content:
                    content = response.choices[0].message.content
                    logger.info(f"📥 Response: {len(content)} chars")
                    yield content

        except Exception as e:
            logger.error(f"❌ API call failed with error: {e}")
            raise e


@lru_cache
def get_openai_client() -> OpenAIClient:
    """Get cached OpenAIClient instance."""
    return create_openai_client()


def create_openai_client(
    api_url: str = settings.openai.api_url,
    api_key: str = settings.openai.api_key,
    chat_completions_model: str = settings.openai.model_name,
    max_tokens: int = settings.openai.max_tokens,
) -> OpenAIClient:
    return OpenAIClient(
        api_url=api_url,
        api_key=api_key,
        chat_completions_model=chat_completions_model,
        max_tokens=max_tokens,
    )
