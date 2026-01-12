from pydantic import BaseModel, Field

from minds.requests.chat_completions_request import ChatCompletionRequestMetadata
from minds.schemas.chat import Message


class ResponsesRequest(BaseModel):
    input: str | list[Message] | None = Field(
        default=None, description="Input for the responses request, either a string or a list of messages"
    )
    # TODO: The OpenAI API also supports a conversation object?
    conversation: str | None = Field(
        default=None,
        description="Conversation ID for the responses request, if not provided, a new conversation will be created",
    )
    model: str = Field(description="Model name for the chat completion request")
    stream: bool | None = Field(
        default=False,
        description="Whether the chat completion request is streaming or not",
    )
    metadata: ChatCompletionRequestMetadata | None = Field(
        default=None,
        description="Metadata for the responses request, including enable_charting",
    )
