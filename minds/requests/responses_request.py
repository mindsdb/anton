from pydantic import BaseModel, ConfigDict, Field

from minds.requests.chat_completions_request import ChatCompletionRequestMetadata
from minds.schemas.chat import Message


class ReasoningParam(BaseModel):
    """OpenAI Responses-shaped ``reasoning`` object; only ``effort`` is honored."""

    model_config = ConfigDict(extra="ignore")

    effort: str | None = Field(
        default=None,
        description="Reasoning effort level; allowed values are per-model (see GET /v1/models reasoning_efforts)",
    )


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
    reasoning: ReasoningParam | None = Field(
        default=None,
        description="Reasoning configuration (OpenAI Responses shape); only ``effort`` is honored",
    )
