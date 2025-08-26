from pydantic import BaseModel, Field

from minds.schemas.chat import Message


class ChatCompletionRequestMetadata(BaseModel):
    mdb_completions_session_id: str | int | None = Field(default=None, description="Session ID for the request")


class ChatCompletionsRequest(BaseModel):
    model: str = Field(description="Model name for the chat completion request")
    messages: list[Message] = Field(
        description="List of messages for the chat completion request",
    )
    metadata: ChatCompletionRequestMetadata | None = Field(
        default=None, description="Make metadata optional with default None"
    )
    stream: bool | None = Field(
        default=False,
        description="Whether the chat completion request is streaming or not",
    )
