from typing import List, Optional

from pydantic import BaseModel, Field

from minds.requests.schemas import Message


class ChatCompletionRequestMetadata(BaseModel):
    mdb_completions_session_id: Optional[str | int] = Field(
        default=None, description="Session ID for the request"
    )


class ChatCompletionsRequest(BaseModel):
    model: str = Field(description="Model name for the chat completion request")
    messages: List[Message] = Field(
        description="List of messages for the chat completion request",
    )
    metadata: Optional[ChatCompletionRequestMetadata] = Field(
        default=None, description="Make metadata optional with default None"
    )
    stream: Optional[bool] = Field(
        default=False,
        description="Whether the chat completion request is streaming or not",
    )
