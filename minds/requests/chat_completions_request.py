from pydantic import BaseModel, Field

from minds.schemas.chat import Message


class ChatCompletionRequestMetadata(BaseModel):
    mdb_completions_session_id: str | int | None = Field(default=None, description="Session ID for the request")
    enable_charting: bool = Field(default=False, description="Whether to enable charting for the request")


class ChatCompletionsRequest(BaseModel):
    model: str = Field(description="Model name for the chat completion request")
    messages: list[Message] = Field(
        description="List of messages for the chat completion request",
    )
    metadata: ChatCompletionRequestMetadata | None = Field(
        default=None,
        description="Make metadata optional with default None",
    )
    stream: bool | None = Field(
        default=False,
        description="Whether the chat completion request is streaming or not",
    )
    tools: list[dict] | None = Field(
        default=None,
        description="List of tools available for the model to call",
    )
    tool_choice: str | dict | None = Field(
        default=None,
        description="Controls which tool the model calls",
    )
    temperature: float | None = Field(
        default=None,
        description="Sampling temperature",
    )
    max_tokens: int | None = Field(
        default=None,
        description="Maximum number of tokens to generate",
    )
    max_completion_tokens: int | None = Field(
        default=None,
        description="Maximum number of completion tokens to generate (OpenAI alias)",
    )
    reasoning_effort: str | None = Field(
        default=None,
        description="Reasoning effort level; allowed values are per-model (see GET /v1/models reasoning_efforts)",
    )
