from sqlmodel import Field, SQLModel


class MessageTracing(SQLModel):
    """Message tracing model."""

    model_name: str | None = Field(default=None, description="Model/mind name")
    request_id: str | None = Field(default=None, description="Request ID")
    langfuse_trace_id: str | None = Field(default=None, description="Langfuse trace ID for cross-referencing")
    input_tokens: int = Field(default=0, description="Number of input (prompt) tokens")
    output_tokens: int = Field(default=0, description="Number of output (completion) tokens")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
