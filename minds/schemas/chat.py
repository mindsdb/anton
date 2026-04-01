import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_serializer


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    function = "function"
    tool = "tool"

    # Roles for thought rendering
    thought_planning = "thought.planning"
    thought_execution = "thought.execution"
    thought_execution_step = "thought.execution.step"
    thought_execution_step_start = "thought.execution.step.start"
    thought_execution_step_sql = "thought.execution.step.sql"
    thought_execution_step_end = "thought.execution.step.end"

    # Anton agent thought roles
    thought_scratchpad_start = "thought.scratchpad.start"
    thought_scratchpad_end = "thought.scratchpad.end"
    thought_scratchpad_result = "thought.scratchpad.result"
    thought_memorize_start = "thought.memorize.start"
    thought_memorize_end = "thought.memorize.end"
    thought_recall_start = "thought.recall.start"
    thought_recall_end = "thought.recall.end"
    thought_context_compacted = "thought.context.compacted"


class Message(BaseModel):
    role: Role
    content: dict | BaseModel | str | list[Any] | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    @model_serializer
    def _serialize(self) -> dict[str, Any]:
        """Omit None tool fields from serialization."""
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls is not None:
            data["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            data["name"] = self.name
        return data


class StreamMessage(Message):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")


class StreamChoice(BaseModel):
    index: int
    delta: Message
    finish_reason: str | None = None


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str | None = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[StreamChoice]
    system_fingerprint: str | None = None

    def dict(self, *args, **kwargs):
        return super().model_dump(*args, **kwargs)


class ChatCompletion(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage | None = None
    system_fingerprint: str | None = None

    def dict(self, *args, **kwargs):
        return super().model_dump(*args, **kwargs)
