import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    function = "function"

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
    content: dict | BaseModel | str | list[Any] = None


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
