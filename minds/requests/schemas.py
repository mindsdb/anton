import uuid
import time
from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, Field


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    function = "function"


class Message(BaseModel):
    role: Role
    content: str | list[Any] = None


class StreamChoice(BaseModel):
    index: int
    delta: Message
    finish_reason: Optional[str] = None


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[StreamChoice]
    system_fingerprint: Optional[str] = None

    def dict(self, *args, **kwargs):
        return super().model_dump(*args, **kwargs)


class ChatCompletion(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None

    def dict(self, *args, **kwargs):
        return super().model_dump(*args, **kwargs)
