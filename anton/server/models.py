"""OpenAI-compatible request/response models for Anton's /v1/responses endpoint.

Matches the shape used by anton_servicesrepo/scratchpad_service/chat_models.py
so the same client code works against either backend.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    # Thought events (tool activity visible to client)
    thought_scratchpad_start = "thought.scratchpad.start"
    thought_scratchpad_progress = "thought.scratchpad.progress"
    thought_scratchpad_result = "thought.scratchpad.result"
    thought_scratchpad_end = "thought.scratchpad.end"
    thought_memorize_start = "thought.memorize.start"
    thought_memorize_end = "thought.memorize.end"
    thought_recall_start = "thought.recall.start"
    thought_recall_end = "thought.recall.end"
    thought_progress = "thought.progress"
    thought_context_compacted = "thought.context_compacted"


class Message(BaseModel):
    role: str
    content: str | list[Any] | None = None


class ResponsesRequest(BaseModel):
    input: str | list[Message]
    model: str = "anton"
    stream: bool = True
    conversation: str | None = None  # session/conversation ID


class ResponseStatus(str, Enum):
    created = "created"
    in_progress = "in_progress"
    completed = "completed"
    failed = "failed"


class ResponseOutputContent(BaseModel):
    type: str = "output_text"
    text: str = ""


class ResponseOutput(BaseModel):
    type: str = "message"
    id: str = Field(default_factory=lambda: f"msg-{uuid.uuid4().hex[:12]}")
    status: ResponseStatus = ResponseStatus.completed
    role: str = "assistant"
    content: list[ResponseOutputContent] = Field(default_factory=list)


class ResponseObject(BaseModel):
    id: str = Field(default_factory=lambda: f"resp-{uuid.uuid4().hex[:12]}")
    object: str = "response"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    status: ResponseStatus = ResponseStatus.created
    model: str = "anton"
    output: list[ResponseOutput] = Field(default_factory=list)
    error: str | None = None


class StreamingResponseEvent(str, Enum):
    created = "response.created"
    in_progress = "response.in_progress"
    output_text_delta = "response.output_text.delta"
    completed = "response.completed"
    failed = "response.failed"
