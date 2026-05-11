"""Pydantic models for the /v1/* API surface.

Two surfaces here:
  - Responses + Conversations (/v1/responses, /v1/conversations)
  - Scratchpad             (/v1/scratchpad/*)

The chat shape mirrors OpenAI's Responses API. The scratchpad shape mirrors
the hosted scratchpad service. Cowork-specific fields (project, attachment_ids)
live alongside the OpenAI fields on the request model because cowork is the
only client today; if we ever generalize, those move to a subclass in the
cowork-specific route layer.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Responses API
# ---------------------------------------------------------------------------


class ResponsesRequest(BaseModel):
    input: str | list[Message]
    model: str | None = "anton"
    stream: bool = True
    conversation: str | None = None  # session/conversation ID
    # Cowork-side extensions — optional, ignored by non-cowork clients.
    project: str | None = None  # project name (folder); None = active project
    attachment_ids: list[str] = Field(default_factory=list)


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


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


class ConversationMeta(BaseModel):
    id: str
    title: str = ""
    turns: int = 0
    preview: str = ""
    created_at: str = ""
    updated_at: str = ""
    project: str | None = None


class ConversationPatch(BaseModel):
    title: str | None = None
    # When set, move the conversation to this project. Handled by
    # conversation_manager.move_conversation rather than the title patch
    # path because it physically relocates files between project dirs.
    project: str | None = None


# ---------------------------------------------------------------------------
# Scratchpad API
# ---------------------------------------------------------------------------


class ScratchpadStartRequest(BaseModel):
    name: str = "default"
    coding_provider: str = ""
    coding_model: str = ""
    coding_api_key: str = ""
    coding_base_url: str = ""


class ScratchpadExecRequest(BaseModel):
    name: str = "default"
    code: str
    description: str = ""
    estimated_time: str = ""
    estimated_seconds: int = 0


class ScratchpadInstallRequest(BaseModel):
    name: str = "default"
    packages: list[str]


class ScratchpadPadRequest(BaseModel):
    name: str = "default"
