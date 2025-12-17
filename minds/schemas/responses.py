import time
import uuid
from enum import Enum

from pydantic import BaseModel, Field

from minds.schemas.chat import Role


class ResponseOutputContent(BaseModel):
    # TODO: There are some other types that have not been included here. Are they needed?
    type: str = "output_text"
    text: str


class ResponseStatus(str, Enum):
    # TODO: Are there other statuses that we are interested in?
    created = "created"
    in_progress = "in_progress"
    completed = "completed"


class ResponseOutput(BaseModel):
    type: str = "message"
    id: str  # The ID here should be the same as the Message stored in the database.
    status: ResponseStatus
    role: str = Role.assistant.value
    content: list[ResponseOutputContent]


class Response(BaseModel):
    # TODO: Should we look at storing the response ID in the database?
    # This is to allow previous_response_id chains.
    id: str = Field(default_factory=lambda: f"resp-{uuid.uuid4()}")
    object: str = "response"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    status: ResponseStatus
    error: str | None = None
    model: str
    output: list[ResponseOutput] = Field(default_factory=list)
    # TODO: There are some other that have not been included here. Are they needed?


class StreamingResponseEvent(str, Enum):
    created = "response.created"
    in_progress = "response.in_progress"
    output_text_delta = "response.output_text.delta"
    completed = "response.completed"


class ResponseDelta(BaseModel):
    item_id: str
    type: str = StreamingResponseEvent.output_text_delta.value
    delta: str


class StreamingResponse(BaseModel):
    type: StreamingResponseEvent
    sequence_number: int
    response: Response | ResponseDelta
