"""
Message management schemas for the Minds API.

This module contains Pydantic models for message management operations
including creation, updates, retrieval, and deletion.
"""

from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from minds.schemas.chat import Role


class MessageContentType(Enum):
    """Type of a message content."""
    input_text = "input_text"
    output_text = "output_text"


class MessageContent(BaseModel):
    """Content of a message."""
    type: MessageContentType = MessageContentType.output_text
    text: str = Field(..., description="Text of the message")


class MessageResponse(BaseModel):
    """Response model for message data."""

    id: UUID = Field(..., description="Message ID")
    type: Literal["message"] = "message"
    role: Role = Field(..., description="Role of the message")
    status: Literal["completed"] = "completed"
    content: MessageContent = Field(..., description="Content of the message")
    created_at: str | None = Field(None, description="Creation timestamp")
    modified_at: str | None = Field(None, description="Last update timestamp")