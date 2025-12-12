"""
Message management schemas for the Minds API.

This module contains Pydantic models for message management operations
including creation, updates, retrieval, and deletion.
"""

from uuid import UUID

from pydantic import BaseModel, Field

from minds.schemas.chat import Role


class MessageResponse(BaseModel):
    """Response model for message data."""

    id: UUID = Field(..., description="Message ID")
    role: Role = Field(..., description="Role of the message")
    content: dict = Field(..., description="Content of the message")
    created_at: str | None = Field(None, description="Creation timestamp")
    modified_at: str | None = Field(None, description="Last update timestamp")