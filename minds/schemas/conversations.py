"""
Conversation management schemas for the Minds API.

This module contains Pydantic models for conversation management operations
including creation, updates, retrieval, and deletion.
"""

from datetime import datetime, timezone
from uuid import UUID

from pydantic import BaseModel, Field

from minds.schemas.messages import MessageResponse


class ConversationCreateRequest(BaseModel):
    """Request model for creating a new conversation."""

    topic: str = Field(
        description="Topic of the conversation. If not provided, defaults to the creation timestamp.",
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    )


class ConversationResponse(BaseModel):
    """Response model for conversation data."""

    id: UUID = Field(..., description="Conversation ID")
    topic: str = Field(..., description="Topic of the conversation")
    created_at: str | None = Field(None, description="Creation timestamp")
    modified_at: str | None = Field(None, description="Last update timestamp")


class ConversationDetailedResponse(ConversationResponse):
    """Response model for conversation data with messages."""

    messages: list[MessageResponse] = Field(..., description="Messages in the conversation")