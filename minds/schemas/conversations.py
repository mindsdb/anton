"""
Conversation management schemas for the Minds API.

This module contains Pydantic models for conversation management operations
including creation, updates, retrieval, and deletion.
"""

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from minds.schemas.chat import Message


class ConversationMetadata(BaseModel):
    """Metadata for a conversation."""
    topic: str = Field(
        description="Topic of the conversation. If not provided, defaults to the creation timestamp.",
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    )


class ConversationItem(Message):
    """Item for a conversation."""
    # TODO: Are there other types to support?
    type: Literal["message"] | None = Field(
        default="message",
        description="Type of the item"
    )


class ConversationCreateRequest(BaseModel):
    """Request model for creating a new conversation."""

    metadata: ConversationMetadata = Field(
        default_factory=ConversationMetadata,
        description="Metadata for the conversation"
    )
    items: list[ConversationItem] = Field(
        ...,
        description="Items in the conversation"
    )


class ConversationResponse(BaseModel):
    """Response model for conversation data."""

    id: UUID = Field(..., description="Conversation ID")
    metadata: ConversationMetadata = Field(..., description="Metadata of the conversation")
    created_at: str | None = Field(None, description="Creation timestamp")
    # TODO: This is not returned by the OpenAI API. Should it be included?
    modified_at: str | None = Field(None, description="Last update timestamp")
    object: Literal["conversation"] = Field(default="conversation", description="Object type")
