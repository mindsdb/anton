"""
Mind management schemas for the Minds API.

This module contains Pydantic models for mind management operations
including creation, updates, and relationship management.
"""

from typing import Any

from pydantic import BaseModel, Field


class MindCreateRequest(BaseModel):
    """Request model for creating a new mind."""

    name: str = Field(..., description="Name of the mind", min_length=1, max_length=256)
    provider: str = Field(default="openai", description="AI provider (openai, google, etc.)")
    model_name: str | None = Field(None, description="Model name to use")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Mind parameters and configuration")
    datasources: list[str] = Field(default_factory=list, description="List of datasource names to attach")


class MindUpdateRequest(BaseModel):
    """Request model for updating an existing mind."""

    name: str | None = Field(None, description="New name for the mind", min_length=1, max_length=256)
    provider: str | None = Field(None, description="AI provider")
    model_name: str | None = Field(None, description="Model name")
    parameters: dict[str, Any] | None = Field(None, description="Mind parameters and configuration")
    datasources: list[str] | None = Field(None, description="List of datasource names")


class MindResponse(BaseModel):
    """Response model for mind data."""

    name: str = Field(..., description="Mind name")
    provider: str = Field(..., description="AI provider")
    model_name: str = Field(..., description="Model name")
    parameters: dict[str, Any] = Field(..., description="Mind parameters and configuration")
    datasources: list[str] = Field(..., description="Attached datasource names")
    created_at: str | None = Field(None, description="Creation timestamp")
    updated_at: str | None = Field(None, description="Last update timestamp")


class DeleteMindRequest(BaseModel):
    """Request model for mind deletion with options."""

    cascade: bool = Field(
        default=False, description="Whether to delete associated resources that aren't used elsewhere"
    )
