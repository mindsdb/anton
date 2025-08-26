"""
Mind management schemas for the Minds API.

This module contains Pydantic models for mind management operations
including creation, updates, and relationship management.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MindCreateRequest(BaseModel):
    """Request model for creating a new mind."""
    
    name: str = Field(..., description="Name of the mind", min_length=1, max_length=256)
    provider: str = Field(default="openai", description="AI provider (openai, google, etc.)")
    model_name: Optional[str] = Field(None, description="Model name to use")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Mind parameters and configuration")
    datasources: List[str] = Field(default_factory=list, description="List of datasource names to attach")


class MindUpdateRequest(BaseModel):
    """Request model for updating an existing mind."""
    
    name: Optional[str] = Field(None, description="New name for the mind", min_length=1, max_length=256)
    provider: Optional[str] = Field(None, description="AI provider")
    model_name: Optional[str] = Field(None, description="Model name")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Mind parameters and configuration")
    datasources: Optional[List[str]] = Field(None, description="List of datasource names")


class MindResponse(BaseModel):
    """Response model for mind data."""
    
    name: str = Field(..., description="Mind name")
    provider: str = Field(..., description="AI provider")
    model_name: str = Field(..., description="Model name")
    parameters: Dict[str, Any] = Field(..., description="Mind parameters and configuration")
    datasources: List[str] = Field(..., description="Attached datasource names")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")

class MindListResponse(BaseModel):
    """Response model for listing minds."""
    
    minds: List[MindResponse] = Field(..., description="List of minds")
    total: int = Field(..., description="Total number of minds")


class AddDatasourceRequest(BaseModel):
    """Request model for adding a datasource to a mind."""
    
    name: str = Field(..., description="Datasource name", min_length=1, max_length=256)
    tables: Optional[List[str]] = Field(None, description="Specific tables to include from the datasource")
    check_connection: bool = Field(default=False, description="Whether to test connection before adding")


class DeleteMindRequest(BaseModel):
    """Request model for mind deletion with options."""
    
    cascade: bool = Field(default=False, description="Whether to delete associated resources that aren't used elsewhere")


class MindListResponse(BaseModel):
    """Response model for listing minds."""
    
    minds: List[MindResponse] = Field(..., description="List of minds")
    total: int = Field(..., description="Total number of minds")


class MindDatasourceResponse(BaseModel):
    """Response model for mind-datasource operations."""
    
    success: bool = Field(..., description="Whether the operation was successful")
    message: Optional[str] = Field(None, description="Additional information about the operation")


