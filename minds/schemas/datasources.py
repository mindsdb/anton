"""
Request and response schemas for Datasources endpoints.

This module contains Pydantic models for validating requests and responses
for datasource management operations.
"""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class DatasourceCreateRequest(BaseModel):
    """Request model for creating a new datasource (matches MindsDB schema)."""

    name: str = Field(..., description="Datasource name")
    engine: str = Field(..., description="Database engine (postgres, mysql, etc.)")
    connection_data: dict[str, Any] = Field(..., description="Connection parameters")


class DatasourceUpdateRequest(BaseModel):
    """Request model for updating an existing datasource (simplified schema)."""

    connection_data: dict[str, Any] | None = Field(None, description="Updated connection parameters")


class DatasourceResponse(BaseModel):
    """Response model for datasource data (simplified schema)."""

    id: UUID = Field(..., description="Datasource ID")
    name: str = Field(..., description="Datasource name")
    engine: str | None = Field(None, description="Database engine")
    connection_data: dict[str, Any] | None = Field(None, description="Connection parameters")
    created_at: str | None = Field(None, description="Creation timestamp")
    is_demo: bool | None = Field(None, description="Whether this is a demo datasource")


class DatasourceConnectionStatus(BaseModel):
    """Model for datasource connection status."""

    success: bool = Field(..., description="Whether connection is successful")
    error_message: str | None = Field(None, description="Error message if connection failed")


class DatasourceDetailedResponse(DatasourceResponse):
    """Extended response model with connection status."""

    connection_status: DatasourceConnectionStatus | None = Field(None, description="Connection status")


class DeleteDatasourceRequest(BaseModel):
    """Request model for datasource deletion."""

    cascade: bool = Field(default=False, description="Remove from all minds that use it")
