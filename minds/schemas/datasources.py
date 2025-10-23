"""
Request and response schemas for Datasources endpoints.

This module contains Pydantic models for validating requests and responses
for datasource management operations.
"""

from typing import Any
from uuid import UUID

from mindsdb_sdk.databases import Database
from pydantic import BaseModel, Field


class DatasourceCreateRequest(BaseModel):
    """Request model for creating a new datasource (matches MindsDB schema)."""

    name: str = Field(..., description="Datasource name")
    description: str | None = Field(None, description="Description of the datasource")
    engine: str = Field(..., description="Database engine (postgres, mysql, etc.)")
    connection_data: dict[str, Any] = Field(..., description="Connection parameters")


class DatasourceUpdateRequest(BaseModel):
    """Request model for updating an existing datasource (simplified schema)."""

    description: str | None = Field(None, description="Updated description of the datasource")
    connection_data: dict[str, Any] | None = Field(None, description="Updated connection parameters")


class DatasourceResponse(BaseModel):
    """Response model for datasource data (simplified schema)."""

    id: UUID = Field(..., description="Datasource ID")
    description: str | None = Field(None, description="Description of the datasource")
    name: str = Field(..., description="Datasource name")
    engine: str | None = Field(None, description="Database engine")
    created_at: str | None = Field(None, description="Creation timestamp")
    modified_at: str | None = Field(None, description="Last update timestamp")
    is_demo: bool | None = Field(None, description="Whether this is a demo datasource")


class DatasourceConnectionStatus(BaseModel):
    """Model for datasource connection status."""

    success: bool = Field(..., description="Whether connection is successful")
    error_message: str | None = Field(None, description="Error message if connection failed")
    mindsdb_database: Database | None = Field(None, description="Underlying MindsDB Database object if available", exclude=True)


class DatasourceDetailedResponse(DatasourceResponse):
    """Extended response model with connection status."""

    connection_data: dict[str, Any] | None = Field(None, description="Connection parameters")
    connection_status: DatasourceConnectionStatus | None = Field(None, description="Connection status")


class DeleteDatasourceRequest(BaseModel):
    """Request model for datasource deletion."""

    cascade: bool = Field(default=False, description="Remove from all minds that use it")


class DatasourceTableSampleResponse(BaseModel):
    """Response model for datasource table sample data."""

    data: list[list[Any]] = Field(..., description="Sample data as array of arrays")
    column_names: list[str] = Field(..., description="Column names for the data")
