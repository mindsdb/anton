"""
Request and response schemas for Datasources endpoints.

This module contains Pydantic models for validating requests and responses
for datasource management operations.
"""

from typing import Any
from uuid import UUID

from mindsdb_sdk.databases import Database
from pydantic import BaseModel, ConfigDict, Field, field_validator


class DatasourceCreateRequest(BaseModel):
    """Request model for creating a new datasource (matches MindsDB schema)."""

    name: str = Field(..., description="Datasource name")
    description: str | None = Field(None, description="Description of the datasource")
    engine: str = Field(..., description="Database engine (postgres, mysql, etc.)")
    connection_data: dict[str, Any] = Field(..., description="Connection parameters")
    is_sample: bool = Field(default=False, description="Whether the datasource is a sample")

    @field_validator("name", "engine", mode="before")
    def lowercase_fields(cls, v: str) -> str:
        if isinstance(v, str):
            return v.lower()
        return v


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
    is_sample: bool = Field(default=False, description="Whether the datasource is a sample")
    created_at: str | None = Field(None, description="Creation timestamp")
    modified_at: str | None = Field(None, description="Last update timestamp")


class DatasourceConnectionStatus(BaseModel):
    """Model for datasource connection status."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool = Field(..., description="Whether connection is successful")
    error_message: str | None = Field(None, description="Error message if connection failed")
    mindsdb_database: Database | None = Field(
        None, description="Underlying MindsDB Database object if available", exclude=True
    )


class DatasourceDetailedResponse(DatasourceResponse):
    """Extended response model with connection status."""

    connection_data: dict[str, Any] | None = Field(None, description="Connection parameters")
    connection_status: DatasourceConnectionStatus | None = Field(None, description="Connection status")


class DeleteDatasourceRequest(BaseModel):
    """Request model for datasource deletion."""

    cascade: bool = Field(default=False, description="Remove from all minds that use it")


class DatasourceQueryRequest(BaseModel):
    """Request model for executing a native query against a datasource."""

    query: str = Field(..., description="SQL query to execute against the datasource")
    native_query: bool = Field(
        default=True, description="Send raw SQL directly to the database, bypassing MindsDB parsing"
    )


class DatasourceQueryResponse(BaseModel):
    """Response model for native query results."""

    data: list[list[Any]] | None = Field(default=None, description="Query result rows")
    column_names: list[str] | None = Field(default=None, description="Column names for the data")


class DatasourceTableSampleResponse(DatasourceQueryResponse):
    """Response model for table sample data."""


class ColumnStatisticsResponse(BaseModel):
    """Response model for column statistics."""

    most_common_values: list[Any] | None = Field(default=None, description="List of most common values")
    most_common_frequencies: list[float] | None = Field(default=None, description="List of most common frequencies")
    null_percentage: float | None = Field(default=None, description="Null percentage")
    distinct_values_count: int | None = Field(default=None, description="Distinct values count")
    min_value: str | None = Field(default=None, description="Minimum value")
    max_value: str | None = Field(default=None, description="Maximum value")


class ColumnResponse(BaseModel):
    """Response model for column metadata."""

    name: str = Field(..., description="Column name")
    data_type: str = Field(..., description="Column data type")
    description: str | None = Field(default=None, description="Column description/comment")
    default_value: str | None = Field(default=None, description="Column default value")
    is_nullable: bool = Field(default=True, description="Whether the column is nullable")
    statistics: ColumnStatisticsResponse | None = Field(default=None, description="Column statistics")


class PrimaryKeyConstraintResponse(BaseModel):
    """Response model for primary key constraint."""

    column_name: str = Field(..., description="Column name")
    ordinal_position: int | None = Field(default=None, description="Ordinal position")
    constraint_name: str | None = Field(default=None, description="Constraint name")


class ForeignKeyConstraintResponse(BaseModel):
    """Response model for foreign key constraint."""

    column_name: str = Field(..., description="Column name")
    referenced_table_name: str = Field(..., description="Referenced table name")
    referenced_column_name: str = Field(..., description="Referenced column name")
    constraint_name: str | None = Field(default=None, description="Constraint name")
    ordinal_position: int | None = Field(default=None, description="Ordinal position")


class TableResponse(BaseModel):
    """Response model for table metadata."""

    name: str = Field(..., description="Table name")
    # Note: 'schema' field name shadows BaseModel.schema, but this is intentional
    # for database schema names. The field works correctly despite the warning.
    schema: str | None = Field(default=None, description="Schema name")
    description: str | None = Field(default=None, description="Table description/comment")
    type: str | None = Field(default=None, description="Table type")
    row_count: int | None = Field(default=None, description="Row count")
    columns: list[ColumnResponse] = Field(default_factory=list, description="Table columns")
    primary_key_constraints: list[PrimaryKeyConstraintResponse] = Field(
        default_factory=list, description="Primary key constraints"
    )
    foreign_key_constraints: list[ForeignKeyConstraintResponse] = Field(
        default_factory=list, description="Foreign key constraints"
    )


class DataCatalogResponse(BaseModel):
    """Response model for data catalog."""

    datasource: DatasourceResponse = Field(..., description="Datasource information")
    tables: list[TableResponse] = Field(default_factory=list, description="List of cataloged tables")


class UpdateTableDescriptionRequest(BaseModel):
    """Request model for updating table description."""

    description: str | None = Field(
        None, description="New description for the table. Set to null to clear the description."
    )


class UpdateColumnDescriptionRequest(BaseModel):
    """Request model for updating column description."""

    description: str | None = Field(
        None, description="New description for the column. Set to null to clear the description."
    )
