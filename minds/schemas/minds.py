"""
Mind management schemas for the Minds API.

This module contains Pydantic models for mind management operations
including creation, updates, and relationship management.
"""

from typing import Any

from pydantic import BaseModel, Field, computed_field, field_validator

from minds.model.mind_datasource import DataCatalogStatus, DetailedDataCatalogStatus


class DatasourceConfig(BaseModel):
    """Reference to a datasource with optional table specification."""

    name: str = Field(..., description="Name of the datasource")
    tables: list[str] | None = Field(None, description="Specific tables to use (None = all tables)")
    status: DetailedDataCatalogStatus = Field(
        default_factory=lambda: DetailedDataCatalogStatus(
            tasks=[], progress=0.0, overall_status=DataCatalogStatus.PENDING
        ),
        description="Loading status of the datasource",
    )


class DetailedDatasourceConfig(DatasourceConfig):
    """Extended datasource config with connection details."""

    engine: str | None = Field(None, description="Database engine (postgres, mysql, etc.)")
    description: str | None = Field(None, description="Description of the datasource")
    connection_data: dict[str, Any] | None = Field(None, description="Connection parameters")
    created_at: str | None = Field(None, description="Creation timestamp")
    modified_at: str | None = Field(None, description="Last update timestamp")


class MindCreateRequest(BaseModel):
    """Request model for creating a new mind."""

    name: str = Field(..., description="Name of the mind", min_length=1, max_length=256)
    provider: str = Field(default="openai", description="AI provider (openai, google, etc.)")
    model_name: str | None = Field(None, description="Model name to use")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Mind parameters and configuration")
    datasources: list[DatasourceConfig] = Field(default_factory=list, description="List of datasource names to attach")

    @field_validator("name", mode="before")
    def lowercase_name(cls, v: str) -> str:
        if isinstance(v, str):
            return v.lower()
        return v

    @field_validator("datasources", mode="after")
    def lowercase_datasource_names(cls, v: list[DatasourceConfig]) -> list[DatasourceConfig]:
        for ds in v:
            ds.name = ds.name.lower()
        return v


class MindUpdateRequest(BaseModel):
    """Request model for updating an existing mind."""

    name: str | None = Field(None, description="New name for the mind", min_length=1, max_length=256)
    provider: str | None = Field(None, description="AI provider")
    model_name: str | None = Field(None, description="Model name")
    parameters: dict[str, Any] | None = Field(None, description="Mind parameters and configuration")
    datasources: list[DatasourceConfig] | None = Field(None, description="List of datasource names")

    @field_validator("name", mode="before")
    def lowercase_name(cls, v: str) -> str:
        if isinstance(v, str):
            return v.lower()
        return v

    @field_validator("datasources", mode="after")
    def lowercase_datasource_names(cls, v: list[DatasourceConfig]) -> list[DatasourceConfig]:
        for ds in v:
            ds.name = ds.name.lower()
        return v


class MindResponse(BaseModel):
    """Response model for mind data."""

    name: str = Field(..., description="Mind name")
    provider: str = Field(..., description="AI provider")
    model_name: str = Field(..., description="Model name")
    parameters: dict[str, Any] = Field(..., description="Mind parameters and configuration")
    datasources: list[DatasourceConfig] | list[DetailedDatasourceConfig] = Field(
        ..., description="List of attached datasources"
    )
    created_at: str | None = Field(None, description="Creation timestamp")
    modified_at: str | None = Field(None, description="Last update timestamp")

    @computed_field
    @property
    def status(self) -> DataCatalogStatus:
        """
        Status of the mind based on loading status of the attached datasources.
        """
        # TODO: Is this correct? We could have a failed load and others that are running?
        for datasource in self.datasources:
            if (
                datasource.status.overall_status in [DataCatalogStatus.PENDING, DataCatalogStatus.LOADING]
                or datasource.status.overall_status == DataCatalogStatus.FAILED
            ):
                return datasource.status.overall_status
        return DataCatalogStatus.COMPLETED
