from typing import Any
from uuid import UUID

from sqlmodel import JSON, Field
from sqlmodel import Column as SQLModelColumn

from minds.model.base import BaseSQLModel


class ColumnStatistics(BaseSQLModel, table=True):
    """Data source column statistics."""

    __tablename__ = "column_statistics"

    column_id: UUID = Field(..., description="Column ID", foreign_key="columns.id")
    most_common_values: list[Any] | None = Field(
        default_factory=list, sa_column=SQLModelColumn(JSON), description="List of most common values"
    )
    most_common_frequencies: list[float] | None = Field(
        default_factory=list, sa_column=SQLModelColumn(JSON), description="List of most common frequencies"
    )
    null_percentage: float | None = Field(default=None, description="Null percentage")
    distinct_values_count: int | None = Field(default=None, description="Distinct values count")
    min_value: str | None = Field(default=None, description="Minimum value")
    max_value: str | None = Field(default=None, description="Maximum value")
