from typing import TYPE_CHECKING
from uuid import UUID

from sqlmodel import Field, Relationship

from minds.model.base import BaseSQLModel

if TYPE_CHECKING:
    from minds.model.data_catalog.column_statistics import ColumnStatistics


class Column(BaseSQLModel, table=True):
    """Data source column metadata."""

    __tablename__ = "columns"

    table_id: UUID = Field(..., description="Table ID", foreign_key="tables.id")
    name: str = Field(..., description="Column name")
    data_type: str = Field(..., description="Column data type")
    description: str | None = Field(default=None, description="Column description/comment")
    default_value: str | None = Field(default=None, description="Column default value")
    is_nullable: bool = Field(default=True, description="Whether the column is nullable")

    statistics: "ColumnStatistics" = Relationship()
