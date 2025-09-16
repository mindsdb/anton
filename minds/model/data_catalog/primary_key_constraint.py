from typing import TYPE_CHECKING
from uuid import UUID

from sqlmodel import Field, Relationship

from minds.model.base import BaseSQLModel

if TYPE_CHECKING:
    from minds.model.data_catalog.column import Column


class PrimaryKeyConstraint(BaseSQLModel, table=True):
    """Data source primary key metadata."""

    __tablename__ = "primary_key_constraints"

    table_id: UUID = Field(..., description="Table ID", foreign_key="tables.id")
    column_id: UUID = Field(..., description="Column ID", foreign_key="columns.id")
    ordinal_position: int | None = Field(default=None, description="Ordinal position")
    constraint_name: str | None = Field(default=None, description="Constraint name")

    column: "Column" = Relationship()
