from typing import TYPE_CHECKING
from uuid import UUID

from sqlmodel import Field, Relationship

from minds.model.base import BaseSQLModel

if TYPE_CHECKING:
    from minds.model.data_catalog.column import Column
    from minds.model.data_catalog.table import Table


class ForeignKeyConstraint(BaseSQLModel, table=True):
    """Data source foreign key metadata."""

    __tablename__ = "foreign_key_constraints"

    table_id: UUID = Field(..., description="Table ID", foreign_key="tables.id")
    column_id: UUID = Field(..., description="Column ID", foreign_key="columns.id")
    referenced_table_id: UUID = Field(..., description="Referenced table ID", foreign_key="tables.id")
    referenced_column_id: UUID = Field(..., description="Referenced column ID", foreign_key="columns.id")
    constraint_name: str | None = Field(default=None, description="Constraint name")
    ordinal_position: int | None = Field(default=None, description="Ordinal position")

    column: "Column" = Relationship(
        sa_relationship_kwargs={
            "foreign_keys": "ForeignKeyConstraint.column_id",
        }
    )
    referenced_table: "Table" = Relationship(
        sa_relationship_kwargs={
            "foreign_keys": "ForeignKeyConstraint.referenced_table_id",
        }
    )
    referenced_column: "Column" = Relationship(
        sa_relationship_kwargs={
            "foreign_keys": "ForeignKeyConstraint.referenced_column_id",
        }
    )
