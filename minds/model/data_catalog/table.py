from typing import TYPE_CHECKING
from uuid import UUID

from sqlmodel import Field, Relationship

from minds.model.base import BaseSQLModel

if TYPE_CHECKING:
    from minds.model.data_catalog.column import Column
    from minds.model.data_catalog.foreign_key_constraint import ForeignKeyConstraint
    from minds.model.data_catalog.primary_key_constraint import PrimaryKeyConstraint


class Table(BaseSQLModel, table=True):
    """Data source table metadata."""

    __tablename__ = "tables"

    datasource_id: UUID = Field(..., description="Datasource ID", foreign_key="datasources.id")
    name: str = Field(..., description="Table name")
    schema: str | None = Field(default=None, description="Schema name")
    description: str | None = Field(default=None, description="Table description/comment")
    type: str | None = Field(default=None, description="Table type")
    row_count: int | None = Field(default=None, description="Row count")

    @property
    def qualified_name(self) -> str:
        """Schema-qualified table name, e.g. 'SalesLT.Orders'."""
        return f"{self.schema}.{self.name}" if self.schema else self.name

    columns: list["Column"] = Relationship()
    primary_key_constraints: list["PrimaryKeyConstraint"] = Relationship()
    foreign_key_constraints: list["ForeignKeyConstraint"] = Relationship(
        sa_relationship_kwargs={
            "foreign_keys": "ForeignKeyConstraint.table_id",
        }
    )
