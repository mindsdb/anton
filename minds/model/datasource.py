"""
Datasource model for internal storage.

This module defines the SQLModel for datasources, matching MindsDB's schema
while adding user attribution for multi-user support.
"""

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, Relationship

from minds.model.base import BaseSQLModel

if TYPE_CHECKING:
    from minds.model.mind_datasource import MindDatasource


if TYPE_CHECKING:
    from minds.model.data_catalog import Table


class Datasource(BaseSQLModel, table=True):
    """
    Datasource model matching MindsDB schema with user attribution.

    Schema matches MindsDB: id, updated_at, created_at, name, data, engine
    Plus user_id for user attribution that MindsDB lacks.
    """

    __tablename__ = "datasources"

    # Core fields (matching MindsDB)
    name: str = Field(..., max_length=255, description="Datasource name (unique per company)")
    description: str | None = Field(None, description="Description of the datasource")
    engine: str = Field(..., max_length=50, description="Database engine (postgres, mysql, etc.)")

    user_id: UUID = Field(..., description="ID of the user who created this datasource")
    engine_info: str | None = Field(None, description="Engine information")

    # Relationships - Many-to-many with minds through junction table
    mind_datasources: list["MindDatasource"] = Relationship(back_populates="datasource")

    # Database constraints
    __table_args__ = (UniqueConstraint("name", "user_id", name="unique_datasource_name_per_user"),)

    tables: list["Table"] = Relationship(sa_relationship_kwargs={"foreign_keys": "Table.datasource_id"})

    def __repr__(self) -> str:
        """String representation of the datasource."""
        return f"Datasource(name='{self.name}', engine='{self.engine}', user_id='{self.user_id}')"
