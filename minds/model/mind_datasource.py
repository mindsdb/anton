"""
Junction table for Many-to-Many relationship between Minds and Datasources.

This allows:
- Multiple minds to use the same datasource
- A mind to use multiple datasources
- Proper referential integrity
"""

from enum import Enum
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import Enum as SAEnum, JSON, UniqueConstraint
from sqlmodel import Column as SQLModelColumn
from sqlmodel import Field, Relationship

from minds.model.base import BaseSQLModel

if TYPE_CHECKING:
    from minds.model.datasource import Datasource
    from minds.model.mind import Mind


class DataCatalogStatus(str, Enum):
    PENDING = "PENDING"
    LOADING = "LOADING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class MindDatasource(BaseSQLModel, table=True):
    """
    Junction table linking Minds to Datasources in a many-to-many relationship.

    This enables:
    - Multiple minds to share the same datasource (e.g., company database)
    - A mind to access multiple datasources (e.g., customer + product data)
    - Proper foreign key relationships and referential integrity
    """

    __tablename__ = "mind_datasources"

    mind_id: UUID = Field(..., foreign_key="minds.id", description="ID of the mind", index=True)

    datasource_id: UUID = Field(..., foreign_key="datasources.id", description="ID of the datasource", index=True)

    tables: list[str] | None = Field(
        default_factory=list, sa_column=SQLModelColumn(JSON), description="Specific tables to use (None = all tables)"
    )

    status: DataCatalogStatus = Field(
        default=DataCatalogStatus.PENDING,
        sa_column=SQLModelColumn(
            SAEnum(DataCatalogStatus, name="data_catalog_status", native_enum=False), nullable=False
        ),
        description="Status of the data catalog loading",
    )

    # Relationships back to parent models
    mind: "Mind" = Relationship(back_populates="mind_datasources")
    datasource: "Datasource" = Relationship(back_populates="mind_datasources")

    # Ensure each mind-datasource pair is unique
    __table_args__ = (UniqueConstraint("mind_id", "datasource_id", name="unique_mind_datasource_pair"),)

    def __repr__(self) -> str:
        """String representation of the mind-datasource relationship."""
        return f"MindDatasource(mind_id='{self.mind_id}', datasource_id='{self.datasource_id}')"
