"""
Junction table for Many-to-Many relationship between Minds and Datasources.

This allows:
- Multiple minds to use the same datasource
- A mind to use multiple datasources
- Proper referential integrity
"""

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, Relationship

from minds.model.base import BaseSQLModel

if TYPE_CHECKING:
    from minds.model.datasource import Datasource
    from minds.model.mind import Mind


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

    # Relationships back to parent models
    mind: "Mind" = Relationship(back_populates="mind_datasources")
    datasource: "Datasource" = Relationship(back_populates="mind_datasources")

    # Ensure each mind-datasource pair is unique
    __table_args__ = (UniqueConstraint("mind_id", "datasource_id", name="unique_mind_datasource_pair"),)

    def __repr__(self) -> str:
        """String representation of the mind-datasource relationship."""
        return f"MindDatasource(mind_id='{self.mind_id}', datasource_id='{self.datasource_id}')"
