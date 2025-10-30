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

from async_property import async_property
from pydantic import computed_field
from sqlalchemy import UniqueConstraint
from sqlmodel import Field, Relationship

from minds.model.base import BaseSQLModel
from minds.client.prefect import PrefectClient
from minds.model.mind_datasource_table import MindDatasourceTable

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

    flow_run_id: UUID | None = Field(
        default=None, description="ID of the flow run of the Prefect deployment for loading the data catalog"
    )

    # Relationships back to parent models
    mind: "Mind" = Relationship(back_populates="mind_datasources")
    datasource: "Datasource" = Relationship(back_populates="mind_datasources")
    mind_datasource_tables: list["MindDatasourceTable"] = Relationship()

    # Ensure each mind-datasource pair is unique
    __table_args__ = (UniqueConstraint("mind_id", "datasource_id", name="unique_mind_datasource_pair"),)

    @computed_field(return_type=DataCatalogStatus)
    @async_property
    async def status(self) -> DataCatalogStatus:
        """
        Async: get status of the mind-datasource relationship.

        This awaits the Prefect client which exposes async methods.
        Callers should `await mind_datasource.status` from async code.
        """
        prefect_client = PrefectClient()
        if self.flow_run_id:
            states = await prefect_client.get_flow_run_state(str(self.flow_run_id))
            # Get the latest state of the flow run
            state = states[-1]
            if state.is_running():
                return DataCatalogStatus.LOADING
            elif state.is_completed():
                return DataCatalogStatus.COMPLETED
            elif state.is_failed():
                return DataCatalogStatus.FAILED
        return DataCatalogStatus.PENDING

    def __repr__(self) -> str:
        """String representation of the mind-datasource relationship."""
        return f"MindDatasource(mind_id='{self.mind_id}', datasource_id='{self.datasource_id}')"
