from typing import TYPE_CHECKING
from uuid import UUID

from sqlmodel import Field, Relationship

from minds.model.base import BaseSQLModel

if TYPE_CHECKING:
    from minds.model.data_catalog.table import Table


class MindDatasourceTable(BaseSQLModel, table=True):
    """
    Junction table linking MindDatasource to Table.
    """

    __tablename__ = "mind_datasource_tables"

    mind_datasource_id: UUID = Field(..., foreign_key="mind_datasources.id")
    table_id: UUID = Field(..., foreign_key="tables.id")

    table: "Table" = Relationship()
