"""
Datasource model for internal storage.

This module defines the SQLModel for datasources, matching MindsDB's schema
while adding user attribution for multi-user support.
"""

from typing import Any

from sqlalchemy import JSON, UniqueConstraint
from sqlmodel import Column, Field

from minds.model.base import BaseSQLModel


class Datasource(BaseSQLModel, table=True):
    """
    Datasource model matching MindsDB schema with user attribution.

    Schema matches MindsDB: id, updated_at, created_at, name, data, engine
    Plus user_id for user attribution that MindsDB lacks.
    """

    __tablename__ = "datasources"

    # Core fields (matching MindsDB)
    name: str = Field(..., max_length=255, description="Datasource name (unique per company)")
    engine: str = Field(..., max_length=50, description="Database engine (postgres, mysql, etc.)")
    connection_data: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON), description="Connection parameters (matches MindsDB 'data' field)"
    )

    user_id: str = Field(..., max_length=255, description="ID of the user who created this datasource")

    # Database constraints
    __table_args__ = (UniqueConstraint("name", "user_id", name="unique_datasource_name_per_user"),)

    def __repr__(self) -> str:
        """String representation of the datasource."""
        return f"Datasource(name='{self.name}', engine='{self.engine}', user_id='{self.user_id}')"
