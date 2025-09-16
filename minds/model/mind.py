from typing import TYPE_CHECKING, Any

from sqlalchemy import Text, UniqueConstraint
from sqlmodel import JSON, Column, Field, Relationship

from minds.model.base import BaseSQLModel

if TYPE_CHECKING:
    from minds.model.mind_datasource import MindDatasource


class Mind(BaseSQLModel, table=True):
    __tablename__ = "minds"

    # Basic mind information
    name: str = Field(description="Name of the mind", max_length=256, index=True)
    provider: str = Field(description="AI provider (openai, google, etc.)", max_length=50)
    model_name: str = Field(description="Model name to use", max_length=256)

    # User ownership
    user_id: str = Field(description="ID of the user who owns this mind", max_length=256, index=True)

    # Configuration
    parameters: dict[str, Any] | None = Field(
        default_factory=dict, sa_column=Column(JSON), description="Mind parameters and configuration as JSON"
    )

    # Optional metadata
    description: str | None = Field(
        default=None, sa_column=Column(Text), description="Optional description of the mind"
    )

    # Relationships - Many-to-many with datasources through junction table
    mind_datasources: list["MindDatasource"] = Relationship(back_populates="mind")

    # Database constraints
    __table_args__ = (UniqueConstraint("name", "user_id", name="unique_mind_name_per_user"),)
