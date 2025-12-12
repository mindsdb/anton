from typing import TYPE_CHECKING
from uuid import UUID

from sqlmodel import Field, Relationship

from minds.model.base import BaseSQLModel

if TYPE_CHECKING:
    from minds.model.message import Message


class Conversation(BaseSQLModel, table=True):
    __tablename__ = "conversations"

    user_id: UUID = Field(description="ID of the user who owns this mind", index=True)

    messages: list["Message"] = Relationship(back_populates="conversation")