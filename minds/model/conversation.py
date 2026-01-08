from typing import TYPE_CHECKING
from uuid import UUID

from sqlmodel import Field, Relationship

from minds.model.base import BaseSQLModel

if TYPE_CHECKING:
    from minds.model.message import Message
    from minds.model.mind import Mind


class Conversation(BaseSQLModel, table=True):
    __tablename__ = "conversations"

    user_id: UUID = Field(description="ID of the user who owns this conversation", index=True)
    mind_id: UUID = Field(
        description="ID of the mind that this conversation belongs to",
        index=True,
        foreign_key="minds.id",
    )
    topic: str = Field(description="Topic of the conversation", max_length=255)

    messages: list["Message"] = Relationship()
    mind: "Mind" = Relationship(back_populates="conversations")
