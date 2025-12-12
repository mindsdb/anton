from uuid import UUID

from sqlmodel import JSONB, Field

from minds.model.base import BaseSQLModel
from minds.schemas.chat import Role


class Message(BaseSQLModel, table=True):
    __tablename__ = "messages"

    conversation_id: UUID = Field(description="ID of the conversation that this message belongs to", index=True)
    role: Role = Field(description="Role of the message")
    content: JSONB = Field(description="Content of the message")
