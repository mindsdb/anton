from typing import Any
from uuid import UUID


from pydantic import BaseModel
from sqlmodel import Column, Field
from sqlalchemy.dialects.postgresql import JSONB

from minds.model.base import BaseSQLModel
from minds.schemas.chat import Message as ChatMessage, Role


class Message(BaseSQLModel, table=True):
    __tablename__ = "messages"

    conversation_id: UUID = Field(
        ..., foreign_key="conversations.id", description="ID of the conversation that this message belongs to", index=True
    )
    role: Role = Field(description="Role of the message")
    content: dict[str, Any] | BaseModel | str | list[Any] = Field(
        default_factory=dict, sa_column=Column(JSONB), description="Content of the message as JSON"
    )

