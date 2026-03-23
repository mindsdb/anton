from __future__ import annotations

from uuid import UUID

import sqlalchemy as sa
from sqlalchemy import Index
from sqlmodel import JSON, Column, Field

from minds.model.base import BaseSQLModel


class MemoryTopic(BaseSQLModel, table=True):
    __tablename__ = "memory_topics"

    mind_id: UUID = Field(..., foreign_key="minds.id", index=True, description="Mind this topic belongs to")

    title: str = Field(
        max_length=256,
        description="Human-readable title used in scoring and prompt display",
    )
    tags: list[str] | None = Field(
        default=None,
        sa_column=Column(JSON),
        description="Keyword tags used for relevance scoring",
    )
    description: str | None = Field(
        default=None,
        sa_column=Column(sa.Text),
        description="Short summary used for scoring — not injected into the prompt",
    )
    body: str = Field(
        sa_column=Column(sa.Text, nullable=False),
        description="Full markdown content injected into the system prompt when topic is selected",
    )

    __table_args__ = (
        Index(
            "unique_memory_topic_title_per_mind",
            "mind_id",
            "title",
            unique=True,
            postgresql_where=sa.text("deleted_at IS NULL"),
        ),
    )
