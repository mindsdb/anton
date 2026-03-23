from __future__ import annotations

import enum
from uuid import UUID

import sqlalchemy as sa
from sqlmodel import Column, Field

from minds.model.base import BaseSQLModel


class RuleType(str, enum.Enum):
    """
    We keep this to follow the .md flow when injecting in the prompt as
    # Always
    do x
    # Never
    do y
    """

    always = "always"
    never = "never"
    when = "when"


class MemoryRule(BaseSQLModel, table=True):
    __tablename__ = "memory_rules"

    mind_id: UUID = Field(..., foreign_key="minds.id", index=True, description="Mind this rule belongs to")

    rule_type: RuleType = Field(
        sa_column=Column(sa.Enum(RuleType, name="rule_type_enum"), nullable=False),
        description="Type of rule: always (do this), never (don't do this), when (conditional instruction)",
    )
    content: str = Field(
        sa_column=Column(sa.Text, nullable=False),
        description="The rule instruction text injected into the system prompt",
    )
