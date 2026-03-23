"""
Request and response schemas for mind memory (rules and topics).
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from minds.model.memory_rule import RuleType


class MemoryRuleCreateRequest(BaseModel):
    rule_type: RuleType = Field(..., description="Type of rule: always, never, or when")
    content: str = Field(..., min_length=1, description="The rule instruction text")


class MemoryRuleUpdateRequest(BaseModel):
    rule_type: RuleType | None = Field(default=None)
    content: str | None = Field(default=None, min_length=1)


class MemoryRuleResponse(BaseModel):
    id: UUID | None
    mind_id: UUID
    rule_type: RuleType
    content: str
    created_at: datetime | None
    modified_at: datetime | None


class MemoryTopicCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=256, description="Human-readable title")
    tags: list[str] | None = Field(default=None, description="Keyword tags used for relevance scoring")
    description: str | None = Field(default=None, description="Short summary used for scoring")
    body: str = Field(..., min_length=1, description="Full markdown content injected into the system prompt")


class MemoryTopicUpdateRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=256)
    tags: list[str] | None = Field(default=None)
    description: str | None = Field(default=None)
    body: str | None = Field(default=None, min_length=1)


class MemoryTopicResponse(BaseModel):
    id: UUID | None
    mind_id: UUID
    title: str
    tags: list[str] | None
    description: str | None
    body: str
    created_at: datetime | None
    modified_at: datetime | None
