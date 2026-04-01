"""
Pydantic models for resource usage limits.

These models represent the structure of limits (from Statsig dynamic config)
and usage tracking across all resource types: tokens, minds, datasources, and questions.
"""

from pydantic import BaseModel, Field

UNLIMITED = -1  # Unlimited is represented as -1


class LimitsConfig(BaseModel):
    """Lifetime and monthly limits for a resource."""

    lifetime: int = Field(default=UNLIMITED, description="Lifetime limit.")
    monthly: int = Field(default=UNLIMITED, description="Monthly limit.")


class UsageConfig(BaseModel):
    """Usage counts split by lifetime and current billing cycle."""

    lifetime: int = Field(default=0, description="Total usage across all time")
    billing_cycle: int = Field(default=0, description="Usage in the current billing cycle")


class ResourceUsageConfig(BaseModel):
    """Limits and current usage for a single resource type."""

    limit: LimitsConfig = Field(default_factory=LimitsConfig, description="The limits for this resource")
    usage: UsageConfig = Field(default_factory=UsageConfig, description="Current usage counts")


class MindLimitsConfig(BaseModel):
    """Aggregate limits and usage across all resource types."""

    tokens: ResourceUsageConfig = Field(default_factory=ResourceUsageConfig, description="Limits and usage for tokens")
    minds: ResourceUsageConfig = Field(default_factory=ResourceUsageConfig, description="Limits and usage for minds")
    datasources: ResourceUsageConfig = Field(
        default_factory=ResourceUsageConfig, description="Limits and usage for datasources"
    )
