"""
Usage guard for enforcing resource consumption limits.

Provides a generic ``require_usage_available`` function that endpoints call
before performing operations that consume a counted resource (minds,
datasources, questions, tokens). The guard fetches the current limits and
usage from the LimitsService and raises an HTTP 429 error when the user
has exhausted their allowance.

Usage::

    from minds.common.guards import require_usage_available, ResourceType

    await require_usage_available(limits_service, ResourceType.MINDS)
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from fastapi import HTTPException

from minds.common.logger import get_logger
from minds.schemas.limits import UNLIMITED, ResourceUsageConfig

if TYPE_CHECKING:
    from minds.services.limits import LimitsService

logger = get_logger(__name__)


class ResourceType(str, Enum):
    """Identifiers for the resource types tracked by the limits system."""

    TOKENS = "tokens"


class UsageLimitExceededError(HTTPException):
    """Raised when a user has exceeded their usage limit for a resource."""

    def __init__(self, resource: ResourceType, detail: str | None = None):
        detail = detail or f"Usage limit exceeded for {resource.value}"
        super().__init__(status_code=429, detail=detail)


def _is_limit_exceeded(usage: int, limit: int) -> bool:
    """Check whether *usage* has reached or exceeded *limit*.

    A limit of ``UNLIMITED`` (-1) means the resource is uncapped.
    """
    if limit == UNLIMITED:
        return False
    return usage >= limit


async def require_usage_available(
    limits_service: LimitsService,
    resource: ResourceType,
) -> None:
    """Ensure the user has remaining capacity for *resource*.

    Fetches the full limits/usage snapshot from *limits_service*, then
    checks both the **monthly** (billing-cycle) and **lifetime** limits
    for the requested *resource*.

    Args:
        limits_service: Service that provides the current limits and usage.
        resource: The type of resource to check.

    Raises:
        UsageLimitExceededError: If the user has exhausted their allowance.
    """
    limits = await limits_service.get_mind_limits()
    config: ResourceUsageConfig = getattr(limits, resource.value)

    if _is_limit_exceeded(usage=config.usage.billing_cycle, limit=config.limit.monthly):
        logger.warning(
            f"Monthly usage limit exceeded for {resource.value}: "
            f"usage={config.usage.billing_cycle}, limit={config.limit.monthly}"
        )
        raise UsageLimitExceededError(
            resource,
            detail=(
                f"Monthly limit exceeded for {resource.value}: {config.usage.billing_cycle}/{config.limit.monthly}"
            ),
        )

    if _is_limit_exceeded(usage=config.usage.lifetime, limit=config.limit.lifetime):
        logger.warning(
            f"Lifetime usage limit exceeded for {resource.value}: "
            f"usage={config.usage.lifetime}, limit={config.limit.lifetime}"
        )
        raise UsageLimitExceededError(
            resource,
            detail=(f"Lifetime limit exceeded for {resource.value}: {config.usage.lifetime}/{config.limit.lifetime}"),
        )
