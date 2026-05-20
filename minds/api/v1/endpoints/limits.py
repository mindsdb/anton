"""
Limits API endpoints for API v1.

This module contains endpoints for handling feature limits,
providing a v1 API interface for limits management.
"""

from fastapi import APIRouter, Depends, HTTPException

from minds.api.v1.deps import get_limits_service
from minds.common.logger import get_logger
from minds.schemas.limits import MindLimitsConfig
from minds.services.limits import LimitsService

logger = get_logger(__name__)

router = APIRouter()


@router.get("/")
async def get_limits(
    limits_service: LimitsService = Depends(get_limits_service),
) -> MindLimitsConfig:
    """
    Get the limits for the current user.
    """
    try:
        limits = await limits_service.get_mind_limits()
        logger.debug(f"Limits: {limits.model_dump_json()}")
        return limits
    except Exception as e:
        logger.error(f"Error getting limits: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from None
