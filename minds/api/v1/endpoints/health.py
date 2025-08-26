"""
Health check endpoints for API v1.

This module contains endpoints for monitoring the health and status
of the Minds API service.
"""

from fastapi import APIRouter

from minds.common.logger import setup_logging

# Set up logging
logger = setup_logging()

# Create router for health endpoints
router = APIRouter()


@router.get("/")
async def healthz():
    """
    Health check endpoint.
    
    Returns:
        dict: Status of the service
    """
    logger.debug("Health check requested")
    return {"status": "ok", "version": "v1"}


@router.get("/ready")
async def readiness():
    """
    Readiness check endpoint.
    
    Returns:
        dict: Readiness status of the service
    """
    logger.debug("Readiness check requested")
    # In a real implementation, you might check database connectivity,
    # external service availability, etc.
    return {"status": "ready", "version": "v1"}


@router.get("/live")
async def liveness():
    """
    Liveness check endpoint.
    
    Returns:
        dict: Liveness status of the service
    """
    logger.debug("Liveness check requested")
    return {"status": "alive", "version": "v1"}
