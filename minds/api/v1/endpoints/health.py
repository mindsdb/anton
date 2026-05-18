"""
Health check endpoints for API v1.

This module contains endpoints for monitoring the health and status
of the Minds API service.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlmodel import Session

from minds.common.logger import get_logger
from minds.db.pg_session import get_session

logger = get_logger(__name__)

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
async def readiness(session: Session = Depends(get_session)):
    """
    Readiness check. Verifies the service can talk to its primary dependency
    (Postgres) by issuing a cheap `SELECT 1`. K8s gates traffic on this — so
    if the DB is unreachable, the pod is removed from the Service endpoints
    instead of returning errors to users.
    """
    logger.debug("Readiness check requested")
    try:
        session.execute(text("SELECT 1"))
    except Exception as e:
        logger.warning(f"Readiness check failed: DB unreachable: {e}")
        raise HTTPException(status_code=503, detail="database unreachable") from e
    return {"status": "ready", "version": "v1"}


@router.get("/live")
async def liveness():
    """
    Liveness check. Stays minimal — succeeds as long as the event loop is
    responsive. K8s restarts the pod if this fails, so it must not depend on
    external services (DB hiccup should not trigger restarts).
    """
    logger.debug("Liveness check requested")
    return {"status": "alive", "version": "v1"}
