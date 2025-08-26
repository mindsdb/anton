"""
API v1 router for the Minds API.

This module aggregates all v1 endpoints into a single router
that can be included in the main FastAPI application.
"""

from fastapi import APIRouter

from minds.api.v1.endpoints import chat, datasources, health, minds

# Create the v1 API router
api_router = APIRouter(prefix="/api/v1")

# Include all endpoint routers
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(minds.router, prefix="/minds", tags=["minds"])
api_router.include_router(datasources.router, prefix="/datasources", tags=["datasources"])
