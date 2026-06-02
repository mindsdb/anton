"""
API v1 router — inference-only endpoints.

Aggregates inference endpoints (/v1/chat/completions, /v1/responses, /v1/models)
and health checks.
"""

from fastapi import APIRouter

from minds.api.v1.endpoints import (
    chat,
    health,
    models,
    responses,
)

api_router = APIRouter(prefix="/v1")

api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(responses.router, prefix="/responses", tags=["responses"])
