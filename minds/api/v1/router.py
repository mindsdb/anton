"""
API v1 router for the Minds API.

This module aggregates all v1 endpoints into a single router
that can be included in the main FastAPI application.
"""

# from collections.abc import Callable
# from typing import Any

# from fastapi import APIRouter as FastAPIRouter
from fastapi import APIRouter

from minds.api.v1.endpoints import chat, datasources, health, minds, tree


# class APIRouter(FastAPIRouter):
#     def add_api_route(
#         self, path: str, endpoint: Callable[..., Any], *, include_in_schema: bool = True, **kwargs: Any
#     ) -> None:
#         alternate_path = path[:-1] if path.endswith("/") else path + "/"
#         super().add_api_route(alternate_path, endpoint, include_in_schema=False, **kwargs)
#         return super().add_api_route(path, endpoint, include_in_schema=include_in_schema, **kwargs)


# Create the v1 API router
api_router = APIRouter(prefix="/api/v1")

# Include all endpoint routers
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(minds.router, prefix="/minds", tags=["minds"])
api_router.include_router(datasources.router, prefix="/datasources", tags=["datasources"])
api_router.include_router(tree.router, prefix="/tree", tags=["tree"])
