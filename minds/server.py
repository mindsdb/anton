"""
Minds FastAPI Application Server.

This module sets up the FastAPI application with middleware, routing,
and all necessary configurations for the Minds service.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from minds.api.v1.router import api_router as v1_router
from minds.common.logger import setup_logging
from minds.common.settings.app_settings import get_app_settings
from minds.common.statsig import init_statsig, shutdown_statsig

# Set up logging
logger = setup_logging()


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application instance
    """

    settings = get_app_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        statsig = init_statsig(settings=settings)
        app.state.statsig = statsig
        yield
        # Shutdown
        shutdown_statsig()

    # Create FastAPI app
    app = FastAPI(
        title="Minds API",
        description="FastAPI-based service providing OpenAI-compatible chat completions with MindsDB integration",
        version="1.9.1",
        lifespan=lifespan,
    )

    # Configure CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow any origin. This will be controlled by the ingress controller.
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=3600,
    )

    # Include v1 API routes at the canonical /v1/* mount, plus a legacy
    # /api/v1/* alias for older clients (mdb-ai axios fallback, anton CLI,
    # cowork, Terraform-managed CI integration tests). Drop the alias once
    # all callers are migrated.
    app.include_router(v1_router)
    app.include_router(v1_router, prefix="/api")

    logger.info("Minds application created successfully")
    return app


# Create the FastAPI application instance for ASGI
app = create_app()
