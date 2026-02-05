"""
Minds FastAPI Application Server.

This module sets up the FastAPI application with middleware, routing,
and all necessary configurations for the Minds service.
"""

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from minds.api.v1.router import api_router as v1_router
from minds.common.launch_darkly import close_launchdarkly, init_launchdarkly
from minds.common.logger import setup_logging

# Set up logging
logger = setup_logging()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Initialize after worker process starts (avoids fork-related issues).
    init_launchdarkly()
    try:
        yield
    finally:
        close_launchdarkly()


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application instance
    """
    # Create FastAPI app
    app = FastAPI(
        title="Minds API",
        description="FastAPI-based service providing OpenAI-compatible chat completions with MindsDB integration",
        version="1.0.0",
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

    # Include v1 API routes
    app.include_router(v1_router)

    logger.info("Minds application created successfully")
    return app


_app: FastAPI | None = None
_app_lock = threading.Lock()


def get_app() -> FastAPI:
    """
    Lazily create (and cache) the FastAPI app instance.

    This avoids constructing the app at import-time which is important when running
    Uvicorn with `--factory` (otherwise the module-level app would be created and
    then the factory would create a second instance).
    """
    global _app
    if _app is not None:
        return _app
    with _app_lock:
        if _app is None:
            _app = create_app()
    return _app


async def app(scope, receive, send):
    """
    ASGI entrypoint for `uvicorn minds.server:app`.

    This wrapper keeps compatibility with existing run commands while allowing
    `uvicorn minds.server:create_app --factory` without duplicate initialization.
    """
    await get_app()(scope, receive, send)
